import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader,  Dataset
from torch.optim.lr_scheduler import ChainedScheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import sqlite3
from  sklearn.decomposition import PCA
from  sklearn.manifold import TSNE
from scipy.stats import pearsonr
import seaborn as sns

from src.DataPreprocess.RawDataVisualizer import subject_ids

# --- パス設定とインポート ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from models.beta_vae import BetaVAE

class TrajectoryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = 100, feature_cols=['HandlePosX', 'HandlePosY']):
        self.seq_len = seq_len
        self.feature_cols = feature_cols

        # 試行毎にデータをグループ化
        self.trials = list(df.groupby(['subject_id', 'trial_num']))

    def __len__(self) -> int:
        """データセットの総試行数を返す"""
        return len(self.trials)

    def __getitem__(self, idx) -> tuple:
        """
        指定されたインデックスのデータを取得する

        :return: tuple(軌道テンソル, 被験者ID, 熟練度ラベル)
        """
        (subject_id, _), trial_df = self.trials[idx]

        # --- 各試行データに対する前処理 ---
        # 1. 軌道データをNumpy配列に変換
        trajectory_abs = trial_df[self.feature_cols].values

        # 2. 差分計算
        trajectory_diff = np.diff(trajectory_abs, axis=0)
        trajectory_diff = np.insert(trajectory_diff, 0, [0, 0], axis=0)

        # 3. 固定長化 (パディング/切り捨て)
        if len(trajectory_diff) > self.seq_len:
            processed_trajectory = trajectory_diff[:self.seq_len]
        else:
            padding = np.zeros((self.seq_len - len(trajectory_diff), len(self.feature_cols)))
            processed_trajectory = np.vstack([trajectory_diff, padding])

        # 4. ラベルを取得
        is_expert = trial_df['is_expert'].iloc[0]

        # 5. テンソルに変換して返す
        return (
            torch.tensor(processed_trajectory, dtype=torch.float32),
            subject_id,  # 分析用にIDも文字列として返す
            torch.tensor(is_expert, dtype=torch.long)
        )


def create_dataloaders(master_data_path: str, seq_len: int, batch_size: int, random_seed:int = 42) -> tuple:
    """
    マスターデータファイルから学習・検証・テスト用のDataLoaderを一度に作成する。

    Args:
        master_data_path (str): 'master_data.parquet'へのパス。
        seq_len (int): 軌道の固定長。
        batch_size (int): バッチサイズ。

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # データを読み込み
    try:
        master_df = pd.read_parquet(master_data_path)
    except FileNotFoundError:
        print(f"エラー: ファイル '{master_data_path}' が見つかりません。")
        return None, None, None

    # --- 被験者ベースでのデータ分割 ---
    np.random.seed(random_seed)
    subject_ids = master_df['subject_id'].unique()
    np.random.shuffle(subject_ids)  # 毎回ランダムに分割

    # 例: 4人学習、1人検証、1人テスト
    # 被験者数が少ないので、人数をハードコーディング
    if len(subject_ids) < 3:
        raise ValueError("データセットの分割には最低3人の被験者が必要です。")

    train_ids = subject_ids[:-2]
    val_ids = subject_ids[-2:-1]
    test_ids = subject_ids[-1:]

    train_df = master_df[master_df['subject_id'].isin(train_ids)]
    val_df = master_df[master_df['subject_id'].isin(val_ids)]
    test_df = master_df[master_df['subject_id'].isin(test_ids)]

    print(f"データ分割: 学習用={len(train_ids)}人, 検証用={len(val_ids)}人, テスト用={len(test_ids)}人")

    # --- DatasetとDataLoaderの作成 ---
    train_dataset = TrajectoryDataset(train_df, seq_len=seq_len)
    val_dataset = TrajectoryDataset(val_df, seq_len=seq_len)
    test_dataset = TrajectoryDataset(test_df, seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, test_df


def update_db(db_path: str, experiment_id: int, data: dict):
    """データベースの指定された実験IDのレコードを更新する"""
    try:
        with sqlite3.connect(db_path) as conn:
            set_clause = ", ".join([f"{key} = ?" for key in data.keys()])
            query = f"UPDATE experiments SET {set_clause} WHERE id = ?"
            values = tuple(data.values()) + (experiment_id, )
            conn.execute(query, values)
            conn.commit()
    except Exception as e:
        print(f"!!! DB更新エラー (ID: {experiment_id}): {e} !!!")


def validate_config(config:dict) -> None:
    """設定ファイルの妥当性を検証"""
    required_sections = {
        'data': ['data_path'],
        'model': ['input_dim', 'seq_len', 'hidden_dim', 'style_latent_dim', 'skill_latent_dim'],
        'training': ['batch_size', 'num_epochs', 'lr'],
        'logging': ['output_dir']
    }

    for section, keys in required_sections.items():
        if section not in config:
            raise ValueError(f"設定ファイルに'{section}'セクションがありません")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"'{section}.{key}'が設定されていません")

    # ファイル存在チェック（オプション）
    data_path = config['data']['data_path']
    if not os.path.exists(data_path):
        raise ValueError(f"データファイルが見つかりません: {data_path}")

    # 値の妥当性をチェック
    model_config = config['model']
    training_config = config['training']

    if model_config['seq_len'] <= 0:
        raise ValueError("seq_lenは正の整数である必要があります")
    if training_config['batch_size'] <= 0:
        raise ValueError("batch_sizeは正の整数である必要があります")
    if training_config['lr'] <= 0:
        raise ValueError("学習率は正の値である必要があります")
    if training_config['num_epochs'] <= 0:
        raise ValueError("num_epochsは正の整数である必要があります")

def extract_latent_variables(model, test_loader, device):
    """テストデータから潜在変数を抽出"""
    model.eval()
    all_z_style = []
    all_z_skill =[]
    all_subject_ids = []
    all_is_expert = []
    all_reconstructions = []
    all_originals = []

    with torch.no_grad():
        for trajectories, subject_ids, is_expert in tqdm(test_loader, decs="潜在変数抽出中"):
            trajectories = trajectories.to(device)

            # エンコード
            encoded = model.encoder(trajectories)
            z_style = encoded['z_style']
            z_skill = encoded['z_skill']

            # 再構成
            reconstructed = model.decode(z_style, z_skill)

            # CPUに移動して保存
            all_z_style.append(z_style.cpu().numpy())
            all_z_skill.append(z_skill.cpu().numpy())
            all_subject_ids.append(subject_ids)
            all_is_expert.append(is_expert.cpu().numpy())
            all_reconstructions.append(reconstructed.cpu().numpy())
            all_originals.append(trajectories.cpu().numpy())

    return {
        'z_style': np.vstack(all_z_style),
        'z_skill': np.vstack(all_z_skill),
        'subject_ids': all_subject_ids,
        'is_expert': np.concatenate(is_expert),
        'reconstructions': np.vstack(all_reconstructions),
        'originals': np.vstack(all_originals)
    }

def quantitative_evaluation(latent_data: dict, test_df: pd.DataFrame, output_dir: str):
    """定量的評価を実行"""
    print("=== 定量的評価開始 ===")

    # 1. 再構成性能
    mse = np.mean((latent_data['original'] - latent_data['reconstructions']) ** 2)
    print(f"再構成MSE: {mse:.6f}")

    # 2. 分離表現の検証 dfから取ってくる

    # z_skillの各次元とパフォーマンス指標の相関
    correlations = {}
    z_skill = latent_data['z_skill']

    for metric_name, metric_value in performance_metrics.items():
        correlations[metric_name] = []
        for dim in range(z_skill.shape[1]):
            if len(metric_value) == len(z_skill):
                corr, p_value = pearsonr(z_skill[:, dim], metric_value)
                correlations[metric_name].append((corr, p_value))
            else:
                correlations[metric_name].append((0.0, 1.0))

    # 結果を保存
    eval_results = {
        'reconstruction_mse': mse,
        'correlations': correlations
    }

    # todo 結果をDBに保存

def visualize_latent_space(latent_data, test_df, output_dir):
    """潜在空間の可視化"""
    print("=== 潜在空間可視化開始 ===")

    # todo 保存したデータをDBに紐づけ

    z_style = latent_data['z_style']
    z_skill = latent_data['z_skill']
    subject_ids = latent_data['subject_ids']
    is_expert = latent_data['is_expert']

    # PCAで２次元に圧縮
    pca_style = PCA(n_components=2)
    pca_skill = PCA(n_components=2)

    z_style_2d = pca_style.fit_transform(z_style)
    z_skill_2d = pca_skill.fit_transform(z_skill)

    # todo 総合スキルスコアの計算
    skill_scores = np.mean(z_skill, axis=1)  # 簡単な平均値をスコアとする

    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. z_style空間: 被験者IDで色分け
    unique_subjects = list(set(subject_ids))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_subjects)))
    subject_color_map = {subj: colors[i] for i, subj in enumerate(unique_subjects)}

    for subject in unique_subjects:
        mask = [sid == subject for sid in subject_ids]
        axes[0].scatter(z_style_2d[mask, 0], z_style_2d[mask, 1],
                        c=[subject_color_map[subject]],
                        label=f'Subject {subject}', alpha=0.7)

    axes[0].set_title('z_style space(per subject) ')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. z_skill空間: 熟練度ラベルで色分け
    expert_mask = is_expert == 1
    novice_mask = is_expert == 0

    axes[1].scatter(z_skill_2d[novice_mask, 0], z_skill_2d[novice_mask, 1],
                    c='blue',
                    label='Novice', alpha=0.7)
    axes[1].scatter(z_skill_2d[expert_mask, 0], z_skill_2d[expert_mask, 1],
                    c='red',
                    label='Expert', alpha=0.7)

    axes[1].set_title('z_skill空間（熟達度別）')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. z_skill空間: スキルスコアでグラデーション
    scatter = axes[2].scatter(z_skill_2d[:, 0], z_skill_2d[:, 1],
                              c=skill_scores, cmap='viridis', alpha=0.7)

    axes[2].set_title('z_skill空間（Skill Score）')
    axes[2].set_xlabel('PC1')
    axes[2].set_ylabel('PC2')
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[2], label='Skill Score')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latent_space_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("潜在空間可視化完了")

def generate_qualitative_trajectories(model, latent_data, output_dir, device):
    """定性評価用の軌道生成"""
    print("=== 定性評価用軌道生成開始 ===")

    model.eval()

    # 代表的な被験者を選択(最初の被験者)
    subject_ids = latent_data['subject_ids']
    unique_subjects = list(set(subject_ids))
    representative_subject = unique_subjects[0]

    # 代表被験者のz_styleの平均を計算
    subject_mask = [sid == representative_subject for sid in subject_ids]
    z_style_mean = np.mean(latent_data['z_style'][subject_mask], axis=0)
    z_style_tensor = torch.tensor(z_style_mean, dtype=torch.float32).unsqueeze(0).to(device)

    # スキル軸の定義(主成分分析で最大分散方向を求める)
    z_skill = latent_data['z_skill']
    pca = PCA(n_components=1)
    pca.fit(z_skill)
    skill_direction = pca.components_[0] # 第1主成分

    # z_skillの平均点を基準とする
    z_skill_mean = np.mean(z_skill, axis=0)

    # α値の範囲
    alpha_values = np.arange(-0.2, 0.31, 0.1)

    generated_trajectories = []

    with torch.no_grad():
        for alpha in alpha_values:
            # z_skillを計算
            z_skill_modified = z_skill_mean + alpha * skill_direction
            z_skill_tensor = torch.tensor(z_skill_modified, dtype=torch.float32).unsqueeze(0).to(device)

            generated_traj = model.decode(z_style_tensor, z_skill_tensor)
            generated_trajectories.append(generated_traj.cpu().numpy().squeeze())

    # 軌道をプロット
    fig, axes = plt.subplots(1, len(alpha_values), figsize=(20, 4))
    if len(alpha_values) == 1:
        axes = [axes]

    for i, (alpha, traj) in enumerate(zip(alpha_values, generated_trajectories)):
        # 差分から絶対座標に変換
        cumsum_traj = np.cumsum(traj, axis=0)

        axes[i].plot(cumsum_traj[:, 0], cumsum_traj[:, 1], 'b-', linewidth=2)
        axes[i].scatter(cumsum_traj[0, 0], cumsum_traj[0, 1], c='green', s=100, label='Start', zorder=5)
        axes[i].scatter(cumsum_traj[-1, 0], cumsum_traj[-1, 1], c='red', s=100, label='End', zorder=5)
        axes[i].set_title(f'α = {alpha:.1f}')
        axes[i].set_xlabel('X Position')
        axes[i].set_ylabel('Y Position')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        axes[i].set_aspect('equal')

    plt.suptitle(f'Generated Trajectories (Subject: {representative_subject})', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generated_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("定性評価用軌道生成完了")

def run_evaluation(model, test_loader, test_df, output_dir, device):
    """評価プロセス全体を実行"""
    print("\n" + "="*50)
    print("評価開始")
    print("="*50)

    # 潜在変数抽出
    latent_data = extract_latent_variables(model, test_loader, device)

    # 定量的評価
    eval_results = quantitative_evaluation(latent_data, test_df, output_dir)

    # 潜在空間可視化
    visualize_latent_space(latent_data, test_df, output_dir)

    # 定性的評価用軌道生成
    generate_qualitative_trajectories(model, latent_data, output_dir, device)

    print("="*50)
    print("評価完了")
    print("="*50)

    return eval_results

def train_model(config_path: str, experiment_id: int, db_path: str):
    """
    単一の学習タスクを実行し，結果をDBに報告する．
    """
    # 1. 設定の読み込み
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    try:
        validate_config(config)
        print("設定ファイルの検証: ✅ 正常")
    except ValueError as e:
        print(f"❌ 設定ファイルエラー: {e}")
        # DB にエラー報告（設定エラー）
        update_db(db_path, experiment_id, {
            'status': 'failed',
            'end_time': datetime.now().isoformat()
        })
        raise e

    # DBに学習開始を通知
    update_db(db_path, experiment_id, {
        'status': 'running',
        'start_time': datetime.now().isoformat()
    })

    try:
        # 2. セットアップ(ディレクトリ作成，デバイス設定)
        output_dir = config['logging']['output_dir']
        setup_directories(output_dir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"-- 実験ID: {experiment_id} | デバイス: {device} ---")

        # 3. データ準備
        try:
            train_loader, val_loader, test_loader, test_df = create_dataloaders(
                config['data']['data_path'],
                config['model']['seq_len'],
                config['training']['batch_size']
            )
            if train_loader is None:
                raise ValueError("データローダーの作成に失敗しました")
        except Exception as e:
            print(f"データ準備エラー:{e}")
            update_db(db_path,experiment_id,{
                'status': 'failed',
                'end_time': datetime.now().isoformat()
            })
            raise e

        # 4. モデル、オプティマイザ、スケジューラの初期化
        model = BetaVAE(**config['model']).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )

        main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['training']['scheduler_T_0'],
            T_mult=config['training']['scheduler_T_mult'],
            eta_min=config['training']['scheduler_eta_min']
        )
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=config['training']['warmup_epochs']
        )
        scheduler = ChainedScheduler([warmup_scheduler, main_scheduler])

        # 5. 学習ループの準備
        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = config['training'].get('patience', 10)
        history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}

        # 学習ループ
        for epoch in range(config['training']['num_epochs']):
            model.train()
            model.update_epoch(epoch, config['training']['num_epochs'])

            epoch_train_losses = []
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["num_epochs"]} [Train]')
            for trajectories, subject_id, expertise in progress_bar:
                trajectories = trajectories.to(device)

                optimizer.zero_grad()
                losses = model(trajectories)
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_grad_norm'])
                optimizer.step()

                epoch_train_losses.append(losses['total_loss'].item())
                progress_bar.set_postfix({'Loss': np.mean(epoch_train_losses)})

            history['train_loss'].append(np.mean(epoch_train_losses))
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])

            # --- 検証ループ ---
            model.eval()
            epoch_val_losses = []
            with torch.no_grad():
                for trajectories, subject_id, expertise in val_loader:
                    trajectories = trajectories.to(device)
                    losses = model(trajectories)
                    epoch_val_losses.append(losses['total_loss'].item())

            current_val_loss = np.mean(epoch_val_losses)
            history['val_loss'].append(current_val_loss)
            print(f"Epoch {epoch + 1}: Train Loss: {history['train_loss'][-1]:.4f}, Val Loss: {current_val_loss:.4f}")

            # 学習率の更新
            scheduler.step()

            # ベストモデル保存 & アーリーストップ
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                epochs_no_improve = 0
                model.save_model(os.path.join(output_dir,'checkpoints', f'best_model_exp{experiment_id}.pth'))
                print(f" -> New best model saved!")
            else:
                epochs_no_improve += 1

            if epochs_no_improve > patience:
                print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
                break

        # 6. 学習終了後，結果を保存，記録
        final_model_path = os.path.join(output_dir, 'checkpoints', f'final_model_exp{experiment_id}.pth')
        model.save_model(final_model_path)

        plot_path = os.path.join(output_dir, 'plots', f'plot_exp{experiment_id}.png')
        plot_training_curves(history['train_loss'], history['val_loss'], history['learning_rates'], plot_path)

        # 7. 評価実行
        try:
            eval_results = run_evaluation(model, test_loader, test_df, output_dir, device)

            # 評価結果をDBに記録
            update_db(db_path, experiment_id, {
                'status': 'completed',
                'end_time': datetime.now().isoformat(),
                'final_total_loss': history['train_loss'][-1],
                'best_val_loss': best_val_loss,
                'model_path': final_model_path,
                'image_path': plot_path,
                'notes': f"Reconstruction MSE: {eval_results['reconstruction_mse']:.6f}"
            })
        except Exception as eval_error:
            print(f"評価時にエラーが発生: {eval_error}")
            # DBに完了報告
            update_db(db_path, experiment_id, {
                'status': 'completed',
                'end_time': datetime.now().isoformat(),
                'final_total_loss': history['train_loss'][-1],
                'best_val_loss': best_val_loss,
                'model_path': final_model_path,
                'image_path': plot_path,
                'notes': f"Training completed, but evaluation failed: {str(eval_error)}"
            })

        print(f"--- 実験ID: {experiment_id} 正常に終了 ---")

    except Exception as e:
        print(f"!!! 実験ID: {experiment_id} 中にエラーが発生: {e} !!!")
        # DB にエラー報告
        update_db(db_path, experiment_id, {
            'status': 'failed',
            'end_time': datetime.now().isoformat()
        })
        # エラーをrun_experiments.pyに送信
        raise e

def setup_directories(output_dir):
    """出力ディレクトリを作成"""
    dirs = [
        output_dir,
        os.path.join(output_dir, 'checkpoints'),
        os.path.join(output_dir, 'plots'),
        os.path.join(output_dir, 'logs')
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def plot_training_curves(train_losses, val_losses, learning_rates, save_path):
    """学習曲線と検証曲線をプロット"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale('log')

    ax2.plot(learning_rates)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"学習曲線を保存: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="単一のタスクを実行するワーカー")
    parser.add_argument('--config', type=str, required=True, help='設定ファイルのパス')
    parser.add_argument('--experiment_id', type=int, required=True, help='実験管理DBのID')
    parser.add_argument('--db_path', type=str, required=True, help='実験管理DBのパス')

    args = parser.parse_args()

    train_model(config_path=args.config, experiment_id=args.experiment_id, db_path=args.db_path)