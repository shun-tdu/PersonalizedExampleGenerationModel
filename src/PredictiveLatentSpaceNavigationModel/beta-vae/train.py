import os
import sys
import argparse
import warnings
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
from scipy.stats import pearsonr
import seaborn as sns

# ===== numpy型のSQLite自動変換を登録 =====
sqlite3.register_adapter(np.float32, float)
sqlite3.register_adapter(np.float64, float)
sqlite3.register_adapter(np.int32, int)
sqlite3.register_adapter(np.int64, int)
sqlite3.register_adapter(np.bool_, bool)

# --- パス設定とインポート ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from models.beta_vae import BetaVAE
from src.PredictiveLatentSpaceNavigationModel.DataPreprocess.data_preprocess import calculate_jerk, calculate_path_efficiency, calculate_approach_angle, calculate_sparc


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
    # DB更新前にデバッグ情報を出力
    # print(f"=== DB更新デバッグ ===")
    # for key, value in data.items():
    #     print(f"{key}: {type(value)} - {repr(value)}")

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
        for trajectories, subject_ids, is_expert in tqdm(test_loader, desc="潜在変数抽出中"):
            trajectories = trajectories.to(device)

            # エンコード
            encoded = model.encode(trajectories)
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
        'is_expert': np.concatenate(all_is_expert),
        'reconstructions': np.vstack(all_reconstructions),
        'originals': np.vstack(all_originals)
    }

def create_performance_dataframe(test_df: pd.DataFrame):
    """パフォーマンス指標を計算"""

    performance_data = []
    for (subject_id, trial_num), trial_df in test_df.groupby(['subject_id', 'trial_num']):
        trajectory = trial_df[['HandlePosX', 'HandlePosY']].values
        velocity = trial_df[['HandleVelX', 'HandleVelY']].values if 'HandleVelX' in trial_df.columns else np.zeros_like(
            trajectory)
        acceleration = trial_df[
            ['HandleAccX', 'HandleAccY']].values if 'HandleAccX' in trial_df.columns else np.zeros_like(trajectory)
        time_stamps = trial_df['Timestamp'].values if 'Timestamp' in trial_df.columns else np.arange(
            len(trajectory)) * 0.01

        # 目標位置
        if 'TargetEndPosX' in trial_df.columns and 'TargetEndPosY' in trial_df.columns:
            target_end_pos = trial_df[['TargetEndPosX', 'TargetEndPosY']].iloc[-1].values
        else:
            target_end_pos = trajectory[-1]  # 最終位置を目標位置とする

        dt = np.mean(np.diff(time_stamps)) if len(time_stamps) > 1 else 0.01
        if np.isnan(dt) or dt == 0:
            dt = 0.01

        performance_data.append({
            'subject_id': subject_id,
            'trial_num': trial_num,
            'trial_time': time_stamps[-1] - time_stamps[0],
            'trial_error': np.linalg.norm(trajectory[-1] - target_end_pos),
            'jerk': calculate_jerk(acceleration, dt),
            'path_efficiency': calculate_path_efficiency(trajectory),
            'approach_angle': calculate_approach_angle(trajectory),
            'sparc': calculate_sparc(velocity, dt),
            'is_expert': trial_df['is_expert'].iloc[0] if 'is_expert' in trial_df.columns else 0
        })

    return pd.DataFrame(performance_data)

def quantitative_evaluation(latent_data: dict, test_df: pd.DataFrame, output_dir: str):
    """定量的評価を実行"""
    print("=== 定量的評価開始 ===")

    # 1. 再構成性能
    mse = np.mean((latent_data['originals'] - latent_data['reconstructions']) ** 2)
    print(f"再構成MSE: {mse:.6f}")

    # 2. 事前計算されたパフォーマンス指標を抽出
    perf_cols = [col for col in test_df.columns if col.startswith('perf_')]
    if not perf_cols:
        raise ValueError("パフォーマンス指標列（perf_で始まる列）が見つかりません。data_preprocess.pyでパフォーマンス指標を計算してください。")
    
    print(f"利用可能なパフォーマンス指標: {perf_cols}")
    
    # 試行ごとのパフォーマンス指標を取得
    performance_df = test_df.groupby(['subject_id', 'trial_num']).first()[['is_expert'] + perf_cols].reset_index()
    
    print(f"パフォーマンス指標データの形状: {performance_df.shape}")

    # 3. 潜在変数をDataFrameに変換
    z_skill_df = pd.DataFrame(
        latent_data['z_skill'],
        columns=[f'z_skill_{i}' for i in range(latent_data['z_skill'].shape[1])]
    )
    z_style_df = pd.DataFrame(
        latent_data['z_style'],
        columns=[f'z_style_{i}' for i in range(latent_data['z_style'].shape[1])]
    )

    # データの長さが一致するかチェック
    if len(performance_df) != len(z_skill_df):
        warnings.warn(
            f"警告: パフォーマンス指標の試行数 ({len(performance_df)}) と "
            f"潜在変数の試行数 ({len(z_skill_df)}) が一致しません。相関分析が不正確になる可能性があります。"
        )

    # パフォーマンス指標と潜在変数を結合
    analysis_df = pd.concat([
        performance_df.reset_index(drop=True),
        z_skill_df,
        z_style_df
    ], axis=1)

    print(f"分析用データフレームを作成しました。形状: {analysis_df.shape}")

    # --- 4. 分離表現の検証（相関分析） ---
    print("\n--- スキル潜在変数とパフォーマンス指標の相関分析 ---")
    correlations = {}
    # 'perf_'プレフィックスを除いた指標名で相関分析
    metric_names = ['trial_time', 'trial_error', 'jerk', 'path_efficiency', 'approach_angle', 'sparc', 'trial_variability']
    available_metrics = [metric for metric in metric_names if f'perf_{metric}' in analysis_df.columns]
    z_skill_cols = [col for col in analysis_df.columns if 'z_skill' in str(col)]

    for metric_name in available_metrics:
        metric_col = f'perf_{metric_name}'
        correlations[metric_name] = []
        
        for z_col in z_skill_cols:
            # NaN値を除外して計算
            subset = analysis_df[[metric_col, z_col]].dropna()

            # 定数値の列でないことを確認（標準偏差がほぼゼロでないか）
            if len(subset) > 1 and subset[metric_col].std() > 1e-6 and subset[z_col].std() > 1e-6:
                corr, p_value = pearsonr(subset[z_col], subset[metric_col])
            else:
                corr, p_value = 0.0, 1.0  # 計算不能な場合は無相関とする

            correlations[metric_name].append((corr, p_value))

        # この指標で最も相関が高かった次元を表示
        valid_corrs = [(abs(c[0]), c) for c in correlations[metric_name] if not np.isnan(c[0])]
        if valid_corrs:
            max_abs_corr, (max_corr_val, max_corr_p) = max(valid_corrs)
            max_abs_corr_idx = [i for i, c in enumerate(correlations[metric_name]) if abs(c[0]) == max_abs_corr][0]
            print(
                f"{metric_name}の最大相関: z_skill_{max_abs_corr_idx} (r={max_corr_val:.4f}, p={max_corr_p:.4f})")
        else:
            print(f"{metric_name}の最大相関: 0.0000 (p=1.0000) - 相関計算不可")

    # --- 5. 結果のパッケージ化 ---
    eval_results = {
        'reconstruction_mse': mse,
        'correlations': correlations,
        'performance_data': performance_df.to_dict('records')
    }

    print("=" * 20 + " 定量的評価完了 " + "=" * 20)

    return eval_results, analysis_df


def visualize_latent_space(
        analysis_df: pd.DataFrame,
        save_path: str
    ):
    """
    分析用DataFrameから潜在空間の可視化グラフを生成し、指定されたパスに保存する。

    Args:
        analysis_df (pd.DataFrame): パフォーマンス指標と潜在変数が結合されたDataFrame。
        save_path (str): 生成したグラフを保存するファイルパス (例: './plots/latent_space.png')。
    """
    print("=" * 20 + " 潜在空間の可視化開始 " + "=" * 20)

    # --- 1. 潜在変数の抽出と次元削減 ---
    # z_styleとz_skillの列名を特定
    z_style_cols = [col for col in analysis_df.columns if 'z_style' in str(col)]
    z_skill_cols = [col for col in analysis_df.columns if 'z_skill' in str(col)]

    if not z_style_cols or not z_skill_cols:
        print("エラー: 潜在変数の列が見つかりません。")
        return

    # PCAで2次元に圧縮
    pca_style = PCA(n_components=2)
    z_style_2d = pca_style.fit_transform(analysis_df[z_style_cols])

    pca_skill = PCA(n_components=2)
    z_skill_2d = pca_skill.fit_transform(analysis_df[z_skill_cols])

    # 可視化のためにDataFrameに次元削減後のデータを追加
    analysis_df['z_style_pc1'] = z_style_2d[:, 0]
    analysis_df['z_style_pc2'] = z_style_2d[:, 1]
    analysis_df['z_skill_pc1'] = z_skill_2d[:, 0]
    analysis_df['z_skill_pc2'] = z_skill_2d[:, 1]

    # --- 2. グラフの描画 ---
    # スタイルの設定
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle("Latent Space Visualization", fontsize=16)

    # --- グラフ1: z_style空間を被験者IDで色分け ---
    sns.scatterplot(
        data=analysis_df,
        x='z_style_pc1',
        y='z_style_pc2',
        hue='subject_id',
        palette='viridis',  # 色覚多様性に対応したカラーマップ
        style='subject_id',  # マーカーの形も変える
        s=100,  # マーカーのサイズ
        alpha=0.8,
        ax=axes[0]
    )
    axes[0].set_title('z_style Space (colored by Subject ID)')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left')

    # --- グラフ2: z_skill空間を熟達度ラベルで色分け ---
    # 'is_expert'列がない場合はスキップ
    if 'is_expert' in analysis_df.columns:
        analysis_df['Expertise'] = analysis_df['is_expert'].map({0: 'Novice', 1: 'Expert'})
        sns.scatterplot(
            data=analysis_df,
            x='z_skill_pc1',
            y='z_skill_pc2',
            hue='Expertise',
            palette={'Novice': 'blue', 'Expert': 'red'},
            style='Expertise',
            s=100,
            alpha=0.8,
            ax=axes[1]
        )
    axes[1].set_title('z_skill Space (colored by Expertise)')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    axes[1].legend(title='Expertise')

    # --- グラフ3: z_skill空間をパフォーマンス指標でグラデーション ---
    # perf_trial_timeをスコアとして使用
    performance_col = 'perf_trial_time' if 'perf_trial_time' in analysis_df.columns else None
    
    if performance_col and not analysis_df[performance_col].isna().all():
        scatter = sns.scatterplot(
            data=analysis_df,
            x='z_skill_pc1',
            y='z_skill_pc2',
            hue=performance_col,
            palette='plasma',  # グラデーション用のカラーマップ
            s=100,
            alpha=0.8,
            ax=axes[2]
        )
        axes[2].set_title('z_skill Space (colored by Trial Time)')
        axes[2].set_xlabel('Principal Component 1')
        axes[2].set_ylabel('Principal Component 2')
        # カラーバーの追加
        norm = plt.Normalize(analysis_df[performance_col].min(), analysis_df[performance_col].max())
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes[2], label='Trial Time (s)')
    else:
        # パフォーマンス指標がない場合は熟達度で色分け
        if 'is_expert' in analysis_df.columns:
            sns.scatterplot(
                data=analysis_df,
                x='z_skill_pc1',
                y='z_skill_pc2',
                hue='is_expert',
                palette={0: 'blue', 1: 'red'},
                s=100,
                alpha=0.8,
                ax=axes[2]
            )
            axes[2].set_title('z_skill Space (colored by Expertise)')
        else:
            # 単色でプロット
            axes[2].scatter(analysis_df['z_skill_pc1'], analysis_df['z_skill_pc2'], alpha=0.8)
            axes[2].set_title('z_skill Space')
        axes[2].set_xlabel('Principal Component 1')
        axes[2].set_ylabel('Principal Component 2')

    # --- 3. 保存と終了 ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # suptitleとの重なりを調整
    # 保存先のディレクトリがなければ作成
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"潜在空間の可視化グラフを保存しました: {save_path}")

    plt.close()
    print("=" * 20 + " 潜在空間の可視化完了 " + "=" * 20)

def generate_qualitative_trajectories(model, latent_data, output_dir, device, save_path=None):
    """定性評価用の軌道生成"""
    print("=== 定性評価用軌道生成開始 ===")
    
    if save_path is None:
        save_path = os.path.join(output_dir, 'generated_trajectories.png')

    model.eval()

    # 代表的な被験者を選択(最初の被験者)
    subject_ids = latent_data['subject_ids']
    # subject_idsを平坦化
    flat_subject_ids = []
    for batch in subject_ids:
        if isinstance(batch, (list, tuple, np.ndarray)):
            flat_subject_ids.extend(batch)
        else:
            flat_subject_ids.append(batch)
    
    unique_subjects = list(set(flat_subject_ids))
    representative_subject = unique_subjects[0]

    # 代表被験者のz_styleの平均を計算
    subject_mask = np.array([sid == representative_subject for sid in flat_subject_ids])
    z_style_mean = np.mean(latent_data['z_style'][subject_mask], axis=0)
    z_style_tensor = torch.tensor(z_style_mean, dtype=torch.float32).unsqueeze(0).to(device)

    # スキル軸の定義(主成分分析で最大分散方向を求める)
    # todo 最も相関の強かった次元を用いた方がストーリー的に良い
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"定性評価用軌道生成完了: {save_path}")

def run_evaluation(model, test_loader, test_df, output_dir, device, experiment_id):
    """評価プロセス全体を実行"""
    print("\n" + "="*50)
    print("評価開始")
    print("="*50)

    # 潜在変数抽出
    latent_data = extract_latent_variables(model, test_loader, device)

    # 定量的評価
    eval_results, performance_df_with_latents = quantitative_evaluation(latent_data, test_df, output_dir)

    # 潜在空間可視化
    latent_viz_path = os.path.join(output_dir, 'plots', f'latent_space_visualization_exp{experiment_id}.png')
    visualize_latent_space(performance_df_with_latents, latent_viz_path)

    # 定性的評価用軌道生成
    traj_gen_path = os.path.join(output_dir, 'plots', f'generated_trajectories_exp{experiment_id}.png')
    generate_qualitative_trajectories(model, latent_data, eval_results, device, traj_gen_path)

    # 評価結果をファイルに保存
    import json
    eval_results_path = os.path.join(output_dir, 'results', f'evaluation_results_exp{experiment_id}.json')
    os.makedirs(os.path.dirname(eval_results_path), exist_ok=True)
    
    # JSON serializable形式に変換
    def convert_to_serializable(obj):
        """NumPy型やその他の非JSON型をPython標準型に変換"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = {
        'reconstruction_mse': float(eval_results['reconstruction_mse']),
        'correlations': {k: [(float(corr) if not np.isnan(corr) else 0.0, 
                             float(p) if not np.isnan(p) else 1.0) for corr, p in v] 
                        for k, v in eval_results['correlations'].items()},
        'performance_data': convert_to_serializable(eval_results['performance_data'])
    }
    
    with open(eval_results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    # パスを追加
    eval_results['latent_visualization_path'] = latent_viz_path
    eval_results['generated_trajectories_path'] = traj_gen_path
    eval_results['evaluation_results_path'] = eval_results_path

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
            eval_results = run_evaluation(model, test_loader, test_df, output_dir, device, experiment_id)

            # 相関結果をJSON形式で保存
            import json
            skill_correlations_json = json.dumps({
                k: [(float(corr), float(p)) for corr, p in v] 
                for k, v in eval_results['correlations'].items()
            })

            # 評価結果をDBに記録
            update_db(db_path, experiment_id, {
                'status': 'completed',
                'end_time': datetime.now().isoformat(),
                'final_total_loss': history['train_loss'][-1],
                'best_val_loss': best_val_loss,
                'model_path': final_model_path,
                'image_path': plot_path,
                'reconstruction_mse': eval_results['reconstruction_mse'],
                'latent_visualization_path': eval_results['latent_visualization_path'],
                'generated_trajectories_path': eval_results['generated_trajectories_path'],
                'evaluation_results_path': eval_results['evaluation_results_path'],
                'skill_correlations': skill_correlations_json,
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