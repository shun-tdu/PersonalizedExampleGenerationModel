import os
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import sqlite3
from sklearn.model_selection import train_test_split

# --- パス設定とインポート ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from DataPreprocess.DataPreprocessForOverFitting import load_processed_data
from HybridModelExperiment.models.hybrid_transfomer_v1_0 import HybridModel


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

def train_model(config_path: str, experiment_id: int, db_path: str):
    """
    単一の学習タスクを実行し，結果をDBに報告する．
    """
    # 1. 設定の読み込み
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

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
        trajectories, conditions = load_processed_data(config['data']['data_path'])
        trajectories_tensor = torch.tensor(trajectories, dtype=torch.float32)
        conditions_tensor = torch.tensor(conditions, dtype=torch.float32)

        # 訓練データと検証データを8:2に分割
        tr_traj, val_traj, tr_cond, val_cond = train_test_split(
            trajectories_tensor, conditions_tensor, test_size=0.2, random_state=42
        )
        train_dataset = TensorDataset(tr_traj, tr_cond)
        val_dataset = TensorDataset(val_traj, val_cond)

        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])

        # 4. モデル、オプティマイザ、スケジューラの初期化
        model = HybridModel(**config['model']).to(device)
        optimizer = optim.AdamW([
            {'params': model.low_freq_model.parameters(), 'lr': config['training']['lr_low_freq']},
            {'params': model.high_freq_model.parameters(), 'lr': config['training']['lr_high_freq']},
            {'params': model.decomposer.parameters(), 'lr': config['training']['lr_decomposer']}
        ], weight_decay=config['training']['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config['training']['scheduler_T_0'], T_mult=config['training']['scheduler_T_mult'],
            eta_min=config['training']['scheduler_eta_min']
        )
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=config['training']['warmup_epochs']
        )

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
            for trajectories, conditions in progress_bar:
                trajectories, conditions = trajectories.to(device), conditions.to(device)

                optimizer.zero_grad()
                losses = model(trajectories, conditions)
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
                for trajectories, conditions in val_loader:
                    trajectories, conditions = trajectories.to(device), conditions.to(device)
                    losses = model(trajectories, conditions)
                    epoch_val_losses.append(losses['total_loss'].item())

            current_val_loss = np.mean(epoch_val_losses)
            history['val_loss'].append(current_val_loss)
            print(f"Epoch {epoch + 1}: Train Loss: {history['train_loss'][-1]:.4f}, Val Loss: {current_val_loss:.4f}")

            # 学習率の更新
            if epoch < config['training']['warmup_epochs']:
                warmup_scheduler.step()
            else:
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

        # DBに完了報告
        update_db(db_path, experiment_id, {
            'status': 'completed',
            'end_time': datetime.now().isoformat(),
            'final_total_loss': history['train_loss'][-1],
            'best_val_loss': best_val_loss,
            'model_path': final_model_path,
            'image_path': plot_path
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
        os.path.join(output_dir, 'plots')
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def plot_training_curves(train_losses, val_losses, learning_rates, save_path):
    """学習曲線と検証曲線をプロット"""
    fig, (ax1, ax2) = plt.subplot(1, 2, figsize=(15,5))

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