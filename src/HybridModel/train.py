
# CLAUDE_ADDED
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import yaml

# パスを追加
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(src_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, current_dir)

# 直接インポート
from hybrid_model import HybridTrajectoryModel
sys.path.insert(0, os.path.join(src_dir, 'DataPreprocess'))
from DataPreprocessForOverFitting import load_processed_data


def create_config():
    """デフォルト設定を作成"""
    config = {
        'model': {
            'input_dim': 2,
            'condition_dim': 5,  # MovementTime, EndpointError, Jerk, GoalX, GoalY
            'lstm_hidden_dim': 128,
            'lstm_num_layers': 2,
            'diffusion_hidden_dim': 256,
            'diffusion_num_layers': 4,
            'moving_average_window': 10,
            'num_diffusion_steps': 1000
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 50,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'scheduler_T_max': 200,
            'scheduler_eta_min': 1e-6,
            'clip_grad_norm': 1.0
        },
        'data': {
            'data_path': '../../data/Datasets/overfitting_dataset.npz'
        },
        'logging': {
            'log_interval': 10,
            'save_interval': 20,
            'output_dir': 'outputs'
        }
    }
    return config


def setup_directories(output_dir):
    """出力ディレクトリを作成"""
    dirs = [
        output_dir,
        os.path.join(output_dir, 'checkpoints'),
        os.path.join(output_dir, 'generated_trajectories'),
        os.path.join(output_dir, 'plots')
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def train_model(config_path=None):
    """モデルを学習"""
    
    # 設定を読み込み
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_config()
        # デフォルト設定を保存
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print("デフォルト設定を config.yaml に保存しました")
    
    # ディレクトリ作成
    setup_directories(config['logging']['output_dir'])
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # データ読み込み
    print("データを読み込み中...")
    data_path = config['data']['data_path']
    trajectories, conditions = load_processed_data(data_path)
    
    print(f"軌道データ形状: {trajectories.shape}")
    print(f"条件データ形状: {conditions.shape}")
    
    # データローダー準備
    trajectories_tensor = torch.tensor(trajectories, dtype=torch.float32)
    conditions_tensor = torch.tensor(conditions, dtype=torch.float32)
    dataset = TensorDataset(trajectories_tensor, conditions_tensor)
    train_loader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=0
    )
    
    # モデル初期化
    print("モデルを初期化中...")
    model = HybridTrajectoryModel(**config['model']).to(device)
    
    # モデル情報表示
    model_info = model.get_model_info()
    print("\nモデル情報:")
    for key, value in model_info.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
    
    # オプティマイザーとスケジューラー
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['scheduler_T_max'],
        eta_min=config['training']['scheduler_eta_min']
    )
    
    # 学習履歴
    train_losses = []
    low_freq_losses = []
    high_freq_losses = []
    learning_rates = []
    
    # 学習開始
    print("学習を開始します...")
    model.train()
    
    for epoch in range(config['training']['num_epochs']):
        epoch_losses = []
        epoch_low_freq_losses = []
        epoch_high_freq_losses = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["num_epochs"]}')
        
        for batch_trajectories, batch_conditions in progress_bar:
            batch_trajectories = batch_trajectories.to(device)
            batch_conditions = batch_conditions.to(device)
            
            optimizer.zero_grad()
            
            # 順方向計算
            outputs = model(batch_trajectories, batch_conditions)
            
            # 損失を取得
            total_loss = outputs['total_loss']
            low_freq_loss = outputs['low_freq_loss']
            high_freq_loss = outputs['high_freq_loss']
            
            # バックプロパゲーション
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=config['training']['clip_grad_norm']
            )
            optimizer.step()
            
            # 損失を記録
            epoch_losses.append(total_loss.item())
            epoch_low_freq_losses.append(low_freq_loss.item())
            epoch_high_freq_losses.append(high_freq_loss.item())
            
            # プログレスバー更新
            progress_bar.set_postfix({
                'Total': f'{total_loss.item():.4f}',
                'Low': f'{low_freq_loss.item():.4f}',
                'High': f'{high_freq_loss.item():.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        # エポック終了処理
        avg_loss = np.mean(epoch_losses)
        avg_low_freq_loss = np.mean(epoch_low_freq_losses)
        avg_high_freq_loss = np.mean(epoch_high_freq_losses)
        current_lr = scheduler.get_last_lr()[0]
        
        train_losses.append(avg_loss)
        low_freq_losses.append(avg_low_freq_loss)
        high_freq_losses.append(avg_high_freq_loss)
        learning_rates.append(current_lr)
        
        scheduler.step()
        
        # ログ出力
        if (epoch + 1) % config['logging']['log_interval'] == 0:
            print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}:")
            print(f"  Average Total Loss: {avg_loss:.6f}")
            print(f"  Average Low Freq Loss: {avg_low_freq_loss:.6f}")
            print(f"  Average High Freq Loss: {avg_high_freq_loss:.6f}")
            print(f"  Learning Rate: {current_lr:.2e}")
        
        # モデル保存
        if (epoch + 1) % config['logging']['save_interval'] == 0:
            checkpoint_path = os.path.join(
                config['logging']['output_dir'], 
                'checkpoints', 
                f'hybrid_model_epoch_{epoch+1}.pth'
            )
            model.save_model(checkpoint_path)
            print(f"  Model saved: {checkpoint_path}")
    
    # 最終モデル保存
    final_model_path = os.path.join(config['logging']['output_dir'], 'hybrid_model_final.pth')
    model.save_model(final_model_path)
    print(f"\n最終モデルを保存: {final_model_path}")
    
    # 学習曲線を描画・保存
    plot_training_curves(
        train_losses, low_freq_losses, high_freq_losses, learning_rates,
        save_path=os.path.join(config['logging']['output_dir'], 'plots', 'training_curves.png')
    )
    
    # 学習統計を保存
    training_stats = {
        'config': config,
        'model_info': model_info,
        'final_losses': {
            'total': float(train_losses[-1]),
            'low_freq': float(low_freq_losses[-1]),
            'high_freq': float(high_freq_losses[-1])
        },
        'min_loss': float(min(train_losses)),
        'epochs_trained': len(train_losses),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(config['logging']['output_dir'], 'training_stats.json'), 'w') as f:
        json.dump(training_stats, f, indent=2, ensure_ascii=False)
    
    print("学習完了!")
    return model, training_stats


def plot_training_curves(train_losses, low_freq_losses, high_freq_losses, learning_rates, save_path):
    """学習曲線を描画"""
    plt.figure(figsize=(15, 5))
    
    # 損失曲線
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Total Loss', linewidth=2)
    plt.plot(low_freq_losses, label='Low Freq Loss', linewidth=2)
    plt.plot(high_freq_losses, label='High Freq Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # 学習率
    plt.subplot(1, 3, 2)
    plt.plot(learning_rates, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.yscale('log')
    
    # 損失比率
    plt.subplot(1, 3, 3)
    total_losses = np.array(train_losses)
    low_ratio = np.array(low_freq_losses) / total_losses
    high_ratio = np.array(high_freq_losses) / total_losses
    plt.plot(low_ratio, label='Low Freq Ratio', linewidth=2)
    plt.plot(high_ratio, label='High Freq Ratio', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Ratio')
    plt.title('Loss Component Ratio')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"学習曲線を保存: {save_path}")


def generate_samples(model_path, config_path=None, num_samples=5):
    """学習済みモデルで軌道を生成"""
    
    # 設定読み込み
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_config()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデル読み込み
    print(f"モデルを読み込み中: {model_path}")
    model = HybridTrajectoryModel.load_model(model_path, device=device)
    model.eval()
    
    # テストデータ読み込み
    data_path = config['data']['data_path']
    trajectories, conditions = load_processed_data(data_path)
    
    # テスト条件を準備
    test_conditions = torch.tensor(conditions[:num_samples], dtype=torch.float32).to(device)
    original_trajectories = trajectories[:num_samples]
    
    # 軌道生成
    print("軌道を生成中...")
    with torch.no_grad():
        generated_trajectories = model.generate(
            condition=test_conditions,
            sequence_length=trajectories.shape[1],
            num_samples=3
        )
    
    print(f"生成された軌道のテンソル形状: {generated_trajectories.shape}")
    generated_trajectories = generated_trajectories.cpu().numpy()
    
    # 結果を可視化（最大9個まで表示）
    max_display_samples = min(9, num_samples)
    
    if max_display_samples <= 6:
        fig_rows, fig_cols = 2, 3
    elif max_display_samples <= 9:
        fig_rows, fig_cols = 3, 3
    else:
        fig_rows, fig_cols = 3, 3
        max_display_samples = 9
    
    plt.figure(figsize=(5*fig_cols, 4*fig_rows))
    
    for i in range(max_display_samples):
        plt.subplot(fig_rows, fig_cols, i+1)
        
        # 元の軌道
        plt.plot(original_trajectories[i, :, 0], original_trajectories[i, :, 1], 
                'k-', linewidth=3, label='Original', alpha=0.8)
        
        # 生成された軌道
        for j in range(3):
            gen_idx = i * 3 + j
            if gen_idx < len(generated_trajectories):
                plt.plot(generated_trajectories[gen_idx, :, 0], generated_trajectories[gen_idx, :, 1],
                        '--', linewidth=2, alpha=0.7, label=f'Generated {j+1}')
        
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'Condition {i+1}')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
    
    if num_samples > max_display_samples:
        plt.suptitle(f'生成結果（{max_display_samples}/{num_samples}サンプルを表示）', fontsize=16)
    
    output_dir = config['logging']['output_dir']
    save_path = os.path.join(output_dir, 'generated_trajectories', 'sample_generation.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"生成結果を保存: {save_path}")
    
    # 数値データも保存
    np.save(os.path.join(output_dir, 'generated_trajectories', 'generated_samples.npy'), 
            generated_trajectories)
    np.save(os.path.join(output_dir, 'generated_trajectories', 'test_conditions.npy'), 
            test_conditions.cpu().numpy())
    
    return generated_trajectories


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ハイブリッドモデルの学習')
    parser.add_argument('--config', type=str, default=None, 
                       help='設定ファイルのパス (デフォルト: config.yaml)')
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], default='train',
                       help='実行モード: train (学習) または generate (生成)')
    parser.add_argument('--model_path', type=str, default='outputs/hybrid_model_final.pth',
                       help='生成モードで使用するモデルのパス')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='生成モードで生成するサンプル数')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        model, stats = train_model(args.config)
        print("\n=== 学習完了 ===")
        print(f"最終損失: {stats['final_losses']['total']:.6f}")
        print(f"最小損失: {stats['min_loss']:.6f}")
        
    elif args.mode == 'generate':
        if not os.path.exists(args.model_path):
            print(f"エラー: モデルファイルが見つかりません: {args.model_path}")
            print("先に学習を実行してください: python train.py --mode train")
        else:
            generated = generate_samples(args.model_path, args.config, args.num_samples)
            print(f"生成完了: {generated.shape[0]} 軌道を生成")