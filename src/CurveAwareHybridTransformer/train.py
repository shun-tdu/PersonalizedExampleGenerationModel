
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
            'transformer_inner_dim': 256,
            'transformer_num_heads': 8,
            'transformer_num_layers': 4,
            'diffusion_hidden_dim': 32,
            'diffusion_num_layers': 4,
            'moving_average_window': 10,
            'num_diffusion_steps': 1000
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 200,
            'lr_low_freq': 1e-3,
            'lr_high_freq': 5e-4,
            'lr_decomposer': 1e-4,
            'weight_decay': 1e-4,
            'scheduler_T_0': 20,
            'scheduler_T_mult': 2,
            'scheduler_eta_min': 1e-6,
            'clip_grad_norm': 1.0,
            'warmup_epochs': 10,
            'augment_after_epoch':20
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
    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr=config['training']['learning_rate'],
    #     weight_decay=config['training']['weight_decay']
    # )
    # オプティマイザーの設定
    optimizer = optim.AdamW([
        {'params': model.low_freq_model.parameters(), 'lr': config['training']['lr_low_freq']},
        {'params': model.high_freq_model.parameters(), 'lr': config['training']['lr_high_freq']},
        {'params': model.decomposer.parameters(), 'lr': config['training']['lr_decomposer']}
    ], weight_decay=config['training']['weight_decay'])

    # 学習率スケジューラ-
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['training']['scheduler_T_0'],
        T_mult=config['training']['scheduler_T_mult'],
        eta_min=config['training']['scheduler_eta_min']
    )

    # ウォームアップスケジューラ
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=config['training']['warmup_epochs']
    )

    # 学習履歴
    history = {
        'train_loss': [], 'val_loss': [],
        'low_freq_loss': [], 'high_freq_loss': [],
        'goal_loss': [], 'curvature_loss': [],
        'path_efficiency_loss': [],
        'learning_rates': []
    }

    # ベストモデルの保存用
    best_val_loss = float('inf')
    
    # 学習開始
    print("学習を開始します...")
    model.train()
    
    for epoch in range(config['training']['num_epochs']):
        # epoch_losses = []
        # epoch_low_freq_losses = []
        # epoch_high_freq_losses = []

        # エポック更新(Scheduler sampling用)
        model.update_epoch(epoch, config['training']['num_epochs'])

        train_losses = {
            'total_loss': 0.0,
            'low_freq_loss': 0.0,
            'low_freq_mse': 0.0,
            'high_freq_loss': 0.0,
            'goal_loss': 0.0,
            'curvature_loss': 0.0,
            'path_efficiency_loss': 0.0,
            'velocity_consistency_loss': 0.0,
            'low_freq_smoothness': 0.0,
            'high_freq_smoothness': 0.0
        }

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["num_epochs"]}')
        
        for batch_idx, (trajectories, conditions) in enumerate(progress_bar):
            trajectories = trajectories.to(device)
            conditions = conditions.to(device)

            # データ拡張(曲線強調)
            if epoch > config['training']['augment_after_epoch']:
                trajectories = augment_curved_trajectories(
                    trajectories, conditions,
                    strength=min(0.3, epoch / config["training"]["num_epochs"])
                )

            # フォワードパス
            losses = model(trajectories, conditions)

            # 勾配計算
            optimizer.zero_grad()

            # 損失を取得
            # total_loss = outputs['total_loss']
            # low_freq_loss = outputs['low_freq_loss']
            # high_freq_loss = outputs['high_freq_loss']
            losses['total_loss'].backward()

            # 勾配クリップ
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=config['training']['clip_grad_norm']
            )

            optimizer.step()
            
            # 損失を記録
            for key in train_losses:
                if key in losses:
                    train_losses[key] += losses[key].item()

            # epoch_losses.append(total_loss.item())
            # epoch_low_freq_losses.append(low_freq_loss.item())
            # epoch_high_freq_losses.append(high_freq_loss.item())
            
            # プログレスバー更新
            progress_bar.set_postfix({
                'Loss': f"{losses['total_loss'].item():.4f}",
                'Low': f"{losses['low_freq_loss'].item():.4f}",
                'High': f"{losses['high_freq_loss'].item():.4f}",
                'goal': f"{losses['goal_loss'].item():.4f}"
            })
        
        # エポック終了処理
        # avg_loss = np.mean(epoch_losses)
        # avg_low_freq_loss = np.mean(epoch_low_freq_losses)
        # avg_high_freq_loss = np.mean(epoch_high_freq_losses)
        current_lr = scheduler.get_last_lr()[0]
        num_batches = len(train_loader)
        for key in train_losses:
            train_losses[key] /= num_batches

        # Historyに記録
        history['train_loss'].append(train_losses['total_loss'])
        history['low_freq_loss'].append(train_losses['low_freq_loss'])
        history['high_freq_loss'].append(train_losses['high_freq_loss'])
        history['goal_loss'].append(train_losses['goal_loss'])
        history['curvature_loss'].append(train_losses['curvature_loss'])
        history['path_efficiency_loss'].append(train_losses['path_efficiency_loss'])
        history['learning_rates'].append(current_lr)

        # 学習率更新
        if epoch < config['training']['warmup_epochs']:
            warmup_scheduler.step()
        else:
            scheduler.step()

        # ログ出力
        if (epoch + 1) % config['logging']['log_interval'] == 0:
            print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}:")
            print(f"  Train Loss: {train_losses['total_loss']:.4f}")
            print(f"  Low Freq Loss: {train_losses['low_freq_loss']:.4f}")
            print(f"  High Freq Loss: {train_losses['high_freq_loss']:.4f}")
            print(f"  Goal Loss: {train_losses['goal_loss']:.4f}")
            print(f"  Curvature Loss: {train_losses['curvature_loss']:.4f}")
            print(f"  Path Efficiency: {train_losses['path_efficiency_loss']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Sampling Prob: {model.low_freq_model.sampling_prob.item():.2f}")
        
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
        history['train_losses'],
        history['low_freq_losses'],
        history['high_freq_losses'],
        history['learning_rates'],
        save_path=os.path.join(config['logging']['output_dir'], 'plots', 'training_curves.png')
    )
    
    # 学習統計を保存
    training_stats = {
        'config': config,
        'model_info': model_info,
        'final_losses': {
            'total': float(history['train_losses'][-1]),
            'low_freq': float(history['low_freq_losses'][-1]),
            'high_freq': float(history['high_freq_losses'][-1])
        },
        'min_loss': float(min(history['train_losses'])),
        'epochs_trained': len(history['train_losses']),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(config['logging']['output_dir'], 'training_stats.json'), 'w') as f:
        json.dump(training_stats, f, indent=2, ensure_ascii=False)
    
    print("学習完了!")
    return model, training_stats

def augment_curved_trajectories(
        trajectories: torch.Tensor,
        conditions: torch.Tensor,
        strength: float = 0.2
    ) -> torch.Tensor:
    """曲線を強調するデータ拡張"""
    batch_size, seq_len, dim = trajectories.shape
    device = trajectories.device

    augmented = trajectories.clone()

    for i in range(batch_size):
        # ランダムに曲線を追加するか決定
        if torch.rand(1).item() < 0.5:
            # 制御点を使ったベジェ曲線による変形
            t = torch.linspace(0, 1, seq_len, device=device).unsqueeze(1)

            # ランダムな制御点
            mid_idx = seq_len // 2
            control_point = trajectories[i, mid_idx].clone()

            # 制御点をランダムに移動
            offset = torch.randn(2, device=device)*strength
            control_point = control_point + offset

            # 3次ベジェ曲線
            p0 = trajectories[i, 0] # 始点
            p3 = conditions[i, 3:5] # 終点

            # 中間点制御
            p1 = p0 + (control_point - p0) * 0.7
            p2 = p3 + (control_point - p3) * 0.7

            # ベジェ曲線の計算
            bezier = (
                (1 - t)**3 * p0 +
                3 * (1 - t)**2 * t * p1 +
                3 * (1 - t) * t**2 * p2 +
                t**3 * p3
            )

            # 元の軌道とブレンド
            alpha = torch.rand(1, device=device).item() * 0.5 + 0.5
            augmented[i] = alpha * bezier + (1 - alpha) * trajectories[i]

    return augmented

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


def generate_samples(model_path, config_path=None, num_samples=9):
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

    # データセットからランダムにインデックスを選択
    num_total_data = conditions.shape[0]
    random_indices = np.random.permutation(num_total_data)[:num_samples]

    # ランダムなインデックスを使って条件と元の軌道を取得
    # サンプルされた条件ベクトル
    sampled_conditions = torch.tensor(conditions[random_indices],dtype=torch.float32).to(device)
    # 対応する正解の軌道
    original_trajectories = trajectories[random_indices]

    print(f"データセットから{num_samples}個の条件をランダムに選択しました")
    
    # 軌道生成
    print("軌道を生成中...")
    with torch.no_grad():
        generated_trajectories = model.generate(
            condition=sampled_conditions,
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
                        '-', linewidth=2, alpha=0.7, label=f'Generated {j+1}')
        
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
    np.save(os.path.join(output_dir, 'generated_trajectories', 'sampled_conditions.npy'),
            sampled_conditions.cpu().numpy())
    
    return generated_trajectories


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ハイブリッドモデルの学習')
    parser.add_argument('--config', type=str, default=None, 
                       help='設定ファイルのパス (デフォルト: config.yaml)')
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], default='train',
                       help='実行モード: train (学習) または generate (生成)')
    parser.add_argument('--model_path', type=str, default='outputs/hybrid_model_final.pth',
                       help='生成モードで使用するモデルのパス')
    parser.add_argument('--num_samples', type=int, default=9,
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