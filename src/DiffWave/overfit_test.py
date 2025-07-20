# CLAUDE_ADDED
"""
DiffWaveモデルの1バッチ過学習テスト
モデル構造がデータに適しているかを検証
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse
from typing import Dict, List

# DiffWaveモデルをインポート
from Model import DiffWave1D, DiffWaveTrainer


class DDPMScheduler:
    """
    DDPM (Denoising Diffusion Probabilistic Models) のノイズスケジューラ
    """
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        
        # βのスケジュール（線形）
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # αの計算
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # 逆プロセス用のパラメータ
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
        """
        x0にノイズを追加してxtを生成
        """
        device = x0.device
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps.cpu()].to(device).view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps.cpu()].to(device).view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise


class DiffWaveOverfitTester:
    """
    DiffWaveモデルの過学習テスター
    """
    def __init__(self, model: DiffWave1D, device: torch.device, learning_rate: float = 1e-3):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.losses = []
        
    def compute_loss(self, trajectories: torch.Tensor, conditions: torch.Tensor, scheduler) -> torch.Tensor:
        """
        損失計算
        """
        batch_size = trajectories.shape[0]
        
        # Random timesteps
        timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise
        noise = torch.randn_like(trajectories)
        noisy_trajectories = scheduler.add_noise(trajectories, noise, timesteps)
        
        # Predict noise
        predicted_noise = self.model(noisy_trajectories, timesteps, conditions)
        
        # Compute loss
        loss = self.criterion(predicted_noise, noise)
        return loss
    
    def overfit_single_batch(self, 
                           batch: tuple, 
                           scheduler,
                           num_epochs: int = 1000,
                           target_loss: float = 1e-4,
                           verbose: bool = True) -> Dict:
        """
        単一バッチでの過学習実行
        """
        trajectories, conditions = batch
        trajectories = trajectories.to(self.device)
        conditions = conditions.to(self.device)
        
        self.model.train()
        
        results = {
            'losses': [],
            'converged': False,
            'final_loss': None,
            'convergence_epoch': None,
            'batch_info': {
                'batch_size': trajectories.shape[0],
                'trajectory_shape': trajectories.shape,
                'condition_shape': conditions.shape
            }
        }
        
        if verbose:
            print(f"Starting DiffWave overfit test...")
            print(f"Batch size: {trajectories.shape[0]}")
            print(f"Trajectory shape: {trajectories.shape}")
            print(f"Condition shape: {conditions.shape}")
            print(f"Target loss: {target_loss}")
        
        with tqdm(range(num_epochs), desc='Overfitting') as pbar:
            for epoch in pbar:
                # Forward pass
                loss = self.compute_loss(trajectories, conditions, scheduler)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                loss_value = loss.item()
                self.losses.append(loss_value)
                results['losses'].append(loss_value)
                
                pbar.set_postfix({'Loss': f'{loss_value:.6f}'})
                
                # 収束チェック
                if loss_value < target_loss:
                    results['converged'] = True
                    results['convergence_epoch'] = epoch + 1
                    if verbose:
                        print(f"\nConverged at epoch {epoch + 1} with loss {loss_value:.6f}")
                    break
        
        results['final_loss'] = self.losses[-1]
        
        return results
    
    def analyze_results(self, results: Dict, save_dir: str = 'outputs/overfit_results'):
        """
        過学習結果の分析と可視化
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 損失曲線をプロット
        plt.figure(figsize=(12, 8))
        
        # 損失曲線
        plt.subplot(2, 2, 1)
        plt.plot(results['losses'], 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('DiffWave Overfitting Loss Curve')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        if results['converged']:
            plt.axvline(x=results['convergence_epoch'], color='r', linestyle='--', 
                       label=f"Converged at epoch {results['convergence_epoch']}")
            plt.legend()
        
        # 後半の損失 (より詳細)
        plt.subplot(2, 2, 2)
        if len(results['losses']) > 100:
            plt.plot(results['losses'][-100:], 'g-', linewidth=2)
            plt.xlabel('Epoch (Last 100)')
        else:
            plt.plot(results['losses'], 'g-', linewidth=2)
            plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Recent Loss (Linear Scale)')
        plt.grid(True, alpha=0.3)
        
        # 損失の改善率
        plt.subplot(2, 2, 3)
        if len(results['losses']) > 1:
            improvements = [
                (results['losses'][i-1] - results['losses'][i]) / results['losses'][i-1] * 100
                for i in range(1, len(results['losses']))
            ]
            plt.plot(improvements, 'r-', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Improvement (%)')
            plt.title('Loss Improvement Rate')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.grid(True, alpha=0.3)
        
        # 収束分析
        plt.subplot(2, 2, 4)
        if len(results['losses']) > 10:
            # 移動平均でスムージング
            window = min(10, len(results['losses']) // 10)
            smoothed = []
            for i in range(window, len(results['losses'])):
                smoothed.append(np.mean(results['losses'][i-window:i]))
            
            plt.plot(range(window, len(results['losses'])), smoothed, 'purple', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Smoothed Loss')
            plt.title(f'Smoothed Loss (window={window})')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        plot_path = os.path.join(save_dir, 'diffwave_overfit_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 結果の要約をJSON保存
        summary = {
            'converged': results['converged'],
            'final_loss': results['final_loss'],
            'convergence_epoch': results['convergence_epoch'],
            'total_epochs': len(results['losses']),
            'batch_info': results['batch_info'],
            'loss_statistics': {
                'min_loss': min(results['losses']),
                'max_loss': max(results['losses']),
                'initial_loss': results['losses'][0],
                'final_loss': results['losses'][-1],
                'loss_reduction_ratio': results['losses'][0] / results['losses'][-1]
            }
        }
        
        summary_path = os.path.join(save_dir, 'diffwave_overfit_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nDiffWave Overfit Analysis:")
        print(f"  Converged: {results['converged']}")
        print(f"  Final Loss: {results['final_loss']:.8f}")
        if results['converged']:
            print(f"  Convergence Epoch: {results['convergence_epoch']}")
        print(f"  Loss Reduction: {summary['loss_statistics']['loss_reduction_ratio']:.2f}x")
        print(f"  Results saved to: {save_dir}")
        
        return summary


def create_test_batch(batch_size: int = 4, seq_len: int = 101, condition_dim: int = 5):
    """
    テスト用の1バッチデータ作成
    """
    # 特定のパターンを持つ軌道データ（学習しやすいように）
    trajectories = []
    conditions = []
    
    for i in range(batch_size):
        # 円弧や直線など、学習しやすいパターンを生成
        t = np.linspace(0, 1, seq_len)
        
        if i % 4 == 0:  # 円弧
            x = 0.5 * np.cos(2 * np.pi * t)
            y = 0.5 * np.sin(2 * np.pi * t)
        elif i % 4 == 1:  # 直線
            x = np.linspace(-0.5, 0.5, seq_len)
            y = 0.1 * x
        elif i % 4 == 2:  # S字カーブ
            x = np.linspace(-0.5, 0.5, seq_len)
            y = 0.3 * np.sin(4 * np.pi * t)
        else:  # 楕円
            x = 0.7 * np.cos(2 * np.pi * t)
            y = 0.3 * np.sin(2 * np.pi * t)
        
        trajectory = np.stack([x, y], axis=0)
        
        # 軌道に少しノイズを追加
        trajectory += np.random.normal(0, 0.01, trajectory.shape)
        
        trajectories.append(trajectory)
        
        # 対応する個人特性（軌道の特徴から計算）
        movement_time = seq_len * 0.01  # 仮の動作時間
        endpoint_error = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
        jerk = np.mean(np.abs(np.diff(x, n=3))) + np.mean(np.abs(np.diff(y, n=3)))
        smoothness = 1.0 / (1.0 + jerk)
        complexity = np.std(x) + np.std(y)
        
        condition = np.array([movement_time, endpoint_error, jerk, smoothness, complexity])
        conditions.append(condition)
    
    trajectories = np.array(trajectories, dtype=np.float32)
    conditions = np.array(conditions, dtype=np.float32)
    
    return torch.FloatTensor(trajectories), torch.FloatTensor(conditions)


def main():
    """
    DiffWave過学習テストのメイン関数
    """
    parser = argparse.ArgumentParser(description='DiffWave Overfit Test')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for overfitting')
    parser.add_argument('--seq_len', type=int, default=101, help='Sequence length')
    parser.add_argument('--condition_dim', type=int, default=5, help='Condition dimension')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Maximum epochs')
    parser.add_argument('--target_loss', type=float, default=1e-4, help='Target loss for convergence')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output_dir', type=str, default='outputs/overfit_results', help='Output directory')
    
    # DiffWaveモデルパラメータ
    parser.add_argument('--residual_channels', type=int, default=64, help='Residual channels')
    parser.add_argument('--skip_channels', type=int, default=64, help='Skip channels')
    parser.add_argument('--condition_channels', type=int, default=128, help='Condition channels')
    parser.add_argument('--num_layers', type=int, default=20, help='Number of layers')
    parser.add_argument('--cycles', type=int, default=4, help='Dilation cycles')
    parser.add_argument('--time_embed_dim', type=int, default=128, help='Time embedding dimension')
    
    args = parser.parse_args()
    
    # デバイス設定
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # テストデータ作成
    print("Creating test batch...")
    trajectories, conditions = create_test_batch(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        condition_dim=args.condition_dim
    )
    
    # DiffWaveモデル作成
    print("Creating DiffWave model...")
    model = DiffWave1D(
        input_dim=2,
        condition_dim=args.condition_dim,
        residual_channels=args.residual_channels,
        skip_channels=args.skip_channels,
        condition_channels=args.condition_channels,
        num_layers=args.num_layers,
        cycles=args.cycles,
        time_embed_dim=args.time_embed_dim
    ).to(device)
    
    # モデル情報表示
    total_params = sum(p.numel() for p in model.parameters())
    print(f'DiffWave Model created with {total_params:,} parameters')
    
    # スケジューラ作成
    scheduler = DDPMScheduler(num_timesteps=1000)
    
    # 過学習テスター作成
    tester = DiffWaveOverfitTester(model, device, args.learning_rate)
    
    # 過学習実行
    print("\nStarting overfit test...")
    results = tester.overfit_single_batch(
        batch=(trajectories, conditions),
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        target_loss=args.target_loss,
        verbose=True
    )
    
    # 結果分析
    print("\nAnalyzing results...")
    summary = tester.analyze_results(results, args.output_dir)
    
    # モデル保存（過学習済み）
    checkpoint_path = os.path.join(args.output_dir, 'diffwave_overfitted_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': tester.optimizer.state_dict(),
        'results': results,
        'summary': summary,
        'args': vars(args),
        'test_batch': (trajectories.cpu(), conditions.cpu())
    }, checkpoint_path)
    
    print(f"\nOverfitted model saved to: {checkpoint_path}")
    
    if results['converged']:
        print(f"✅ DiffWave successfully overfitted the batch!")
        print(f"   Converged in {results['convergence_epoch']} epochs")
        print(f"   Final loss: {results['final_loss']:.8f}")
    else:
        print(f"⚠️  DiffWave did not fully converge")
        print(f"   Final loss: {results['final_loss']:.8f}")
        print(f"   Consider increasing epochs or adjusting learning rate")


if __name__ == '__main__':
    main()