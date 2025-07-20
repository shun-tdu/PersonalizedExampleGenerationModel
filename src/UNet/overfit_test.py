# CLAUDE_ADDED
"""
1バッチ過学習テスト用スクリプト
モデル構造がデータに適しているかを検証するため、少数のサンプルで過学習させる
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from typing import Dict, Any

from Model import UNet1D
from train import DDPMScheduler, DiffusionTrainer
from TrajectoryDataset import TrajectoryDataset
from generate import TrajectoryGenerator, visualize_trajectories


class OverfitTester:
    """
    1バッチ過学習テスト用クラス
    """
    def __init__(self, 
                 model: UNet1D,
                 scheduler: DDPMScheduler,
                 device: torch.device,
                 learning_rate: float = 1e-3):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        
        # 過学習用に高い学習率を設定
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # 記録用
        self.losses = []
        
    def overfit_single_batch(self, 
                           batch_data: tuple,
                           num_epochs: int = 1000,
                           target_loss: float = 1e-5,
                           print_interval: int = 100) -> Dict[str, Any]:
        """
        単一バッチで過学習させる
        
        Args:
            batch_data: (trajectories, conditions) のタプル
            num_epochs: 最大エポック数
            target_loss: 目標損失値（これ以下になったら停止）
            print_interval: ログ出力間隔
            
        Returns:
            Dict: 過学習結果の辞書
        """
        trajectories, conditions = batch_data
        trajectories = trajectories.to(self.device)
        conditions = conditions.to(self.device)
        
        batch_size = trajectories.shape[0]
        print(f"Starting overfit test with batch size: {batch_size}")
        print(f"Trajectories shape: {trajectories.shape}")
        print(f"Conditions shape: {conditions.shape}")
        print(f"Target loss: {target_loss}")
        
        self.model.train()
        self.losses = []
        
        for epoch in range(num_epochs):
            # ランダムなタイムステップをサンプル
            timesteps = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=self.device)
            
            # ガウシアンノイズを生成
            noise = torch.randn_like(trajectories)
            
            # ノイズを追加した軌道を生成
            noisy_trajectories = self.scheduler.add_noise(trajectories, noise, timesteps)
            
            # モデルにノイズ予測をさせる
            predicted_noise = self.model(noisy_trajectories, timesteps, conditions)
            
            # 損失を計算
            loss = self.criterion(predicted_noise, noise)
            
            # 逆伝播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.losses.append(loss.item())
            
            # ログ出力
            if (epoch + 1) % print_interval == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
            
            # 目標損失に到達したら停止
            if loss.item() < target_loss:
                print(f"Reached target loss {target_loss} at epoch {epoch+1}")
                break
        
        final_loss = self.losses[-1]
        converged = final_loss < target_loss
        
        result = {
            'final_loss': final_loss,
            'converged': converged,
            'epochs_trained': len(self.losses),
            'target_loss': target_loss,
            'losses': self.losses,
            'batch_size': batch_size
        }
        
        print(f"\nOverfit test completed:")
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Converged: {converged}")
        print(f"  Epochs trained: {len(self.losses)}")
        
        return result
    
    def test_reconstruction(self, batch_data: tuple, num_inference_steps: int = 50) -> np.ndarray:
        """
        過学習後のモデルで軌道を再構成して、元データとの一致を確認
        
        Args:
            batch_data: 過学習に使用したバッチデータ
            num_inference_steps: 推論ステップ数
            
        Returns:
            np.ndarray: 生成された軌道
        """
        trajectories, conditions = batch_data
        conditions = conditions.to(self.device)
        
        # 生成器を作成
        generator = TrajectoryGenerator(self.model, self.scheduler, self.device)
        
        print(f"Generating trajectories with {num_inference_steps} inference steps...")
        
        # 軌道生成
        with torch.no_grad():
            generated_trajectories = generator.generate_trajectories(
                conditions=conditions,
                seq_len=trajectories.shape[2],
                method='ddim',
                num_inference_steps=num_inference_steps
            )
        
        return generated_trajectories.cpu().numpy()
    
    def save_results(self, result: Dict[str, Any], output_dir: str):
        """
        過学習テスト結果を保存
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 損失曲線を保存
        plt.figure(figsize=(10, 6))
        plt.semilogy(result['losses'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title(f'Overfit Test: Loss Curve (Final: {result["final_loss"]:.6f})')
        plt.grid(True, alpha=0.3)
        
        loss_path = os.path.join(output_dir, 'overfit_loss_curve.png')
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss curve saved to: {loss_path}")
        
        # 結果をJSONで保存
        result_copy = result.copy()
        result_copy.pop('losses', None)  # 長いリストは除外
        
        json_path = os.path.join(output_dir, 'overfit_results.json')
        with open(json_path, 'w') as f:
            json.dump(result_copy, f, indent=2)
        print(f"Results saved to: {json_path}")
        
        # 詳細な損失データを保存
        loss_data_path = os.path.join(output_dir, 'overfit_losses.npy')
        np.save(loss_data_path, np.array(result['losses']))
        print(f"Loss data saved to: {loss_data_path}")


def run_overfit_test(data_path: str, 
                    batch_size: int = 4,
                    num_epochs: int = 1000,
                    target_loss: float = 1e-4,
                    learning_rate: float = 1e-3,
                    output_dir: str = 'overfit_test_results',
                    device: str = 'cuda'):
    """
    過学習テストを実行するメイン関数
    
    Args:
        data_path: データセットのパス
        batch_size: バッチサイズ（小さく設定）
        num_epochs: 最大エポック数
        target_loss: 目標損失値
        learning_rate: 学習率（通常より高め）
        output_dir: 出力ディレクトリ
        device: 使用デバイス
    """
    # デバイス設定
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # データセット読み込み
    print(f'Loading dataset from: {data_path}')
    dataset = TrajectoryDataset(data_path)
    
    # 少数サンプルのサブセットを作成
    indices = list(range(min(batch_size, len(dataset))))
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=len(subset), shuffle=False)
    
    # バッチデータを取得
    batch_data = next(iter(dataloader))
    trajectories, conditions = batch_data
    
    print(f"Selected {len(indices)} samples for overfit test")
    print(f"Sample indices: {indices}")
    
    # 条件次元を自動検出
    condition_dim = conditions.shape[1]
    print(f'Detected condition dimension: {condition_dim}')
    
    # モデル作成
    model = UNet1D(
        input_dim=2,
        condition_dim=condition_dim,
        time_embed_dim=128,
        base_channels=64  # 過学習テストでは小さめでも良い
    ).to(device)
    
    # スケジューラ作成
    scheduler = DDPMScheduler(num_timesteps=1000)
    
    # 過学習テスター作成
    tester = OverfitTester(model, scheduler, device, learning_rate)
    
    # 過学習実行
    print("\n" + "="*50)
    print("Starting overfit test...")
    print("="*50)
    
    result = tester.overfit_single_batch(
        batch_data, 
        num_epochs=num_epochs,
        target_loss=target_loss,
        print_interval=100
    )
    
    # 結果保存
    tester.save_results(result, output_dir)
    
    # 軌道再構成テスト
    print("\n" + "="*50)
    print("Testing trajectory reconstruction...")
    print("="*50)
    
    generated_trajectories = tester.test_reconstruction(batch_data, num_inference_steps=50)
    
    # 元データと生成データの比較可視化
    original_trajectories = trajectories.numpy()
    conditions_np = conditions.numpy()
    
    # 元データの可視化
    original_vis_path = os.path.join(output_dir, 'original_trajectories.png')
    visualize_trajectories(original_trajectories, conditions_np, original_vis_path, denormalized=True)
    
    # 生成データの可視化
    generated_vis_path = os.path.join(output_dir, 'generated_trajectories.png')
    visualize_trajectories(generated_trajectories, conditions_np, generated_vis_path, denormalized=True)
    
    # 比較可視化
    comparison_vis_path = os.path.join(output_dir, 'comparison_trajectories.png')
    create_comparison_visualization(
        original_trajectories, 
        generated_trajectories, 
        conditions_np, 
        comparison_vis_path
    )
    
    # 数値的な比較
    mse_error = np.mean((original_trajectories - generated_trajectories) ** 2)
    mae_error = np.mean(np.abs(original_trajectories - generated_trajectories))
    
    print(f"\nReconstruction Error:")
    print(f"  MSE: {mse_error:.6f}")
    print(f"  MAE: {mae_error:.6f}")
    
    # 詳細結果を保存
    detailed_results = {
        **result,
        'reconstruction_mse': float(mse_error),
        'reconstruction_mae': float(mae_error),
        'learning_rate': learning_rate,
        'model_params': {
            'input_dim': 2,
            'condition_dim': condition_dim,
            'time_embed_dim': 128,
            'base_channels': 64
        }
    }
    
    detailed_results_path = os.path.join(output_dir, 'detailed_results.json')
    with open(detailed_results_path, 'w') as f:
        json.dump({k: v for k, v in detailed_results.items() if k != 'losses'}, f, indent=2)
    
    print(f"\nOverfit test completed! Results saved to: {output_dir}")
    
    return detailed_results


def create_comparison_visualization(original: np.ndarray, 
                                  generated: np.ndarray, 
                                  conditions: np.ndarray,
                                  save_path: str):
    """
    元データと生成データの比較可視化
    """
    batch_size = original.shape[0]
    
    fig, axes = plt.subplots(2, batch_size, figsize=(4*batch_size, 8))
    if batch_size == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(batch_size):
        # 元データ
        ax_orig = axes[0, i]
        x_orig = original[i, 0, :]
        y_orig = original[i, 1, :]
        ax_orig.plot(x_orig, y_orig, 'b-', linewidth=2, label='Original')
        ax_orig.plot(x_orig[0], y_orig[0], 'go', markersize=8, label='Start')
        ax_orig.plot(x_orig[-1], y_orig[-1], 'ro', markersize=8, label='End')
        ax_orig.set_title(f'Original Trajectory {i+1}')
        ax_orig.set_xlabel('X Position (m)')
        ax_orig.set_ylabel('Y Position (m)')
        ax_orig.grid(True, alpha=0.3)
        ax_orig.legend()
        ax_orig.set_aspect('equal')
        ax_orig.set_xlim(-0.4, 0.4)
        ax_orig.set_ylim(-0.4, 0.4)
        
        # 生成データ
        ax_gen = axes[1, i]
        x_gen = generated[i, 0, :]
        y_gen = generated[i, 1, :]
        ax_gen.plot(x_gen, y_gen, 'r-', linewidth=2, label='Generated')
        ax_gen.plot(x_gen[0], y_gen[0], 'go', markersize=8, label='Start')
        ax_gen.plot(x_gen[-1], y_gen[-1], 'ro', markersize=8, label='End')
        
        # 元データも薄く表示
        ax_gen.plot(x_orig, y_orig, 'b-', linewidth=1, alpha=0.3, label='Original (ref)')
        
        # 誤差計算
        mse = np.mean((x_orig - x_gen)**2 + (y_orig - y_gen)**2)
        ax_gen.set_title(f'Generated Trajectory {i+1}\nMSE: {mse:.6f}')
        ax_gen.set_xlabel('X Position (m)')
        ax_gen.set_ylabel('Y Position (m)')
        ax_gen.grid(True, alpha=0.3)
        ax_gen.legend()
        ax_gen.set_aspect('equal')
        ax_gen.set_xlim(-0.4, 0.4)
        ax_gen.set_ylim(-0.4, 0.4)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison visualization saved to: {save_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Overfit Test for UNet Diffusion Model')
    parser.add_argument('--data_path', type=str, default='/data/Datasets/overfitting_dataset.npz',
                        help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for overfit test (small number)')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='Maximum number of epochs')
    parser.add_argument('--target_loss', type=float, default=1e-4,
                        help='Target loss to achieve')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate (higher than usual)')
    parser.add_argument('--output_dir', type=str, default='overfit_test_results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    results = run_overfit_test(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        target_loss=args.target_loss,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        device=args.device
    )