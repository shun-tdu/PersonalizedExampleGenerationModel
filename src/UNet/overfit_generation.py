# CLAUDE_ADDED
"""
過学習したモデルを使った詳細な軌道生成・分析スクリプト
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple, Optional

from Model import UNet1D
from train import DDPMScheduler
from TrajectoryDataset import TrajectoryDataset
from generate import TrajectoryGenerator, visualize_trajectories


class OverfitGenerationAnalyzer:
    """
    過学習したモデルでの生成分析クラス
    """
    def __init__(self, 
                 model: UNet1D,
                 scheduler: DDPMScheduler,
                 device: torch.device):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.generator = TrajectoryGenerator(model, scheduler, device)
    
    def analyze_condition_sensitivity(self, 
                                    base_condition: np.ndarray,
                                    condition_variations: Dict[int, List[float]],
                                    seq_len: int = 101,
                                    method: str = 'ddim',
                                    num_inference_steps: int = 50) -> Dict[str, np.ndarray]:
        """
        条件ベクトルの各次元を変化させて、生成への影響を分析
        
        Args:
            base_condition: ベースとなる条件ベクトル [condition_dim,]
            condition_variations: {次元インデックス: 変化値リスト} の辞書
            seq_len: 軌道長
            method: サンプリング手法
            num_inference_steps: 推論ステップ数
            
        Returns:
            Dict[str, np.ndarray]: 各条件変化での生成結果
        """
        results = {}
        
        for dim_idx, variations in condition_variations.items():
            print(f"Analyzing condition dimension {dim_idx} variations: {variations}")
            
            # 各変化値での軌道生成
            trajectories_list = []
            conditions_list = []
            
            for variation in variations:
                # 条件ベクトルをコピーして指定次元を変更
                modified_condition = base_condition.copy()
                modified_condition[dim_idx] = variation
                
                # テンソルに変換
                condition_tensor = torch.FloatTensor(modified_condition).unsqueeze(0).to(self.device)
                
                # 軌道生成
                with torch.no_grad():
                    trajectory = self.generator.generate_trajectories(
                        conditions=condition_tensor,
                        seq_len=seq_len,
                        method=method,
                        num_inference_steps=num_inference_steps
                    )
                
                trajectories_list.append(trajectory.cpu().numpy()[0])
                conditions_list.append(modified_condition)
            
            results[f'dim_{dim_idx}'] = {
                'trajectories': np.array(trajectories_list),
                'conditions': np.array(conditions_list),
                'variations': variations
            }
        
        return results
    
    def generate_multiple_samples(self,
                                condition: np.ndarray,
                                num_samples: int = 10,
                                seq_len: int = 101,
                                method: str = 'ddpm',
                                num_inference_steps: Optional[int] = None) -> np.ndarray:
        """
        同じ条件で複数回生成して、確率的な変動を確認
        """
        print(f"Generating {num_samples} samples with the same condition...")
        
        trajectories_list = []
        condition_tensor = torch.FloatTensor(condition).unsqueeze(0).to(self.device)
        
        for i in range(num_samples):
            with torch.no_grad():
                trajectory = self.generator.generate_trajectories(
                    conditions=condition_tensor,
                    seq_len=seq_len,
                    method=method,
                    num_inference_steps=num_inference_steps
                )
            trajectories_list.append(trajectory.cpu().numpy()[0])
        
        return np.array(trajectories_list)
    
    def analyze_reconstruction_quality(self,
                                     original_trajectories: np.ndarray,
                                     original_conditions: np.ndarray,
                                     num_reconstructions: int = 5,
                                     method: str = 'ddim',
                                     num_inference_steps: int = 50) -> Dict[str, any]:
        """
        元データの再構成品質を詳細分析
        """
        print(f"Analyzing reconstruction quality with {num_reconstructions} attempts per sample...")
        
        batch_size = original_trajectories.shape[0]
        results = {
            'original': original_trajectories,
            'reconstructions': [],
            'errors': [],
            'statistics': {}
        }
        
        conditions_tensor = torch.FloatTensor(original_conditions).to(self.device)
        
        # 各サンプルに対して複数回再構成
        for i in range(batch_size):
            sample_reconstructions = []
            sample_errors = []
            
            condition_single = conditions_tensor[i:i+1]  # [1, condition_dim]
            original_single = original_trajectories[i]  # [2, seq_len]
            
            for j in range(num_reconstructions):
                with torch.no_grad():
                    reconstructed = self.generator.generate_trajectories(
                        conditions=condition_single,
                        seq_len=original_trajectories.shape[2],
                        method=method,
                        num_inference_steps=num_inference_steps
                    )
                
                reconstructed_np = reconstructed.cpu().numpy()[0]
                sample_reconstructions.append(reconstructed_np)
                
                # 誤差計算
                mse = np.mean((original_single - reconstructed_np) ** 2)
                mae = np.mean(np.abs(original_single - reconstructed_np))
                sample_errors.append({'mse': mse, 'mae': mae})
            
            results['reconstructions'].append(np.array(sample_reconstructions))
            results['errors'].append(sample_errors)
        
        # 統計計算
        all_mse = [error['mse'] for errors in results['errors'] for error in errors]
        all_mae = [error['mae'] for errors in results['errors'] for error in errors]
        
        results['statistics'] = {
            'mse_mean': np.mean(all_mse),
            'mse_std': np.std(all_mse),
            'mse_min': np.min(all_mse),
            'mse_max': np.max(all_mse),
            'mae_mean': np.mean(all_mae),
            'mae_std': np.std(all_mae),
            'mae_min': np.min(all_mae),
            'mae_max': np.max(all_mae)
        }
        
        return results


def visualize_condition_sensitivity(results: Dict[str, Dict], 
                                  save_path: str,
                                  condition_names: Optional[List[str]] = None):
    """
    条件感度分析の結果を可視化
    """
    n_conditions = len(results)
    fig, axes = plt.subplots(1, n_conditions, figsize=(6*n_conditions, 5))
    
    if n_conditions == 1:
        axes = [axes]
    
    for idx, (condition_key, data) in enumerate(results.items()):
        ax = axes[idx]
        trajectories = data['trajectories']
        variations = data['variations']
        
        # 各変化値での軌道をプロット
        for i, (traj, var) in enumerate(zip(trajectories, variations)):
            x_traj = traj[0, :]
            y_traj = traj[1, :]
            
            ax.plot(x_traj, y_traj, linewidth=2, alpha=0.7, label=f'{var:.3f}')
            ax.plot(x_traj[0], y_traj[0], 'o', markersize=6)
            ax.plot(x_traj[-1], y_traj[-1], 's', markersize=6)
        
        dim_idx = int(condition_key.split('_')[1])
        condition_name = condition_names[dim_idx] if condition_names else f'Condition {dim_idx}'
        
        ax.set_title(f'{condition_name} Sensitivity')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')
        ax.set_xlim(-0.4, 0.4)
        ax.set_ylim(-0.4, 0.4)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Condition sensitivity visualization saved to: {save_path}")


def visualize_multiple_samples(trajectories: np.ndarray,
                             condition: np.ndarray,
                             save_path: str):
    """
    同じ条件での複数サンプル生成結果を可視化
    """
    num_samples = trajectories.shape[0]
    
    plt.figure(figsize=(10, 8))
    
    # すべての軌道をプロット
    for i in range(num_samples):
        x_traj = trajectories[i, 0, :]
        y_traj = trajectories[i, 1, :]
        
        plt.plot(x_traj, y_traj, linewidth=2, alpha=0.6, label=f'Sample {i+1}')
        plt.plot(x_traj[0], y_traj[0], 'go', markersize=6, alpha=0.6)
        plt.plot(x_traj[-1], y_traj[-1], 'ro', markersize=6, alpha=0.6)
    
    # 平均軌道を計算してプロット
    mean_trajectory = np.mean(trajectories, axis=0)
    plt.plot(mean_trajectory[0, :], mean_trajectory[1, :], 'k-', linewidth=3, label='Mean')
    plt.plot(mean_trajectory[0, 0], mean_trajectory[1, 0], 'ko', markersize=8, label='Mean Start')
    plt.plot(mean_trajectory[0, -1], mean_trajectory[1, -1], 'ks', markersize=8, label='Mean End')
    
    plt.title(f'Multiple Samples (N={num_samples})\nCondition: {condition[:3]}...')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('equal')
    plt.xlim(-0.4, 0.4)
    plt.ylim(-0.4, 0.4)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Multiple samples visualization saved to: {save_path}")


def visualize_reconstruction_analysis(analysis_results: Dict,
                                    save_path: str):
    """
    再構成分析結果の可視化
    """
    original = analysis_results['original']
    reconstructions = analysis_results['reconstructions']
    errors = analysis_results['errors']
    stats = analysis_results['statistics']
    
    batch_size = len(reconstructions)
    num_reconstructions = len(reconstructions[0])
    
    fig, axes = plt.subplots(2, batch_size, figsize=(5*batch_size, 10))
    if batch_size == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(batch_size):
        # 上段: 軌道比較
        ax_traj = axes[0, i]
        
        # 元データ
        x_orig = original[i, 0, :]
        y_orig = original[i, 1, :]
        ax_traj.plot(x_orig, y_orig, 'k-', linewidth=3, label='Original', alpha=0.8)
        ax_traj.plot(x_orig[0], y_orig[0], 'ko', markersize=8)
        ax_traj.plot(x_orig[-1], y_orig[-1], 'ks', markersize=8)
        
        # 再構成結果
        for j in range(num_reconstructions):
            recon = reconstructions[i][j]
            x_recon = recon[0, :]
            y_recon = recon[1, :]
            ax_traj.plot(x_recon, y_recon, '--', linewidth=2, alpha=0.6, 
                        label=f'Recon {j+1}')
        
        ax_traj.set_title(f'Sample {i+1}: Original vs Reconstructions')
        ax_traj.set_xlabel('X Position (m)')
        ax_traj.set_ylabel('Y Position (m)')
        ax_traj.grid(True, alpha=0.3)
        ax_traj.legend()
        ax_traj.set_aspect('equal')
        ax_traj.set_xlim(-0.4, 0.4)
        ax_traj.set_ylim(-0.4, 0.4)
        
        # 下段: 誤差分布
        ax_error = axes[1, i]
        
        sample_mse = [error['mse'] for error in errors[i]]
        sample_mae = [error['mae'] for error in errors[i]]
        
        x_pos = np.arange(num_reconstructions)
        width = 0.35
        
        ax_error.bar(x_pos - width/2, sample_mse, width, label='MSE', alpha=0.7)
        ax_error.bar(x_pos + width/2, sample_mae, width, label='MAE', alpha=0.7)
        
        ax_error.set_title(f'Reconstruction Errors\nMSE: {np.mean(sample_mse):.4f}±{np.std(sample_mse):.4f}')
        ax_error.set_xlabel('Reconstruction Attempt')
        ax_error.set_ylabel('Error')
        ax_error.set_xticks(x_pos)
        ax_error.set_xticklabels([f'R{j+1}' for j in range(num_reconstructions)])
        ax_error.legend()
        ax_error.grid(True, alpha=0.3)
    
    # 全体統計をタイトルに追加
    fig.suptitle(f'Reconstruction Analysis\n'
                f'Overall MSE: {stats["mse_mean"]:.4f}±{stats["mse_std"]:.4f}, '
                f'MAE: {stats["mae_mean"]:.4f}±{stats["mae_std"]:.4f}', 
                fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Reconstruction analysis visualization saved to: {save_path}")


def run_overfit_generation_analysis(model_path: str,
                                  data_path: str,
                                  output_dir: str = 'overfit_generation_analysis',
                                  device: str = 'cuda'):
    """
    過学習モデルでの詳細生成分析を実行
    """
    # デバイス設定
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # データセット読み込み
    dataset = TrajectoryDataset(data_path)
    
    # モデル読み込み（過学習テストで保存されたもの）
    checkpoint = torch.load(model_path, map_location=device)
    
    # モデル構造の推定
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    else:
        model_state = checkpoint
    
    # 条件次元の推定
    condition_dim = model_state['encoder_blocks.0.cross_attention.to_k.weight'].shape[1]
    base_channels = model_state['input_proj.weight'].shape[0]
    
    # モデル作成
    model = UNet1D(
        input_dim=2,
        condition_dim=condition_dim,
        time_embed_dim=128,
        base_channels=base_channels
    ).to(device)
    
    model.load_state_dict(model_state)
    model.eval()
    
    # スケジューラとアナライザー作成
    scheduler = DDPMScheduler(num_timesteps=1000)
    analyzer = OverfitGenerationAnalyzer(model, scheduler, device)
    
    # テスト用サンプルを取得（過学習に使ったものと同じ）
    sample_trajectory, sample_condition = dataset[0]
    sample_condition_np = sample_condition.numpy()
    sample_trajectory_np = sample_trajectory.numpy()
    
    print(f"Analysis target - Condition: {sample_condition_np}")
    print(f"Trajectory shape: {sample_trajectory_np.shape}")
    
    # 1. 条件感度分析
    print("\n1. Condition Sensitivity Analysis...")
    condition_variations = {
        0: [sample_condition_np[0] - 0.5, sample_condition_np[0], sample_condition_np[0] + 0.5],  # 動作時間
        1: [sample_condition_np[1] - 0.5, sample_condition_np[1], sample_condition_np[1] + 0.5],  # 終点誤差
        2: [sample_condition_np[2] - 0.5, sample_condition_np[2], sample_condition_np[2] + 0.5],  # ジャーク
    }
    
    sensitivity_results = analyzer.analyze_condition_sensitivity(
        sample_condition_np, condition_variations
    )
    
    condition_names = ['Movement Time', 'Endpoint Error', 'Jerk']
    visualize_condition_sensitivity(
        sensitivity_results, 
        os.path.join(output_dir, 'condition_sensitivity.png'),
        condition_names
    )
    
    # 2. 複数サンプル生成（確率的変動の確認）
    print("\n2. Multiple Sample Generation...")
    multiple_samples = analyzer.generate_multiple_samples(
        sample_condition_np, num_samples=10
    )
    
    visualize_multiple_samples(
        multiple_samples,
        sample_condition_np,
        os.path.join(output_dir, 'multiple_samples.png')
    )
    
    # 3. 再構成品質分析
    print("\n3. Reconstruction Quality Analysis...")
    reconstruction_analysis = analyzer.analyze_reconstruction_quality(
        sample_trajectory_np.reshape(1, 2, -1),
        sample_condition_np.reshape(1, -1),
        num_reconstructions=5
    )
    
    visualize_reconstruction_analysis(
        reconstruction_analysis,
        os.path.join(output_dir, 'reconstruction_analysis.png')
    )
    
    # 結果をJSONで保存
    analysis_summary = {
        'condition_sensitivity': {
            k: {
                'variations': v['variations'],
                'num_trajectories': len(v['trajectories'])
            } for k, v in sensitivity_results.items()
        },
        'multiple_samples': {
            'num_samples': len(multiple_samples),
            'trajectory_variance': {
                'x_var': float(np.var(multiple_samples[:, 0, :])),
                'y_var': float(np.var(multiple_samples[:, 1, :]))
            }
        },
        'reconstruction_quality': reconstruction_analysis['statistics']
    }
    
    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    
    print(f"\nOverfit generation analysis completed! Results saved to: {output_dir}")
    
    return analysis_summary


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Overfit Model Generation Analysis')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to overfitted model')
    parser.add_argument('--data_path', type=str, default='/data/Datasets/overfitting_dataset.npz',
                        help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='overfit_generation_analysis',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    run_overfit_generation_analysis(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        device=args.device
    )