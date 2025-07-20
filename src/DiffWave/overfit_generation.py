# CLAUDE_ADDED
"""
過学習したDiffWaveモデルでの軌道生成テスト
再構成能力と条件付き生成の検証
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict
import argparse
import json

# DiffWaveモデルをインポート
from Model import DiffWave1D


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


class DiffWaveOverfitGenerator:
    """
    過学習したDiffWaveモデル用の軌道生成クラス
    """
    def __init__(self, 
                 model: DiffWave1D, 
                 scheduler: DDPMScheduler, 
                 device: torch.device):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        
        # 逆プロセス用のパラメータをデバイスに移動
        self.sqrt_recip_alphas = torch.sqrt(1.0 / scheduler.alphas).to(device)
        self.sqrt_one_minus_alphas_cumprod = scheduler.sqrt_one_minus_alphas_cumprod.to(device)
        
    @torch.no_grad()
    def ddpm_sample(self, 
                    shape: Tuple[int, ...], 
                    condition: torch.Tensor,
                    num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """
        DDPM サンプリング
        """
        batch_size = shape[0]
        
        # 推論ステップ数の設定
        if num_inference_steps is None:
            timesteps = list(range(self.scheduler.num_timesteps))[::-1]
        else:
            step_size = self.scheduler.num_timesteps // num_inference_steps
            timesteps = list(range(0, self.scheduler.num_timesteps, step_size))[::-1]
        
        # 純粋なノイズから開始
        x = torch.randn(shape, device=self.device)
        
        for i, t in enumerate(timesteps):
            # 現在のタイムステップ
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # モデルでノイズを予測
            predicted_noise = self.model(x, t_tensor, condition)
            
            # DDPM更新式
            if t > 0:
                sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
                beta_t = self.scheduler.betas[t]
                sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
                
                # 平均項の計算
                mean = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alphas_cumprod_t * predicted_noise)
                
                # 分散項の計算
                alpha_t = self.scheduler.alphas[t].to(self.device)
                alpha_cumprod_t = self.scheduler.alphas_cumprod[t].to(self.device)
                alpha_cumprod_prev = self.scheduler.alphas_cumprod_prev[t].to(self.device)
                
                variance = beta_t.to(self.device) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
                sigma = torch.sqrt(variance)
                
                # ランダムノイズ
                noise = torch.randn_like(x)
                
                x = mean + sigma * noise
            else:
                # 最後のステップ（t=0）ではノイズを加えない
                sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
                beta_t = self.scheduler.betas[t]
                sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
                
                x = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alphas_cumprod_t * predicted_noise)
        
        return x
    
    def generate_trajectories(self, 
                            conditions: torch.Tensor,
                            seq_len: int = 101,
                            num_inference_steps: Optional[int] = None,
                            num_samples_per_condition: int = 1) -> torch.Tensor:
        """
        指定された個人特性に基づいて軌道を生成
        """
        all_trajectories = []
        
        for condition in conditions:
            # 条件を複製して複数サンプル生成
            repeated_condition = condition.unsqueeze(0).repeat(num_samples_per_condition, 1)
            shape = (num_samples_per_condition, 2, seq_len)
            
            # 拡散モデルで軌道生成
            trajectories = self.ddpm_sample(shape, repeated_condition, num_inference_steps)
            all_trajectories.append(trajectories)
        
        return torch.cat(all_trajectories, dim=0)


def analyze_reconstruction_quality(original_trajectories: torch.Tensor,
                                 original_conditions: torch.Tensor,
                                 generated_trajectories: torch.Tensor,
                                 save_dir: str) -> Dict:
    """
    再構成品質の分析
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 軌道を numpy に変換
    orig_traj = original_trajectories.cpu().numpy()
    gen_traj = generated_trajectories.cpu().numpy()
    conditions = original_conditions.cpu().numpy()
    
    batch_size = orig_traj.shape[0]
    
    # 各軌道の比較メトリクス計算
    mse_errors = []
    endpoint_errors = []
    length_errors = []
    
    for i in range(batch_size):
        # MSE誤差
        mse = np.mean((orig_traj[i] - gen_traj[i])**2)
        mse_errors.append(mse)
        
        # 終点誤差
        orig_end = orig_traj[i, :, -1]
        gen_end = gen_traj[i, :, -1]
        endpoint_error = np.linalg.norm(orig_end - gen_end)
        endpoint_errors.append(endpoint_error)
        
        # 軌道長誤差
        orig_length = np.sum(np.sqrt(np.diff(orig_traj[i, 0])**2 + np.diff(orig_traj[i, 1])**2))
        gen_length = np.sum(np.sqrt(np.diff(gen_traj[i, 0])**2 + np.diff(gen_traj[i, 1])**2))
        length_error = abs(orig_length - gen_length)
        length_errors.append(length_error)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 軌道比較プロット
    ax = axes[0, 0]
    for i in range(batch_size):
        ax.plot(orig_traj[i, 0], orig_traj[i, 1], 'b-', linewidth=2, alpha=0.7, 
                label='Original' if i == 0 else '')
        ax.plot(gen_traj[i, 0], gen_traj[i, 1], 'r--', linewidth=2, alpha=0.7,
                label='Generated' if i == 0 else '')
        
        # 開始点と終点をマーク
        ax.plot(orig_traj[i, 0, 0], orig_traj[i, 1, 0], 'go', markersize=8)
        ax.plot(orig_traj[i, 0, -1], orig_traj[i, 1, -1], 'ro', markersize=8)
        ax.plot(gen_traj[i, 0, 0], gen_traj[i, 1, 0], 'g^', markersize=8)
        ax.plot(gen_traj[i, 0, -1], gen_traj[i, 1, -1], 'r^', markersize=8)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('DiffWave: Original vs Generated Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # MSE誤差分布
    ax = axes[0, 1]
    ax.bar(range(batch_size), mse_errors, alpha=0.7, color='orange')
    ax.set_xlabel('Trajectory Index')
    ax.set_ylabel('MSE Error')
    ax.set_title('Reconstruction MSE Errors')
    ax.grid(True, alpha=0.3)
    
    # 終点誤差分布
    ax = axes[1, 0]
    ax.bar(range(batch_size), endpoint_errors, alpha=0.7, color='green')
    ax.set_xlabel('Trajectory Index')
    ax.set_ylabel('Endpoint Error')
    ax.set_title('Endpoint Reconstruction Errors')
    ax.grid(True, alpha=0.3)
    
    # 軌道長誤差分布
    ax = axes[1, 1]
    ax.bar(range(batch_size), length_errors, alpha=0.7, color='purple')
    ax.set_xlabel('Trajectory Index')
    ax.set_ylabel('Length Error')
    ax.set_title('Trajectory Length Errors')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    plot_path = os.path.join(save_dir, 'diffwave_reconstruction_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 結果サマリー
    analysis_results = {
        'mse_errors': {
            'mean': float(np.mean(mse_errors)),
            'std': float(np.std(mse_errors)),
            'max': float(np.max(mse_errors)),
            'min': float(np.min(mse_errors)),
            'values': [float(x) for x in mse_errors]
        },
        'endpoint_errors': {
            'mean': float(np.mean(endpoint_errors)),
            'std': float(np.std(endpoint_errors)),
            'max': float(np.max(endpoint_errors)),
            'min': float(np.min(endpoint_errors)),
            'values': [float(x) for x in endpoint_errors]
        },
        'length_errors': {
            'mean': float(np.mean(length_errors)),
            'std': float(np.std(length_errors)),
            'max': float(np.max(length_errors)),
            'min': float(np.min(length_errors)),
            'values': [float(x) for x in length_errors]
        },
        'reconstruction_quality': {
            'excellent': sum(1 for x in mse_errors if x < 0.001),
            'good': sum(1 for x in mse_errors if 0.001 <= x < 0.01),
            'fair': sum(1 for x in mse_errors if 0.01 <= x < 0.1),
            'poor': sum(1 for x in mse_errors if x >= 0.1)
        }
    }
    
    # JSON保存
    with open(os.path.join(save_dir, 'diffwave_reconstruction_results.json'), 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\nDiffWave Reconstruction Analysis:")
    print(f"  MSE Error: {analysis_results['mse_errors']['mean']:.6f} ± {analysis_results['mse_errors']['std']:.6f}")
    print(f"  Endpoint Error: {analysis_results['endpoint_errors']['mean']:.6f} ± {analysis_results['endpoint_errors']['std']:.6f}")
    print(f"  Length Error: {analysis_results['length_errors']['mean']:.6f} ± {analysis_results['length_errors']['std']:.6f}")
    print(f"  Quality Distribution:")
    print(f"    Excellent (MSE < 0.001): {analysis_results['reconstruction_quality']['excellent']}")
    print(f"    Good (0.001 ≤ MSE < 0.01): {analysis_results['reconstruction_quality']['good']}")
    print(f"    Fair (0.01 ≤ MSE < 0.1): {analysis_results['reconstruction_quality']['fair']}")
    print(f"    Poor (MSE ≥ 0.1): {analysis_results['reconstruction_quality']['poor']}")
    
    return analysis_results


def test_condition_sensitivity(model: DiffWave1D,
                             scheduler: DDPMScheduler,
                             original_conditions: torch.Tensor,
                             device: torch.device,
                             save_dir: str) -> Dict:
    """
    条件パラメータへの感度テスト
    """
    print("\nTesting DiffWave condition sensitivity...")
    
    generator = DiffWaveOverfitGenerator(model, scheduler, device)
    
    # 元の条件で生成
    original_trajectories = generator.generate_trajectories(
        original_conditions, num_samples_per_condition=3
    )
    
    # 条件を少し変更して生成
    perturbed_conditions = original_conditions.clone()
    perturbed_conditions[:, 0] += 0.1  # 最初の条件パラメータを変更
    
    perturbed_trajectories = generator.generate_trajectories(
        perturbed_conditions, num_samples_per_condition=3
    )
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 元の条件での生成結果
    ax = axes[0, 0]
    for i, traj in enumerate(original_trajectories.cpu().numpy()):
        ax.plot(traj[0], traj[1], linewidth=2, alpha=0.7, label=f'Sample {i+1}')
    ax.set_title('DiffWave: Original Conditions')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 変更した条件での生成結果
    ax = axes[0, 1]
    for i, traj in enumerate(perturbed_trajectories.cpu().numpy()):
        ax.plot(traj[0], traj[1], linewidth=2, alpha=0.7, label=f'Sample {i+1}')
    ax.set_title('DiffWave: Perturbed Conditions (+0.1 to first param)')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 軌道の差異分析
    ax = axes[1, 0]
    differences = []
    for i in range(min(len(original_trajectories), len(perturbed_trajectories))):
        diff = torch.mean((original_trajectories[i] - perturbed_trajectories[i])**2).item()
        differences.append(diff)
    
    ax.bar(range(len(differences)), differences, alpha=0.7, color='red')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('MSE Difference')
    ax.set_title('Trajectory Differences Due to Condition Change')
    ax.grid(True, alpha=0.3)
    
    # 条件パラメータの影響度
    ax = axes[1, 1]
    condition_names = ['Movement Time', 'Endpoint Error', 'Jerk', 'Smoothness', 'Complexity']
    if len(condition_names) > original_conditions.shape[1]:
        condition_names = [f'Condition {i}' for i in range(original_conditions.shape[1])]
    
    # 各条件パラメータの値をプロット
    ax.bar(condition_names[:original_conditions.shape[1]], 
           original_conditions[0].cpu().numpy(), 
           alpha=0.7, label='Original')
    ax.bar(condition_names[:original_conditions.shape[1]], 
           perturbed_conditions[0].cpu().numpy(), 
           alpha=0.7, label='Perturbed')
    ax.set_ylabel('Condition Value')
    ax.set_title('Condition Parameters Comparison')
    ax.legend()
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 保存
    plot_path = os.path.join(save_dir, 'diffwave_condition_sensitivity.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    sensitivity_results = {
        'mean_trajectory_difference': float(np.mean(differences)),
        'std_trajectory_difference': float(np.std(differences)),
        'max_difference': float(np.max(differences)),
        'condition_change_magnitude': 0.1,
        'sensitivity_score': float(np.mean(differences) / 0.1)  # 変化量に対する軌道変化の比
    }
    
    print(f"  Condition Sensitivity Score: {sensitivity_results['sensitivity_score']:.4f}")
    print(f"  Mean Trajectory Difference: {sensitivity_results['mean_trajectory_difference']:.6f}")
    
    return sensitivity_results


def main():
    """
    過学習DiffWaveモデルの生成テスト
    """
    parser = argparse.ArgumentParser(description='DiffWave Overfit Generation Test')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to overfitted DiffWave model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output_dir', type=str, default='diffwave_overfit_generation_results', 
                       help='Output directory')
    parser.add_argument('--num_inference_steps', type=int, default=None, 
                       help='Number of inference steps (None for full)')
    parser.add_argument('--num_samples_per_condition', type=int, default=5, 
                       help='Number of samples per condition')
    
    args = parser.parse_args()
    
    # デバイス設定
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # チェックポイント読み込み
    print(f"Loading overfitted DiffWave model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # テストバッチデータ
    test_trajectories, test_conditions = checkpoint['test_batch']
    test_trajectories = test_trajectories.to(device)
    test_conditions = test_conditions.to(device)
    
    # モデル復元
    model_args = checkpoint['args']
    model = DiffWave1D(
        input_dim=2,
        condition_dim=model_args['condition_dim'],
        residual_channels=model_args['residual_channels'],
        skip_channels=model_args['skip_channels'],
        condition_channels=model_args['condition_channels'],
        num_layers=model_args['num_layers'],
        cycles=model_args['cycles'],
        time_embed_dim=model_args['time_embed_dim']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # スケジューラ作成
    scheduler = DDPMScheduler(num_timesteps=1000)
    
    # ジェネレータ作成
    generator = DiffWaveOverfitGenerator(model, scheduler, device)
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nGenerating trajectories...")
    print(f"Test batch shape: {test_trajectories.shape}")
    print(f"Test conditions shape: {test_conditions.shape}")
    
    # 軌道生成（再構成テスト）
    with torch.no_grad():
        generated_trajectories = generator.generate_trajectories(
            test_conditions,
            seq_len=test_trajectories.shape[-1],
            num_inference_steps=args.num_inference_steps,
            num_samples_per_condition=1
        )
    
    print(f"Generated trajectories shape: {generated_trajectories.shape}")
    
    # 再構成品質分析
    reconstruction_results = analyze_reconstruction_quality(
        test_trajectories, test_conditions, generated_trajectories, args.output_dir
    )
    
    # 条件感度テスト
    sensitivity_results = test_condition_sensitivity(
        model, scheduler, test_conditions, device, args.output_dir
    )
    
    # 複数サンプル生成テスト
    print(f"\nGenerating multiple samples per condition...")
    with torch.no_grad():
        multi_generated = generator.generate_trajectories(
            test_conditions,
            seq_len=test_trajectories.shape[-1],
            num_inference_steps=args.num_inference_steps,
            num_samples_per_condition=args.num_samples_per_condition
        )
    
    # 多様性可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i in range(min(4, len(test_conditions))):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # 元の軌道
        orig_traj = test_trajectories[i].cpu().numpy()
        ax.plot(orig_traj[0], orig_traj[1], 'k-', linewidth=3, alpha=0.8, label='Original')
        
        # 生成された複数サンプル
        start_idx = i * args.num_samples_per_condition
        end_idx = start_idx + args.num_samples_per_condition
        
        for j, gen_traj in enumerate(multi_generated[start_idx:end_idx].cpu().numpy()):
            ax.plot(gen_traj[0], gen_traj[1], '--', linewidth=2, alpha=0.7, 
                   label=f'Generated {j+1}')
        
        ax.set_title(f'DiffWave Condition {i+1}: Original vs Multiple Generated')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'diffwave_multiple_samples.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 総合結果保存
    final_results = {
        'reconstruction_results': reconstruction_results,
        'sensitivity_results': sensitivity_results,
        'model_info': {
            'parameters': sum(p.numel() for p in model.parameters()),
            'overfitting_converged': checkpoint['results']['converged'],
            'final_training_loss': checkpoint['results']['final_loss']
        },
        'generation_info': {
            'num_samples_per_condition': args.num_samples_per_condition,
            'inference_steps': args.num_inference_steps
        }
    }
    
    with open(os.path.join(args.output_dir, 'diffwave_generation_summary.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nDiffWave Generation Test Complete!")
    print(f"Results saved to: {args.output_dir}")
    
    # 軌道データ保存
    np.save(os.path.join(args.output_dir, 'diffwave_original_trajectories.npy'), 
            test_trajectories.cpu().numpy())
    np.save(os.path.join(args.output_dir, 'diffwave_generated_trajectories.npy'), 
            generated_trajectories.cpu().numpy())
    np.save(os.path.join(args.output_dir, 'diffwave_test_conditions.npy'), 
            test_conditions.cpu().numpy())


if __name__ == '__main__':
    main()