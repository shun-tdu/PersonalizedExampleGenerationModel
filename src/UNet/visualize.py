# CLAUDE_ADDED
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import argparse
import os
from Model import UNet1D
from train import DDPMScheduler, TrajectoryDataset
from generate import TrajectoryGenerator, load_model
from torch.utils.data import DataLoader


class TrainingVisualizer:
    """
    訓練プロセスの可視化クラス
    """
    def __init__(self, output_dir: str = 'visualization_outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def log_model_architecture(self, model: UNet1D):
        """
        モデルアーキテクチャの可視化
        """
        # パラメータ数の計算
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # レイヤー別パラメータ数
        layer_params = {}
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 末端のモジュールのみ
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    layer_params[name] = params
        
        # パラメータ数をファイルに保存
        import pandas as pd
        layer_df = pd.DataFrame(list(layer_params.items()), columns=['Layer', 'Parameters'])
        layer_df.to_csv(os.path.join(self.output_dir, 'model_parameters.csv'), index=False)
        print(f'Model parameters saved to {self.output_dir}/model_parameters.csv')
    
    def log_noise_schedule(self, scheduler: DDPMScheduler):
        """
        ノイズスケジュールの可視化
        """
        timesteps = np.arange(scheduler.num_timesteps)
        
        # βスケジュール
        betas = scheduler.betas.numpy()
        fig = self._create_line_plot(timesteps, betas, 'Timesteps', 'Beta Values', 'Beta Schedule')
        fig.savefig(os.path.join(self.output_dir, 'noise_schedule_betas.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # α累積積
        alphas_cumprod = scheduler.alphas_cumprod.numpy()
        fig = self._create_line_plot(timesteps, alphas_cumprod, 'Timesteps', 'Alpha Cumprod', 'Alpha Cumulative Product')
        fig.savefig(os.path.join(self.output_dir, 'noise_schedule_alphas_cumprod.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # ノイズレベル
        noise_levels = scheduler.sqrt_one_minus_alphas_cumprod.numpy()
        fig = self._create_line_plot(timesteps, noise_levels, 'Timesteps', 'Noise Level', 'Noise Level Schedule')
        fig.savefig(os.path.join(self.output_dir, 'noise_schedule_levels.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f'Noise schedule plots saved to {self.output_dir}/')
    
    def log_training_samples(self, 
                           original_trajectories: torch.Tensor,
                           noisy_trajectories: torch.Tensor,
                           predicted_noise: torch.Tensor,
                           actual_noise: torch.Tensor,
                           conditions: torch.Tensor,
                           timesteps: torch.Tensor,
                           step: int):
        """
        訓練サンプルの可視化
        """
        # バッチから最初の4サンプルを選択
        batch_size = min(4, original_trajectories.shape[0])
        
        fig, axes = plt.subplots(2, batch_size, figsize=(4*batch_size, 8))
        
        for i in range(batch_size):
            # 軌道の可視化
            ax1 = axes[0, i]
            orig_traj = original_trajectories[i].cpu().numpy()
            noisy_traj = noisy_trajectories[i].cpu().numpy()
            
            ax1.plot(orig_traj[0], orig_traj[1], 'b-', label='Original', linewidth=2)
            ax1.plot(noisy_traj[0], noisy_traj[1], 'r--', label=f'Noisy (t={timesteps[i].item()})', alpha=0.7)
            ax1.set_title(f'Sample {i+1}: Trajectory\nCondition: {conditions[i, :3].cpu().numpy()}')
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            
            # ノイズ予測の比較
            ax2 = axes[1, i]
            pred_noise = predicted_noise[i].cpu().numpy()
            true_noise = actual_noise[i].cpu().numpy()
            
            time_axis = np.arange(pred_noise.shape[1])
            ax2.plot(time_axis, pred_noise[0], 'g-', label='Predicted (X)', linewidth=2)
            ax2.plot(time_axis, true_noise[0], 'g--', label='Actual (X)', alpha=0.7)
            ax2.plot(time_axis, pred_noise[1], 'orange', label='Predicted (Y)', linewidth=2)
            ax2.plot(time_axis, true_noise[1], 'orange', linestyle='--', label='Actual (Y)', alpha=0.7)
            
            ax2.set_title(f'Noise Prediction Comparison')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Noise Value')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f'training_samples_step_{step}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Training samples plot saved to {self.output_dir}/training_samples_step_{step}.png')
    
    def log_generation_process(self, 
                             generator: TrajectoryGenerator,
                             condition: torch.Tensor,
                             seq_len: int = 101,
                             step: int = 0):
        """
        生成プロセスの可視化（段階的なノイズ除去）
        """
        device = condition.device
        shape = (1, 2, seq_len)
        
        # 特定のタイムステップでの中間結果を保存
        timesteps_to_visualize = [999, 800, 600, 400, 200, 100, 50, 0]
        intermediate_results = []
        
        # DDPM サンプリングの改造版（中間結果保存）
        with torch.no_grad():
            x = torch.randn(shape, device=device)
            
            for t in range(generator.scheduler.num_timesteps-1, -1, -1):
                if t in timesteps_to_visualize:
                    intermediate_results.append((t, x.clone()))
                
                t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
                predicted_noise = generator.model(x, t_tensor, condition)
                
                if t > 0:
                    sqrt_recip_alpha_t = generator.sqrt_recip_alphas[t]
                    beta_t = generator.scheduler.betas[t]
                    sqrt_one_minus_alphas_cumprod_t = generator.sqrt_one_minus_alphas_cumprod[t]
                    
                    mean = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alphas_cumprod_t * predicted_noise)
                    
                    alpha_t = generator.scheduler.alphas[t]
                    alpha_cumprod_t = generator.scheduler.alphas_cumprod[t]
                    alpha_cumprod_prev = generator.scheduler.alphas_cumprod_prev[t]
                    
                    variance = beta_t * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
                    sigma = torch.sqrt(variance)
                    noise = torch.randn_like(x)
                    
                    x = mean + sigma * noise
                else:
                    sqrt_recip_alpha_t = generator.sqrt_recip_alphas[t]
                    beta_t = generator.scheduler.betas[t]
                    sqrt_one_minus_alphas_cumprod_t = generator.sqrt_one_minus_alphas_cumprod[t]
                    x = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alphas_cumprod_t * predicted_noise)
        
        # 最終結果も追加
        intermediate_results.append((0, x.clone()))
        
        # 可視化
        num_steps = len(intermediate_results)
        cols = min(4, num_steps)
        rows = (num_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (timestep, trajectory) in enumerate(intermediate_results):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            traj_np = trajectory[0].cpu().numpy()
            ax.plot(traj_np[0], traj_np[1], 'b-', linewidth=2)
            ax.plot(traj_np[0, 0], traj_np[1, 0], 'go', markersize=8)
            ax.plot(traj_np[0, -1], traj_np[1, -1], 'ro', markersize=8)
            
            ax.set_title(f't = {timestep}')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        # 空のサブプロットを非表示
        for idx in range(num_steps, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f'generation_process_step_{step}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Generation process plot saved to {self.output_dir}/generation_process_step_{step}.png')
    
    def log_condition_analysis(self, 
                             trajectories: torch.Tensor,
                             conditions: torch.Tensor,
                             step: int):
        """
        個人特性と生成軌道の関係分析
        """
        trajectories_np = trajectories.cpu().numpy()
        conditions_np = conditions.cpu().numpy()
        
        # 軌道の特徴量計算
        trajectory_features = self._extract_trajectory_features(trajectories_np)
        
        # 相関分析
        condition_names = [f'Condition_{i}' for i in range(conditions_np.shape[1])]
        feature_names = list(trajectory_features.keys())
        
        # 相関行列の計算
        all_data = np.column_stack([conditions_np] + list(trajectory_features.values()))
        corr_matrix = np.corrcoef(all_data.T)
        
        # 条件と軌道特徴の相関のみ抽出
        condition_feature_corr = corr_matrix[:len(condition_names), len(condition_names):]
        
        # ヒートマップ作成
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(condition_feature_corr, 
                   xticklabels=feature_names,
                   yticklabels=condition_names,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   ax=ax)
        ax.set_title('Correlation between Conditions and Trajectory Features')
        plt.tight_layout()
        
        fig.savefig(os.path.join(self.output_dir, f'condition_correlation_step_{step}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Condition correlation plot saved to {self.output_dir}/condition_correlation_step_{step}.png')
    
    def _extract_trajectory_features(self, trajectories: np.ndarray) -> Dict[str, np.ndarray]:
        """
        軌道から特徴量を抽出
        """
        features = {}
        
        # 移動距離
        distances = np.sqrt(np.sum(np.diff(trajectories, axis=2)**2, axis=1))
        features['Total_Distance'] = np.sum(distances, axis=1)
        
        # 最大速度
        velocities = np.sqrt(np.sum(np.diff(trajectories, axis=2)**2, axis=1))
        features['Max_Velocity'] = np.max(velocities, axis=1)
        
        # 終点誤差（原点からの距離と仮定）
        end_points = trajectories[:, :, -1]
        features['End_Point_Error'] = np.sqrt(np.sum(end_points**2, axis=1))
        
        # 軌道の滑らかさ（ジャーク）
        acceleration = np.diff(velocities, axis=1)
        jerk = np.diff(acceleration, axis=1)
        features['Jerk_Mean'] = np.mean(np.abs(jerk), axis=1)
        
        # 軌道の範囲
        features['X_Range'] = np.ptp(trajectories[:, 0, :], axis=1)
        features['Y_Range'] = np.ptp(trajectories[:, 1, :], axis=1)
        
        return features
    
    def _create_line_plot(self, x: np.ndarray, y: np.ndarray, 
                         xlabel: str, ylabel: str, title: str) -> plt.Figure:
        """
        線グラフの作成
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig
    
    def close(self):
        """
        可視化完了
        """
        print(f'All visualizations saved to {self.output_dir}/')


def create_enhanced_training_visualizer(model_path: str,
                                      data_loader: DataLoader,
                                      device: torch.device) -> None:
    """
    拡張された訓練可視化の実行
    """
    # モデルとスケジューラの準備
    model = load_model(model_path, condition_dim=5, device=device)
    scheduler = DDPMScheduler()
    generator = TrajectoryGenerator(model, scheduler, device)
    
    # 可視化器の初期化
    visualizer = TrainingVisualizer('visualization_outputs')
    
    # モデルアーキテクチャの可視化
    visualizer.log_model_architecture(model)
    
    # ノイズスケジュールの可視化
    visualizer.log_noise_schedule(scheduler)
    
    # サンプルデータでの可視化
    with torch.no_grad():
        batch_trajectories, batch_conditions = next(iter(data_loader))
        batch_trajectories = batch_trajectories.to(device)
        batch_conditions = batch_conditions.to(device)
        
        # 訓練サンプルの可視化
        batch_size = batch_trajectories.shape[0]
        timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(batch_trajectories)
        noisy_trajectories = scheduler.add_noise(batch_trajectories, noise, timesteps)
        predicted_noise = model(noisy_trajectories, timesteps, batch_conditions)
        
        visualizer.log_training_samples(
            batch_trajectories, noisy_trajectories, predicted_noise, noise, 
            batch_conditions, timesteps, 0
        )
        
        # 生成プロセスの可視化
        sample_condition = batch_conditions[:1]
        visualizer.log_generation_process(generator, sample_condition, step=0)
        
        # 複数のサンプルを生成して条件分析
        generated_trajectories = generator.generate_trajectories(
            batch_conditions, method='ddim', num_inference_steps=50
        )
        visualizer.log_condition_analysis(generated_trajectories, batch_conditions, 0)
    
    visualizer.close()
    print("Enhanced visualization completed! Check visualization_outputs/ for results.")


def main():
    parser = argparse.ArgumentParser(description='Training Visualization')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--condition_dim', type=int, default=5, help='Condition dimension')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for visualization')
    parser.add_argument('--seq_len', type=int, default=101, help='Sequence length')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_samples', type=int, default=200, help='Number of samples for visualization')
    
    args = parser.parse_args()
    
    # デバイス設定
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # ダミーデータローダーの作成
    dummy_trajectories = np.random.randn(args.num_samples, 2, args.seq_len).astype(np.float32)
    dummy_conditions = np.random.randn(args.num_samples, args.condition_dim).astype(np.float32)
    
    from train import TrajectoryDataset
    dataset = TrajectoryDataset(dummy_trajectories, dummy_conditions)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 拡張可視化の実行
    create_enhanced_training_visualizer(args.checkpoint, data_loader, device)
    
    print("Visualization completed! Check visualization_outputs/ for PNG files.")


if __name__ == '__main__':
    main()