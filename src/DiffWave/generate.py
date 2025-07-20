# CLAUDE_ADDED
"""
DiffWaveベースモデルの軌道生成スクリプト
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import argparse
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.pytorch

# UNetの共通モジュールを参照
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'UNet'))

from Model import DiffWave1D
from TrajectoryDataset import TrajectoryDataset

# UNetのスケジューラを使用
import train as unet_train
DDPMScheduler = unet_train.DDPMScheduler


class DiffWaveGenerator:
    """
    DiffWaveベースの軌道生成クラス
    """
    def __init__(self, 
                 model: DiffWave1D, 
                 scheduler: DDPMScheduler, 
                 device: torch.device,
                 denormalize_trajectories: bool = True,
                 trajectory_scale_factor: float = 0.15):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.denormalize_trajectories = denormalize_trajectories
        self.trajectory_scale_factor = trajectory_scale_factor
        
        # 逆プロセス用のパラメータをデバイスに移動
        self.sqrt_recip_alphas = torch.sqrt(1.0 / scheduler.alphas).to(device)
        self.sqrt_one_minus_alphas_cumprod = scheduler.sqrt_one_minus_alphas_cumprod.to(device)
        
    @torch.no_grad()
    def ddpm_sample(self, 
                    shape: Tuple[int, ...], 
                    condition: torch.Tensor,
                    num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """
        DDPM サンプリング（DiffWave版）
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
            
            # DDPM更新式でx_{t-1}を計算
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
    
    @torch.no_grad()
    def ddim_sample(self, 
                    shape: Tuple[int, ...], 
                    condition: torch.Tensor,
                    num_inference_steps: int = 50,
                    eta: float = 0.0) -> torch.Tensor:
        """
        DDIM サンプリング（DiffWave版）
        """
        batch_size = shape[0]
        
        # 等間隔でタイムステップを選択
        step_size = self.scheduler.num_timesteps // num_inference_steps
        timesteps = list(range(0, self.scheduler.num_timesteps, step_size))[::-1]
        
        # 純粋なノイズから開始
        x = torch.randn(shape, device=self.device)
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # 次のタイムステップ
            prev_timestep = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            
            # モデルでノイズを予測
            predicted_noise = self.model(x, t_tensor, condition)
            
            # DDIM更新式
            alpha_cumprod_t = self.scheduler.alphas_cumprod[t].to(self.device)
            alpha_cumprod_prev = self.scheduler.alphas_cumprod[prev_timestep].to(self.device) if prev_timestep > 0 else torch.tensor(1.0, device=self.device)
            
            # 予測されたoriginal sample
            pred_original_sample = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            # 方向ベクトル
            pred_sample_direction = torch.sqrt(1 - alpha_cumprod_prev - eta**2 * (1 - alpha_cumprod_t) / alpha_cumprod_t * (1 - alpha_cumprod_prev)) * predicted_noise
            
            # DDIMステップ
            prev_sample = torch.sqrt(alpha_cumprod_prev) * pred_original_sample + pred_sample_direction
            
            # ランダム項（eta > 0の場合）
            if eta > 0:
                variance = eta**2 * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_prev)
                noise = torch.randn_like(x)
                prev_sample += torch.sqrt(variance) * noise
            
            x = prev_sample
        
        return x
    
    def denormalize_trajectory(self, normalized_trajectory: torch.Tensor) -> torch.Tensor:
        """
        正規化された軌道を元のスケールに戻す
        """
        if not self.denormalize_trajectories:
            return normalized_trajectory
            
        denormalized = normalized_trajectory * self.trajectory_scale_factor
        
        return denormalized
    
    def generate_trajectories(self, 
                            conditions: torch.Tensor,
                            seq_len: int = 101,
                            method: str = 'ddpm',
                            num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """
        指定された個人特性に基づいて軌道を生成
        """
        batch_size = conditions.shape[0]
        shape = (batch_size, 2, seq_len)
        
        # 拡散モデルで軌道生成（正規化済み）
        if method == 'ddpm':
            normalized_trajectories = self.ddpm_sample(shape, conditions, num_inference_steps)
        elif method == 'ddim':
            steps = num_inference_steps if num_inference_steps is not None else 50
            normalized_trajectories = self.ddim_sample(shape, conditions, steps)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        # 正規化を元に戻す
        trajectories = self.denormalize_trajectory(normalized_trajectories)
        
        return trajectories


def load_diffwave_model(checkpoint_path: str, 
                       condition_dim: int = None,
                       device: torch.device = torch.device('cpu')) -> DiffWave1D:
    """
    チェックポイントからDiffWaveモデルをロード
    """
    # チェックポイントを読み込んで構造を確認
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # condition_dimを推定（DiffWaveの場合はConditionEncoderから）
    if condition_dim is None:
        condition_proj_key = 'condition_encoder.condition_proj.weight'
        if condition_proj_key in checkpoint['model_state_dict']:
            condition_dim = checkpoint['model_state_dict'][condition_proj_key].shape[1]
            print(f'Detected condition_dim from checkpoint: {condition_dim}')
        else:
            print('Warning: Could not detect condition_dim from checkpoint, using default 5')
            condition_dim = 5
    
    # residual_channelsを推定
    residual_channels = 64  # デフォルト値
    input_proj_key = 'input_proj.weight'
    if input_proj_key in checkpoint['model_state_dict']:
        residual_channels = checkpoint['model_state_dict'][input_proj_key].shape[0]
        print(f'Detected residual_channels from checkpoint: {residual_channels}')
    
    # その他のパラメータも推定可能であれば推定
    model = DiffWave1D(
        input_dim=2,
        condition_dim=condition_dim,
        residual_channels=residual_channels,
        skip_channels=residual_channels,  # 通常同じ値
        condition_channels=128,  # デフォルト
        num_layers=20,  # デフォルト
        cycles=4,  # デフォルト
        time_embed_dim=128  # デフォルト
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def visualize_diffwave_trajectories(trajectories: np.ndarray, 
                                   conditions: np.ndarray,
                                   save_path: Optional[str] = None,
                                   denormalized: bool = True):
    """
    DiffWave生成軌道の可視化
    """
    batch_size = trajectories.shape[0]
    cols = min(4, batch_size)
    rows = (batch_size + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    unit = 'm' if denormalized else 'normalized'
    scale_info = 'Real Scale' if denormalized else 'Normalized (-1 to 1)'
    
    for i in range(batch_size):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        x_traj = trajectories[i, 0, :]
        y_traj = trajectories[i, 1, :]
        
        ax.plot(x_traj, y_traj, 'b-', linewidth=2, alpha=0.7)
        ax.plot(x_traj[0], y_traj[0], 'go', markersize=8, label='Start')
        ax.plot(x_traj[-1], y_traj[-1], 'ro', markersize=8, label='End')
        
        # 軌道の統計情報
        trajectory_length = np.sum(np.sqrt(np.diff(x_traj)**2 + np.diff(y_traj)**2))
        endpoint_distance = np.sqrt((x_traj[-1] - x_traj[0])**2 + (y_traj[-1] - y_traj[0])**2)
        
        title = f'DiffWave Trajectory {i+1}\n'
        if conditions.shape[1] >= 3:
            title += f'MT:{conditions[i, 0]:.2f}, EE:{conditions[i, 1]:.2f}, J:{conditions[i, 2]:.2f}\n'
        title += f'Length: {trajectory_length:.3f}{unit}, End-dist: {endpoint_distance:.3f}{unit}'
        
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(f'X Position ({unit})')
        ax.set_ylabel(f'Y Position ({unit})')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
        
        if denormalized:
            ax.set_xlim(-0.4, 0.4)
            ax.set_ylim(-0.4, 0.4)
        else:
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
    
    # 空のサブプロットを非表示
    for i in range(batch_size, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)
    
    fig.suptitle(f'DiffWave Generated Trajectories ({scale_info})', fontsize=14, y=0.95)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'DiffWave trajectories visualization saved to: {save_path}')
    else:
        plt.show()


@hydra.main(version_base=None, config_path=".", config_name="config")
def hydra_generate(cfg: DictConfig) -> None:
    """
    Hydra統合DiffWave生成関数
    """
    print(f"DiffWave Generation Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # デバイス設定
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 出力ディレクトリ作成
    output_dir = cfg.output.get('generation_dir', 'diffwave_generated_trajectories')
    os.makedirs(output_dir, exist_ok=True)
    
    # MLFlowセットアップ（生成ログ用）
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    with mlflow.start_run(run_name=f"diffwave_generation_{cfg.generation.method}"):
        # チェックポイントパスを決定（最新のものを使用）
        checkpoint_dir = cfg.output.checkpoint_dir
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth') and 'diffwave' in f]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
                print(f'Using latest DiffWave checkpoint: {checkpoint_path}')
            else:
                raise FileNotFoundError(f"No DiffWave checkpoints found in {checkpoint_dir}")
        else:
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # モデルロード
        model = load_diffwave_model(checkpoint_path, None, device)
        
        # スケジューラ作成
        scheduler = DDPMScheduler(
            num_timesteps=cfg.scheduler.num_timesteps,
            beta_start=cfg.scheduler.beta_start,
            beta_end=cfg.scheduler.beta_end
        )
        
        # ジェネレータ作成
        generator = DiffWaveGenerator(model, scheduler, device)
        
        # 条件次元の自動検出
        checkpoint = torch.load(checkpoint_path, map_location=device)
        condition_proj_key = 'condition_encoder.condition_proj.weight'
        if condition_proj_key in checkpoint['model_state_dict']:
            detected_condition_dim = checkpoint['model_state_dict'][condition_proj_key].shape[1]
        else:
            detected_condition_dim = cfg.model.condition_dim
        
        print(f'Using condition dimension: {detected_condition_dim}')
        
        # 条件データの準備
        if cfg.training.use_dummy or not cfg.data.train_data:
            print('Using dummy condition data...')
            conditions = np.random.randn(cfg.generation.num_samples, detected_condition_dim).astype(np.float32)
            conditions_source = 'dummy'
        else:
            print(f'Loading condition data from: {cfg.data.train_data}')
            try:
                dataset = TrajectoryDataset(cfg.data.train_data)
                
                total_samples = len(dataset)
                if cfg.generation.num_samples > total_samples:
                    indices = list(range(total_samples))
                else:
                    indices = np.random.choice(total_samples, cfg.generation.num_samples, replace=False).tolist()
                
                conditions_list = []
                for idx in indices:
                    _, condition = dataset[idx]
                    conditions_list.append(condition.numpy())
                
                conditions = np.array(conditions_list, dtype=np.float32)
                conditions_source = f'real_data (indices: {indices})'
                
            except Exception as e:
                print(f'Error loading real data: {e}')
                print('Falling back to dummy data...')
                conditions = np.random.randn(cfg.generation.num_samples, detected_condition_dim).astype(np.float32)
                conditions_source = 'dummy (fallback)'
        
        conditions_tensor = torch.FloatTensor(conditions).to(device)
        print(f'Condition data source: {conditions_source}')
        print(f'Condition data shape: {conditions.shape}')
        
        # 軌道生成
        print(f'Generating {len(conditions)} trajectories using DiffWave {cfg.generation.method} method...')
        import time
        start_time = time.time()
        
        with torch.no_grad():
            generated_trajectories = generator.generate_trajectories(
                conditions=conditions_tensor,
                seq_len=cfg.generation.seq_len,
                method=cfg.generation.method,
                num_inference_steps=cfg.generation.num_inference_steps
            )
        
        generation_time = time.time() - start_time
        print(f'DiffWave generation completed in {generation_time:.2f} seconds')
        
        # CPU に移動してnumpy配列に変換
        trajectories_np = generated_trajectories.cpu().numpy()
        
        # 可視化と保存
        vis_path = os.path.join(output_dir, 'diffwave_generated_trajectories.png')
        visualize_diffwave_trajectories(trajectories_np, conditions, vis_path)
        
        # 軌道データと条件データを保存
        np.save(os.path.join(output_dir, 'diffwave_trajectories.npy'), trajectories_np)
        np.save(os.path.join(output_dir, 'diffwave_conditions.npy'), conditions)
        
        # メタデータも保存
        metadata = {
            'model_type': 'DiffWave',
            'condition_source': conditions_source,
            'method': cfg.generation.method,
            'num_inference_steps': cfg.generation.num_inference_steps,
            'condition_dim': detected_condition_dim,
            'seq_len': cfg.generation.seq_len,
            'checkpoint_path': checkpoint_path,
            'generation_time': generation_time
        }
        
        import json
        with open(os.path.join(output_dir, 'diffwave_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # MLFlowにメトリクスとアーティファクトをログ
        mlflow.log_metrics({
            "generation_time": generation_time,
            "num_generated": len(trajectories_np),
            "trajectory_x_mean": float(trajectories_np[:, 0, :].mean()),
            "trajectory_y_mean": float(trajectories_np[:, 1, :].mean()),
            "trajectory_x_std": float(trajectories_np[:, 0, :].std()),
            "trajectory_y_std": float(trajectories_np[:, 1, :].std())
        })
        
        mlflow.log_artifact(vis_path, "diffwave_plots")
        mlflow.log_artifact(os.path.join(output_dir, 'diffwave_metadata.json'), "metadata")
        
        print(f'DiffWave generation completed! Results saved to: {output_dir}')


def main():
    """
    argparse対応メイン関数（後方互換性のため）
    """
    parser = argparse.ArgumentParser(description='DiffWave Trajectory Generation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to DiffWave model checkpoint')
    parser.add_argument('--data_path', type=str, default=None, help='Path to real data (.npz file) for conditions')
    parser.add_argument('--condition_dim', type=int, default=5, help='Condition dimension (used when generating dummy data)')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of trajectories to generate')
    parser.add_argument('--seq_len', type=int, default=101, help='Sequence length')
    parser.add_argument('--method', type=str, default='ddpm', choices=['ddpm', 'ddim'], help='Sampling method')
    parser.add_argument('--steps', type=int, default=None, help='Number of inference steps')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output_dir', type=str, default='diffwave_generated_trajectories', help='Output directory')
    parser.add_argument('--use_dummy', action='store_true', help='Use dummy conditions instead of real data')
    
    args = parser.parse_args()
    
    # デバイス設定
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # モデルロード
    print(f'Loading DiffWave model from: {args.checkpoint}')
    model = load_diffwave_model(args.checkpoint, None, device)
    
    # スケジューラ作成
    scheduler = DDPMScheduler(num_timesteps=1000)
    
    # ジェネレータ作成
    generator = DiffWaveGenerator(model, scheduler, device)
    
    # 条件データの準備
    if args.use_dummy or args.data_path is None:
        print('Using dummy condition data...')
        conditions = np.random.randn(args.batch_size, args.condition_dim).astype(np.float32)
        conditions_source = 'dummy'
    else:
        print(f'Loading condition data from: {args.data_path}')
        try:
            dataset = TrajectoryDataset(args.data_path)
            
            total_samples = len(dataset)
            if args.batch_size > total_samples:
                indices = list(range(total_samples))
            else:
                indices = np.random.choice(total_samples, args.batch_size, replace=False).tolist()
            
            conditions_list = []
            for idx in indices:
                _, condition = dataset[idx]
                conditions_list.append(condition.numpy())
            
            conditions = np.array(conditions_list, dtype=np.float32)
            conditions_source = f'real_data (indices: {indices})'
            
        except Exception as e:
            print(f'Error loading real data: {e}')
            print('Falling back to dummy data...')
            conditions = np.random.randn(args.batch_size, args.condition_dim).astype(np.float32)
            conditions_source = 'dummy (fallback)'
    
    conditions_tensor = torch.FloatTensor(conditions).to(device)
    print(f'Condition data source: {conditions_source}')
    print(f'Condition data shape: {conditions.shape}')
    
    # 軌道生成
    print(f'Generating {len(conditions)} trajectories using DiffWave {args.method} method...')
    with torch.no_grad():
        generated_trajectories = generator.generate_trajectories(
            conditions=conditions_tensor,
            seq_len=args.seq_len,
            method=args.method,
            num_inference_steps=args.steps
        )
    
    # CPU に移動してnumpy配列に変換
    trajectories_np = generated_trajectories.cpu().numpy()
    
    # 可視化と保存
    vis_path = os.path.join(args.output_dir, 'diffwave_generated_trajectories.png')
    visualize_diffwave_trajectories(trajectories_np, conditions, vis_path)
    
    # 軌道データと条件データを保存
    np.save(os.path.join(args.output_dir, 'diffwave_trajectories.npy'), trajectories_np)
    np.save(os.path.join(args.output_dir, 'diffwave_conditions.npy'), conditions)
    
    print(f'DiffWave generation completed! Results saved to: {args.output_dir}')


if __name__ == '__main__':
    main()