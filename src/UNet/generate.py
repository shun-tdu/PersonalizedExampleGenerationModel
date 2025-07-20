# CLAUDE_ADDED
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import argparse
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.pytorch
from Model import UNet1D
from train import DDPMScheduler
from TrajectoryDataset import TrajectoryDataset


class TrajectoryGenerator:
    """
    拡散モデルを使った軌道生成クラス
    """
    def __init__(self, 
                 model: UNet1D, 
                 scheduler: DDPMScheduler, 
                 device: torch.device,
                 denormalize_trajectories: bool = True,
                 trajectory_scale_factor: float = 0.15):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.denormalize_trajectories = denormalize_trajectories
        self.trajectory_scale_factor = trajectory_scale_factor  # 正規化時に使用された最大値
        
        # 逆プロセス用のパラメータをデバイスに移動
        self.sqrt_recip_alphas = torch.sqrt(1.0 / scheduler.alphas).to(device)
        self.sqrt_one_minus_alphas_cumprod = scheduler.sqrt_one_minus_alphas_cumprod.to(device)
        
    @torch.no_grad()
    def ddpm_sample(self, 
                    shape: Tuple[int, ...], 
                    condition: torch.Tensor,
                    num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """
        DDPM サンプリング（逆拡散プロセス）
        
        :param shape: 生成する軌道の形状 [batch_size, 2, seq_len]
        :param condition: 個人特性ベクトル [batch_size, condition_dim]
        :param num_inference_steps: 推論ステップ数（Noneの場合は全ステップ）
        :return: 生成された軌道 [batch_size, 2, seq_len]
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
                # x_{t-1} = 1/sqrt(α_t) * (x_t - β_t/sqrt(1-α̅_t) * ε_θ) + σ_t * z
                sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
                beta_t = self.scheduler.betas[t]
                sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
                
                # 平均項の計算
                mean = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alphas_cumprod_t * predicted_noise)
                
                # 分散項の計算
                alpha_t = self.scheduler.alphas[t].to(self.device)
                alpha_cumprod_t = self.scheduler.alphas_cumprod[t].to(self.device)
                alpha_cumprod_prev = self.scheduler.alphas_cumprod_prev[t].to(self.device)
                
                # σ_t^2 = β_t * (1 - α̅_{t-1}) / (1 - α̅_t)
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
        DDIM サンプリング（より高速な推論）
        
        :param shape: 生成する軌道の形状 [batch_size, 2, seq_len]
        :param condition: 個人特性ベクトル [batch_size, condition_dim]
        :param num_inference_steps: 推論ステップ数
        :param eta: DDIMパラメータ（0.0でdeterministic, 1.0でDDPM相当）
        :return: 生成された軌道 [batch_size, 2, seq_len]
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
        
        :param normalized_trajectory: 正規化済み軌道 [batch_size, 2, seq_len] (-1~1範囲)
        :return: 元スケールの軌道 [batch_size, 2, seq_len] (メートル単位)
        """
        if not self.denormalize_trajectories:
            return normalized_trajectory
            
        # 正規化の逆変換：各軌道が個別に正規化されていたので、
        # データ生成時の最大値(0.15m)をスケール基準として使用
        denormalized = normalized_trajectory * self.trajectory_scale_factor
        
        return denormalized
    
    def generate_trajectories(self, 
                            conditions: torch.Tensor,
                            seq_len: int = 101,
                            method: str = 'ddpm',
                            num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """
        指定された個人特性に基づいて軌道を生成
        
        :param conditions: 個人特性ベクトル [batch_size, condition_dim]
        :param seq_len: 軌道の長さ
        :param method: サンプリング手法 ('ddpm' or 'ddim')
        :param num_inference_steps: 推論ステップ数
        :return: 生成された軌道 [batch_size, 2, seq_len]
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


def load_model(checkpoint_path: str, 
               condition_dim: int = None,
               device: torch.device = torch.device('cpu')) -> UNet1D:
    """
    チェックポイントからモデルをロード
    """
    # まずチェックポイントを読み込んで構造を確認
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # チェックポイントから condition_dim を推定
    if condition_dim is None:
        # CrossAttentionのto_k.weightから条件次元を推定
        # encoder_blocks.0.cross_attention.to_k.weight の形状: [inner_dim, condition_dim]
        sample_key = 'encoder_blocks.0.cross_attention.to_k.weight'
        if sample_key in checkpoint['model_state_dict']:
            condition_dim = checkpoint['model_state_dict'][sample_key].shape[1]
            print(f'Detected condition_dim from checkpoint: {condition_dim}')
        else:
            print('Warning: Could not detect condition_dim from checkpoint, using default 5')
            condition_dim = 5
    
    # チェックポイントからbase_channelsを推定
    # input_proj.weight の形状: [base_channels, input_dim, 1]
    base_channels = 64  # デフォルト値
    input_proj_key = 'input_proj.weight'
    if input_proj_key in checkpoint['model_state_dict']:
        base_channels = checkpoint['model_state_dict'][input_proj_key].shape[0]
        print(f'Detected base_channels from checkpoint: {base_channels}')
    
    model = UNet1D(
        input_dim=2,
        condition_dim=condition_dim,
        time_embed_dim=128,
        base_channels=base_channels
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def visualize_trajectories(trajectories: np.ndarray, 
                         conditions: np.ndarray,
                         save_path: Optional[str] = None,
                         denormalized: bool = True):
    """
    生成された軌道を可視化（改善版）
    
    :param trajectories: [batch_size, 2, seq_len] の軌道データ
    :param conditions: [batch_size, condition_dim] の個人特性データ
    :param save_path: 保存パス（Noneの場合は表示のみ）
    :param denormalized: 軌道が既に正規化解除されているかどうか
    """
    batch_size = trajectories.shape[0]
    cols = min(4, batch_size)
    rows = (batch_size + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # 単位とスケールの情報
    unit = 'm' if denormalized else 'normalized'
    scale_info = 'Real Scale' if denormalized else 'Normalized (-1 to 1)'
    
    for i in range(batch_size):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        x_traj = trajectories[i, 0, :]  # X軸軌道
        y_traj = trajectories[i, 1, :]  # Y軸軌道
        
        ax.plot(x_traj, y_traj, 'b-', linewidth=2, alpha=0.7)
        ax.plot(x_traj[0], y_traj[0], 'go', markersize=8, label='Start')
        ax.plot(x_traj[-1], y_traj[-1], 'ro', markersize=8, label='End')
        
        # 軌道の統計情報
        trajectory_length = np.sum(np.sqrt(np.diff(x_traj)**2 + np.diff(y_traj)**2))
        endpoint_distance = np.sqrt((x_traj[-1] - x_traj[0])**2 + (y_traj[-1] - y_traj[0])**2)
        
        # タイトルに統計情報を含める
        title = f'Trajectory {i+1}\n'
        if conditions.shape[1] >= 3:
            title += f'MT:{conditions[i, 0]:.2f}, EE:{conditions[i, 1]:.2f}, J:{conditions[i, 2]:.2f}\n'
        title += f'Length: {trajectory_length:.3f}{unit}, End-dist: {endpoint_distance:.3f}{unit}'
        
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(f'X Position ({unit})')
        ax.set_ylabel(f'Y Position ({unit})')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
        
        # 軸の範囲を適切に設定
        if denormalized:
            # 実スケールの場合、-0.2m to 0.2m程度の範囲
            ax.set_xlim(-0.4, 0.4)
            ax.set_ylim(-0.4, 0.4)
        else:
            # 正規化済みの場合、-1.5 to 1.5程度
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
    
    # 空のサブプロットを非表示
    for i in range(batch_size, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)
    
    # 全体のタイトル
    fig.suptitle(f'Generated Trajectories ({scale_info})', fontsize=14, y=0.95)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Trajectories visualization saved to: {save_path}')
        
        # 統計情報をテキストファイルにも保存
        stats_path = save_path.replace('.png', '_stats.txt')
        with open(stats_path, 'w') as f:
            f.write(f'Trajectory Generation Statistics ({scale_info})\n')
            f.write('='*50 + '\n')
            f.write(f'Number of trajectories: {batch_size}\n')
            f.write(f'Sequence length: {trajectories.shape[2]}\n')
            f.write(f'X range: [{trajectories[:, 0, :].min():.4f}, {trajectories[:, 0, :].max():.4f}] {unit}\n')
            f.write(f'Y range: [{trajectories[:, 1, :].min():.4f}, {trajectories[:, 1, :].max():.4f}] {unit}\n')
            f.write(f'X mean: {trajectories[:, 0, :].mean():.4f} {unit}\n')
            f.write(f'Y mean: {trajectories[:, 1, :].mean():.4f} {unit}\n')
            f.write(f'X std: {trajectories[:, 0, :].std():.4f} {unit}\n')
            f.write(f'Y std: {trajectories[:, 1, :].std():.4f} {unit}\n')
            f.write('\nPer-trajectory statistics:\n')
            for i in range(batch_size):
                x_traj = trajectories[i, 0, :]
                y_traj = trajectories[i, 1, :]
                length = np.sum(np.sqrt(np.diff(x_traj)**2 + np.diff(y_traj)**2))
                end_dist = np.sqrt((x_traj[-1] - x_traj[0])**2 + (y_traj[-1] - y_traj[0])**2)
                f.write(f'  Traj {i+1}: Length={length:.4f}{unit}, End-dist={end_dist:.4f}{unit}\n')
        
        print(f'Statistics saved to: {stats_path}')
    else:
        plt.show()


@hydra.main(version_base=None, config_path=".", config_name="config")
def hydra_generate(cfg: DictConfig) -> None:
    """
    Hydra統合生成関数
    """
    print(f"Generation Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # デバイス設定
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 出力ディレクトリ作成
    output_dir = cfg.output.get('generated_dir', 'outputs/generated_trajectories')
    os.makedirs(output_dir, exist_ok=True)
    
    # MLFlowセットアップ（生成ログ用）
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    with mlflow.start_run(run_name=f"generation_{cfg.generation.method}"):
        # 設定をMLFlowにログ
        generation_params = {
            "generation_method": cfg.generation.method,
            "num_inference_steps": cfg.generation.num_inference_steps,
            "seq_len": cfg.generation.seq_len,
            "num_samples": cfg.generation.num_samples
        }
        mlflow.log_params(generation_params)
        
        # チェックポイントパスを決定（最新のものを使用）
        checkpoint_dir = cfg.output.checkpoint_dir
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if checkpoints:
                # エポック番号でソートして最新を取得
                checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
                print(f'Using latest checkpoint: {checkpoint_path}')
            else:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        else:
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # モデルロード
        model = load_model(checkpoint_path, None, device)
        
        # スケジューラ作成
        scheduler = DDPMScheduler(
            num_timesteps=cfg.scheduler.num_timesteps,
            beta_start=cfg.scheduler.beta_start,
            beta_end=cfg.scheduler.beta_end
        )
        
        # ジェネレータ作成
        generator = TrajectoryGenerator(model, scheduler, device)
        
        # 条件次元の自動検出
        detected_condition_dim = None
        checkpoint = torch.load(checkpoint_path, map_location=device)
        sample_key = 'encoder_blocks.0.cross_attention.to_k.weight'
        if sample_key in checkpoint['model_state_dict']:
            detected_condition_dim = checkpoint['model_state_dict'][sample_key].shape[1]
        
        actual_condition_dim = detected_condition_dim if detected_condition_dim is not None else cfg.model.condition_dim
        print(f'Using condition dimension: {actual_condition_dim}')
        
        # 条件データの準備
        if cfg.training.use_dummy or not cfg.data.train_data:
            # ダミー個人特性データ生成
            print('Using dummy condition data...')
            conditions = np.random.randn(cfg.generation.num_samples, actual_condition_dim).astype(np.float32)
            conditions_source = 'dummy'
        else:
            # 実データから条件を取得
            print(f'Loading condition data from: {cfg.data.train_data}')
            try:
                dataset = TrajectoryDataset(cfg.data.train_data)
                
                # ランダムサンプリング
                total_samples = len(dataset)
                if cfg.generation.num_samples > total_samples:
                    print(f'Warning: num_samples ({cfg.generation.num_samples}) > dataset size ({total_samples}), using all data')
                    indices = list(range(total_samples))
                else:
                    indices = np.random.choice(total_samples, cfg.generation.num_samples, replace=False).tolist()
                print(f'Randomly sampled indices: {indices}')
                
                # 条件データを取得
                conditions_list = []
                for idx in indices:
                    _, condition = dataset[idx]
                    conditions_list.append(condition.numpy())
                
                conditions = np.array(conditions_list, dtype=np.float32)
                conditions_source = f'real_data (indices: {indices})'
                
            except Exception as e:
                print(f'Error loading real data: {e}')
                print('Falling back to dummy data...')
                conditions = np.random.randn(cfg.generation.num_samples, actual_condition_dim).astype(np.float32)
                conditions_source = 'dummy (fallback)'
        
        conditions_tensor = torch.FloatTensor(conditions).to(device)
        print(f'Condition data source: {conditions_source}')
        print(f'Condition data shape: {conditions.shape}')
        
        # 軌道生成
        print(f'Generating {len(conditions)} trajectories using {cfg.generation.method} method...')
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
        print(f'Generation completed in {generation_time:.2f} seconds')
        
        # CPU に移動してnumpy配列に変換
        trajectories_np = generated_trajectories.cpu().numpy()
        
        # 可視化と保存
        vis_path = os.path.join(output_dir, 'generated_trajectories.png')
        visualize_trajectories(trajectories_np, conditions, vis_path)
        
        # 軌道データと条件データを保存
        np.save(os.path.join(output_dir, 'trajectories.npy'), trajectories_np)
        np.save(os.path.join(output_dir, 'conditions.npy'), conditions)
        
        # メタデータも保存
        metadata = {
            'condition_source': conditions_source,
            'method': cfg.generation.method,
            'num_inference_steps': cfg.generation.num_inference_steps,
            'condition_dim': actual_condition_dim,
            'seq_len': cfg.generation.seq_len,
            'checkpoint_path': checkpoint_path,
            'generation_time': generation_time
        }
        
        import json
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
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
        
        # アーティファクトを保存
        mlflow.log_artifact(vis_path, "generated_plots")
        mlflow.log_artifact(os.path.join(output_dir, 'metadata.json'), "metadata")
        
        print(f'Generation completed! Results saved to: {output_dir}')


def main():
    """
    argparse対応メイン関数（後方互換性のため）
    """
    parser = argparse.ArgumentParser(description='Trajectory Generation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default=None, help='Path to real data (.npz file) for conditions')
    parser.add_argument('--condition_dim', type=int, default=5, help='Condition dimension (used when generating dummy data)')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of trajectories to generate')
    parser.add_argument('--seq_len', type=int, default=101, help='Sequence length')
    parser.add_argument('--method', type=str, default='ddpm', choices=['ddpm', 'ddim'], help='Sampling method')
    parser.add_argument('--steps', type=int, default=None, help='Number of inference steps')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output_dir', type=str, default='generated_trajectories', help='Output directory')
    parser.add_argument('--use_dummy', action='store_true', help='Use dummy conditions instead of real data')
    parser.add_argument('--sample_indices', type=str, default=None, help='Comma-separated indices to sample from data (e.g., "0,5,10")')
    
    args = parser.parse_args()
    
    # デバイス設定
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # モデルロード
    print(f'Loading model from: {args.checkpoint}')
    model = load_model(args.checkpoint, None, device)  # condition_dimを自動検出
    
    # スケジューラ作成
    scheduler = DDPMScheduler(num_timesteps=1000)
    
    # ジェネレータ作成
    generator = TrajectoryGenerator(model, scheduler, device)
    
    # 自動検出されたcondition_dimを取得
    detected_condition_dim = None
    checkpoint = torch.load(args.checkpoint, map_location=device)
    sample_key = 'encoder_blocks.0.cross_attention.to_k.weight'
    if sample_key in checkpoint['model_state_dict']:
        detected_condition_dim = checkpoint['model_state_dict'][sample_key].shape[1]
    
    actual_condition_dim = detected_condition_dim if detected_condition_dim is not None else args.condition_dim
    print(f'Using condition dimension: {actual_condition_dim}')
    
    # 条件データの準備
    if args.use_dummy or args.data_path is None:
        # ダミー個人特性データ生成
        print('Using dummy condition data...')
        conditions = np.random.randn(args.batch_size, actual_condition_dim).astype(np.float32)
        conditions_source = 'dummy'
    else:
        # 実データから条件を取得
        print(f'Loading condition data from: {args.data_path}')
        try:
            dataset = TrajectoryDataset(args.data_path)
            
            # サンプル対象のインデックスを決定
            if args.sample_indices:
                # 指定されたインデックスを使用
                indices = [int(idx.strip()) for idx in args.sample_indices.split(',')]
                indices = [idx for idx in indices if 0 <= idx < len(dataset)]  # 範囲チェック
                if len(indices) == 0:
                    raise ValueError("No valid indices provided")
                print(f'Using specified indices: {indices}')
            else:
                # ランダムサンプリング
                total_samples = len(dataset)
                if args.batch_size > total_samples:
                    print(f'Warning: batch_size ({args.batch_size}) > dataset size ({total_samples}), using all data')
                    indices = list(range(total_samples))
                else:
                    indices = np.random.choice(total_samples, args.batch_size, replace=False).tolist()
                print(f'Randomly sampled indices: {indices}')
            
            # 条件データを取得
            conditions_list = []
            for idx in indices:
                _, condition = dataset[idx]
                conditions_list.append(condition.numpy())
            
            conditions = np.array(conditions_list, dtype=np.float32)
            conditions_source = f'real_data (indices: {indices})'
            
        except Exception as e:
            print(f'Error loading real data: {e}')
            print('Falling back to dummy data...')
            conditions = np.random.randn(args.batch_size, actual_condition_dim).astype(np.float32)
            conditions_source = 'dummy (fallback)'
    
    conditions_tensor = torch.FloatTensor(conditions).to(device)
    print(f'Condition data source: {conditions_source}')
    print(f'Condition data shape: {conditions.shape}')
    
    # 軌道生成
    print(f'Generating {len(conditions)} trajectories using {args.method} method...')
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
    vis_path = os.path.join(args.output_dir, 'generated_trajectories.png')
    visualize_trajectories(trajectories_np, conditions, vis_path)
    
    # 軌道データと条件データを保存
    np.save(os.path.join(args.output_dir, 'trajectories.npy'), trajectories_np)
    np.save(os.path.join(args.output_dir, 'conditions.npy'), conditions)
    
    # メタデータも保存
    metadata = {
        'condition_source': conditions_source,
        'method': args.method,
        'num_inference_steps': args.steps,
        'condition_dim': actual_condition_dim,
        'seq_len': args.seq_len,
        'checkpoint_path': args.checkpoint
    }
    
    import json
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f'Generation completed! Results saved to: {args.output_dir}')
    
    # 統計情報を表示
    print(f'\nTrajectory Statistics:')
    print(f'  Shape: {trajectories_np.shape}')
    print(f'  X-axis range: [{trajectories_np[:, 0, :].min():.3f}, {trajectories_np[:, 0, :].max():.3f}]')
    print(f'  Y-axis range: [{trajectories_np[:, 1, :].min():.3f}, {trajectories_np[:, 1, :].max():.3f}]')
    
    # 条件データの統計も表示
    print(f'\nCondition Statistics:')
    print(f'  Mean: {conditions.mean(axis=0)}')
    print(f'  Std: {conditions.std(axis=0)}')
    print(f'  Min: {conditions.min(axis=0)}')
    print(f'  Max: {conditions.max(axis=0)}')


if __name__ == '__main__':
    main()