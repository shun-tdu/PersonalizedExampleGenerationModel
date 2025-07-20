# CLAUDE_ADDED
"""
Transformerベースモデルの軌道生成スクリプト
実データと実スケールに対応
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

from Model import TransformerTrajectoryGenerator
from TrajectoryDataset import TrajectoryDataset


class TransformerTrajectoryGenerator:
    """
    Transformerベースの軌道生成クラス（実データ対応）
    """
    def __init__(self, 
                 model: TransformerTrajectoryGenerator, 
                 device: torch.device,
                 denormalize_trajectories: bool = True,
                 trajectory_scale_factor: float = 0.15):
        self.model = model
        self.device = device
        self.denormalize_trajectories = denormalize_trajectories
        self.trajectory_scale_factor = trajectory_scale_factor
        
    @torch.no_grad()
    def generate_trajectories(self, 
                            conditions: torch.Tensor,
                            start_points: Optional[torch.Tensor] = None,
                            seq_len: int = 101,
                            num_start_points: int = 3) -> torch.Tensor:
        """
        指定された個人特性に基づいて軌道を生成
        
        Args:
            conditions: [batch_size, condition_dim] 個人特性
            start_points: [batch_size, num_start_points, 2] 開始点（Noneの場合はランダム）
            seq_len: 生成する軌道の長さ
            num_start_points: 開始点の数
            
        Returns:
            [batch_size, seq_len, 2] 生成された軌道
        """
        self.model.eval()
        batch_size = conditions.shape[0]
        
        if start_points is None:
            # ランダムな開始点を生成（正規化範囲内）
            start_points = torch.randn(batch_size, num_start_points, 2, device=self.device) * 0.1
        
        # Transformerで軌道生成
        generated_trajectories = self.model.generate_trajectory(
            start_points, conditions, max_length=seq_len
        )
        
        return generated_trajectories
    
    def denormalize_trajectory(self, normalized_trajectory: torch.Tensor) -> torch.Tensor:
        """
        正規化された軌道を元のスケールに戻す
        
        Args:
            normalized_trajectory: [batch_size, seq_len, 2] 正規化済み軌道
            
        Returns:
            [batch_size, seq_len, 2] 元スケールの軌道（メートル単位）
        """
        if not self.denormalize_trajectories:
            return normalized_trajectory
            
        # 正規化の逆変換：データ生成時の最大値(0.15m)をスケール基準として使用
        denormalized = normalized_trajectory * self.trajectory_scale_factor
        
        return denormalized


def load_transformer_model(checkpoint_path: str, 
                          condition_dim: int = None,
                          device: torch.device = torch.device('cpu')) -> TransformerTrajectoryGenerator:
    """
    チェックポイントからTransformerモデルをロード
    """
    # チェックポイントを読み込んで構造を確認
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # condition_dimを推定
    if condition_dim is None:
        # モデルのstate_dictから推定
        for key, value in checkpoint['model_state_dict'].items():
            if 'condition_encoder.condition_proj.0.weight' in key:
                condition_dim = value.shape[1]
                print(f'Detected condition_dim from checkpoint: {condition_dim}')
                break
        
        if condition_dim is None:
            print('Warning: Could not detect condition_dim from checkpoint, using default 5')
            condition_dim = 5
    
    # その他のパラメータを推定
    d_model = 256  # デフォルト値
    for key, value in checkpoint['model_state_dict'].items():
        if 'trajectory_embedding.embedding.weight' in key:
            d_model = value.shape[1]
            print(f'Detected d_model from checkpoint: {d_model}')
            break
    
    model = TransformerTrajectoryGenerator(
        input_dim=2,
        condition_dim=condition_dim,
        d_model=d_model,
        nhead=8,  # デフォルト
        num_encoder_layers=6,  # デフォルト
        num_decoder_layers=6,  # デフォルト
        dim_feedforward=1024,  # デフォルト
        max_seq_len=101,  # デフォルト
        dropout=0.1  # デフォルト
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def visualize_transformer_trajectories(trajectories: np.ndarray, 
                                      conditions: np.ndarray,
                                      save_path: Optional[str] = None,
                                      denormalized: bool = True):
    """
    Transformer生成軌道の可視化
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
        
        x_traj = trajectories[i, :, 0]
        y_traj = trajectories[i, :, 1]
        
        ax.plot(x_traj, y_traj, 'b-', linewidth=2, alpha=0.7)
        ax.plot(x_traj[0], y_traj[0], 'go', markersize=8, label='Start')
        ax.plot(x_traj[-1], y_traj[-1], 'ro', markersize=8, label='End')
        
        # 軌道の統計情報
        trajectory_length = np.sum(np.sqrt(np.sum(np.diff(trajectories[i], axis=0)**2, axis=1)))
        endpoint_distance = np.sqrt((x_traj[-1] - x_traj[0])**2 + (y_traj[-1] - y_traj[0])**2)
        
        title = f'Transformer Trajectory {i+1}\\n'
        if conditions.shape[1] >= 3:
            title += f'MT:{conditions[i, 0]:.2f}, EE:{conditions[i, 1]:.2f}, J:{conditions[i, 2]:.2f}\\n'
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
    
    fig.suptitle(f'Transformer Generated Trajectories ({scale_info})', fontsize=14, y=0.95)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Transformer trajectories visualization saved to: {save_path}')
    else:
        plt.show()


@hydra.main(version_base=None, config_path=".", config_name="config")
def hydra_generate(cfg: DictConfig) -> None:
    """
    Hydra統合Transformer生成関数
    """
    print(f"Transformer Generation Configuration:\\n{OmegaConf.to_yaml(cfg)}")
    
    # デバイス設定
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 出力ディレクトリ作成
    output_dir = cfg.output.get('generated_dir', 'outputs/generated_trajectories')
    os.makedirs(output_dir, exist_ok=True)
    
    # MLFlowセットアップ（生成ログ用）
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    with mlflow.start_run(run_name=f"transformer_generation"):
        # チェックポイントパスを決定（最新のものを使用）
        checkpoint_dir = cfg.output.checkpoint_dir
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth') and 'transformer' in f]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
                print(f'Using latest Transformer checkpoint: {checkpoint_path}')
            else:
                raise FileNotFoundError(f"No Transformer checkpoints found in {checkpoint_dir}")
        else:
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # モデルロード
        model = load_transformer_model(checkpoint_path, None, device)
        
        # ジェネレータ作成
        generator = TransformerTrajectoryGenerator(model, device)
        
        # 条件次元の自動検出
        checkpoint = torch.load(checkpoint_path, map_location=device)
        detected_condition_dim = None
        for key, value in checkpoint['model_state_dict'].items():
            if 'condition_encoder.condition_proj.0.weight' in key:
                detected_condition_dim = value.shape[1]
                break
        
        if detected_condition_dim is None:
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
        print(f'Generating {len(conditions)} trajectories using Transformer...')
        import time
        start_time = time.time()
        
        with torch.no_grad():
            generated_trajectories = generator.generate_trajectories(
                conditions=conditions_tensor,
                seq_len=cfg.generation.max_length,
                num_start_points=cfg.generation.get('num_start_points', 3)
            )
            
            # 正規化を元に戻す
            generated_trajectories = generator.denormalize_trajectory(generated_trajectories)
        
        generation_time = time.time() - start_time
        print(f'Transformer generation completed in {generation_time:.2f} seconds')
        
        # CPU に移動してnumpy配列に変換
        trajectories_np = generated_trajectories.cpu().numpy()
        
        # 可視化と保存
        vis_path = os.path.join(output_dir, 'transformer_generated_trajectories.png')
        visualize_transformer_trajectories(trajectories_np, conditions, vis_path)
        
        # 軌道データと条件データを保存
        np.save(os.path.join(output_dir, 'transformer_trajectories.npy'), trajectories_np)
        np.save(os.path.join(output_dir, 'transformer_conditions.npy'), conditions)
        
        # 統計情報の計算と保存
        trajectory_stats = {
            'mean_x': float(trajectories_np[:, :, 0].mean()),
            'mean_y': float(trajectories_np[:, :, 1].mean()),
            'std_x': float(trajectories_np[:, :, 0].std()),
            'std_y': float(trajectories_np[:, :, 1].std()),
            'max_x': float(trajectories_np[:, :, 0].max()),
            'min_x': float(trajectories_np[:, :, 0].min()),
            'max_y': float(trajectories_np[:, :, 1].max()),
            'min_y': float(trajectories_np[:, :, 1].min()),
        }
        
        # 軌道長とエンドポイント距離の計算
        trajectory_lengths = []
        endpoint_distances = []
        
        for i in range(trajectories_np.shape[0]):
            traj = trajectories_np[i]
            # 軌道長
            length = np.sum(np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1)))
            trajectory_lengths.append(length)
            
            # エンドポイント距離
            endpoint_dist = np.sqrt((traj[-1, 0] - traj[0, 0])**2 + (traj[-1, 1] - traj[0, 1])**2)
            endpoint_distances.append(endpoint_dist)
        
        trajectory_stats.update({
            'mean_length': float(np.mean(trajectory_lengths)),
            'std_length': float(np.std(trajectory_lengths)),
            'mean_endpoint_distance': float(np.mean(endpoint_distances)),
            'std_endpoint_distance': float(np.std(endpoint_distances))
        })
        
        # 統計情報をファイルに保存
        stats_path = os.path.join(output_dir, 'transformer_generated_trajectories_stats.txt')
        with open(stats_path, 'w') as f:
            f.write("Transformer Generated Trajectories Statistics (Real Scale)\\n")
            f.write("=" * 60 + "\\n")
            f.write(f"Number of trajectories: {trajectories_np.shape[0]}\\n")
            f.write(f"Trajectory length: {trajectories_np.shape[1]} points\\n")
            f.write(f"Coordinate dimension: {trajectories_np.shape[2]}\\n\\n")
            
            f.write("Position Statistics (meters):\\n")
            f.write(f"  X: mean={trajectory_stats['mean_x']:.4f}, std={trajectory_stats['std_x']:.4f}, ")
            f.write(f"range=[{trajectory_stats['min_x']:.4f}, {trajectory_stats['max_x']:.4f}]\\n")
            f.write(f"  Y: mean={trajectory_stats['mean_y']:.4f}, std={trajectory_stats['std_y']:.4f}, ")
            f.write(f"range=[{trajectory_stats['min_y']:.4f}, {trajectory_stats['max_y']:.4f}]\\n\\n")
            
            f.write("Trajectory Metrics:\\n")
            f.write(f"  Mean trajectory length: {trajectory_stats['mean_length']:.4f} ± {trajectory_stats['std_length']:.4f} m\\n")
            f.write(f"  Mean endpoint distance: {trajectory_stats['mean_endpoint_distance']:.4f} ± {trajectory_stats['std_endpoint_distance']:.4f} m\\n")
        
        # メタデータも保存
        metadata = {
            'model_type': 'Transformer',
            'condition_source': conditions_source,
            'condition_dim': detected_condition_dim,
            'trajectory_length': cfg.generation.max_length,
            'checkpoint_path': checkpoint_path,
            'generation_time': generation_time,
            'denormalized': True,
            'scale_factor': 0.15,
            'statistics': trajectory_stats
        }
        
        import json
        with open(os.path.join(output_dir, 'transformer_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # MLFlowにメトリクスとアーティファクトをログ
        mlflow.log_metrics({
            "generation_time": generation_time,
            "num_generated": len(trajectories_np),
            **{f"trajectory_{k}": v for k, v in trajectory_stats.items()}
        })
        
        mlflow.log_artifact(vis_path, "transformer_plots")
        mlflow.log_artifact(os.path.join(output_dir, 'transformer_metadata.json'), "metadata")
        mlflow.log_artifact(stats_path, "statistics")
        
        print(f'Transformer generation completed! Results saved to: {output_dir}')
        print(f'Real scale trajectories with mean length: {trajectory_stats["mean_length"]:.4f}m')


def main():
    """
    argparse対応メイン関数（後方互換性のため）
    """
    parser = argparse.ArgumentParser(description='Transformer Trajectory Generation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to Transformer model checkpoint')
    parser.add_argument('--data_path', type=str, default=None, help='Path to real data (.npz file) for conditions')
    parser.add_argument('--condition_dim', type=int, default=5, help='Condition dimension (used when generating dummy data)')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of trajectories to generate')
    parser.add_argument('--seq_len', type=int, default=101, help='Sequence length')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output_dir', type=str, default='outputs/generated_trajectories', help='Output directory')
    parser.add_argument('--use_dummy', action='store_true', help='Use dummy conditions instead of real data')
    parser.add_argument('--num_start_points', type=int, default=3, help='Number of start points for generation')
    
    args = parser.parse_args()
    
    # デバイス設定
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # モデルロード
    print(f'Loading Transformer model from: {args.checkpoint}')
    model = load_transformer_model(args.checkpoint, None, device)
    
    # ジェネレータ作成
    generator = TransformerTrajectoryGenerator(model, device)
    
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
    print(f'Generating {len(conditions)} trajectories using Transformer...')
    with torch.no_grad():
        generated_trajectories = generator.generate_trajectories(
            conditions=conditions_tensor,
            seq_len=args.seq_len,
            num_start_points=args.num_start_points
        )
        
        # 正規化を元に戻す
        generated_trajectories = generator.denormalize_trajectory(generated_trajectories)
    
    # CPU に移動してnumpy配列に変換
    trajectories_np = generated_trajectories.cpu().numpy()
    
    # 可視化と保存
    vis_path = os.path.join(args.output_dir, 'transformer_generated_trajectories.png')
    visualize_transformer_trajectories(trajectories_np, conditions, vis_path)
    
    # 軌道データと条件データを保存
    np.save(os.path.join(args.output_dir, 'transformer_trajectories.npy'), trajectories_np)
    np.save(os.path.join(args.output_dir, 'transformer_conditions.npy'), conditions)
    
    print(f'Transformer generation completed! Results saved to: {args.output_dir}')


if __name__ == '__main__':
    main()