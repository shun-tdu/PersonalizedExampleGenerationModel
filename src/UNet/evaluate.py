# CLAUDE_ADDED
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.pytorch
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import pandas as pd

from Model import UNet1D
from train import DDPMScheduler
from generate import TrajectoryGenerator, load_model
from TrajectoryDataset import TrajectoryDataset


class TrajectoryEvaluator:
    """
    軌道生成モデルの評価クラス
    """
    
    def __init__(self, 
                 model: UNet1D,
                 scheduler: DDPMScheduler,
                 device: torch.device,
                 output_dir: str = "evaluation_results"):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.generator = TrajectoryGenerator(model, scheduler, device)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_trajectory_diversity(self, 
                                    generated_trajectories: np.ndarray) -> Dict[str, float]:
        """
        軌道の多様性を評価
        
        Args:
            generated_trajectories: [batch_size, 2, seq_len] の軌道データ
            
        Returns:
            多様性メトリクスの辞書
        """
        batch_size, _, seq_len = generated_trajectories.shape
        
        # 軌道を平坦化して距離行列を計算
        flattened_trajectories = generated_trajectories.reshape(batch_size, -1)
        
        # ペアワイズ距離
        pairwise_dist = pairwise_distances(flattened_trajectories, metric='euclidean')
        
        # 多様性メトリクス
        metrics = {
            'mean_pairwise_distance': float(np.mean(pairwise_dist[np.triu_indices(batch_size, k=1)])),
            'std_pairwise_distance': float(np.std(pairwise_dist[np.triu_indices(batch_size, k=1)])),
            'min_pairwise_distance': float(np.min(pairwise_dist[np.triu_indices(batch_size, k=1)])),
            'max_pairwise_distance': float(np.max(pairwise_dist[np.triu_indices(batch_size, k=1)])),
        }
        
        # 軌道の開始点と終点の多様性
        start_points = generated_trajectories[:, :, 0]  # [batch_size, 2]
        end_points = generated_trajectories[:, :, -1]   # [batch_size, 2]
        
        start_distances = pdist(start_points, metric='euclidean')
        end_distances = pdist(end_points, metric='euclidean')
        
        metrics.update({
            'start_point_diversity': float(np.mean(start_distances)),
            'end_point_diversity': float(np.mean(end_distances)),
        })
        
        return metrics
    
    def evaluate_trajectory_smoothness(self, 
                                     generated_trajectories: np.ndarray) -> Dict[str, float]:
        """
        軌道の滑らかさを評価
        
        Args:
            generated_trajectories: [batch_size, 2, seq_len] の軌道データ
            
        Returns:
            滑らかさメトリクスの辞書
        """
        # 1次微分（速度）
        velocities = np.diff(generated_trajectories, axis=2)
        
        # 2次微分（加速度）
        accelerations = np.diff(velocities, axis=2)
        
        # 3次微分（ジャーク）
        jerks = np.diff(accelerations, axis=2)
        
        # 各軌道の速度、加速度、ジャークの大きさ
        velocity_magnitudes = np.sqrt(np.sum(velocities**2, axis=1))  # [batch_size, seq_len-1]
        acceleration_magnitudes = np.sqrt(np.sum(accelerations**2, axis=1))  # [batch_size, seq_len-2]
        jerk_magnitudes = np.sqrt(np.sum(jerks**2, axis=1))  # [batch_size, seq_len-3]
        
        metrics = {
            'mean_velocity': float(np.mean(velocity_magnitudes)),
            'std_velocity': float(np.std(velocity_magnitudes)),
            'max_velocity': float(np.max(velocity_magnitudes)),
            
            'mean_acceleration': float(np.mean(acceleration_magnitudes)),
            'std_acceleration': float(np.std(acceleration_magnitudes)),
            'max_acceleration': float(np.max(acceleration_magnitudes)),
            
            'mean_jerk': float(np.mean(jerk_magnitudes)),
            'std_jerk': float(np.std(jerk_magnitudes)),
            'max_jerk': float(np.max(jerk_magnitudes)),
            
            # 平均ジャーク（滑らかさの逆指標）
            'smoothness_score': 1.0 / (1.0 + float(np.mean(jerk_magnitudes))),
        }
        
        return metrics
    
    def evaluate_condition_consistency(self, 
                                     generated_trajectories: np.ndarray,
                                     conditions: np.ndarray) -> Dict[str, float]:
        """
        条件と生成軌道の一貫性を評価
        
        Args:
            generated_trajectories: [batch_size, 2, seq_len] の軌道データ
            conditions: [batch_size, condition_dim] の条件データ
            
        Returns:
            一貫性メトリクスの辞書
        """
        batch_size, _, seq_len = generated_trajectories.shape
        
        # 軌道から特徴量を抽出
        trajectory_features = self._extract_trajectory_features(generated_trajectories)
        
        # 条件と軌道特徴の相関を計算
        correlations = {}
        for i, feature_name in enumerate(trajectory_features.keys()):
            feature_values = trajectory_features[feature_name]
            
            # 各条件次元との相関
            for j in range(conditions.shape[1]):
                condition_values = conditions[:, j]
                
                # ピアソン相関係数を計算
                if np.std(feature_values) > 1e-8 and np.std(condition_values) > 1e-8:
                    corr, p_value = pearsonr(feature_values, condition_values)
                    correlations[f'corr_condition_{j}_feature_{feature_name}'] = float(corr)
                    correlations[f'pvalue_condition_{j}_feature_{feature_name}'] = float(p_value)
        
        # 平均絶対相関
        abs_correlations = [abs(v) for k, v in correlations.items() if k.startswith('corr_')]
        metrics = {
            'mean_abs_correlation': float(np.mean(abs_correlations)) if abs_correlations else 0.0,
            'max_abs_correlation': float(np.max(abs_correlations)) if abs_correlations else 0.0,
        }
        
        metrics.update(correlations)
        return metrics
    
    def evaluate_endpoint_accuracy(self,
                                 generated_trajectories: np.ndarray,
                                 target_endpoints: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        終点精度を評価
        
        Args:
            generated_trajectories: [batch_size, 2, seq_len] の軌道データ
            target_endpoints: [batch_size, 2] の目標終点（Noneの場合は原点を目標とする）
            
        Returns:
            終点精度メトリクスの辞書
        """
        actual_endpoints = generated_trajectories[:, :, -1]  # [batch_size, 2]
        
        if target_endpoints is None:
            target_endpoints = np.zeros_like(actual_endpoints)  # 原点を目標とする
        
        # 終点誤差
        endpoint_errors = np.sqrt(np.sum((actual_endpoints - target_endpoints)**2, axis=1))
        
        metrics = {
            'mean_endpoint_error': float(np.mean(endpoint_errors)),
            'std_endpoint_error': float(np.std(endpoint_errors)),
            'max_endpoint_error': float(np.max(endpoint_errors)),
            'min_endpoint_error': float(np.min(endpoint_errors)),
            'median_endpoint_error': float(np.median(endpoint_errors)),
        }
        
        return metrics
    
    def _extract_trajectory_features(self, trajectories: np.ndarray) -> Dict[str, np.ndarray]:
        """
        軌道から特徴量を抽出
        
        Args:
            trajectories: [batch_size, 2, seq_len] の軌道データ
            
        Returns:
            特徴量の辞書
        """
        features = {}
        
        # 移動距離
        distances = np.sqrt(np.sum(np.diff(trajectories, axis=2)**2, axis=1))
        features['total_distance'] = np.sum(distances, axis=1)
        
        # 最大速度
        velocities = np.sqrt(np.sum(np.diff(trajectories, axis=2)**2, axis=1))
        features['max_velocity'] = np.max(velocities, axis=1)
        
        # 終点誤差（原点からの距離）
        end_points = trajectories[:, :, -1]
        features['endpoint_error'] = np.sqrt(np.sum(end_points**2, axis=1))
        
        # 軌道の範囲
        features['x_range'] = np.ptp(trajectories[:, 0, :], axis=1)
        features['y_range'] = np.ptp(trajectories[:, 1, :], axis=1)
        
        # 軌道の曲率（近似）
        x_traj = trajectories[:, 0, :]
        y_traj = trajectories[:, 1, :]
        
        # 2次微分を使った曲率の近似
        d2x = np.diff(x_traj, n=2, axis=1)
        d2y = np.diff(y_traj, n=2, axis=1)
        curvature_approx = np.sqrt(d2x**2 + d2y**2)
        features['mean_curvature'] = np.mean(curvature_approx, axis=1)
        
        return features
    
    def comprehensive_evaluation(self,
                               test_dataset: TrajectoryDataset,
                               num_samples: int = 100,
                               generation_method: str = 'ddim',
                               num_inference_steps: int = 50) -> Dict[str, float]:
        """
        包括的な評価を実行
        
        Args:
            test_dataset: テストデータセット
            num_samples: 評価に使用するサンプル数
            generation_method: 生成手法
            num_inference_steps: 推論ステップ数
            
        Returns:
            全評価メトリクスの辞書
        """
        print(f"Starting comprehensive evaluation with {num_samples} samples...")
        
        # テストデータからランダムサンプリング
        total_samples = len(test_dataset)
        if num_samples > total_samples:
            indices = list(range(total_samples))
        else:
            indices = np.random.choice(total_samples, num_samples, replace=False)
        
        # 条件データを収集
        conditions_list = []
        ground_truth_trajectories_list = []
        
        for idx in indices:
            trajectory, condition = test_dataset[idx]
            conditions_list.append(condition.numpy())
            ground_truth_trajectories_list.append(trajectory.numpy())
        
        conditions = np.array(conditions_list)
        ground_truth_trajectories = np.array(ground_truth_trajectories_list)
        
        # 軌道生成
        print(f"Generating {len(conditions)} trajectories...")\n        conditions_tensor = torch.FloatTensor(conditions).to(self.device)
        
        with torch.no_grad():
            generated_trajectories = self.generator.generate_trajectories(
                conditions=conditions_tensor,
                seq_len=ground_truth_trajectories.shape[2],
                method=generation_method,
                num_inference_steps=num_inference_steps
            )
        
        generated_trajectories_np = generated_trajectories.cpu().numpy()
        
        # 各評価メトリクスを計算
        all_metrics = {}
        
        # 1. 軌道多様性
        diversity_metrics = self.evaluate_trajectory_diversity(generated_trajectories_np)
        all_metrics.update({f"diversity_{k}": v for k, v in diversity_metrics.items()})
        
        # 2. 軌道滑らかさ
        smoothness_metrics = self.evaluate_trajectory_smoothness(generated_trajectories_np)
        all_metrics.update({f"smoothness_{k}": v for k, v in smoothness_metrics.items()})
        
        # 3. 条件一貫性
        consistency_metrics = self.evaluate_condition_consistency(generated_trajectories_np, conditions)
        all_metrics.update({f"consistency_{k}": v for k, v in consistency_metrics.items()})
        
        # 4. 終点精度
        endpoint_metrics = self.evaluate_endpoint_accuracy(generated_trajectories_np)
        all_metrics.update({f"endpoint_{k}": v for k, v in endpoint_metrics.items()})
        
        # 5. 実軌道との比較（可能な場合）
        if ground_truth_trajectories is not None:
            comparison_metrics = self._compare_with_ground_truth(
                generated_trajectories_np, ground_truth_trajectories
            )
            all_metrics.update({f"comparison_{k}": v for k, v in comparison_metrics.items()})
        
        # 結果を保存
        self._save_evaluation_results(all_metrics, generated_trajectories_np, 
                                    conditions, ground_truth_trajectories)
        
        print("Comprehensive evaluation completed!")
        return all_metrics
    
    def _compare_with_ground_truth(self,
                                 generated_trajectories: np.ndarray,
                                 ground_truth_trajectories: np.ndarray) -> Dict[str, float]:
        """
        生成軌道と実軌道を比較
        """
        # 軌道間距離
        trajectory_distances = []
        for i in range(len(generated_trajectories)):
            gen_traj = generated_trajectories[i].flatten()
            gt_traj = ground_truth_trajectories[i].flatten()
            dist = np.linalg.norm(gen_traj - gt_traj)
            trajectory_distances.append(dist)
        
        # 特徴量比較
        gen_features = self._extract_trajectory_features(generated_trajectories)
        gt_features = self._extract_trajectory_features(ground_truth_trajectories)
        
        feature_differences = {}
        for feature_name in gen_features.keys():
            gen_vals = gen_features[feature_name]
            gt_vals = gt_features[feature_name]
            
            feature_differences[f'{feature_name}_mae'] = float(np.mean(np.abs(gen_vals - gt_vals)))
            feature_differences[f'{feature_name}_mse'] = float(np.mean((gen_vals - gt_vals)**2))
        
        metrics = {
            'mean_trajectory_distance': float(np.mean(trajectory_distances)),
            'std_trajectory_distance': float(np.std(trajectory_distances)),
        }
        metrics.update(feature_differences)
        
        return metrics
    
    def _save_evaluation_results(self,
                               metrics: Dict[str, float],
                               generated_trajectories: np.ndarray,
                               conditions: np.ndarray,
                               ground_truth_trajectories: Optional[np.ndarray] = None):
        """
        評価結果を保存
        """
        # メトリクスをJSONで保存
        with open(os.path.join(self.output_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # 軌道を可視化
        self._visualize_evaluation_results(generated_trajectories, conditions, ground_truth_trajectories)
        
        # データを保存
        np.save(os.path.join(self.output_dir, 'generated_trajectories.npy'), generated_trajectories)
        np.save(os.path.join(self.output_dir, 'conditions.npy'), conditions)
        if ground_truth_trajectories is not None:
            np.save(os.path.join(self.output_dir, 'ground_truth_trajectories.npy'), ground_truth_trajectories)
    
    def _visualize_evaluation_results(self,
                                    generated_trajectories: np.ndarray,
                                    conditions: np.ndarray,
                                    ground_truth_trajectories: Optional[np.ndarray] = None):
        """
        評価結果を可視化
        """
        # 1. 軌道プロット
        num_samples = min(16, len(generated_trajectories))
        cols = 4
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # 生成軌道
            gen_traj = generated_trajectories[i]
            ax.plot(gen_traj[0], gen_traj[1], 'b-', linewidth=2, alpha=0.7, label='Generated')
            ax.plot(gen_traj[0, 0], gen_traj[1, 0], 'go', markersize=8)  # Start
            ax.plot(gen_traj[0, -1], gen_traj[1, -1], 'ro', markersize=8)  # End
            
            # 実軌道（もしあれば）
            if ground_truth_trajectories is not None:
                gt_traj = ground_truth_trajectories[i]
                ax.plot(gt_traj[0], gt_traj[1], 'r--', linewidth=2, alpha=0.7, label='Ground Truth')
            
            ax.set_title(f'Sample {i+1}')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_aspect('equal')
        
        # 空のサブプロットを非表示
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'trajectory_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 特徴量分布
        features = self._extract_trajectory_features(generated_trajectories)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (feature_name, feature_values) in enumerate(features.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            ax.hist(feature_values, bins=20, alpha=0.7, density=True)
            ax.set_title(f'{feature_name.replace("_", " ").title()} Distribution')
            ax.set_xlabel(feature_name.replace("_", " ").title())
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra統合評価メイン関数
    """
    print(f"Evaluation Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # デバイス設定
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # MLFlowセットアップ
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    with mlflow.start_run(run_name="model_evaluation"):
        # 最新チェックポイントをロード
        checkpoint_dir = cfg.output.checkpoint_dir
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
                print(f'Using checkpoint: {checkpoint_path}')
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
        
        # 評価器作成
        evaluator = TrajectoryEvaluator(model, scheduler, device, "evaluation_results")
        
        # テストデータセット準備
        if cfg.training.use_dummy:
            # ダミーデータで評価
            print("Creating dummy test data...")
            from train import create_dummy_data
            test_trajectories, test_conditions = create_dummy_data(200, 101, 3)
            
            class DummyTestDataset:
                def __init__(self, trajectories, conditions):
                    self.trajectories = torch.FloatTensor(trajectories)
                    self.conditions = torch.FloatTensor(conditions)
                
                def __len__(self):
                    return len(self.trajectories)
                
                def __getitem__(self, idx):
                    return self.trajectories[idx], self.conditions[idx]
            
            test_dataset = DummyTestDataset(test_trajectories, test_conditions)
        else:
            # 実データで評価
            print(f'Loading test data from: {cfg.data.train_data}')
            test_dataset = TrajectoryDataset(cfg.data.train_data)
        
        # 評価実行
        evaluation_metrics = evaluator.comprehensive_evaluation(
            test_dataset=test_dataset,
            num_samples=cfg.evaluation.get('num_samples', 100),
            generation_method=cfg.generation.method,
            num_inference_steps=cfg.generation.num_inference_steps
        )
        
        # MLFlowにメトリクスをログ
        mlflow.log_params({
            "evaluation_method": cfg.generation.method,
            "num_inference_steps": cfg.generation.num_inference_steps,
            "num_eval_samples": cfg.evaluation.get('num_samples', 100)
        })
        
        mlflow.log_metrics(evaluation_metrics)
        
        # アーティファクトを保存
        mlflow.log_artifacts("evaluation_results", "evaluation")
        
        print("Evaluation completed successfully!")
        print(f"Key metrics:")
        for metric_name in ['diversity_mean_pairwise_distance', 'smoothness_smoothness_score', 
                          'endpoint_mean_endpoint_error', 'consistency_mean_abs_correlation']:
            if metric_name in evaluation_metrics:
                print(f"  {metric_name}: {evaluation_metrics[metric_name]:.4f}")


if __name__ == '__main__':
    main()