from typing import List, Dict, Any, Union, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from base_evaluator import BaseEvaluator
from src.PredictiveLatentSpaceNavigationModel.TransformerBaseEndToEndVAE.evaluator import EnhancedEvaluationResult
from src.PredictiveLatentSpaceNavigationModel.DataPreprocess.analyze_skill_metrics import SkillMetricCalculator


class TrajectoryGenerationEvaluator(BaseEvaluator):
    """再構築軌道の統計的なデータの再現度を評価"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def evaluate(self, model: torch.nn.Module, test_data: Dict[str, Any], device: torch.device, result: EnhancedEvaluationResult):
        originals = test_data.get('originals')
        reconstructed = test_data.get('reconstructed')

        print("=" * 60)
        print("再構築軌道誤差評価完了")
        print("=" * 60)

        reconstructed_trajectory_rmser = self._evaluate_reconstruction_rmse(originals, reconstructed)
        reconstructed_trajectory_skill_metrics_rmse = self._evaluate_skill_metric_rmse(originals, reconstructed)


        result.add_metric(name='reconstructed_trajectory_rmse',
                          value=reconstructed_trajectory_rmser,
                          description='元軌道と再構築軌道のRMSE',
                          category='baseline')

        result.add_metric(name='reconstructed_trajectory_skill_metrics_rmse',
                          value=reconstructed_trajectory_skill_metrics_rmse,
                          description='元軌道と再構築軌道のスキル指標のRMSE',
                          category='baseline')

        print("✅ 再構築軌道誤差評価完了")

    def _evaluate_reconstruction_rmse(self, originals: np.ndarray, reconstructions: np.ndarray) -> float:
        """軌道の再構成誤差の平均"""
        rmse_per_batch = np.sqrt(np.mean((originals - reconstructions) ** 2, axis=(1, 2)))
        mean_rmse = np.mean(rmse_per_batch)

        return mean_rmse

    def _evaluate_skill_metric_rmse(self, originals: np.ndarray, reconstructions: np.ndarray) -> float:
        """
        軌道から計算したスキル指標のRMSEを計算する

        :param originals: モデルに入力する軌道データ [batch, seq_len, dim]
        :param reconstructions: モデルから出力された軌道データ [batch, seq_len, dim]
        :return:
        """
        # 両方のデータセットのスキル指標を生データで計算
        original_skill_metrics_raw = self._calculate_skill_metrics_raw(originals)
        reconstructions_skill_metrics_raw = self._calculate_skill_metrics_raw(reconstructions)
        
        # 元データの統計量で統一スケーラを作成
        combined_metrics = np.vstack([original_skill_metrics_raw, reconstructions_skill_metrics_raw])
        scaler = StandardScaler()
        scaler.fit(combined_metrics)
        
        # 同じスケーラで両方を標準化
        original_skill_metrics = scaler.transform(original_skill_metrics_raw)
        reconstructions_skill_metrics = scaler.transform(reconstructions_skill_metrics_raw)

        rmse_per_batch = np.mean((original_skill_metrics - reconstructions_skill_metrics) ** 2, axis=1)
        mean_rmse = np.mean(rmse_per_batch)

        return mean_rmse

    def _calculate_skill_metrics_raw(self, trajectory: np.ndarray) -> np.ndarray:
        """
        軌道データのスキル指標を計算する（標準化なし）
        :param trajectory: 軌道データ[batch, seq_len, dim]
        :return: 各batchのスキル指標 [batch, num_skill_metrics]（生データ）
        """
        columns = ['HandlePosX','HandlePosY','HandleVelX','HandleVelY','HandleAccX','HandleAccY']

        assert trajectory.shape[2] == len(columns), \
            f"入力データの次元数が合いません。期待値: {len(columns)}, 実際値: {trajectory.shape[2]}"

        batch_size = trajectory.shape[0]
        seq_len = trajectory.shape[1]

        # DataFrameに変換
        list_of_trajectories = [pd.DataFrame(trajectory[i], columns=columns).assign(Timestamp=np.arange(seq_len)*0.01) for i in range(batch_size)]

        # 全テストデータに対してスキル指標を計算
        all_skill_metrics = []
        for trajectory_df in list_of_trajectories:
            skill_metrics_per_batch = [SkillMetricCalculator.calculate_curvature(trajectory_df),
                                       SkillMetricCalculator.calculate_velocity_smoothness(trajectory_df),
                                       SkillMetricCalculator.calculate_acceleration_smoothness(trajectory_df),
                                       SkillMetricCalculator.calculate_jerk(trajectory_df),
                                       SkillMetricCalculator.calculate_control_stability(trajectory_df),
                                       SkillMetricCalculator.calculate_temporal_consistency(trajectory_df),
                                       SkillMetricCalculator.calculate_trial_time(trajectory_df),
                                       SkillMetricCalculator.calculate_endpoint_error(trajectory_df)]
            all_skill_metrics.append(skill_metrics_per_batch)

        all_skill_metrics_array = np.array(all_skill_metrics)  # shape: [batch_size, num_metrics]

        # NaN/inf値の処理
        if np.any(np.isnan(all_skill_metrics_array)) or np.any(np.isinf(all_skill_metrics_array)):
            print("Warning: NaN or inf values found in skill metrics. Replacing with zeros.")
            all_skill_metrics_array = np.nan_to_num(all_skill_metrics_array, nan=0.0, posinf=0.0, neginf=0.0)

        return all_skill_metrics_array  # 標準化なしで返す

    def get_required_data(self) -> List[str]:
        return ['originals', 'reconstructions']

class OrthogonalityEvaluator(BaseEvaluator):
    """潜在空間の直交性評価"""
    def __init__(self, config:Dict[str, Any]):
        super().__init__(config)

    def evaluate(self, model: torch.nn.Module, test_data: Dict[str, Any], device: torch.device, result: EnhancedEvaluationResult):
        """潜在空間の直交性評価を実行"""
        z_style = test_data.get('z_style')
        z_skill = test_data.get('z_skill')

        print("=" * 60)
        print("潜在空間の直交性評価実行")
        print("=" * 60)

        # コサイン類似度を計算
        cosine_sim = self._calc_cosine_similarity(z_style, z_skill)

        # 相関行列のヒートマップ作成
        latent_space_crr_heatmap = self._create_latent_space_correlation_heatmap(z_style, z_skill)

        result.add_metric(name='latent_space_cosine_similarity',
                          value=cosine_sim,
                          description='潜在変数のコサイン類似度',
                          category='baseline')

        result.add_visualization(name='latent_space_correlation',
                                 fig_or_path=latent_space_crr_heatmap,
                                 description="潜在変数の相関ヒートマップ",
                                 category='baseline')

        print("✅ 潜在空間の直交性評価完了")

    def _calc_cosine_similarity(self, z_style: np.ndarray, z_skill: np.ndarray) -> float:
        """各潜在変数のコサイン類似度を計算"""
        similarities = []
        for i in range(len(z_style)):
            # ベクトルを2次元配列にreshape
            sim = cosine_similarity(z_style[i].reshape(-1, 1), z_skill[i].reshape(-1, 1))
            similarities.append(sim[0, 0])

        # 全サンプルのコサイン類似度の平均値の絶対値を計算
        return np.mean(np.abs(similarities))

    def _create_latent_space_correlation_heatmap(self, z_style: np.ndarray, z_skill: np.ndarray) -> plt.Figure:
        """各潜在変数の相関行列ヒートマップを作成"""
        # 各潜在変数をDataFrameに変換
        style_cols = [f'style_{i}' for i in range(z_style.shape[1])]
        skill_cols = [f'skill_{i}' for i in range(z_skill.shape[1])]
        df_style = pd.DataFrame(z_style, columns=style_cols)
        df_skill = pd.DataFrame(z_skill, columns=skill_cols)

        # 2つのDataFrameをカラム方向に結合
        combined_df = pd.concat([df_style, df_skill], axis=1)

        # 全体の相関行列を計算
        correlation_matrix = combined_df.corr()

        # スタイル次元とスキル次元の相関だけを抽出
        cross_correlation = correlation_matrix.loc[style_cols, skill_cols]

        # ヒートマップを作成
        fig_latent_corr, ax_latent_corr = plt.subplots(figsize=(6, 8))
        sns.heatmap(cross_correlation, annot=True, cmap='RdBu_r', center=0, fmt='.2f', ax=ax_latent_corr)
        ax_latent_corr.set_title('Cross-Correlation between Style (8D) and Skill (2D)')

        return fig_latent_corr

    def get_required_data(self) -> List[str]:
        return ['z_style', 'z_skill']


