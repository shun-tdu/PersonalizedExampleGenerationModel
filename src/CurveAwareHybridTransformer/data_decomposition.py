# CLAUDE_ADDED
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class DataDecomposer:
    """
    時系列データを低周波成分と高周波成分（残差）に分解するクラス
    """
    
    def __init__(self, window_size: int = 10):
        """
        Args:
            window_size: 移動平均のウィンドウサイズ
        """
        self.window_size = window_size
    
    def decompose(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        データを低周波成分と高周波成分に分解
        
        Args:
            data: 入力データ [batch_size, sequence_length, feature_dim]
            
        Returns:
            low_freq: 低周波成分（移動平均）
            high_freq: 高周波成分（残差）
        """
        batch_size, seq_len, feature_dim = data.shape
        
        # 移動平均フィルタを適用
        low_freq = self._moving_average(data)
        
        # 残差を計算
        high_freq = data - low_freq
        
        return low_freq, high_freq
    
    def _moving_average(self, data: torch.Tensor) -> torch.Tensor:
        """
        移動平均を計算
        
        Args:
            data: 入力データ [batch_size, sequence_length, feature_dim]
            
        Returns:
            移動平均された低周波成分
        """
        batch_size, seq_len, feature_dim = data.shape
        
        # 適切なパディングサイズを計算
        # window_size が偶数の場合、左右非対称になる可能性があるので調整
        left_pad = self.window_size // 2
        right_pad = self.window_size - 1 - left_pad
        
        # パディングを追加して境界を処理
        padded_data = torch.nn.functional.pad(
            data, 
            (0, 0, left_pad, right_pad), 
            mode='reflect'
        )
        
        # 畳み込みを使って移動平均を計算
        kernel = torch.ones(self.window_size, device=data.device) / self.window_size
        kernel = kernel.view(1, 1, self.window_size)
        
        # 各特徴次元について移動平均を計算
        smoothed = torch.zeros_like(data)
        for i in range(feature_dim):
            feature_data = padded_data[:, :, i:i+1].transpose(1, 2)  # [batch, 1, seq+padding]
            smoothed_feature = torch.nn.functional.conv1d(
                feature_data, kernel, padding=0
            ).transpose(1, 2)  # [batch, conv_output_len, 1]
            
            # 出力サイズを元のサイズに合わせて調整
            if smoothed_feature.size(1) != seq_len:
                # 中央部分を抽出して元のサイズに合わせる
                start_idx = (smoothed_feature.size(1) - seq_len) // 2
                end_idx = start_idx + seq_len
                smoothed_feature = smoothed_feature[:, start_idx:end_idx, :]
            
            smoothed[:, :, i] = smoothed_feature.squeeze(-1)
        
        return smoothed
    
    def reconstruct(self, low_freq: torch.Tensor,
                    high_freq: torch.Tensor
                    ) -> torch.Tensor:
        """
        低周波成分と高周波成分から元のデータを復元
        
        Args:
            low_freq: 低周波成分
            high_freq: 高周波成分
            
        Returns:
            復元されたデータ
        """
        # return low_freq
        return low_freq + high_freq[:,1:low_freq.size(1)+1,:]


class ImprovedDataDecomposer(nn.Module):
    """
    軌道データを低周波成分と高周波成分に分解するクラス
    学習可能なパラメータを持ち、最適な分解方法を学習する
    """

    def __init__(
            self,
            window_size: int = 10,
            learnable: bool = True,
            decomposition_method: str = 'adaptive'  # 'moving_average', 'gaussian', 'adaptive'
    ):
        """
        Args:
            window_size: 移動平均のウィンドウサイズ
            learnable: 学習可能なパラメータを使用するか
            decomposition_method: 分解方法
        """
        super().__init__()

        self.window_size = window_size
        self.learnable = learnable
        self.decomposition_method = decomposition_method

        if learnable:
            # 学習可能な重み（移動平均の重み）
            self.window_weights = nn.Parameter(
                torch.ones(window_size) / window_size
            )

            # 適応的分解のためのネットワーク
            if decomposition_method == 'adaptive':
                self.adaptive_net = nn.Sequential(
                    nn.Conv1d(2, 32, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(32, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(32, 1, kernel_size=3, padding=1),
                    nn.Sigmoid()
                )

            # ガウシアンフィルタのσを学習
            self.gaussian_sigma = nn.Parameter(torch.tensor(1.0))

        else:
            # 固定の重み
            self.register_buffer(
                'window_weights',
                torch.ones(window_size) / window_size
            )

    def decompose(self, trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        軌道を低周波と高周波成分に分解

        Args:
            trajectory: [batch_size, seq_len, 2] の軌道データ

        Returns:
            low_freq: 低周波成分
            high_freq: 高周波成分
        """
        if self.decomposition_method == 'moving_average':
            low_freq = self._moving_average_decompose(trajectory)
        elif self.decomposition_method == 'gaussian':
            low_freq = self._gaussian_decompose(trajectory)
        elif self.decomposition_method == 'adaptive':
            low_freq = self._adaptive_decompose(trajectory)
        else:
            raise ValueError(f"Unknown decomposition method: {self.decomposition_method}")

        # 高周波成分 = 元の軌道 - 低周波成分
        high_freq = trajectory - low_freq

        return low_freq, high_freq

    def _moving_average_decompose(self, trajectory: torch.Tensor) -> torch.Tensor:
        """移動平均による分解"""
        batch_size, seq_len, dim = trajectory.shape
        device = trajectory.device

        # 重みの正規化（学習可能な場合）
        if self.learnable:
            weights = F.softmax(self.window_weights, dim=0)
        else:
            weights = self.window_weights

        # パディング
        pad_size = self.window_size // 2
        # [B, L, D] -> [B, D, L] for conv1d
        traj_transposed = trajectory.transpose(1, 2)

        # 反射パディング（端点での振る舞いを改善）
        # padded = F.pad(traj_transposed, (pad_size, pad_size), mode='reflect')

        # 各次元に対して畳み込み
        low_freq_list = []
        for d in range(dim):
            # 1次元畳み込みで移動平均
            kernel = weights.view(1, 1, -1)
            filtered = F.conv1d(
                traj_transposed[:, d:d + 1, :],
                kernel,
                padding='same'
            )
            low_freq_list.append(filtered)

        # 結合して元の形状に戻す
        low_freq = torch.cat(low_freq_list, dim=1)
        low_freq = low_freq.transpose(1, 2)

        return low_freq

    def _gaussian_decompose(self, trajectory: torch.Tensor) -> torch.Tensor:
        """ガウシアンフィルタによる分解"""
        batch_size, seq_len, dim = trajectory.shape
        device = trajectory.device

        # 学習可能なσまたは固定値
        if self.learnable:
            sigma = F.softplus(self.gaussian_sigma) + 0.1  # 正の値を保証
        else:
            sigma = float(self.window_size) / 4.0

        # ガウシアンカーネルの作成
        kernel_size = self.window_size
        x = torch.arange(kernel_size, dtype=torch.float32, device=device)
        x = x - kernel_size // 2

        # ガウシアン重み
        weights = torch.exp(-x ** 2 / (2 * sigma ** 2))
        weights = weights / weights.sum()

        # 移動平均と同様の処理
        pad_size = kernel_size // 2
        traj_transposed = trajectory.transpose(1, 2)
        padded = F.pad(traj_transposed, (pad_size, pad_size), mode='reflect')

        low_freq_list = []
        for d in range(dim):
            kernel = weights.view(1, 1, -1)
            filtered = F.conv1d(
                padded[:, d:d + 1, :],
                kernel,
                padding=0
            )
            low_freq_list.append(filtered)

        low_freq = torch.cat(low_freq_list, dim=1)
        low_freq = low_freq.transpose(1, 2)

        return low_freq

    def _adaptive_decompose(self, trajectory: torch.Tensor) -> torch.Tensor:
        """適応的な分解（ニューラルネットワークベース）"""
        if not self.learnable:
            # 学習不可の場合は移動平均にフォールバック
            return self._moving_average_decompose(trajectory)

        batch_size, seq_len, dim = trajectory.shape
        device = trajectory.device

        # まず基本的な移動平均を計算
        base_low_freq = self._moving_average_decompose(trajectory)

        # 軌道の特徴から適応的な重みを計算
        # [B, L, D] -> [B, D, L]
        traj_transposed = trajectory.transpose(1, 2)

        # 適応的な重み（0-1の範囲）
        adaptive_weights = self.adaptive_net(traj_transposed)  # [B, 1, L]
        adaptive_weights = adaptive_weights.transpose(1, 2)  # [B, L, 1]

        # 元の軌道と移動平均の間を適応的に補間
        low_freq = (
                adaptive_weights * base_low_freq +
                (1 - adaptive_weights) * trajectory
        )

        return low_freq

    def reconstruct(
            self,
            low_freq: torch.Tensor,
            high_freq: torch.Tensor
    ) -> torch.Tensor:
        """
        低周波と高周波成分から元の軌道を再構成

        Args:
            low_freq: 低周波成分 [batch_size, seq_len, 2]
            high_freq: 高周波成分 [batch_size, seq_len, 2]

        Returns:
            trajectory: 再構成された軌道
        """
        # 単純な加算
        trajectory = low_freq + high_freq

        # オプション：学習可能な結合重み
        if self.learnable and hasattr(self, 'combine_weight'):
            alpha = torch.sigmoid(self.combine_weight)
            trajectory = alpha * low_freq + (1 - alpha) * high_freq

        return trajectory

    def get_frequency_analysis(
            self,
            trajectory: torch.Tensor
    ) -> dict:
        """
        周波数分析を実行（デバッグ・可視化用）

        Args:
            trajectory: 軌道データ

        Returns:
            分析結果の辞書
        """
        with torch.no_grad():
            low_freq, high_freq = self.decompose(trajectory)

            # パワーの計算
            low_power = torch.norm(low_freq, dim=-1).mean()
            high_power = torch.norm(high_freq, dim=-1).mean()
            total_power = torch.norm(trajectory, dim=-1).mean()

            # 滑らかさの指標
            low_smoothness = self._compute_smoothness(low_freq)
            high_smoothness = self._compute_smoothness(high_freq)

            return {
                'low_freq_power': low_power.item(),
                'high_freq_power': high_power.item(),
                'low_freq_ratio': (low_power / total_power).item(),
                'high_freq_ratio': (high_power / total_power).item(),
                'low_freq_smoothness': low_smoothness.item(),
                'high_freq_smoothness': high_smoothness.item(),
            }

    def _compute_smoothness(self, trajectory: torch.Tensor) -> torch.Tensor:
        """軌道の滑らかさを計算"""
        if trajectory.shape[1] < 2:
            return torch.tensor(0.0)

        # 1階微分（速度）
        velocity = trajectory[:, 1:] - trajectory[:, :-1]

        # 速度の変化率
        if velocity.shape[1] > 1:
            acceleration = velocity[:, 1:] - velocity[:, :-1]
            smoothness = torch.mean(torch.norm(acceleration, dim=-1))
        else:
            smoothness = torch.tensor(0.0)

        return smoothness


class MultiScaleDecomposer(nn.Module):
    """
    複数スケールでの分解を行う拡張版
    """

    def __init__(
            self,
            scales: list = [5, 10, 20],
            learnable: bool = True,
            combination_method: str = 'weighted'  # 'weighted', 'cascade'
    ):
        super().__init__()

        self.scales = scales
        self.combination_method = combination_method

        # 各スケールのデコンポーザー
        self.decomposers = nn.ModuleList([
            ImprovedDataDecomposer(
                window_size=scale,
                learnable=learnable,
                decomposition_method='gaussian'
            )
            for scale in scales
        ])

        if learnable and combination_method == 'weighted':
            # 各スケールの重み
            self.scale_weights = nn.Parameter(
                torch.ones(len(scales)) / len(scales)
            )

    def decompose(self, trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        マルチスケール分解

        Args:
            trajectory: 入力軌道

        Returns:
            low_freq: 複数スケールを組み合わせた低周波成分
            high_freq: 高周波成分
        """
        if self.combination_method == 'weighted':
            # 重み付き平均
            low_freq_components = []
            for decomposer in self.decomposers:
                low_freq, _ = decomposer.decompose(trajectory)
                low_freq_components.append(low_freq)

            # 重みの正規化
            weights = F.softmax(self.scale_weights, dim=0)

            # 重み付き結合
            low_freq = torch.zeros_like(trajectory)
            for i, component in enumerate(low_freq_components):
                low_freq = low_freq + weights[i] * component

        elif self.combination_method == 'cascade':
            # カスケード分解
            remaining = trajectory
            low_freq = torch.zeros_like(trajectory)

            for decomposer in self.decomposers:
                low_component, high_component = decomposer.decompose(remaining)
                low_freq = low_freq + low_component
                remaining = high_component

        high_freq = trajectory - low_freq

        return low_freq, high_freq