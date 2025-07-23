# CLAUDE_ADDED
import torch
import torch.nn as nn
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
    
    def reconstruct(self, low_freq: torch.Tensor, high_freq: torch.Tensor) -> torch.Tensor:
        """
        低周波成分と高周波成分から元のデータを復元
        
        Args:
            low_freq: 低周波成分
            high_freq: 高周波成分
            
        Returns:
            復元されたデータ
        """
        return low_freq + high_freq