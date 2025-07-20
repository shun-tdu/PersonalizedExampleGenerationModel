# CLAUDE_ADDED
"""
Transformer用の軌道データセット
UNetと同じデータセットを使用するためのラッパー
"""

import sys
import os
import torch
from torch.utils.data import Dataset
import numpy as np

# 親ディレクトリのUNetモジュールを参照
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'UNet'))

# UNetのTrajectoryDatasetをインポート（モジュール名を明示的に指定）
import TrajectoryDataset as unet_trajectory_dataset
UNetTrajectoryDataset = unet_trajectory_dataset.TrajectoryDataset


class TrajectoryDataset(Dataset):
    """
    Transformerに適したデータセット
    UNetのデータセットを[batch, seq_len, features]形式に変換
    """
    def __init__(self, data_path: str):
        # UNetのデータセットを使用
        self.unet_dataset = UNetTrajectoryDataset(data_path)
        
    def __len__(self):
        return len(self.unet_dataset)
    
    def __getitem__(self, idx):
        trajectory, condition = self.unet_dataset[idx]
        
        # UNetの形状: [2, seq_len] -> Transformerの形状: [seq_len, 2]
        trajectory_transposed = trajectory.transpose(0, 1)
        
        return trajectory_transposed, condition


# UNetのTrajectoryDatasetをそのまま使用可能にする
__all__ = ['TrajectoryDataset', 'UNetTrajectoryDataset']