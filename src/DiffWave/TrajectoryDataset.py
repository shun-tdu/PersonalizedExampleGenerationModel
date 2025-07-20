# CLAUDE_ADDED
"""
DiffWave用の軌道データセット
UNetと同じデータセットを使用するためのシンボリックリンクまたはコピー
"""

import sys
import os

# 親ディレクトリのUNetモジュールを参照
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'UNet'))

# UNetのTrajectoryDatasetをインポート（モジュール名を明示的に指定）
import TrajectoryDataset as unet_trajectory_dataset
TrajectoryDataset = unet_trajectory_dataset.TrajectoryDataset

# UNetのTrajectoryDatasetをそのまま使用
__all__ = ['TrajectoryDataset']