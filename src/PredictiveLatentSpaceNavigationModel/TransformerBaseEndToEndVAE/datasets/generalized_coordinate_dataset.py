import os
import pandas as pd
import numpy as np
import joblib
import torch
from typing import Dict, Any, Tuple, List
from .base_dataset import BaseExperimentDataset

def apply_physics_based_scaling(features: np.ndarray, scalers: dict, feature_cols: list) -> np.ndarray:
    """物理量別にスケーリングを適用"""
    scaled_feature = features.copy()

    for i, col in enumerate(feature_cols):
        if col in scalers:
            scaled_feature[:, i] = scalers[col].transform(features[:, i].reshape(-1, 1)).flatten()

    return scaled_feature


class GeneralizedCoordinateDataset(BaseExperimentDataset):
    """一般化座標データセット"""

    def __init__(self, df: pd.DataFrame, scalers: dict, feature_cols: list, config: Dict[str, Any]):
        super().__init__(config)
        self.df = df
        self.scalers = scalers
        self.feature_cols = feature_cols
        self.load_data()
        self.preprocess()

    def load_data(self):
        """データを試行ごとにグループ化"""
        self.trials = list(self.df.groupby(['subject_id', 'trial_num']))
        print(f"Dataset initialized with {len(self.trials)} trials")
        print(f"Feature columns: {self.feature_cols}")
        print(f"Available scalers: {list(self.scalers.keys())}")

    def preprocess(self):
        pass

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, torch.Tensor]:
        """指定されたインデックスのデータを取得"""

        # 1. インデックスに対応する試行データを取得
        (subject_id, _), trial_df = self.trials[idx]

        # 2. 特徴量データをNumpy配列に変換
        features = trial_df[self.feature_cols].values

        # 3. 物理量別にスケーリング
        scaled_feature = apply_physics_based_scaling(features, self.scalers, self.feature_cols)\

        # 4. 固定長化
        if len(scaled_feature) > self.seq_len:
            processed_trajectory = scaled_feature[:self.seq_len]
        else:
            padding_length = self.seq_len - len(scaled_feature)
            last_value = scaled_feature[-1:].repeat(padding_length, axis=0)
            processed_trajectory = np.vstack([scaled_feature, last_value])

        # 5. ラベルを取得
        is_expert = trial_df['is_expert'].iloc[0]

        # 6. テンソルに変換して返す
        return (
            torch.tensor(processed_trajectory, dtype=torch.float32),
            subject_id,
            torch.tensor(is_expert, dtype=torch.long)
        )

    def get_info(self) -> Dict[str, Any]:
        """データセット情報を取得"""
        info = super().get_info()
        info.update({
            'num_trials': len(self.trials),
            'num_subjects': len(self.df['subject_id'].unique()),
            'feature_columns': self.feature_cols,
            'scalers': list(self.scalers.keys())
        })
        return info