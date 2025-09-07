"""スキル指標データセット"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, Tuple, List

from .base_dataset import BaseExperimentDataset


class SkillMetricsDataset(BaseExperimentDataset):
    """スキル指標を用いたデータセット"""
    
    def __init__(self, df: pd.DataFrame, scalers: dict, feature_cols: list, config: Dict[str, Any]):
        super().__init__(config)
        self.df = df
        self.scalers = scalers
        self.feature_cols = feature_cols
        self.load_data()
        self.preprocess()
    
    # TODO(human) - ここにload_data, preprocess, __len__, __getitem__メソッドを実装してください
    def load_data(self):
        self.trials = list(self.df.groupby(['subject_id','trial_num']))
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

        # DataPreprocess analyze_skill_metrics.pyで前処理を行う
        # 以下のようなカラム構造、Pos, Vel, Accは事前にカラム毎にスケーリング、固定長化
        # subject_id, skill_score, HandlePosX, HandlePosY, HandleVelX, HandleVelY, HandleAccX, HandleAAccY
        # 2. 特徴量データをNumpy配列に変換
        trajectories = trial_df['HandlePosX', 'HandlePosY', 'HandleVelX', 'HandleVelY', 'HandleAccX', 'HandleAccY'].values
        skill_score = trial_df['SkillScore']

        # 3. テンソルに変換して返す
        return (
            torch.tensor(trajectories, dtype=torch.float32),
            subject_id,
            torch.tensor(skill_score, dtype=torch.float32)
        )

    def get_info(self) -> Dict[str, Any]:
        """データセット情報を取得"""
        info = super().get_info()
        info.update({
            'num_trials': len(self.trials),
            'num_subjects': len(self.df['subject_id'].unique()),
            'feature_columns': self.feature_cols,
        })
        return info


    