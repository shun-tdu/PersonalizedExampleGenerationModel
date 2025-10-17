"""因子スキル指標データセット"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, Tuple, List

from .base_dataset import BaseExperimentDataset

class FactorScoreMetricsDataset(BaseExperimentDataset):
    """因子スキル指標データセット"""
    def __init__(self, df: pd.DataFrame, scalers: dict, feature_cols: list, config: Dict[str, Any]):
        super().__init__(config)
        self.df = df
        self.scalers = scalers
        self.feature_cols = feature_cols
        self.load_data()
        self.preprocess()

    def load_data(self):
        # CLAUDE_ADDED: ブロック情報も含めてグループ化（各ブロックの各試行を別々に扱う）
        self.trials = list(self.df.groupby(['subject_id', 'trial_num', 'block']))
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
        # CLAUDE_ADDED: ブロック情報も含む3つの要素を展開
        (subject_id, trial_num, block), trial_df = self.trials[idx]

        # CLAUDE_ADDED: 軌道特徴量（6次元：位置、速度、加速度のx,y）
        trajectory_features = ['HandlePosX', 'HandlePosY', 'HandleVelX',
                             'HandleVelY', 'HandleAccX', 'HandleAccY']

        # 2. 軌道データを時系列として取得 (seq_len, num_features)
        trajectory_data = trial_df[trajectory_features].values

        # CLAUDE_ADDED: 因子スコアを取得（データセットに含まれている場合）
        factor_score_cols = [col for col in trial_df.columns if col.startswith('factor_') and col.endswith('_score')]

        if factor_score_cols:
            # 因子スコアが存在する場合、ベクトルとして返す
            skill_factors = []
            for col in sorted(factor_score_cols):  # factor_1_score, factor_2_score, ...の順にソート
                skill_factors.append(trial_df[col].iloc[0] if len(trial_df) > 0 else 0.0)
            skill_factor_tensor = torch.tensor(skill_factors, dtype=torch.float32)  # [factor_num]
        else:
            # 因子スコアが存在しない場合、skill_scoreのみ（スカラー）
            skill_score = trial_df['skill_score'].iloc[0] if len(trial_df) > 0 else 0.0
            skill_factor_tensor = torch.tensor(skill_score, dtype=torch.float32)  # scalar

        # 4. テンソルに変換して返す
        # trajectory: [seq_len, 6] -> モデルが期待する形状
        # subject_id: 文字列
        # skill_factor: [factor_num] または scalar
        return (
            torch.tensor(trajectory_data, dtype=torch.float32),  # [seq_len, 6]
            subject_id,                                           # string
            skill_factor_tensor                                   # [factor_num] or scalar
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

