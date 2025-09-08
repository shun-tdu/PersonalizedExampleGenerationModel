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

    def load_data(self):
        # CLAUDE_ADDED: ブロック情報も含めてグループ化（各ブロックの各試行を別々に扱う）
        self.trials = list(self.df.groupby(['subject_id','trial_num', 'block']))
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
        
        # 3. スキルスコアを取得（全タイムステップで同じ値）
        skill_score = trial_df['skill_score'].iloc[0] if len(trial_df) > 0 else 0.0

        # 4. テンソルに変換して返す
        # trajectory: [seq_len, 6] -> モデルが期待する形状
        # subject_id: 文字列
        # skill_score: スカラー値
        return (
            torch.tensor(trajectory_data, dtype=torch.float32),  # [seq_len, 6]
            subject_id,                                           # string
            torch.tensor(skill_score, dtype=torch.float32)       # scalar
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


    