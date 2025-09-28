# CLAUDE_ADDED: Hierarchical VAE用の階層的データセット
"""階層型VAE用データセット - SkillVAE学習とStyleVAE学習の両対応"""

import pandas as pd
import numpy as np
import torch
import joblib
import os
from typing import Dict, Any, Tuple, List, Optional
from torch.utils.data import Dataset

from .base_dataset import BaseExperimentDataset


class HierarchicalSkillDataset(BaseExperimentDataset):
    """階層型VAE用データセット - 学習段階に応じてデータ形式を切り替え"""

    def __init__(self, df: pd.DataFrame, scalers: dict, feature_cols: list,
                 config: Dict[str, Any], stage: str = 'skill'):
        super().__init__(config)
        self.df = df
        self.scalers = scalers
        self.feature_cols = feature_cols
        self.stage = stage  # 'skill' or 'style'
        self.block_len = config.get('block_len', 35)

        if stage == 'skill':
            self._prepare_skill_data()
        elif stage == 'style':
            self._prepare_style_data()
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 'skill' or 'style'")

    def _prepare_skill_data(self):
        """SkillVAE学習用：単一軌道データの準備"""
        # 各試行を個別にリスト化（従来のSkillMetricsDatasetと同様）
        self.trials = list(self.df.groupby(['subject_id', 'trial_num', 'block']))
        print(f"SkillVAE Dataset: {len(self.trials)} individual trials")

    def _prepare_style_data(self):
        """StyleVAE学習用：ブロック単位のスキル系列データの準備"""
        # 被験者とブロックでグループ化（ブロック内の全試行を1系列として扱う）
        self.blocks = []

        for (subject_id, block), block_df in self.df.groupby(['subject_id', 'block']):
            # ブロック内の試行を順序付け
            trials_in_block = []
            for trial_num in sorted(block_df['trial_num'].unique()):
                trial_df = block_df[block_df['trial_num'] == trial_num]
                if len(trial_df) > 0:
                    trials_in_block.append((trial_num, trial_df))

            if len(trials_in_block) >= self.block_len:
                # 十分な試行数がある場合
                trials_in_block = trials_in_block[:self.block_len]
            elif len(trials_in_block) > 0:
                # 不足分をパディング（最後の試行を繰り返し）
                while len(trials_in_block) < self.block_len:
                    trials_in_block.append(trials_in_block[-1])
            else:
                continue  # 空のブロックはスキップ

            self.blocks.append((subject_id, block, trials_in_block))

        print(f"StyleVAE Dataset: {len(self.blocks)} blocks from {len(set(b[0] for b in self.blocks))} subjects")

    def __len__(self) -> int:
        if self.stage == 'skill':
            return len(self.trials)
        else:  # style
            return len(self.blocks)

    def __getitem__(self, idx: int) -> Tuple:
        if self.stage == 'skill':
            return self._get_skill_item(idx)
        else:  # style
            return self._get_style_item(idx)

    def _get_skill_item(self, idx: int) -> Tuple[torch.Tensor, str, torch.Tensor]:
        """SkillVAE学習用アイテム取得"""
        # 従来のSkillMetricsDatasetと同様
        (subject_id, trial_num, block), trial_df = self.trials[idx]

        # 軌道特徴量（6次元：位置、速度、加速度のx,y）
        trajectory_features = ['HandlePosX', 'HandlePosY', 'HandleVelX',
                              'HandleVelY', 'HandleAccX', 'HandleAccY']

        # 軌道データ [seq_len, 6]
        trajectory_data = trial_df[trajectory_features].values

        # スキルスコア
        skill_score = trial_df['skill_score'].iloc[0] if len(trial_df) > 0 else 0.0

        return (
            torch.tensor(trajectory_data, dtype=torch.float32),  # [seq_len, 6]
            subject_id,                                          # string
            torch.tensor(skill_score, dtype=torch.float32)      # scalar
        )

    def _get_style_item(self, idx: int) -> Tuple[torch.Tensor, str]:
        """StyleVAE学習用アイテム取得 - スキル潜在変数系列を返す"""
        subject_id, block, trials_in_block = self.blocks[idx]

        # 各試行の軌道データからスキル潜在変数を抽出
        # 注意：実際には事前学習済みSkillVAEが必要
        # ここでは軌道データの特徴量を代用（実装時要修正）

        skill_features = []
        trajectory_features = ['HandlePosX', 'HandlePosY', 'HandleVelX',
                              'HandleVelY', 'HandleAccX', 'HandleAccY']

        for trial_num, trial_df in trials_in_block:
            # 軌道データの統計的特徴量を計算（スキル潜在変数の代用）
            trajectory_data = trial_df[trajectory_features].values

            if len(trajectory_data) > 0:
                # 統計的特徴量：平均、標準偏差、最大値、最小値など
                mean_vals = np.mean(trajectory_data, axis=0)  # [6,]
                std_vals = np.std(trajectory_data, axis=0)    # [6,]
                max_vals = np.max(trajectory_data, axis=0)    # [6,]
                min_vals = np.min(trajectory_data, axis=0)    # [6,]

                # 特徴量結合 [24,]
                features = np.concatenate([mean_vals, std_vals, max_vals, min_vals])
            else:
                # フォールバック
                features = np.zeros(24)

            skill_features.append(features)

        # スキル系列 [block_len, feature_dim]
        skill_sequence = np.array(skill_features)

        return (
            torch.tensor(skill_sequence, dtype=torch.float32),  # [block_len, feature_dim]
            subject_id                                          # string
        )

    def get_info(self) -> Dict[str, Any]:
        """データセット情報を取得"""
        info = super().get_info()
        info.update({
            'stage': self.stage,
            'feature_columns': self.feature_cols,
        })

        if self.stage == 'skill':
            info.update({
                'num_trials': len(self.trials),
                'num_subjects': len(self.df['subject_id'].unique()),
            })
        else:  # style
            info.update({
                'num_blocks': len(self.blocks),
                'block_len': self.block_len,
                'num_subjects': len(set(b[0] for b in self.blocks)),
            })

        return info

    def load_data(self):
        """BaseExperimentDatasetの抽象メソッド実装 - データは初期化時に渡されるため何もしない"""
        pass

    def preprocess(self):
        """BaseExperimentDatasetの抽象メソッド実装 - 前処理は__init__で実行済みのため何もしない"""
        pass


class StyleVAESkillSequenceDataset(BaseExperimentDataset):
    """StyleVAE学習用：事前学習済みSkillVAEから生成されたスキル潜在変数系列を使用"""

    def __init__(self, skill_sequences_path: str, config: Dict[str, Any]):
        super().__init__(config)
        self.skill_sequences_path = skill_sequences_path
        self.block_len = config.get('block_len', 35)
        self._load_skill_sequences()

    def _load_skill_sequences(self):
        """事前計算済みスキル潜在変数系列を読み込み"""
        if not os.path.exists(self.skill_sequences_path):
            raise FileNotFoundError(f"Skill sequences file not found: {self.skill_sequences_path}")

        # 事前学習済みSkillVAEで変換済みのスキル潜在変数を読み込み
        data = joblib.load(self.skill_sequences_path)

        self.skill_blocks = data['skill_blocks']        # List[(subject_id, block, z_skill_sequence)]
        self.subject_mapping = data['subject_mapping']  # Dict[str, int]

        print(f"StyleVAE Dataset: Loaded {len(self.skill_blocks)} skill sequence blocks")

    def __len__(self) -> int:
        return len(self.skill_blocks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """スキル潜在変数系列と被験者IDを返す"""
        subject_id, block, z_skill_sequence = self.skill_blocks[idx]

        # z_skill_sequence: [block_len, skill_latent_dim]
        return (
            torch.tensor(z_skill_sequence, dtype=torch.float32),  # [block_len, skill_dim]
            subject_id                                            # string
        )

    def get_info(self) -> Dict[str, Any]:
        """データセット情報を取得"""
        info = super().get_info()
        info.update({
            'num_blocks': len(self.skill_blocks),
            'block_len': self.block_len,
            'num_subjects': len(self.subject_mapping),
            'skill_latent_dim': self.skill_blocks[0][2].shape[1] if self.skill_blocks else 0
        })
        return info

    def load_data(self):
        """BaseExperimentDatasetの抽象メソッド実装 - データは初期化時に読み込み済み"""
        pass

    def preprocess(self):
        """BaseExperimentDatasetの抽象メソッド実装 - 前処理は不要"""
        pass