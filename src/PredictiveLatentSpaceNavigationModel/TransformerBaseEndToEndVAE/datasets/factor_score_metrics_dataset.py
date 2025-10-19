"""因子スキル指標データセット"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, Tuple, List

from torch.distributed import isend
from torch.nn.utils.rnn import pad_sequence

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


class FactorScoreMetricsNoInterpolateDataset(BaseExperimentDataset):
    """因子スキル指標時間補完なしデータセット"""
    def __init__(self, df: pd.DataFrame, scalers: dict, feature_cols: list, config: Dict[str, Any]):
        super().__init__(config)
        self.df = df
        self.scalers = scalers
        self.feature_cols = feature_cols

        # パッチ処理の設定をconfigファイルから読み込む
        patch_config = self.config.get('patch', {})
        self.patch_size = patch_config.get('size', 20)
        self.patch_step = patch_config.get('step', 10)

        print(f"💡 Patching configuration: size={self.patch_size}, step={self.patch_step}")
        if self.patch_step < self.patch_size:
            print(f"   -> Overlap is enabled ({(self.patch_size - self.patch_step) / self.patch_size * 100:.1f}%).")

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
        (subject_id, trial_num, block), trial_df = self.trials[idx]

        # CLAUDE_ADDED: 軌道特徴量（6次元：位置、速度、加速度のx,y）
        trajectory_features = ['HandlePosX', 'HandlePosY', 'HandleVelX',
                             'HandleVelY', 'HandleAccX', 'HandleAccY']

        # 2. 軌道データをテンソルに変換 [seq_len, num_features]
        trajectory_tensor = torch.tensor(trial_df[trajectory_features].values,dtype=torch.float32)

        # 3. 軌道データに対してパッチ処理を実行 [num_patches, num_features, patch_size]
        patches = trajectory_tensor.unfold(dimension=0, size=self.patch_size, step=self.patch_step)
        patches = patches.permute(0, 2, 1) # [num_patches, patch_size, num_features]

        # 4. 因子スコアを取得
        factor_score_cols = [col for col in trial_df.columns if col.startswith('factor_') and col.endswith('_score')]
        if factor_score_cols:
            skill_factors = []
            for col in sorted(factor_score_cols):  # factor_1_score, factor_2_score, ...の順にソート
                skill_factors.append(trial_df[col].iloc[0] if len(trial_df) > 0 else 0.0)
            skill_factor_tensor = torch.tensor(skill_factors, dtype=torch.float32)  # [factor_num]
        else:
            # 因子スコアが存在しない場合、skill_scoreのみ（スカラー）
            skill_score = trial_df['skill_score'].iloc[0] if len(trial_df) > 0 else 0.0
            skill_factor_tensor = torch.tensor(skill_score, dtype=torch.float32)  # scalar

        return (
            patches,                # [num_patches, patch_size, 6]
            subject_id,             # string
            skill_factor_tensor     # [factor_num] or scalar
        )

    def get_info(self) -> Dict[str, Any]:
        """データセット情報を取得"""
        info = super().get_info()
        info.update({
            'num_trials': len(self.trials),
            'num_subjects': len(self.df['subject_id'].unique()),
            'feature_columns': self.feature_cols,
            'patch_size': self.patch_size,
            'patch_step': self.patch_step
        })
        return info

def collate_fn_padd(batch:List[Tuple[torch.Tensor, str, torch.Tensor]]) -> Dict[str, Any]:
    """
    可変長のパッチ列をパディングし、アテンションマスクを作成する

    Args:
        batch: Datasetから返される(軌道パッチ、subject_id, スキル因子ベクトル)のタプルのリスト

    Returns:
        A dict containing:
                    'trajectory': パディング済みの軌道テンソル (B, S, P, F)
                    'skill_vector': スキルベクトルのテンソル (B, N_factors)
                    'attention_mask': パディング部分を無視するためのマスク (B, S)
                    'subject_id': subject_idのリスト
    """
    # 1. データを種類ごとに分割
    # trajectories_listは、形状が (S_i, P, F) のテンソルのリストになる (S_iは試行iのパッチ数)
    trajectories_list, subject_ids, skill_factors = zip(*batch)

    # 2. 軌道データをパディング
    padded_trajectories = pad_sequence(trajectories_list, batch_first=True, padding_value=0.0)
    # -> 出力形状: (B, S_max, P, F)
    #    B = バッチサイズ
    #    S_max = バッチ内で最もパッチ数が多い試行のパッチ数
    #    P = パッチサイズ
    #    F = 特徴量数 (6)

    # 3. アテンションマスクを作成
    original_lengths = [t.size(0) for t in trajectories_list]
    max_len = padded_trajectories.size(1)

    # (B, S_max) の形状のマスクを作成
    # PyTorchのTransformerは、Trueの部分を無視(マスク)する
    attention_mask = torch.arange(max_len)[None, :] >= torch.tensor(original_lengths)[:, None]

    # 4. スキル因子を1つのテンソルにまとめる
    skill_factor_tensor = torch.stack(skill_factors)

    # 5.辞書形式で返す
    return {
        'trajectory': padded_trajectories,
        'skill_factor': skill_factor_tensor,
        'attention_mask': attention_mask,
        'subject_id': subject_ids
    }
