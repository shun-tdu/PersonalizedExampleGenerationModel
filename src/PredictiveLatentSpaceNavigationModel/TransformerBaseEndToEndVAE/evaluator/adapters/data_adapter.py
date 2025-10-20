# -*- coding: utf-8 -*-
"""
CLAUDE_ADDED: DataAdapter implementations
Converts batch data to standardized format
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
import torch
import numpy as np
import pandas as pd

from .types import StandardizedBatch


class DataAdapter(ABC):
    """Base class to convert batch data to standardized format"""

    TRAJECTORY_FEATURES = ['HandlePosX', 'HandlePosY', 'HandleVelX',
                          'HandleVelY', 'HandleAccX', 'HandleAccY']

    @abstractmethod
    def extract_batch(self, batch_data: Any) -> StandardizedBatch:
        """Extract batch data to standardized format"""
        pass

    @abstractmethod
    def get_skill_metric_dim(self) -> int:
        """Get skill metric dimension (1=scalar, N=factor vector)"""
        pass

    @abstractmethod
    def get_skill_metric_names(self) -> List[str]:
        """Get skill metric names"""
        pass

    def denormalize_trajectory(self, trajectory: np.ndarray, scalers: Dict) -> np.ndarray:
        """Denormalize trajectory data"""
        batch_size, seq_len, n_features = trajectory.shape
        denormalized = trajectory.copy()

        for feat_idx, feat_name in enumerate(self.TRAJECTORY_FEATURES):
            if feat_name in scalers:
                scaler = scalers[feat_name]
                feature_data = trajectory[:, :, feat_idx].reshape(-1, 1)
                denorm_feature = scaler.inverse_transform(feature_data)
                denormalized[:, :, feat_idx] = denorm_feature.reshape(batch_size, seq_len)

        return denormalized

    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self.TRAJECTORY_FEATURES

    def prepare_for_skill_calculation(self, trajectory: np.ndarray,
                                      time_step: float = 0.01) -> List[pd.DataFrame]:
        """Prepare DataFrames for skill metric calculation"""
        batch_size, seq_len, n_features = trajectory.shape
        list_of_dataframes = []
        for i in range(batch_size):
            df = pd.DataFrame(trajectory[i], columns=self.TRAJECTORY_FEATURES)
            df['Timestamp'] = np.arange(seq_len) * time_step
            list_of_dataframes.append(df)
        return list_of_dataframes


class TupleDataAdapter(DataAdapter):
    """Adapter for tuple format: (trajectory, subject_id, skill_score)"""

    def extract_batch(self, batch_data: Tuple) -> StandardizedBatch:
        trajectory, subject_ids, skill_scores = batch_data
        return StandardizedBatch(
            trajectory=trajectory,
            subject_id=list(subject_ids),
            skill_metric=skill_scores,
            attention_mask=None,
            metadata={}
        )

    def get_skill_metric_dim(self) -> int:
        return 1

    def get_skill_metric_names(self) -> List[str]:
        return ['skill_score']


class DictPatchDataAdapter(DataAdapter):
    """Adapter for dict format with patches (from collate_fn_padd)"""

    def __init__(self, factor_names: Optional[List[str]] = None):
        self.factor_names = factor_names
        self._skill_dim = None
        self._names = None

    def extract_batch(self, batch_data: Dict) -> StandardizedBatch:
        skill_factor = batch_data['skill_factor']

        if skill_factor.dim() == 1:
            self._skill_dim = 1
            self._names = ['skill_score']
        else:
            self._skill_dim = skill_factor.shape[1]
            self._names = self.factor_names or [f'factor_{i+1}' for i in range(self._skill_dim)]

        return StandardizedBatch(
            trajectory=batch_data['trajectory'],
            subject_id=list(batch_data['subject_id']),
            skill_metric=skill_factor,
            attention_mask=batch_data.get('attention_mask'),
            metadata={
                'is_patched': True,
                'original_shape': batch_data['trajectory'].shape
            }
        )

    def get_skill_metric_dim(self) -> int:
        return self._skill_dim if self._skill_dim is not None else 1

    def get_skill_metric_names(self) -> List[str]:
        return self._names if self._names is not None else ['skill_score']