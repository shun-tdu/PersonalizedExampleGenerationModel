"""データセットの基底クラスと共通機能"""

from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd

class BaseExperimentDataset(ABC, Dataset):
    """実験用データセットの基底クラス"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.seq_len = config.get('seq_len', 100)

    @abstractmethod
    def load_data(self)->None:
        """データを読み込む"""
        pass

    @abstractmethod
    def preprocess(self) -> None:
        """前処理を行う"""
        pass

    @abstractmethod
    def __len__(self) -> None:
        """データセットのサイズを返す"""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.T, ...]:
        """インデックスに対応するデータを返す"""
        pass

    def get_info(self) -> Dict[str, Any]:
        """データセットの情報を返す"""
        return {
            'type': self.__class__.__name__,
            'size': len(self),
            'seq_len': self.seq_len,
            'config': self.config
        }