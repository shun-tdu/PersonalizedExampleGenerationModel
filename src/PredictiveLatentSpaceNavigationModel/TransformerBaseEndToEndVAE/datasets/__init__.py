"""データセット関連モジュール"""

from .base_dataset import BaseExperimentDataset
from .generalized_coordinate_dataset import GeneralizedCoordinateDataset, apply_physics_based_scaling
from .dataloader_factory import DataLoaderFactory, DatasetRegistry

__all__ = [
    'BaseExperimentDataset',
    'GeneralizedCoordinateDataset',
    'apply_physics_based_scaling',
    'DataLoaderFactory',
    'DatasetRegistry'
]

# 利用可能なデータセットタイプを表示する便利関数
def list_available_datasets():
    """利用可能なデータセットタイプを表示"""
    return DatasetRegistry.list_available_datasets()

def register_custom_dataset(name: str, dataset_class):
    """カスタムデータセットを登録"""
    DatasetRegistry.register_dataset(name, dataset_class)