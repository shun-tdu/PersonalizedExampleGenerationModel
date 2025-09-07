"""データローダー作成のファクトリークラス"""

import os
import pandas as pd
import joblib
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import  train_test_split
from typing import Dict, Any, Tuple, Optional, Type, List

from .base_dataset import BaseExperimentDataset
from .generalized_coordinate_dataset import  GeneralizedCoordinateDataset
from .skill_metrics_dataset import SkillMetricsDataset  # CLAUDE_ADDED

class DatasetRegistry:
    """データセットタイプの登録システム"""

    _datasets = {
        'generalized_coordinate': GeneralizedCoordinateDataset,
        'skill_metrics': SkillMetricsDataset,  # CLAUDE_ADDED: 新しいスキル指標データセット
    }

    @classmethod
    def register_dataset(cls, name: str, dataset_class: Type[BaseExperimentDataset]):
        """新しいデータセットタイプを登録"""
        cls._datasets[name] = dataset_class

    @classmethod
    def get_dataset_class(cls, name: str) -> Type[BaseExperimentDataset]:
        """データセットクラスを取得"""
        if name not in cls._datasets:
            raise ValueError(f"Unknown dataset type: {name}. Available types: {list(cls._datasets.keys())}")
        return cls._datasets[name]

    @classmethod
    def list_available_datasets(cls) -> List[str]:
        """利用可能なデータセットタイプをリスト"""
        return list(cls._datasets)


class DataLoaderFactory:
    @staticmethod
    def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, None, Dataset, Optional[Dataset], Dataset]:
        """
        データローダーを作成

        :param data_config: コンフィグのdataセクション
        :return: tuple(train_loader, val_loader, test_loader, test_df, train_dataset, val_dataset, test_dataset)
        """

        data_config = config.get('data', {})
        training_config = config.get('training', {})

        dataset_type = data_config.get('type', 'dummy')

        if dataset_type == 'generalized_coordinate':
            return DataLoaderFactory._create_generalized_coordinate_dataloaders(data_config, training_config)
        elif dataset_type == 'skill_metrics':
            return DataLoaderFactory._create_skill_metrics_dataloaders(data_config, training_config)  # CLAUDE_ADDED
        else:
            return DataLoaderFactory._create_custom_dataloaders(data_config, training_config)

    @staticmethod
    def _create_generalized_coordinate_dataloaders(data_config: Dict[str, Any], training_config: Dict[str, Any]) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, pd.DataFrame, Dataset, Optional[Dataset], Dataset]:
        """一般化座標データローダーを作成"""
        print("一般化座標データローダーを作成中...")

        # 設定からパラメータを取得
        if 'data_path' not in data_config:
            raise KeyError("data_configに 'data_path'のキーが見つかりません ")

        data_path = data_config['data_path']
        batch_size = training_config.get('batch_size', 32)
        random_seed = data_config.get('random_seed', 42)

        # データのパス
        train_data_path = os.path.join(data_path, 'train_data.parquet')
        test_data_path = os.path.join(data_path, 'test_data.parquet')
        scalers_path = os.path.join(data_path, 'scalers.joblib')
        feature_config_path = os.path.join(data_path, 'feature_config.joblib')

        try:
            # データとスケーラーを読み込み
            train_val_df = pd.read_parquet(train_data_path)
            test_df = pd.read_parquet(test_data_path)
            scalers = joblib.load(scalers_path)
            feature_config = joblib.load(feature_config_path)
            feature_cols = feature_config['feature_cols']

            print(f"Loaded scalers: {list(scalers.keys())}")
            print(f"Feature columns: {feature_cols}")

        except FileNotFoundError as e:
            print(f"エラー: データローダー作成に必要なファイルがありません")
            raise

        # 学習用データと検証データに分割
        train_val_subject_ids = train_val_df['subject_id'].unique()

        if len(train_val_subject_ids) < 2:
            print("警告: 検証セットを作成するには学習データの被験者が2人以上必要です。検証セットなしで進めます。")
            train_ids = train_val_subject_ids
            val_ids = []
        else:
            train_ids, val_ids = train_test_split(
                train_val_subject_ids,
                test_size=data_config.get('val_split', 0.25),
                random_state=random_seed
            )

        train_df = train_val_df[train_val_df['subject_id'].isin(train_ids)]
        val_df = train_val_df[train_val_df['subject_id'].isin(val_ids)]

        print(
            f"データ分割: 学習用={len(train_ids)}人, 検証用={len(val_ids)}人, テスト用={len(test_df['subject_id'].unique())}人")

        # Datasetの作成
        train_dataset = GeneralizedCoordinateDataset(train_df, scalers, feature_cols, data_config)
        val_dataset = GeneralizedCoordinateDataset(val_df, scalers, feature_cols,
                                                   data_config) if not val_df.empty else None
        test_dataset = GeneralizedCoordinateDataset(test_df, scalers, feature_cols, data_config)

        # DataLoaderの作成
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, test_df, train_dataset, val_dataset, test_dataset

    @staticmethod
    def _create_skill_metrics_dataloaders(data_config: Dict[str, Any], training_config: Dict[str, Any]) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, pd.DataFrame, Dataset, Optional[Dataset], Dataset]:
        """スキル指標データローダーを作成"""
        print("スキル指標データローダーを作成中...")

        # CLAUDE_ADDED: skill_metricsデータセット用のデータローダー作成
        if 'data_path' not in data_config:
            raise KeyError("data_configに 'data_path'のキーが見つかりません ")

        data_path = data_config['data_path']
        batch_size = training_config.get('batch_size', 32)
        random_seed = data_config.get('random_seed', 42)

        # データのパス
        train_data_path = os.path.join(data_path, 'train_data.parquet')
        test_data_path = os.path.join(data_path, 'test_data.parquet')
        scalers_path = os.path.join(data_path, 'scalers.joblib')
        feature_config_path = os.path.join(data_path, 'feature_config.joblib')

        try:
            # データとスケーラーを読み込み
            train_val_df = pd.read_parquet(train_data_path)
            test_df = pd.read_parquet(test_data_path)
            scalers = joblib.load(scalers_path)
            feature_config = joblib.load(feature_config_path)
            feature_cols = feature_config['feature_cols']

            print(f"Loaded scalers: {list(scalers.keys())}")
            print(f"Feature columns: {feature_cols}")
            print(f"Target sequence length: {feature_config.get('target_seq_len', 'N/A')}")

        except FileNotFoundError as e:
            print(f"エラー: データローダー作成に必要なファイルがありません: {e}")
            raise

        # 学習用データと検証データに分割
        train_val_subject_ids = train_val_df['subject_id'].unique()

        if len(train_val_subject_ids) < 2:
            print("警告: 検証セットを作成するには学習データの被験者が2人以上必要です。検証セットなしで進めます。")
            train_ids = train_val_subject_ids
            val_ids = []
        else:
            train_ids, val_ids = train_test_split(
                train_val_subject_ids,
                test_size=data_config.get('val_split', 0.25),
                random_state=random_seed
            )

        train_df = train_val_df[train_val_df['subject_id'].isin(train_ids)]
        val_df = train_val_df[train_val_df['subject_id'].isin(val_ids)]

        print(f"データ分割: 学習用={len(train_ids)}人, 検証用={len(val_ids)}人, テスト用={len(test_df['subject_id'].unique())}人")
        print(f"学習用試行数: {len(train_df)}, 検証用試行数: {len(val_df)}, テスト用試行数: {len(test_df)}")

        # Datasetの作成
        train_dataset = SkillMetricsDataset(train_df, scalers, feature_cols, data_config)
        val_dataset = SkillMetricsDataset(val_df, scalers, feature_cols, data_config) if not val_df.empty else None
        test_dataset = SkillMetricsDataset(test_df, scalers, feature_cols, data_config)

        # DataLoaderの作成
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, test_df, train_dataset, val_dataset, test_dataset

    @staticmethod
    def _create_custom_dataloaders(data_config: Dict[str, Any], training_config: Dict[str, Any]) -> Tuple[
        DataLoader, Optional[DataLoader], DataLoader, Optional[pd.DataFrame]]:
        """カスタムデータセットのデータローダーを作成"""
        dataset_type = data_config['type']
        dataset_class = DatasetRegistry.get_dataset_class(dataset_type)

        # カスタムデータセットの実装に依存
        # 基本的にはGeneralizedCoordinateDatasetと同様の処理
        raise NotImplementedError(f"Custom dataset type '{dataset_type}' requires specific implementation")

    @staticmethod
    def validate_data_config(config: Dict[str, Any]) -> bool:
        """データ設定の妥当性を検証"""
        data_config = config.get('data', {})
        dataset_type = data_config.get('type')

        if not dataset_type:
            print("エラー: 'type'フィールドが設定されていません")
            return False

        if dataset_type not in DatasetRegistry.list_available_datasets():
            print(f"エラー: 不明なデータセットタイプ '{dataset_type}'")
            print(f"利用可能なタイプ: {DatasetRegistry.list_available_datasets()}")
            return False

        # タイプ別の検証
        if dataset_type == 'generalized_coordinate':
            data_path = data_config.get('data_path')
            if not data_path:
                print("エラー: 'data_path'が設定されていません")
                return False
            if not os.path.exists(data_path):
                print(f"エラー: データディレクトリが見つかりません: {data_path}")
                return False

        return True

    @staticmethod
    def get_dataset_info(config: Dict[str, Any]) -> Dict[str, Any]:
        """データセットの情報を取得（実際にロードせずに）"""
        data_config = config.get('data', {})
        training_config = config.get('training', {})

        dataset_type = data_config.get('type', 'unknown')

        info = {
            'type': dataset_type,
            'batch_size': training_config.get('batch_size', 32),
            'seq_len': data_config.get('seq_len', 100),
        }

        if dataset_type == 'generalized_coordinate':
            data_path = data_config.get('data_path')
            if data_path and os.path.exists(data_path):
                try:
                    feature_config_path = os.path.join(data_path, 'feature_config.joblib')
                    if os.path.exists(feature_config_path):
                        feature_config = joblib.load(feature_config_path)
                        info['feature_columns'] = feature_config.get('feature_cols', [])
                        info['num_features'] = len(info['feature_columns'])
                except:
                    pass

        return info
