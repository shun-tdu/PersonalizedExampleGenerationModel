import os
import json
import sqlite3
import importlib.util
from datetime import datetime
from typing import Dict, Any, List, Protocol, Optional
from abc import ABC, abstractmethod
import torch
import yaml


class ModelMetrics(Protocol):
    """モデルの損失・評価指標のインターフェース"""
    def get_loss_dict(self) -> Dict[str, float]:
        """全ての損失値を辞書で返す"""
        pass

    def get_evaluation_dict(self) -> Dict[str, float]:
        """全ての評価指標を辞書で返す"""
        pass

class ExperimentTracker(ABC):
    """実験追跡の抽象基底クラス"""
    @abstractmethod
    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """エポックごとの指標をログ"""
        pass

    @abstractmethod
    def log_final_results(self, results: Dict[str, Any]):
        """最終結果をログ"""
        pass

    @abstractmethod
    def update_status(self, status: str, **kwargs):
        """実験ステータスを更新"""
        pass

class DatabaseTracker(ExperimentTracker):
    """データベース用実験追跡"""
    def __init__(self, db_path: str, experiment_id: int, table_name: str):
        self.db_path = db_path
        self.experiment_id = experiment_id
        self.table_name = table_name

        # テーブル構造を動的に取得
        self._get_table_schema()

    def _get_table_schema(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({self.table_name})")
                self.columns = {row[1]: row[2] for row in cursor.fetchall()}
        except Exception as e:
            print(f"テーブル構造取得エラー: {e}")
            self.columns = {}

    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """エポックごとの指標をログ(オプション: 履歴テーブルに保存)"""
        # 必要に応じて履歴テーブルに保存
        pass

    def log_final_results(self, results: Dict[str, Any]):
        """最終結果をデータベースに保存"""
        self._safe_update(results)

    def update_status(self, status: str, **kwargs):
        """ステータス更新"""
        update_data = {'status': status, **kwargs}
        self._safe_update(update_data)

    def _safe_update(self, data: Dict[str, Any]):
        """安全なDB更新(存在するカラムのみ更新)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 存在するカラムのみフィルタリング
                valid_data = {k: v for k, v in data.items() if k in self.columns}

                if not valid_data:
                    return

                # 動的にUPDATE文を生成
                set_clause = ", ".join([f"{key} = ?" for key in valid_data.keys()])
                query = f"UPDATE {self.table_name} SET {set_clause} WHERE id = ?"
                values = tuple(valid_data.values()) + (self.experiment_id,)

                cursor.execute(query, values)
                conn.commit()
        except Exception as e:
            print(f"DB更新エラー (ID: {self.experiment_id}): {e}")


class ConfigHandler:
    """設定ファイル処理の統一インターフェース"""

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def extract_db_params(config: Dict[str, Any]) -> Dict[str, Any]:
        """設定からDB保存用パラメータを抽出"""
        params = {}

        # 設定の各セクションから値を抽出
        for section_name, section_data in config.items():
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    # フラット化して保存
                    params_name = f"{section_name}_{key}" if section_name != 'experiment' else key
                    params[key] = value
            else:
                params[section_name] = section_data

        return params


class ModelWrapper:
    """モデルとその損失計算をラップ"""
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config

    def compute_losses(self, batch_data) -> Dict[str, Any]:
        """損失を計算して辞書で返す"""
        # CLAUDE_ADDED: データ形式に対応した処理
        if len(batch_data) == 2:
            trajectories, subject_ids = batch_data
            outputs = self.model(trajectories, subject_ids)
        elif len(batch_data) >= 3:
            # 新しいデータローダー形式: [trajectory, subject_id, is_expert, ...]
            # モデルは trajectory と subject_id のみ使用
            trajectories, subject_ids = batch_data[0], batch_data[1]
            # is_expert (batch_data[2]) は現在のモデルでは使用しない
            outputs = self.model(trajectories, subject_ids)
        else:
            outputs = self.model(*batch_data)

        # モデルからの出力から損失を計算
        if hasattr(outputs, 'get_loss_dict'):
            return outputs.get_loss_dict()
        elif isinstance(outputs, dict):
            losses = {}
            for k, v in outputs.items():
                if 'loss' in k.lower():
                    if torch.is_tensor(v) and v.requires_grad:
                        losses[k] = v  # テンソルのまま保持（backwardのため）
                    elif torch.is_tensor(v):
                        losses[k] = v.item()
                    else:
                        losses[k] = v
            return losses
        else:
            # フォールバック (総損失のみ)
            return {'total_loss': outputs.item() if torch.is_tensor(outputs) else outputs}

    def evaluate(self, test_data) -> Dict[str, Any]:
        """評価指標を計算"""

        # モデル固有の評価ロジック
        if hasattr(self.model, 'evaluate'):
            return self.model.evaluate(test_data)
        else:
            return {}


class ExperimentRunner:
    """実験実行の統合クラス"""
    def __init__(self, tracker: ExperimentTracker, config: Dict[str, Any]):
        self.tracker = tracker
        self.config = config
        self.history = {}

    # def setup_model(self) -> ModelWrapper:
    #     """設定に基づいてモデルを初期化"""
    #     model_config = self.config.get('model', {})
    #
    #     # 動的にモデルクラスをロード
    #     model_class_name = model_config.get('class_name', 'MotionTransformerVAE')
    #     # model_module_path = model_config.get('module_path', 'models.ExampleModel')
    #
    #     # モジュールを動的インポート
    #     spec = importlib.util.spec_from_file_location(
    #         "model_module",
    #         model_config.get('file_path', 'models/ExampleModel.py')
    #     )
    #     model_module = importlib.util.module_from_spec(spec)
    #     spec.loader.exec_module(model_module)
    #
    #     # モデルクラスを取得
    #     model_class = getattr(model_module, model_class_name)
    #
    #     # モデルを初期化
    #     model = model_class(**{k: v for k, v in model_config.items() if k not in ['class_name', 'module_path', 'file_path']})
    #
    #     return ModelWrapper(model, self.config)

    def run_training(self, model_wrapper: ModelWrapper, train_loader, val_loader):
        """訓練ループを実行"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_wrapper.model.to(device)

        # オプティマイザ等のセットアップ
        optimizer = self._setup_optimizer(model_wrapper.model)
        scheduler = self._setup_scheduler(optimizer)

        training_config = self.config.get('training', {})
        num_epochs = training_config.get('num_epochs', 100)

        try:
            self.tracker.update_status('running', start_time=datetime.now().isoformat())

            for epoch in range(num_epochs):
                # 訓練フェーズ
                epoch_metrics = self._train_epoch(
                    model_wrapper, train_loader, optimizer, device
                )

                # 検証フェーズ
                val_metrics = self._validate_epoch(
                    model_wrapper, val_loader, device
                )

                # 全指標を統合
                all_metrics = {
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr'],
                    **{f'train_{k}': v for k, v in epoch_metrics.items()},
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                }

                # 履歴に追加
                for key, value in all_metrics.items():
                    if key not in self.history:
                        self.history[key] = []
                    self.history[key].append(value)

                # エポックごとのログ
                self.tracker.log_epoch_metrics(epoch, all_metrics)

                # 学習率を更新
                if scheduler:
                    scheduler.step()

                # 進捗表示
                print(f"Epoch {epoch+1}/{num_epochs}: {epoch_metrics}")

            # 訓練完了
            final_metrics = {
                'status': 'completed',
                'end_time': datetime.now().isoformat(),
                **{f'final_{k}': v[-1] if v else 0 for k, v in self.history.items() if k != 'epoch'}
            }

            self.tracker.log_final_results(final_metrics)

        except Exception as e:
            error_metrics = {
                'status': 'failed',
                'end_time': datetime.now().isoformat(),
                'error_message': str(e)
            }
            self.tracker.log_final_results(error_metrics)
            raise

    def _train_epoch(self, model_wrapper: ModelWrapper, train_loader, optimizer, device):
        """1エポックの訓練"""
        model_wrapper.model.train()
        epoch_losses = {}

        for batch_data in train_loader:
            # デバイスに移動
            batch_data = [data.to(device) if torch.is_tensor(data) else data
                          for data in batch_data]

            optimizer.zero_grad()

            # 損失計算
            losses = model_wrapper.compute_losses(batch_data)
            total_loss = losses.get('total_loss', 0)

            total_loss.backward()
            optimizer.step()

            # 損失を蓄積
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                # テンソルの場合はitemを使用、そうでなければそのまま
                if torch.is_tensor(value):
                    epoch_losses[key].append(value.item())
                else:
                    epoch_losses[key].append(float(value))

        # 平均を計算
        return {k: sum(v) / len(v) for k, v in epoch_losses.items()}

    def _validate_epoch(self, model_wrapper: ModelWrapper, val_loader, device):
        """1エポックの検証"""
        model_wrapper.model.eval()
        epoch_losses = {}

        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = [data.to(device) if torch.is_tensor(data) else data
                              for data in batch_data]

                losses = model_wrapper.compute_losses(batch_data)

                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = []
                    # テンソルの場合はitemを使用、そうでなければそのまま  
                    if torch.is_tensor(value):
                        epoch_losses[key].append(value.item())
                    else:
                        epoch_losses[key].append(float(value))

        return {k: sum(v) / len(v) for k, v in epoch_losses.items()}

    def _setup_optimizer(self, model):
        """オプティマイザをセットアップ"""
        training_config = self.config.get('training', {})
        optimizer_type = training_config.get('optimizer', 'AdamW')
        lr = training_config.get('learning_rate', training_config.get('lr', 1e-3))
        weight_decay = training_config.get('weight_decay', 1e-5)

        if optimizer_type == 'AdamW':
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'Adam':
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _setup_scheduler(self, optimizer):
        """スケジューラをセットアップ"""
        training_config = self.config.get('training', {})
        scheduler_type = training_config.get('scheduler', None)

        if scheduler_type == 'CosineAnnealingWarmRestarts':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=training_config.get('scheduler_T_0', 15),
                T_mult=training_config.get('scheduler_T_mult', 2),
                eta_min=training_config.get('scheduler_eta_min', 1e-6)
            )

        return None


