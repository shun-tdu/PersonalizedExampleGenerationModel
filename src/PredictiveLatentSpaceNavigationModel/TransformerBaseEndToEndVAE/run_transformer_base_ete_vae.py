#!/usr/bin/env python3
# CLAUDE_ADDED: 統合実験管理システム
"""
Transformer Base End-to-End VAE 統合実験管理システム
コンフィグ駆動の疎結合アーキテクチャ
"""
import os
import sys
import argparse
import sqlite3
import yaml
import json
import datetime
import torch
import numpy as np
import subprocess
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import importlib.util
from torch.utils.data import DataLoader

# 相対インポートの設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiment_manager import (
    ExperimentRunner, DatabaseTracker, ConfigHandler, ModelWrapper
)
from evaluator.base_evaluator import EvaluationPipeline

from datasets import DataLoaderFactory, DatasetRegistry


@dataclass
class ExperimentConfig:
    """実験設定のデータクラス"""
    config_path: str
    config_name: str
    priority: int = 0
    dependencies: List[str] = None
    tags: List[str] = None
    estimated_duration: Optional[int] = None  # 分単位

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []

class IntegratedExperimentRunner:
    """CLAUDE_ADDED: 統合実験実行クラス"""
    
    def __init__(self, config_path: str, db_path: str = None, experiment_id: int = None):
        self.config_path = config_path
        self.config = ConfigHandler.load_config(config_path)
        
        # データベース設定
        self.db_path = db_path or 'experiments.db'
        self.experiment_id = experiment_id
        
        # トラッカー初期化
        if self.experiment_id:
            self.tracker = DatabaseTracker(
                self.db_path, self.experiment_id, 
                'transformer_base_e2e_vae_experiment'
            )
        else:
            self.tracker = None
            
        # デバイス設定
        self.device = self._setup_device()
        
        # 出力ディレクトリ
        self.output_dir = self._setup_output_directory()
        
    def _setup_device(self):
        """デバイス設定"""
        device_config = self.config.get('system', {}).get('device', 'auto')
        
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
            
        print(f"使用デバイス: {device}")
        return device
        
    def _setup_output_directory(self):
        """出力ディレクトリを設定"""
        base_dir = self.config.get('output', {}).get('base_dir', 'outputs')
        exp_name = self.config.get('experiment', {}).get('name', 'experiment')
        
        output_dir = os.path.join(base_dir, exp_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # サブディレクトリ作成
        for subdir in ['models', 'plots', 'reports', 'logs']:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
            
        return output_dir
        
    def run_complete_experiment(self):
        """完全な実験を実行"""
        try:
            if self.tracker:
                self.tracker.update_status('running', 
                    start_time=datetime.datetime.now().isoformat()
                )
                
            print(f"実験開始: {self.config['experiment']['name']}")
            
            # 1. データ準備
            train_loader, val_loader, test_loader, test_data = self._prepare_data()
            
            # 2. モデル設定
            model_wrapper = self._setup_model()
            
            # 3. 学習実行
            trained_model = self._run_training(model_wrapper, train_loader, val_loader)
            
            # 4. モデル保存
            model_path = self._save_model(trained_model)
            
            # 5. 評価実行
            evaluation_results = self._run_evaluation(trained_model, test_data)
            
            # 6. 結果保存
            final_results = self._finalize_results(model_path, evaluation_results)
            
            if self.tracker:
                self.tracker.log_final_results(final_results)
                self.tracker.update_status('completed',
                    end_time=datetime.datetime.now().isoformat()
                )
                
            print("実験完了")
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"実験エラー: {error_msg}")
            
            if self.tracker:
                self.tracker.update_status('failed',
                    end_time=datetime.datetime.now().isoformat(),
                    notes=error_msg
                )
                
            raise
            
    def _prepare_data(self):
        """データ準備"""
        print("データ準備中...")

        # データ設定の検証
        if not DataLoaderFactory.validate_data_config(self.config):
            raise ValueError("データ設定が無効です")

        # データセット情報を表示
        dataset_info = DataLoaderFactory.get_dataset_info(self.config)
        print(f"データセット情報: {dataset_info}")

        try:
            train_loader, val_loader, test_loader, test_df, train_dataset, val_dataset, test_dataset = DataLoaderFactory.create_dataloaders(self.config)

            # テストデータ準備(評価用)
            test_data = {
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
                'test_dataset': test_dataset,
                'test_loader': test_loader,
                'test_df': test_df,
                'output_dir': self.output_dir,
                'experiment_id': self.experiment_id or 0,
                'dataset_info': dataset_info
            }

            print(f"データローダー作成完了:")
            print(f"  - 学習データ: {len(train_loader)} batches")
            print(f"  - 検証データ: {len(val_loader) if val_loader else 0} batches")
            print(f"  - テストデータ: {len(test_loader)} batches")

            return train_loader, val_loader, test_loader, test_data

        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            # フォールバックとして例外を再発生（エラーハンドリングは呼び出し元で）
            raise
        
    def _setup_model(self):
        """モデル設定"""
        print("モデル初期化中...")
        
        model_config = self.config['model']
        
        # 動的モデルロード
        spec = importlib.util.spec_from_file_location(
            "model_module", model_config['file_path']
        )
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        
        model_class = getattr(model_module, model_config['class_name'])
        
        # モデルパラメータ
        model_params = {k: v for k, v in model_config.items() 
                       if k not in ['class_name', 'file_path']}
        
        model = model_class(**model_params)
        model.to(self.device)
        
        return ModelWrapper(model, self.config)
        
    def _run_training(self, model_wrapper, train_loader, val_loader):
        """学習実行"""
        print("学習開始...")
        
        # experiment_managerのExperimentRunnerを使用
        from experiment_manager import ExperimentRunner
        runner = ExperimentRunner(self.tracker, self.config)
        
        # 学習実行
        runner.run_training(model_wrapper, train_loader, val_loader)
        
        return model_wrapper.model
        
    def _save_model(self, model):
        """モデル保存"""
        if not self.config.get('output', {}).get('save_model', True):
            return None
            
        model_path = os.path.join(self.output_dir, 'models', 
                                f'model_exp{self.experiment_id or 0}.pth')
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'experiment_id': self.experiment_id
        }, model_path)
        
        print(f"モデル保存: {model_path}")
        return model_path
        
    def _run_evaluation(self, model, test_data):
        """評価実行"""
        print("評価開始...")
        
        # 評価パイプライン設定
        evaluation_pipeline = EvaluationPipeline(self.config)
        
        # 評価実行
        results = evaluation_pipeline.run_evaluation(model, test_data, self.device)
        
        return results
        
    def _finalize_results(self, model_path, evaluation_results):
        """結果の最終化"""
        final_results = {
            'status': 'completed',
            'end_time': datetime.datetime.now().isoformat(),
        }
        
        if model_path:
            final_results['model_path'] = model_path
            
        if evaluation_results:
            # 評価結果をDB用形式に変換
            eval_dict = evaluation_results.get_db_ready_dict() if hasattr(evaluation_results, 'get_db_ready_dict') else {}
            final_results.update(eval_dict)
            
        return final_results


class ExperimentDatabase:
    """実験データベース管理クラス"""

    def __init__(self, db_path: str, schema_path: str = None):
        self.db_path = db_path
        self.schema_path = schema_path
        self._setup_database()

    def _setup_database(self):
        """データベースの初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA foreign_keys = ON")

                # 基本テーブルの確認・作成
                cursor.execute("""
                               SELECT name
                               FROM sqlite_master
                               WHERE type = 'table'
                                 AND name = 'transformer_base_e2e_vae_experiment'
                               """)

                if not cursor.fetchone():
                    self._create_experiment_table(cursor)
                else:
                    self._ensure_required_columns(cursor)

                conn.commit()

        except Exception as e:
            print(f"データベース初期化エラー: {e}")
            raise

    def _create_experiment_table(self, cursor):
        """実験テーブルを作成"""
        cursor.execute("""
                       CREATE TABLE transformer_base_e2e_vae_experiment
                       (
                           id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                           experiment_name        TEXT NOT NULL,
                           status                 TEXT      DEFAULT 'pending' CHECK (status IN
                                                                                     ('pending', 'running', 'completed',
                                                                                      'failed', 'skipped',
                                                                                      'cancelled')),
                           config_path            TEXT,
                           config_backup_path     TEXT,
                           description            TEXT,
                           tags                   TEXT,
                           priority               INTEGER   DEFAULT 0,
                           dependencies           TEXT,

                           -- タイムスタンプ
                           created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                           start_time             TIMESTAMP,
                           end_time               TIMESTAMP,
                           updated_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                           -- 動的フィールド (JSONとして保存)
                           config_parameters      TEXT, -- JSON
                           training_results       TEXT, -- JSON
                           evaluation_results     TEXT, -- JSON

                           -- ファイルパス
                           model_path             TEXT,
                           evaluation_report_path TEXT,

                           -- システム情報
                           python_version         TEXT,
                           system_info            TEXT,

                           -- 備考
                           notes                  TEXT
                       )
                       """)

        # インデックス作成
        cursor.execute("CREATE INDEX idx_status ON transformer_base_e2e_vae_experiment(status)")
        cursor.execute("CREATE INDEX idx_created_at ON transformer_base_e2e_vae_experiment(created_at)")
        cursor.execute("CREATE INDEX idx_priority ON transformer_base_e2e_vae_experiment(priority DESC)")

    def _ensure_required_columns(self, cursor):
        """必要なカラムが存在することを確認"""
        cursor.execute("PRAGMA table_info(transformer_base_e2e_vae_experiment)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        required_columns = {
            'config_parameters': 'TEXT',
            'training_results': 'TEXT',
            'evaluation_results': 'TEXT',
            'priority': 'INTEGER DEFAULT 0',
            'dependencies': 'TEXT'
        }

        for column, column_def in required_columns.items():
            if column not in existing_columns:
                try:
                    cursor.execute(
                        f"ALTER TABLE transformer_base_e2e_vae_experiment ADD COLUMN {column} {column_def}")
                    print(f"カラム追加: {column}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e):
                        print(f"カラム追加失敗 {column}: {e}")

    def register_experiment(self, config: ExperimentConfig, config_data: Dict[str, Any]) -> int:
        """実験を登録"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # システム情報取得
                system_info = self._get_system_info()

                # 設定バックアップ
                backup_path = self._backup_config(config.config_path)

                insert_data = {
                    'experiment_name': config.config_name,
                    'status': 'pending',
                    'config_path': config.config_path,
                    'config_backup_path': backup_path,
                    'description': config_data.get('experiment', {}).get('description', ''),
                    'tags': json.dumps(config.tags),
                    'priority': config.priority,
                    'dependencies': json.dumps(config.dependencies),
                    'config_parameters': json.dumps(self._extract_config_params(config_data)),
                    'python_version': system_info['python_version'],
                    'system_info': json.dumps(system_info),
                    'created_at': datetime.datetime.now().isoformat()
                }

                columns = ', '.join(insert_data.keys())
                placeholders = ', '.join(['?'] * len(insert_data))

                cursor.execute(
                    f"INSERT INTO transformer_base_e2e_vae_experiment ({columns}) VALUES ({placeholders})",
                    tuple(insert_data.values())
                )

                experiment_id = cursor.lastrowid
                conn.commit()

                return experiment_id

        except Exception as e:
            print(f"実験登録エラー: {e}")
            raise

    def update_experiment(self, experiment_id: int, **kwargs):
        """実験情報を更新"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # updated_atを自動追加
                kwargs['updated_at'] = datetime.datetime.now().isoformat()

                set_clause = ', '.join([f"{key} = ?" for key in kwargs.keys()])
                values = tuple(kwargs.values()) + (experiment_id,)

                cursor.execute(
                    f"UPDATE transformer_base_e2e_vae_experiment SET {set_clause} WHERE id = ?",
                    values
                )
                conn.commit()

        except Exception as e:
            print(f"実験更新エラー (ID: {experiment_id}): {e}")

    def get_experiment_status(self, experiment_id: int) -> Optional[str]:
        """実験のステータスを取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT status FROM transformer_base_e2e_vae_experiment WHERE id = ?",
                               (experiment_id,))
                result = cursor.fetchone()
                return result[0] if result else None
        except:
            return None

    def _extract_config_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """設定から主要パラメータを抽出"""
        params = {}

        # 各セクションから主要パラメータを抽出
        sections_to_extract = ['model', 'training', 'evaluation', 'data']
        for section in sections_to_extract:
            if section in config:
                section_data = config[section]
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        params[f"{section}_{key}"] = value

        return params

    def _get_system_info(self) -> Dict[str, str]:
        """システム情報を取得"""
        import platform
        try:
            import torch
            torch_version = torch.__version__
            cuda_version = torch.version.cuda if torch.cuda.is_available() else 'Not available'
            gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'
        except ImportError:
            torch_version = 'Not installed'
            cuda_version = 'Not available'
            gpu_info = 'CPU only'

        return {
            'python_version': sys.version,
            'pytorch_version': torch_version,
            'cuda_version': cuda_version,
            'gpu_info': gpu_info,
            'system_info': f"{platform.system()} {platform.release()}"
        }

    def _backup_config(self, config_path: str) -> str:
        """設定ファイルをバックアップ"""
        import shutil

        backup_dir = Path(config_path).parent / 'backups'
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = Path(config_path).name
        backup_filename = f"{timestamp}_{config_name}"
        backup_path = backup_dir / backup_filename

        shutil.copy2(config_path, backup_path)
        return str(backup_path)


class ExperimentScheduler:
    """実験スケジューラー"""

    def __init__(self, max_parallel: int = 1, respect_dependencies: bool = True):
        self.max_parallel = max_parallel
        self.respect_dependencies = respect_dependencies
        self.completed_experiments = set()
        self.failed_experiments = set()

    def schedule_experiments(self, experiments: List[ExperimentConfig]) -> List[ExperimentConfig]:
        """実験をスケジューリング"""
        if not self.respect_dependencies:
            return sorted(experiments, key=lambda x: x.priority, reverse=True)

        # 依存関係を考慮したトポロジカルソート
        return self._topological_sort(experiments)

    def _topological_sort(self, experiments: List[ExperimentConfig]) -> List[ExperimentConfig]:
        """依存関係を考慮したソート"""
        experiment_map = {exp.config_name: exp for exp in experiments}
        result = []
        visited = set()
        temp_visited = set()

        def visit(exp_name: str):
            if exp_name in temp_visited:
                raise ValueError(f"循環依存が検出されました: {exp_name}")
            if exp_name in visited:
                return

            temp_visited.add(exp_name)

            if exp_name in experiment_map:
                exp = experiment_map[exp_name]
                for dep in exp.dependencies:
                    visit(dep)

                temp_visited.remove(exp_name)
                visited.add(exp_name)
                result.append(exp)

        for exp in experiments:
            if exp.config_name not in visited:
                visit(exp.config_name)

        # 優先度でソート
        return sorted(result, key=lambda x: x.priority, reverse=True)

    def can_run_experiment(self, experiment: ExperimentConfig) -> bool:
        """実験が実行可能かチェック"""
        if not self.respect_dependencies:
            return True

        # 依存関係の確認
        for dep in experiment.dependencies:
            if dep not in self.completed_experiments:
                if dep in self.failed_experiments:
                    print(f"依存実験 {dep} が失敗しているため {experiment.config_name} をスキップ")
                    return False
                return False

        return True


class BatchExperimentRunner:
    """CLAUDE_ADDED: バッチ実験実行クラス（複数設定ファイルの並列実行用）"""

    def __init__(self, db: ExperimentDatabase, train_script_path: str = None):
        self.db = db
        self.train_script_path = train_script_path or self._find_train_script()

    def _find_train_script(self) -> str:
        """CLAUDE_ADDED: 訓練スクリプトのパスを探索（現在は使用せず）"""
        # サブプロセス実行をやめたため、この機能は不要
        return __file__

    def run_experiment(self, experiment_id: int, config_path: str) -> bool:
        """CLAUDE_ADDED: 単一実験を実行（統合された学習実行）"""
        print(f"実験 {experiment_id} 開始: {Path(config_path).name}")

        try:
            # ステータス更新
            self.db.update_experiment(experiment_id,
                                      status='running',
                                      start_time=datetime.datetime.now().isoformat())

            # IntegratedExperimentRunnerを直接使用（サブプロセスなし）
            runner = IntegratedExperimentRunner(config_path, self.db.db_path, experiment_id)
            success = runner.run_complete_experiment()

            if success:
                # 成功
                self.db.update_experiment(experiment_id,
                                          status='completed',
                                          end_time=datetime.datetime.now().isoformat())
                print(f"実験 {experiment_id} 完了")
                return True
            else:
                # 失敗
                self.db.update_experiment(experiment_id,
                                          status='failed',
                                          end_time=datetime.datetime.now().isoformat(),
                                          notes="実験実行中にエラーが発生")
                print(f"実験 {experiment_id} 失敗")
                return False

        except Exception as e:
            # エラー処理
            error_msg = str(e)
            self.db.update_experiment(experiment_id,
                                      status='failed',
                                      end_time=datetime.datetime.now().isoformat(),
                                      notes=error_msg)

            print(f"実験 {experiment_id} でエラー: {error_msg}")
            import traceback
            traceback.print_exc()
            return False

    def run_parallel_experiments(self, experiment_configs: List[tuple], max_workers: int = 2):
        """並列実験実行"""
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for experiment_id, config_path in experiment_configs:
                future = executor.submit(self.run_experiment, experiment_id, config_path)
                futures[future] = (experiment_id, config_path)

            # 結果を収集
            results = {}
            for future in concurrent.futures.as_completed(futures):
                experiment_id, config_path = futures[future]
                try:
                    success = future.result()
                    results[experiment_id] = success
                except Exception as e:
                    print(f"並列実験エラー (ID: {experiment_id}): {e}")
                    results[experiment_id] = False

            return results


class ConfigLoader:
    """設定ファイル管理クラス"""

    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise FileNotFoundError(f"設定ディレクトリが見つかりません: {config_dir}")

    def discover_configs(self, pattern: str = "*.yaml") -> List[ExperimentConfig]:
        """設定ファイルを発見してExperimentConfigリストを生成"""
        config_files = list(self.config_dir.glob(pattern)) + list(self.config_dir.glob("*.yml"))

        experiments = []
        for config_path in sorted(config_files):
            try:
                experiment_config = self._load_experiment_config(config_path)
                experiments.append(experiment_config)
            except Exception as e:
                print(f"設定読み込みエラー {config_path}: {e}")
                continue

        return experiments

    def _load_experiment_config(self, config_path: Path) -> ExperimentConfig:
        """単一設定ファイルからExperimentConfigを作成"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        experiment_section = config_data.get('experiment', {})

        return ExperimentConfig(
            config_path=str(config_path),
            config_name=config_path.stem,
            priority=experiment_section.get('priority', 0),
            dependencies=experiment_section.get('dependencies', []),
            tags=experiment_section.get('tags', []),
            estimated_duration=experiment_section.get('estimated_duration')
        )

    def validate_config(self, config_path: str) -> bool:
        """設定ファイルの妥当性検証"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 必須セクションの確認
            required_sections = ['model', 'training', 'data']
            for section in required_sections:
                if section not in config:
                    print(f"必須セクション '{section}' が見つかりません: {config_path}")
                    return False

            # データパスの確認
            data_path = config.get('data', {}).get('data_path')
            if data_path and not os.path.exists(data_path):
                print(f"警告: データファイルが見つかりません: {data_path}")

            return True

        except Exception as e:
            print(f"設定検証エラー {config_path}: {e}")
            return False


def show_experiment_status(db_path: str, limit: int = 20):
    """実験状況を表示"""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # 最新実験の取得
            cursor.execute("""
                           SELECT id, experiment_name, status, start_time, end_time, priority, tags
                           FROM transformer_base_e2e_vae_experiment
                           ORDER BY created_at DESC LIMIT ?
                           """, (limit,))

            experiments = cursor.fetchall()

            if not experiments:
                print("実験が登録されていません。")
                return

            print(f"\n実験状況 (最新{limit}件)")
            print("=" * 100)
            print(f"{'ID':>3} | {'実験名':20} | {'ステータス':10} | {'優先度':6} | {'開始時刻':16} | {'タグ':15}")
            print("-" * 100)

            for exp in experiments:
                exp_id, name, status, start_time, end_time, priority, tags = exp

                start_str = start_time.split('T')[0] if start_time else "---"
                tags_str = json.loads(tags) if tags else []
                tags_display = ','.join(tags_str[:2]) if tags_str else ""

                print(f"{exp_id:>3} | {name[:20]:20} | {status:10} | {priority:6} | {start_str:16} | {tags_display:15}")

            # 統計情報
            cursor.execute("""
                           SELECT status, COUNT(*)
                           FROM transformer_base_e2e_vae_experiment
                           GROUP BY status
                           """)
            stats = cursor.fetchall()

            print("\n実験統計:")
            for status, count in stats:
                print(f"  {status}: {count}件")

    except Exception as e:
        print(f"状況表示エラー: {e}")


def run_single_experiment(config_path: str, db_path: str = None, experiment_id: int = None):
    """CLAUDE_ADDED: 単一実験を実行"""
    try:
        runner = IntegratedExperimentRunner(config_path, db_path, experiment_id)
        return runner.run_complete_experiment()
    except Exception as e:
        print(f"実験実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Transformer Base End-to-End VAE 統合実験システム"
    )

    # 基本設定
    parser.add_argument('--config', type=str, 
                        help='実験設定ファイルパス（単一実験用）')
    parser.add_argument('--config_dir', type=str, default='configs',
                        help='設定ファイルディレクトリ（バッチ実行用）')
    parser.add_argument('--db_path', type=str, default='PredictiveLatentSpaceNavigationModel/TransformerBaseEndToEndVAE/experiments.db',
                        help='データベースファイルパス')
    parser.add_argument('--experiment_id', type=int,
                        help='実験ID（DBトラッキング用）')

    # 実行オプション
    parser.add_argument('--dry_run', action='store_true',
                        help='実際の実行なしで登録のみ')
    parser.add_argument('--parallel', type=int, default=1,
                        help='並列実行数')
    parser.add_argument('--respect_dependencies', action='store_true', default=True,
                        help='依存関係を考慮')

    # 管理オプション
    parser.add_argument('--show_status', action='store_true',
                        help='実験状況を表示')
    parser.add_argument('--filter_tags', nargs='*',
                        help='指定タグの実験のみ実行')
    parser.add_argument('--skip_validation', action='store_true',
                        help='設定検証をスキップ')

    args = parser.parse_args()

    # ステータス表示のみの場合
    if args.show_status:
        show_experiment_status(args.db_path)
        return

    # 単一実験実行モード
    if args.config:
        print(f"単一実験実行: {args.config}")
        success = run_single_experiment(args.config, args.db_path, args.experiment_id)
        if success:
            print("実験成功")
        else:
            print("実験失敗")
            sys.exit(1)
        return

    # バッチ実行モード
    print("Transformer Base End-to-End VAE 統合実験システム開始")
    print(f"設定ディレクトリ: {args.config_dir}")
    print(f"データベース: {args.db_path}")
    print(f"並列実行数: {args.parallel}")

    try:
        # データベース初期化
        db = ExperimentDatabase(args.db_path)

        # 設定ファイル読み込み
        config_loader = ConfigLoader(args.config_dir)
        experiment_configs = config_loader.discover_configs()

        if not experiment_configs:
            print("実験設定が見つかりませんでした。")
            return

        # フィルタリング
        if args.filter_tags:
            experiment_configs = [
                config for config in experiment_configs
                if any(tag in config.tags for tag in args.filter_tags)
            ]

        print(f"\n{len(experiment_configs)}件の実験を発見:")
        for config in experiment_configs:
            print(f"  - {config.config_name} (優先度: {config.priority}, タグ: {config.tags})")

        # スケジューリング
        scheduler = ExperimentScheduler(
            max_parallel=args.parallel,
            respect_dependencies=args.respect_dependencies
        )

        scheduled_configs = scheduler.schedule_experiments(experiment_configs)

        # 実験登録と実行
        runner = BatchExperimentRunner(db)
        successful = 0
        failed = 0

        experiment_queue = []

        for config in scheduled_configs:
            # 設定検証
            if not args.skip_validation and not config_loader.validate_config(config.config_path):
                print(f"設定検証失敗、スキップ: {config.config_name}")
                failed += 1
                continue

            # 設定データ読み込み
            with open(config.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            # 実験登録
            experiment_id = db.register_experiment(config, config_data)
            print(f"実験登録: {experiment_id} - {config.config_name}")

            if not args.dry_run:
                experiment_queue.append((experiment_id, config.config_path))

        # 実験実行
        if not args.dry_run and experiment_queue:
            if args.parallel > 1:
                print(f"\n並列実験実行開始 (最大{args.parallel}並列)")
                results = runner.run_parallel_experiments(experiment_queue, args.parallel)

                for exp_id, success in results.items():
                    if success:
                        successful += 1
                    else:
                        failed += 1
            else:
                print(f"\n逐次実験実行開始")
                for experiment_id, config_path in experiment_queue:
                    if scheduler.can_run_experiment(
                            next(c for c in scheduled_configs if c.config_path == config_path)
                    ):
                        if runner.run_experiment(experiment_id, config_path):
                            successful += 1
                            scheduler.completed_experiments.add(
                                Path(config_path).stem
                            )
                        else:
                            failed += 1
                            scheduler.failed_experiments.add(
                                Path(config_path).stem
                            )
                    else:
                        print(f"依存関係により実験をスキップ: {experiment_id}")
                        db.update_experiment(experiment_id, status='skipped')

        # 結果サマリー
        print(f"\n実験実行結果:")
        print(f"成功: {successful}件")
        print(f"失敗: {failed}件")

        if successful > 0:
            print(f"\n結果確認: python {__file__} --show_status")

    except Exception as e:
        print(f"実行エラー: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()