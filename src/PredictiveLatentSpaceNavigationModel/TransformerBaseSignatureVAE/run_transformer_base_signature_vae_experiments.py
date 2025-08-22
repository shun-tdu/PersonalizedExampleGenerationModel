import os
import sys
import platform
import subprocess
import yaml
import sqlite3
import argparse
import datetime
from typing import  Dict, Any
import json
import torch

# ---グローバル設定---
SCRIPT_DIR = os.path.dirname((os.path.abspath(__file__)))

DEFAULT_DB_PATH = os.path.join(SCRIPT_DIR, "transformer_base_signature_vae.db")
DEFAULT_CONFIG_DIR = os.path.join(SCRIPT_DIR, "configs")
DEFAULT_PROCESSED_CONFIG_DIR = os.path.join(SCRIPT_DIR, "configs_processed")
DEFAULT_SCHEMA_PATH = os.path.join(SCRIPT_DIR, "transformer_base_signature_vae_schema.sql")


def setup_database_safe(db_path: str, schema_path: str = None):
    """
    安全な階層型VAE用データベース初期化
    """
    print(f"階層型VAE実験データベースを初期化/確認中: {db_path}")

    try:
        # データベースディレクトリの作成
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        # データベース接続テスト
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # 外部キー制約を有効化
            cursor.execute("PRAGMA foreign_keys = ON")

            # 既存テーブルの確認
            cursor.execute("""
                           SELECT name
                           FROM sqlite_master
                           WHERE type = 'table'
                             AND name = 'transformer_base_signature_vae_experiment'
                           """)

            if cursor.fetchone():
                print("既存のtransformer_base_signature_vae_experimentテーブルを発見")
                # テーブル構造の確認
                cursor.execute("PRAGMA table_info(transformer_base_signature_vae_experiment)")
                columns = [row[1] for row in cursor.fetchall()]
                print(f"既存カラム数: {len(columns)}")

                # 必要に応じてカラム追加
                add_missing_columns(cursor, columns)

            else:
                print("新規テーブルを作成中...")
                create_tables_directly(cursor)

            conn.commit()
            print("TransformerBaseSignatureVAE実験データベースの準備完了")

    except Exception as e:
        print(f"データベース初期化エラー: {e}")
        print("基本テーブル作成にフォールバック")
        create_basic_schema_safe(db_path)


def create_tables_directly(cursor):
    """
    テーブルを直接作成（スキーマファイルを使わずに）
    """

    # database_infoテーブル
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS database_info
                   (
                       name
                       TEXT
                       PRIMARY
                       KEY,
                       version
                       TEXT
                       NOT
                       NULL,
                       description
                       TEXT,
                       created_at
                       TIMESTAMP
                       DEFAULT
                       CURRENT_TIMESTAMP
                   )
                   """)

    # 初期データ挿入
    cursor.execute("""
        INSERT OR REPLACE INTO database_info (name, version, description)
        VALUES ('transformer_base_signature_vae_experiments', '1.0', 'Transformer Base Signature VAE experiment tracking database')
    """)

    # transformer_base_signature_vae_experimentテーブル
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS transformer_base_signature_vae_experiment
                   (
                       -- 基本情報
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       experiment_name
                       TEXT
                       NOT
                       NULL,
                       status
                       TEXT
                       DEFAULT
                       'pending'
                       CHECK (
                       status
                       IN
                   (
                       'pending',
                       'running',
                       'completed',
                       'failed',
                       'dry_run'
                   )),
                       config_path TEXT,
                       config_backup_path TEXT,
                       description TEXT,
                       tags TEXT,

                       -- タイムスタンプ
                       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                       start_time TIMESTAMP,
                       end_time TIMESTAMP,
                       updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                       -- データ設定
                       data_path TEXT,

                       -- モデル設定
                       input_dim INTEGER,
                       seq_len INTEGER,
                       hidden_dim INTEGER,
                       style_latent_dim INTEGER,
                       skill_latent_dim INTEGER,
                       beta REAL,
                       n_layers INTEGER,
                       contrastive_weight REAL,
                       use_triplet BOOLEAN,

                       -- 学習設定
                       batch_size INTEGER,
                       num_epochs INTEGER,
                       lr REAL,
                       weight_decay REAL,
                       clip_grad_norm REAL,
                       warmup_epochs INTEGER,
                       scheduler_T_0 INTEGER,
                       scheduler_T_mult INTEGER,
                       scheduler_eta_min REAL,
                       patience INTEGER,

                       -- お手本生成設定
                       skill_enhancement_factor REAL,
                       style_preservation_weight REAL,
                       max_enhancement_steps INTEGER,

                       -- 実験結果
                       final_total_loss REAL,
                       best_val_loss REAL,
                       final_epoch INTEGER,
                       early_stopped BOOLEAN,

                       -- 階層別最終損失
                       final_trajectory_recon_loss REAL,
                       final_encoder_signature_loss REAL,
                       final_decoder_signature_loss REAL,
                       final_contrastive_loss REAL,
                       final_kl_style REAL,
                       final_kl_skill REAL,

                       -- 評価指標
                       reconstruction_mse REAL,
                       style_separation_score REAL,
                       skill_performance_correlation REAL,
                       best_skill_correlation_metric TEXT,

                       -- スキル軸分析結果
                       skill_axis_analysis_completed BOOLEAN DEFAULT FALSE,
                       skill_improvement_directions_count INTEGER DEFAULT 0,
                       axis_based_improvement_enabled BOOLEAN DEFAULT FALSE,

                       -- ファイルパス
                       model_path TEXT,
                       best_model_path TEXT,
                       training_curves_path TEXT,
                       axis_based_exemplars_path TEXT,
                       evaluation_results_path TEXT,

                       -- ログ設定
                       output_dir TEXT,

                       -- システム情報
                       python_version TEXT,
                       pytorch_version TEXT,
                       cuda_version TEXT,
                       gpu_info TEXT,
                       system_info TEXT,

                       -- アブレーション研究
                       is_ablation_study BOOLEAN DEFAULT FALSE,

                       -- 備考
                       notes TEXT
                       )
                   """)

    # インデックス作成
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_status ON transformer_base_signature_vae_experiment(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON transformer_base_signature_vae_experiment(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_tags ON transformer_base_signature_vae_experiment(tags)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_experiments_performance ON transformer_base_signature_vae_experiment(reconstruction_mse)")

    print("全テーブルとインデックスを作成しました")


def add_missing_columns(cursor, existing_columns):
    """
    既存テーブルに不足しているカラムを追加
    """
    required_columns = {
        'skill_axis_analysis_completed': 'BOOLEAN DEFAULT FALSE',
        'skill_improvement_directions_count': 'INTEGER DEFAULT 0',
        'axis_based_improvement_enabled': 'BOOLEAN DEFAULT FALSE',
        'axis_based_exemplars_path': 'TEXT',
        'best_skill_correlation_metric': 'TEXT',
        'final_kl_skill': 'REAL',
        'final_kl_style': 'REAL',
        'final_trajectory_recon_loss': 'REAL',
        'final_encoder_signature_loss': 'REAL',
        'final_decoder_signature_loss': 'REAL',
        'final_contrastive_loss': 'REAL',
    }

    for column_name, column_def in required_columns.items():
        if column_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE transformer_base_signature_vae_experiment ADD COLUMN {column_name} {column_def}")
                print(f"カラム追加: {column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e):
                    print(f"カラム追加失敗 {column_name}: {e}")


def create_basic_schema_safe(db_path: str):
    """基本的な階層型VAEテーブルを安全に作成"""
    basic_sql = """
                CREATE TABLE IF NOT EXISTS transformer_base_signature_vae_experiment \
                ( \
                    id \
                    INTEGER \
                    PRIMARY \
                    KEY \
                    AUTOINCREMENT, \
                    experiment_name \
                    TEXT \
                    NOT \
                    NULL, \
                    status \
                    TEXT \
                    DEFAULT \
                    'pending', \
                    config_path \
                    TEXT, \
                    created_at \
                    TIMESTAMP \
                    DEFAULT \
                    CURRENT_TIMESTAMP, \
                    reconstruction_mse \
                    REAL, \
                    style_separation_score \
                    REAL, \
                    skill_performance_correlation \
                    REAL, \
                    notes \
                    TEXT
                ); \
                """

    try:
        with sqlite3.connect(db_path) as conn:
            conn.executescript(basic_sql)
            print("基本テーブルを作成しました")
    except Exception as e:
        print(f"基本テーブル作成も失敗: {e}")
        raise

def setup_database(db_path: str, schema_path: str = None):
    """
    階層型VAE用データベースとテーブルの初期化（安全版）
    """
    return setup_database_safe(db_path, schema_path)

def extract_config_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    設定ファイルから階層型VAE用のパラメータを抽出
    """
    params = {}

    # 基本情報
    params['config_path'] = config.get('config_path', '')

    # データ設定
    data_config = config.get('data', {})
    params['data_path'] = data_config.get('data_path', '')

    # モデル設定
    model_config = config.get('model', {})
    params['input_dim'] = model_config.get('input_dim')
    params['seq_len'] = model_config.get('seq_len')
    params['hidden_dim'] = model_config.get('hidden_dim')
    params['style_latent_dim'] = model_config.get('style_latent_dim')
    params['skill_latent_dim'] = model_config.get('skill_latent_dim')
    params['beta'] = model_config.get('beta')
    params['n_layers'] = model_config.get('n_layers')
    params['contrastive_weight'] = model_config.get('contrastive_weight')
    params['use_triplet'] = model_config.get('use_triplet')

    # 学習設定
    training_config = config.get('training', {})
    params['batch_size'] = training_config.get('batch_size')
    params['num_epochs'] = training_config.get('num_epochs')
    params['lr'] = training_config.get('lr')
    params['weight_decay'] = training_config.get('weight_decay')
    params['clip_grad_norm'] = training_config.get('clip_grad_norm')
    params['warmup_epochs'] = training_config.get('warmup_epochs')
    params['scheduler_T_0'] = training_config.get('scheduler_T_0')
    params['scheduler_T_mult'] = training_config.get('scheduler_T_mult')
    params['scheduler_eta_min'] = training_config.get('scheduler_eta_min')
    params['patience'] = training_config.get('patience')

    # ログ設定
    logging_config = config.get('logging', {})
    params['output_dir'] = logging_config.get('output_dir', '')

    # 実験設定
    experiment_config = config.get('experiment', {})
    params['description'] = experiment_config.get('description', '')
    params['tags'] = json.dumps(experiment_config.get('tags', []))

    # アブレーション研究
    ablation_studies = experiment_config.get('ablation_studies', [])
    params['is_ablation_study'] = len(ablation_studies) > 0

    return params

def get_system_info() -> Dict[str, str]:
    """システム情報を取得"""
    try:
        return {
            'python_version': sys.version,
            'pytorch_version': torch.__version__ if torch else 'Not installed',
            'cuda_version': torch.version.cuda if torch and torch.cuda.is_available() else 'Not available',
            'gpu_info': torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else 'CPU only',
            'system_info': f"{platform.system()} {platform.release()} {platform.machine()}"
        }
    except Exception as e:
        return {'error': str(e)}


def register_experiment(db_path: str, config: Dict[str, Any], config_filename: str, config_path: str) -> int:
    """
    階層型VAE実験をDBに登録
    """
    # 設定からパラメータを抽出
    params = extract_config_parameters(config)

    # 基本情報を追加
    params['experiment_name'] = os.path.splitext(config_filename)[0]
    params['status'] = 'pending'
    params['config_path'] = config_path

    # システム情報を追加
    system_info = get_system_info()
    params.update(system_info)

    # タイムスタンプ
    params['created_at'] = datetime.datetime.now().isoformat()

    # 設定ファイルのバックアップパス
    backup_dir = os.path.join(os.path.dirname(config_path), 'backups')
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{timestamp}_{config_filename}"
    backup_path = os.path.join(backup_dir, backup_filename)

    # 設定ファイルをバックアップ
    import shutil
    shutil.copy2(config_path, backup_path)
    params['config_backup_path'] = backup_path

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Noneの値を除外
            filtered_params = {k: v for k, v in params.items() if v is not None}

            columns = ', '.join(filtered_params.keys())
            placeholders = ', '.join(['?'] * len(filtered_params))
            query = f"INSERT INTO transformer_base_signature_vae_experiment ({columns}) VALUES ({placeholders})"

            cursor.execute(query, tuple(filtered_params.values()))
            experiment_id = cursor.lastrowid
            conn.commit()

        print(f"階層型VAE実験 {experiment_id} をDBに登録しました")
        print(f"  実験名: {params['experiment_name']}")
        print(f"  説明: {params.get('description', 'なし')}")
        print(f"  タグ: {params.get('tags', '[]')}")

        return experiment_id

    except sqlite3.Error as e:
        print(f"データベース登録エラー: {e}")
        print(f"パラメータ: {filtered_params}")
        raise


def run_experiment_subprocess(experiment_id: int, config_path: str, db_path: str):
    """
    階層型VAE実験をサブプロセスとして実行
    """
    print(f"--- SignatureVAE実験 {experiment_id} をサブプロセスとして開始 ---")

    # 実行前にステータスを更新
    update_experiment_status(db_path, experiment_id, 'running', start_time=datetime.datetime.now().isoformat())

    # train_hierarchical_vae.py を使用
    train_script_path = os.path.join(SCRIPT_DIR, 'train_transformer_base_signature_vae.py')

    if not os.path.exists(train_script_path):
        print(f"警告: {train_script_path} が見つかりません。train.py を使用します")
        train_script_path = os.path.join(SCRIPT_DIR, 'train.py')

    command = [
        'python', train_script_path,
        '--config', config_path,
        '--experiment_id', str(experiment_id),
        '--db_path', db_path
    ]

    try:
        print(f"実行コマンド: {' '.join(command)}")

        # サブプロセスとして実行
        result = subprocess.run(command, check=True)

        print(f"--- SignatureVAE実験 {experiment_id} 正常終了 ---")

        # 成功時のステータス更新
        update_experiment_status(db_path, experiment_id, 'completed', end_time=datetime.datetime.now().isoformat())

        return True

    except subprocess.CalledProcessError as e:
        print(f"!!! エラー: SignatureVAE実験 {experiment_id} が失敗しました !!!")
        print(f"エラーコード: {e.returncode}")
        print(f"標準エラー: {e.stderr}")
        print(f"標準出力: {e.stdout}")

        # 失敗時のステータス更新
        update_experiment_status(
            db_path, experiment_id, 'failed',
            end_time=datetime.datetime.now().isoformat(),
            notes=f"subprocess error: {e.stderr}"
        )

        return False
    except Exception as e:
        print(f"!!! 予期しないエラー: {e} !!!")
        update_experiment_status(
            db_path, experiment_id, 'failed',
            end_time=datetime.datetime.now().isoformat(),
            notes=f"unexpected error: {str(e)}"
        )
        return False

def update_experiment_status(db_path: str, experiment_id: int, status: str, **kwargs):
    """実験のステータスを更新"""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # 基本のステータス更新
            update_fields = ['status = ?']
            update_values = [status]

            # 追加フィールドの更新
            for key, value in kwargs.items():
                if value is not None:
                    update_fields.append(f"{key} = ?")
                    update_values.append(value)

            # updated_at フィールドも更新
            update_fields.append("updated_at = ?")
            update_values.append(datetime.datetime.now().isoformat())

            query = f"UPDATE transformer_base_signature_vae_experiment SET {', '.join(update_fields)} WHERE id = ?"
            update_values.append(experiment_id)

            cursor.execute(query, update_values)
            conn.commit()

    except sqlite3.Error as e:
        print(f"ステータス更新エラー: {e}")

def move_processed_config(config_path: str, processed_dir: str):
    """処理済み設定ファイルを移動"""
    os.makedirs(processed_dir, exist_ok=True)

    filename = os.path.basename(config_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_filename = f"{timestamp}_{filename}"

    processed_path = os.path.join(processed_dir, processed_filename)

    try:
        os.rename(config_path, processed_path)
        print(f"設定ファイルを移動: {config_path} -> {processed_path}")
    except Exception as e:
        print(f"設定ファイル移動エラー: {e}")

def validate_config(config: Dict[str, Any], config_path: str) -> bool:
    """設定ファイルの妥当性検証"""
    required_sections = ['data', 'model', 'training', 'logging']

    for section in required_sections:
        if section not in config:
            print(f"エラー: 設定ファイル {config_path} に '{section}' セクションがありません")
            return False

    # モデル設定の必須パラメータ
    model_required = ['input_dim',
                      'seq_len',
                      'hidden_dim',
                      'style_latent_dim',
                      'skill_latent_dim',
                      'beta',
                      'n_layers',
                      'contrastive_weight',
                      'use_triplet'
                      ]
    model_config = config['model']

    for param in model_required:
        if param not in model_config:
            print(f"エラー: model.{param} が設定されていません")
            return False

    # データパスの存在確認
    data_path = config['data'].get('data_path', '')
    if data_path and not os.path.exists(data_path):
        print(f"警告: データファイル {data_path} が見つかりません")
        # 警告のみで続行

    return True

def show_experiment_status(db_path: str):
    """実験状況を表示"""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # 実験サマリーを取得
            cursor.execute("""
                           SELECT id,
                                  experiment_name,
                                  status,
                                  reconstruction_mse,
                                  style_separation_score,
                                  skill_axis_analysis_completed,
                                  start_time,
                                  end_time,
                                  tags
                           FROM transformer_base_signature_vae_experiment
                           ORDER BY created_at DESC LIMIT 20
                           """)

            experiments = cursor.fetchall()

            print("\n" + "=" * 100)
            print("SignatureVAE実験状況 (最新20件)")
            print("=" * 100)

            if not experiments:
                print("実験が登録されていません。")
                return

            print(
                f"{'ID':>3} | {'実験名':20} | {'ステータス':10} | {'MSE':8} | {'Style ARI':9} | {'軸分析':6} | {'開始時刻':16}")
            print("-" * 100)

            for exp in experiments:
                exp_id, name, status, mse, style_score, axis_completed, start, end, tags = exp

                mse_str = f"{mse:.4f}" if mse else "---"
                style_str = f"{style_score:.3f}" if style_score else "---"
                axis_str = "✓" if axis_completed else "×"
                start_str = start.split('T')[0] if start else "---"

                print(
                    f"{exp_id:>3} | {name[:20]:20} | {status:10} | {mse_str:8} | {style_str:9} | {axis_str:6} | {start_str:16}")

            # 統計情報
            cursor.execute("""
                           SELECT status, COUNT(*)
                           FROM transformer_base_signature_vae_experiment
                           GROUP BY status
                           """)
            stats = cursor.fetchall()

            print("\n実験統計:")
            for status, count in stats:
                print(f"  {status}: {count}件")

    except sqlite3.Error as e:
        print(f"データベース読み取りエラー: {e}")


def main(args):
    print("SignatureVAE実験管理システム開始")
    print(f"データベース: {args.db_path}")
    print(f"設定ディレクトリ: {args.config_dir}")

    # データベース初期化
    setup_database(args.db_path, args.schema_path)

    # ステータス表示のみの場合
    if args.show_status:
        show_experiment_status(args.db_path)
        return

    # 設定ディレクトリの確認
    if not os.path.isdir(args.config_dir):
        print(f"エラー: 設定ファイルディレクトリ '{args.config_dir}' が見つかりません")
        return

    # 設定ファイルを検索
    config_files = sorted([f for f in os.listdir(args.config_dir)
                           if f.endswith((".yaml", ".yml")) and not f.startswith('.')])

    if not config_files:
        print(f"警告: '{args.config_dir}' 内に設定ファイル(.yaml/.yml) が見つかりませんでした")
        return

    print(f"\n{len(config_files)}件のSignatureVAE実験を開始します:")
    for i, filename in enumerate(config_files, 1):
        print(f"  {i}. {filename}")

    successful_experiments = 0
    failed_experiments = 0

    for config_filename in config_files:
        config_path = os.path.join(args.config_dir, config_filename)

        print(f"\n{'=' * 60}")
        print(f"設定ファイル処理中: {config_filename}")
        print('=' * 60)

        try:
            # 設定ファイル読み込み
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 設定の妥当性検証
            if not validate_config(config, config_path):
                print(f"設定ファイル {config_filename} をスキップします")
                failed_experiments += 1
                continue

            # 実験登録
            experiment_id = register_experiment(args.db_path, config, config_filename, config_path)

            # 実験実行
            if args.dry_run:
                print(f"DRY RUN: 実験 {experiment_id} の実行をスキップします")
                update_experiment_status(args.db_path, experiment_id, 'dry_run')
                continue

            is_success = run_experiment_subprocess(experiment_id, config_path, args.db_path)

            if is_success:
                successful_experiments += 1
                if args.move_processed:
                    move_processed_config(config_path, args.processed_config_dir)
            else:
                failed_experiments += 1

        except Exception as e:
            print(f"エラー: 設定ファイル '{config_path}' の処理に失敗しました")
            print(f"詳細: {e}")
            failed_experiments += 1
            continue

    # 結果サマリー
    print(f"\n{'=' * 60}")
    print("実験実行結果サマリー")
    print('=' * 60)
    print(f"成功: {successful_experiments}件")
    print(f"失敗: {failed_experiments}件")
    print(f"合計: {successful_experiments + failed_experiments}件")

    if successful_experiments > 0:
        print(f"\n実験結果を確認するには:")
        print(f"  python {os.path.basename(__file__)} --show-status")

    print("\n全ての実験が完了しました． ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignatureVAE実験管理システム")

    # 基本設定
    parser.add_argument('--db_path', type=str, default=DEFAULT_DB_PATH,
                        help='データベースファイルのパス')
    parser.add_argument('--config_dir', type=str, default=DEFAULT_CONFIG_DIR,
                        help='設定ファイルディレクトリのパス')
    parser.add_argument('--processed_config_dir', type=str, default=DEFAULT_PROCESSED_CONFIG_DIR,
                        help='処理済み設定ファイルの移動先')
    parser.add_argument('--schema_path', type=str, default=DEFAULT_SCHEMA_PATH,
                        help='データベーススキーマファイルのパス')

    # 実行オプション
    parser.add_argument('--dry_run', action='store_true',
                        help='実際の実験を実行せずに登録のみ行う')
    parser.add_argument('--move_processed', action='store_true', default=True,
                        help='処理済み設定ファイルを移動する')
    parser.add_argument('--show_status', action='store_true',
                        help='実験状況を表示して終了')

    args = parser.parse_args()
    main(args)
