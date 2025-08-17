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

DEFAULT_DB_PATH = os.path.join(SCRIPT_DIR, "hierarchical_experiments.db")
DEFAULT_CONFIG_DIR = os.path.join(SCRIPT_DIR, "configs")
DEFAULT_PROCESSED_CONFIG_DIR = os.path.join(SCRIPT_DIR, "configs_processed")
DEFAULT_SCHEMA_PATH = os.path.join(SCRIPT_DIR, "hierarchical_vae_schema.sql")

def setup_database(db_path: str, schema_path: str = DEFAULT_SCHEMA_PATH):
    """
    階層型VAE用データベースとテーブルの初期化
    """
    print(f"階層型VAE実験データベースを初期化/確認中: {db_path}")

    if not os.path.exists(schema_path):
        print(f"警告: スキーマファイル '{schema_path}' が見つかりません。")
        print("階層型VAE用の基本テーブルを作成します...")
        create_basic_schema(db_path)
        return

    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_script = f.read()

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # 外部キー制約を有効化
            cursor.execute("PRAGMA foreign_keys = ON")

            # スキーマを実行
            cursor.executescript(schema_script)
            conn.commit()

            print("階層型VAE実験データベースの準備完了")

            # データベース情報を表示
            cursor.execute("SELECT * FROM database_info")
            db_info = cursor.fetchone()
            if db_info:
                print(f"データベース: {db_info[0]} v{db_info[1]}")
                print(f"説明: {db_info[2]}")

    except Exception as e:
        print(f"データベース初期化エラー: {e}")
        raise


def create_basic_schema(db_path: str):
    """基本的な階層型VAEテーブルを作成（スキーマファイルがない場合）"""
    basic_sql = """
                CREATE TABLE IF NOT EXISTS hierarchical_experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    experiment_name TEXT NOT NULL,
                    status TEXT DEFAULT 'pending', 
                    config_path TEXT, 
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ); 
                """

    with sqlite3.connect(db_path) as conn:
        conn.executescript(basic_sql)
        print("基本テーブルを作成しました")


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
    params['primitive_latent_dim'] = model_config.get('primitive_latent_dim')
    params['skill_latent_dim'] = model_config.get('skill_latent_dim')
    params['style_latent_dim'] = model_config.get('style_latent_dim')

    # 階層別β値
    params['beta_primitive'] = model_config.get('beta_primitive')
    params['beta_skill'] = model_config.get('beta_skill')
    params['beta_style'] = model_config.get('beta_style')
    params['precision_lr'] = model_config.get('precision_lr')

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

    # 階層型VAE特有設定
    hierarchical_config = config.get('hierarchical_settings', {})
    params['primitive_learning_start'] = hierarchical_config.get('primitive_learning_start')
    params['skill_learning_start'] = hierarchical_config.get('skill_learning_start')
    params['style_learning_start'] = hierarchical_config.get('style_learning_start')

    # 予測誤差重み
    prediction_weights = hierarchical_config.get('prediction_error_weights', {})
    params['prediction_error_weight_level1'] = prediction_weights.get('level1')
    params['prediction_error_weight_level2'] = prediction_weights.get('level2')
    params['prediction_error_weight_level3'] = prediction_weights.get('level3')

    # お手本生成設定
    exemplar_config = hierarchical_config.get('exemplar_generation', {})
    params['skill_enhancement_factor'] = exemplar_config.get('skill_enhancement_factor')
    params['style_preservation_weight'] = exemplar_config.get('style_preservation_weight')
    params['max_enhancement_steps'] = exemplar_config.get('max_enhancement_steps')

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
            query = f"INSERT INTO hierarchical_experiments ({columns}) VALUES ({placeholders})"

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
    print(f"--- 階層型VAE実験 {experiment_id} をサブプロセスとして開始 ---")

    # 実行前にステータスを更新
    update_experiment_status(db_path, experiment_id, 'running', start_time=datetime.datetime.now().isoformat())

    # train_hierarchical_vae.py を使用
    train_script_path = os.path.join(SCRIPT_DIR, 'train_hierarchical_vae.py')

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
        result = subprocess.run(command, check=True, capture_output=True, text=True)

        print(f"--- 階層型VAE実験 {experiment_id} 正常終了 ---")
        print("標準出力:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

        # 成功時のステータス更新
        update_experiment_status(db_path, experiment_id, 'completed', end_time=datetime.datetime.now().isoformat())

        return True

    except subprocess.CalledProcessError as e:
        print(f"!!! エラー: 階層型VAE実験 {experiment_id} が失敗しました !!!")
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

            query = f"UPDATE hierarchical_experiments SET {', '.join(update_fields)} WHERE id = ?"
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
    model_required = ['input_dim', 'seq_len', 'hidden_dim', 'primitive_latent_dim', 'skill_latent_dim',
                      'style_latent_dim']
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
                           FROM hierarchical_experiments
                           ORDER BY created_at DESC LIMIT 20
                           """)

            experiments = cursor.fetchall()

            print("\n" + "=" * 100)
            print("階層型VAE実験状況 (最新20件)")
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
                           FROM hierarchical_experiments
                           GROUP BY status
                           """)
            stats = cursor.fetchall()

            print("\n実験統計:")
            for status, count in stats:
                print(f"  {status}: {count}件")

    except sqlite3.Error as e:
        print(f"データベース読み取りエラー: {e}")


def main(args):
    print("階層型VAE実験管理システム開始")
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

    print(f"\n{len(config_files)}件の階層型VAE実験を開始します:")
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
    parser = argparse.ArgumentParser(description="階層型VAE実験管理システム")

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
