import os
import subprocess
import yaml
import sqlite3
import argparse
import datetime
from typing import Dict, Any

# ----グローバルな設定----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DB_PATH = os.path.join(SCRIPT_DIR, "experiments.db")
DEFAULT_CONFIG_DIR = os.path.join(SCRIPT_DIR, "configs")
DEFAULT_SCHEMA_PATH = os.path.join(SCRIPT_DIR, "schema.sql")

def setup_database(db_path : str, schema_path: str = DEFAULT_SCHEMA_PATH):
    """
    データベースとテーブルが存在しない場合に初期化をする
    :param db_path:データベースのパス
    :param schema_path: データベースのスキーマのパス
    """
    print(f"データベースを初期化/確認中: {db_path}")
    if not os.path.exists(schema_path):
        print(f"警告: スキーマファイル '{schema_path}' が見つかりません．テーブルは作成されません．")
        return

    # .sqlファイルを読み込む
    with open(schema_path, 'r', encoding='utf-8') as f:
        # ファイル全体を一つの文字列として読み込む
        schema_script = f.read()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.executescript(schema_script)
        conn.commit()

    print("データベースの準備完了．")


def get_git_commit_hash() -> str:
    """
    現在のGitコミットハッシュを取得する．
    コンテナ内でgitコマンドが使えるように，.gitディレクトリもマウントされている必要がある．
    """
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.PIPE
        ).decode('utf-8').strip()
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("警告: Gitリポジトリが見つからないか、gitコマンドがありません。コミットハッシュは記録されません。")
        return ""


def register_experiment(db_path: str, config: Dict[str, Any], config_filename: str) -> int:
    """
    実験を設定ファイルに基づいてDBに登録し，実験IDを返す．
    :param db_path: データベースのパス
    :param config:  コンフィグファイルのパス
    :param config_filename: コンフィグファイルの名前
    :return: 実験ID
    """
    params = {
        'status': 'queued',
        'config_file': config_filename,
        'git_commit_hash': get_git_commit_hash(),
        'lr': config.get('training', {}).get('lr'),
        'batch_size': config.get('training', {}).get('batch_size'),
        'num_epochs': config.get('training', {}).get('num_epochs'),
        'hidden_dim': config.get('model', {}).get('hidden_dim'),
        'style_latent_dim': config.get('model', {}).get('style_latent_dim'),
        'skill_latent_dim': config.get('model', {}).get('skill_latent_dim'),
        'n_layers': config.get('model', {}).get('n_layers'),
    }

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        columns = ', '.join(params.keys())
        placeholders = ', '.join(['?'] * len(params))
        query = f"INSERT INTO experiments ({columns}) VALUES ({placeholders})"

        cursor.execute(query, tuple(params.values()))
        experiment_id = cursor.lastrowid
        conn.commit()

    print(f"実験 {experiment_id} をDBに登録しました．")
    return experiment_id


def run_experiment_subprocess(experiment_id: int, config_path: str, db_path: str):
    """
    指定された実験IDと設定ファイルで，コンテナ内に学習プロセスを起動する．
    """
    print(f"--- 実験 {experiment_id} をサブプロセスとして開始 ---")

    train_script_path = os.path.join(SCRIPT_DIR, 'train.py')

    # train.pyにわたすコマンドライン引数を構築
    command =[
        "python", train_script_path,
        "--config", config_path,
        "--experiment_id", str(experiment_id),
        "--db_path", db_path
    ]

    try:
        print(f"実行コマンド: {' '.join(command)}")
        # サブプロセスとして学習を実行
        subprocess.run(command, check=True)
        print(f"--- 実験 {experiment_id} 正常終了 ---")
    except subprocess.CalledProcessError as e:
        print(f"!!! エラー: 実験 {experiment_id} が失敗しました !!!")
        print(e)


def main(args):
    setup_database(args.db_path, args.schema_path)

    if not os.path.isdir(args.config_dir):
        print(f"エラー: 設定ファイルディレクトリ '{args.config_dir}' が見つかりません．")
        return

    config_files = sorted([f for f in os.listdir(args.config_dir) if f.endswith(".yaml") or f.endswith(".yml")])

    if not config_files:
        print(f"警告: '{args.config_dir}'内に設定ファイル(.yaml) が見つかりませんでした．")
        return

    print(f"{len(config_files)}件の実験を開始します: {config_files}")

    for config_filename in config_files:
        config_path = os.path.join(args.config_dir, config_filename)

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"エラー: 設定ファイル '{config_path}'の読み込みに失敗しました． スキップします．エラー: {e}")
            continue

        experiment_id = register_experiment(args.db_path, config, config_filename)

        # サブプロセスとして実験を実行
        run_experiment_subprocess(experiment_id, config_path, args.db_path)

    print("\n全ての実験が完了しました． ")


if __name__ == "__main__":
    parser  = argparse.ArgumentParser(description="単一コンテナ内で機械学習の実験を管理・実行するオーケストレーター")

    parser.add_argument(
        '--config-dir',
        type=str,
        default=DEFAULT_CONFIG_DIR,
        help=f'実験設定のYAMLファイルが含まれるディレクトリ (デフォルト: {DEFAULT_CONFIG_DIR})'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default=DEFAULT_DB_PATH,
        help=f'実験結果を保存するSQLiteデータベースのパス (デフォルト: {DEFAULT_DB_PATH})'
    )

    parser.add_argument(
        '--schema-path',
        type=str,
        default=DEFAULT_SCHEMA_PATH,
        help=f'実験結果を保存するSQLiteSchemaのパス (デフォルト: {DEFAULT_SCHEMA_PATH})'
    )

    args =parser.parse_args()
    main(args)
