# CLAUDE_ADDED
"""
Loss Weight Analysis Runner
任意の設定ファイルまたは実験IDから損失重み推移を可視化する統合スクリプト
"""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from loss_weight_visualizer import LossWeightVisualizer
import yaml
import json
import sqlite3

def load_config_from_experiment_id(experiment_id: int, db_path: str = "PredictiveLatentSpaceNavigationModel/TransformerBaseEndToEndVAE/experiments.db") -> dict:
    """実験IDからデータベースの設定を取得"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT config_parameters
            FROM transformer_base_e2e_vae_experiment
            WHERE id = ?
        """, (experiment_id,))

        result = cursor.fetchone()
        if not result:
            raise ValueError(f"実験ID {experiment_id} が見つかりません")

        config_parameters = result[0]
        if not config_parameters:
            raise ValueError(f"実験ID {experiment_id} の設定パラメータが見つかりません")

        config = json.loads(config_parameters)
        conn.close()

        return config

    except sqlite3.Error as e:
        raise RuntimeError(f"データベースエラー: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON解析エラー: {e}")

def convert_flat_config_to_nested(flat_config: dict) -> dict:
    """フラット化された設定を階層化された設定に変換"""
    nested_config = {}

    # training_num_epochsを抽出
    nested_config['training_num_epochs'] = flat_config.get('training_num_epochs', 200)

    # model_loss_schedule_configが直接存在する場合
    if 'model_loss_schedule_config' in flat_config:
        nested_config['model_loss_schedule_config'] = flat_config['model_loss_schedule_config']
        return nested_config

    # 損失スケジュール設定を抽出・変換
    loss_schedule_config = {}
    loss_types = ['beta_skill', 'beta_style', 'contrastive_loss',
                 'manifold_loss', 'orthogonal_loss', 'skill_regression_loss']

    print("🔍 フラット化された設定からスケジュール情報を抽出中...")
    print(f"   利用可能なキー: {list(flat_config.keys())[:10]}...")  # 最初の10個のキーを表示

    for loss_type in loss_types:
        loss_config = {}
        params = ['start_epoch', 'end_epoch', 'start_val', 'end_val', 'schedule']

        for param in params:
            # フラット化されたキーを検索
            key_patterns = [
                f'model_loss_schedule_config_{loss_type}_{param}',
                f'loss_schedule_{loss_type}_{param}',
                f'{loss_type}_{param}'
            ]

            for key_pattern in key_patterns:
                if key_pattern in flat_config:
                    loss_config[param] = flat_config[key_pattern]
                    print(f"   見つかった設定: {key_pattern} = {flat_config[key_pattern]}")
                    break

        if loss_config:
            loss_schedule_config[loss_type] = loss_config
            print(f"   ✅ {loss_type}: {len(loss_config)} parameters found")

    if loss_schedule_config:
        nested_config['model_loss_schedule_config'] = loss_schedule_config
        print(f"   ✅ 合計 {len(loss_schedule_config)} 損失タイプの設定を変換")
    else:
        print("   ⚠️ 損失スケジュール設定が見つかりませんでした")
        # デフォルト設定を使用
        nested_config['model_loss_schedule_config'] = {
            'beta_skill': {
                'start_epoch': 40, 'end_epoch': 70, 'start_val': 0.0, 'end_val': 0.0001, 'schedule': 'linear'
            },
            'beta_style': {
                'start_epoch': 40, 'end_epoch': 70, 'start_val': 0.0, 'end_val': 0.0001, 'schedule': 'linear'
            },
            'contrastive_loss': {
                'start_epoch': 91, 'end_epoch': 110, 'start_val': 0.0, 'end_val': 1.0, 'schedule': 'linear'
            },
            'manifold_loss': {
                'start_epoch': 91, 'end_epoch': 110, 'start_val': 0.0, 'end_val': 0.5, 'schedule': 'linear'
            },
            'orthogonal_loss': {
                'start_epoch': 71, 'end_epoch': 90, 'start_val': 0.0, 'end_val': 2.0, 'schedule': 'linear'
            },
            'skill_regression_loss': {
                'start_epoch': 91, 'end_epoch': 110, 'start_val': 0.0, 'end_val': 1.0, 'schedule': 'linear'
            }
        }
        print("   💡 実験347のデフォルト設定を使用します")

    return nested_config

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="損失重みの推移を可視化")
    parser.add_argument('--config', '-c', type=str,
                       help='設定ファイルのパス')
    parser.add_argument('--experiment-id', '-id', type=int,
                       help='実験ID（データベースから取得）')
    parser.add_argument('--output-dir', '-o', type=str, default='result_extractor/loss_weight_plots',
                       help='出力ディレクトリ')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='詳細プロットも生成')
    parser.add_argument('--db-path', type=str,
                       default='PredictiveLatentSpaceNavigationModel/TransformerBaseEndToEndVAE/experiments.db',
                       help='データベースファイルのパス')

    args = parser.parse_args()

    # 設定データの取得
    config_data = None

    try:
        if args.experiment_id:
            print(f"🔍 実験ID {args.experiment_id} から設定を取得中...")
            flat_config = load_config_from_experiment_id(args.experiment_id, args.db_path)
            config_data = convert_flat_config_to_nested(flat_config)
            output_prefix = f"experiment_{args.experiment_id}"

        elif args.config:
            print(f"📄 設定ファイル {args.config} から設定を取得中...")
            with open(args.config, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            output_prefix = Path(args.config).stem

        else:
            print("❌ --config または --experiment-id のいずれかを指定してください")
            return

        # 出力ディレクトリの作成
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 可視化クラスの初期化
        visualizer = LossWeightVisualizer(config_data=config_data)

        # スケジュール概要の表示
        visualizer.print_schedule_summary()

        # メインプロットの作成
        main_output = output_dir / f"{output_prefix}_loss_weights"
        visualizer.create_weight_progression_plot(str(main_output))

        # 詳細プロットの作成（オプション）
        if args.detailed:
            detailed_output = output_dir / f"{output_prefix}_detailed_schedule"
            visualizer.create_detailed_schedule_plot(str(detailed_output))

        print(f"\n✅ 損失重み可視化完了!")
        print(f"📁 出力ディレクトリ: {output_dir}")
        print(f"📊 生成ファイル:")
        print(f"   - {main_output}.pdf/.png")
        if args.detailed:
            print(f"   - {detailed_output}.pdf/.png")

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        raise

if __name__ == "__main__":
    main()