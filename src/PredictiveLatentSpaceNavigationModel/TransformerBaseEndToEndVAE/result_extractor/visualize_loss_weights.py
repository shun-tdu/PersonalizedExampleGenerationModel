# CLAUDE_ADDED
"""
Loss Weight Visualization Runner
実験347の損失重み推移を可視化する実行スクリプト
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from loss_weight_visualizer import LossWeightVisualizer

def visualize_experiment_347():
    """実験347の損失重み推移を可視化"""

    # 実験347の設定データ（取得済みの設定）
    config_data = {
        'training_num_epochs': 200,
        'model_loss_schedule_config': {
            'beta_skill': {
                'start_epoch': 40,
                'end_epoch': 70,
                'start_val': 0.0,
                'end_val': 0.0001,
                'schedule': 'linear'
            },
            'beta_style': {
                'start_epoch': 40,
                'end_epoch': 70,
                'start_val': 0.0,
                'end_val': 0.0001,
                'schedule': 'linear'
            },
            'contrastive_loss': {
                'start_epoch': 91,
                'end_epoch': 110,
                'start_val': 0.0,
                'end_val': 1.0,
                'schedule': 'linear'
            },
            'manifold_loss': {
                'start_epoch': 91,
                'end_epoch': 110,
                'start_val': 0.0,
                'end_val': 0.5,
                'schedule': 'linear'
            },
            'orthogonal_loss': {
                'start_epoch': 71,
                'end_epoch': 90,
                'start_val': 0.0,
                'end_val': 2.0,
                'schedule': 'linear'
            },
            'skill_regression_loss': {
                'start_epoch': 91,
                'end_epoch': 110,
                'start_val': 0.0,
                'end_val': 1.0,
                'schedule': 'linear'
            }
        }
    }

    print("🎯 実験347の損失重み推移を可視化中...")

    # 可視化クラスの初期化
    visualizer = LossWeightVisualizer(config_data=config_data)

    # 出力ディレクトリの設定
    output_dir = Path("result_extractor/loss_weight_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # スケジュール概要の表示
    visualizer.print_schedule_summary()

    # メインプロットの作成
    main_output = output_dir / "experiment_347_loss_weights"
    visualizer.create_weight_progression_plot(str(main_output))

    # 詳細プロットの作成
    detailed_output = output_dir / "experiment_347_detailed_schedule"
    visualizer.create_detailed_schedule_plot(str(detailed_output))

    print(f"\n✅ 実験347の損失重み可視化完了!")
    print(f"📁 出力ディレクトリ: {output_dir}")
    print(f"📊 生成ファイル:")
    print(f"   - {main_output}.pdf/.png")
    print(f"   - {detailed_output}.pdf/.png")

def main():
    """メイン実行関数"""
    try:
        visualize_experiment_347()
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        raise

if __name__ == "__main__":
    main()