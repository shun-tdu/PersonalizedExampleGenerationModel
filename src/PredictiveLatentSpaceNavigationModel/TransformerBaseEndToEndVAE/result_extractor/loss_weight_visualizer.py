# CLAUDE_ADDED
"""
Loss Weight Progression Visualizer
損失関数の重みの推移を可視化するモジュール
"""

import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
import argparse

# Academic paper formatting settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Liberation Serif']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

def save_academic_figure(fig, save_path):
    """Save figure in both PDF and PNG formats for academic use"""
    save_path = Path(save_path)
    pdf_path = save_path.with_suffix('.pdf')
    png_path = save_path.with_suffix('.png')

    # Save PDF (vector format, scalable, preferred for academic papers)
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300,
               facecolor='white', edgecolor='none')

    # Save PNG (raster format, 300 DPI for print quality)
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300,
               facecolor='white', edgecolor='none')

    print(f"✅ Saved: {pdf_path.name} and {png_path.name}")

class LossWeightScheduler:
    """損失関数の重みスケジュール計算クラス"""

    def __init__(self, schedule_config: Dict[str, Any]):
        """
        Args:
            schedule_config: 損失スケジュール設定
        """
        self.schedule_config = schedule_config

    def calculate_weight(self, loss_name: str, epoch: int) -> float:
        """指定されたエポックでの損失重みを計算"""
        if loss_name not in self.schedule_config:
            return 0.0

        config = self.schedule_config[loss_name]
        start_epoch = config.get('start_epoch', 0)
        end_epoch = config.get('end_epoch', 100)
        start_val = config.get('start_val', 0.0)
        end_val = config.get('end_val', 1.0)
        schedule = config.get('schedule', 'linear')

        if epoch < start_epoch:
            return start_val
        elif epoch > end_epoch:
            return end_val
        else:
            # エポック間での線形補間
            progress = (epoch - start_epoch) / (end_epoch - start_epoch)

            if schedule == 'linear':
                return start_val + (end_val - start_val) * progress
            elif schedule == 'cosine':
                # コサイン減衰
                return start_val + (end_val - start_val) * (1 - np.cos(progress * np.pi)) / 2
            elif schedule == 'exponential':
                # 指数的変化
                return start_val * ((end_val / start_val) ** progress)
            else:
                # デフォルトは線形
                return start_val + (end_val - start_val) * progress

class LossWeightVisualizer:
    """損失重みの推移を可視化するクラス"""

    def __init__(self, config_path: str = None, config_data: Dict = None):
        """
        Args:
            config_path: 設定ファイルのパス
            config_data: 設定データ（直接指定）
        """
        if config_data:
            self.config = config_data
        elif config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError("config_pathまたはconfig_dataのいずれかを指定してください")

        # 損失スケジュール設定を抽出
        self.loss_schedule_config = self._extract_loss_schedule()
        self.total_epochs = self.config.get('training_num_epochs', 200)

        # 色設定（学術論文用の視認性の良い色）
        self.colors = {
            'beta_skill': '#2E86AB',        # 青
            'beta_style': '#A23B72',        # 紫
            'contrastive_loss': '#F18F01',  # オレンジ
            'manifold_loss': '#C73E1D',     # 赤
            'orthogonal_loss': '#592693',   # 深紫
            'skill_regression_loss': '#1B5E20',  # 緑
            'reconstruction_loss': '#795548'     # 茶色
        }

        # 線スタイル
        self.line_styles = {
            'beta_skill': '-',
            'beta_style': '-',
            'contrastive_loss': '--',
            'manifold_loss': '-.',
            'orthogonal_loss': ':',
            'skill_regression_loss': '-',
            'reconstruction_loss': '--'
        }

    def _extract_loss_schedule(self) -> Dict[str, Any]:
        """設定から損失スケジュール情報を抽出"""
        # フラット化された設定から抽出
        schedule_config = {}

        # model_loss_schedule_configが存在する場合
        if 'model_loss_schedule_config' in self.config:
            return self.config['model_loss_schedule_config']

        # フラット化された設定から復元を試行
        loss_types = ['beta_skill', 'beta_style', 'contrastive_loss',
                     'manifold_loss', 'orthogonal_loss', 'skill_regression_loss']

        for loss_type in loss_types:
            loss_config = {}
            for param in ['start_epoch', 'end_epoch', 'start_val', 'end_val', 'schedule']:
                key = f'model_loss_schedule_config_{loss_type}_{param}'
                if key in self.config:
                    loss_config[param] = self.config[key]

            if loss_config:
                schedule_config[loss_type] = loss_config

        return schedule_config

    def create_weight_progression_plot(self, output_path: str = "loss_weight_progression"):
        """損失重みの推移プロットを作成"""
        print("損失重みの推移プロットを作成中...")

        scheduler = LossWeightScheduler(self.loss_schedule_config)
        epochs = np.arange(1, self.total_epochs + 1)

        # 8.5cm = 3.35 inches (1 inch = 2.54 cm)
        fig, ax = plt.subplots(figsize=(3.35, 2.5))

        # 各損失項目の重み推移を計算・プロット
        for loss_name in self.loss_schedule_config.keys():
            weights = [scheduler.calculate_weight(loss_name, epoch) for epoch in epochs]

            color = self.colors.get(loss_name, '#000000')
            line_style = self.line_styles.get(loss_name, '-')

            # 損失名を整形（アンダースコアをスペースに、最初の文字を大文字に）
            display_name = loss_name.replace('_', ' ').title()
            display_name = display_name.replace('Loss', '').strip()  # 'Loss'を除去

            ax.plot(epochs, weights, color=color, linestyle=line_style,
                   linewidth=1.0, label=display_name, alpha=0.8)

        # 再構成損失（常に1.0）を追加
        reconstruction_weights = np.ones_like(epochs)
        ax.plot(epochs, reconstruction_weights, color=self.colors.get('reconstruction_loss', '#795548'),
               linestyle='--', linewidth=1.0, label='Reconstruction', alpha=0.8)

        # グラフの設定
        ax.set_xlabel('Training Epoch')
        ax.set_ylabel('Loss Weight')
        # ax.set_title('Loss Weight Progression During Training')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(loc=(0.06,0.55), frameon=True,
                 fancybox=True, shadow=True, fontsize=5)

        # 軸の範囲設定
        ax.set_xlim(1, self.total_epochs)
        ax.set_ylim(0, max([max([scheduler.calculate_weight(name, epoch) for epoch in epochs])
                           for name in self.loss_schedule_config.keys()] + [1.0]) * 1.1)

        plt.tight_layout()

        # 学術論文用の保存
        save_academic_figure(fig, output_path)
        plt.close(fig)

        return output_path

    def create_detailed_schedule_plot(self, output_path: str = "detailed_loss_schedule"):
        """詳細な損失スケジュール情報を含むプロット"""
        print("詳細な損失スケジュールプロットを作成中...")

        scheduler = LossWeightScheduler(self.loss_schedule_config)
        epochs = np.arange(1, self.total_epochs + 1)

        # サブプロット作成（2x3のグリッド）
        n_losses = len(self.loss_schedule_config)
        n_cols = 3
        n_rows = (n_losses + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        axes = axes.flatten()

        for i, (loss_name, config) in enumerate(self.loss_schedule_config.items()):
            ax = axes[i]

            weights = [scheduler.calculate_weight(loss_name, epoch) for epoch in epochs]
            color = self.colors.get(loss_name, '#000000')

            ax.plot(epochs, weights, color=color, linewidth=2.5)

            # 重要なエポックに垂直線を追加
            start_epoch = config.get('start_epoch', 0)
            end_epoch = config.get('end_epoch', 100)

            ax.axvline(x=start_epoch, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=end_epoch, color='gray', linestyle='--', alpha=0.5)

            # 設定情報をテキストで表示
            info_text = f"Start: Epoch {start_epoch}, Value {config.get('start_val', 0.0):.4f}\n"
            info_text += f"End: Epoch {end_epoch}, Value {config.get('end_val', 1.0):.4f}\n"
            info_text += f"Schedule: {config.get('schedule', 'linear').title()}"

            ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            display_name = loss_name.replace('_', ' ').title()
            ax.set_title(display_name)
            ax.set_xlabel('Training Epoch')
            ax.set_ylabel('Weight Value')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(1, self.total_epochs)

        # 使用しないサブプロットを非表示
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        # plt.suptitle('Detailed Loss Weight Schedules', fontsize=18, y=0.98)
        plt.tight_layout()

        # 学術論文用の保存
        save_academic_figure(fig, output_path)
        plt.close(fig)

        return output_path

    def print_schedule_summary(self):
        """スケジュール設定の概要を表示"""
        print("\n📋 Loss Weight Schedule Summary:")
        print("=" * 60)

        for loss_name, config in self.loss_schedule_config.items():
            print(f"\n🔹 {loss_name.replace('_', ' ').title()}:")
            print(f"   Start: Epoch {config.get('start_epoch', 0)} → {config.get('start_val', 0.0):.4f}")
            print(f"   End:   Epoch {config.get('end_epoch', 100)} → {config.get('end_val', 1.0):.4f}")
            print(f"   Schedule: {config.get('schedule', 'linear').title()}")

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="損失重みの推移を可視化")
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='設定ファイルのパス')
    parser.add_argument('--output-dir', '-o', type=str, default='.',
                       help='出力ディレクトリ')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='詳細プロットも生成')

    args = parser.parse_args()

    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 可視化クラスの初期化
        visualizer = LossWeightVisualizer(config_path=args.config)

        # スケジュール概要の表示
        visualizer.print_schedule_summary()

        # メインプロットの作成
        main_output = output_dir / "loss_weight_progression"
        visualizer.create_weight_progression_plot(str(main_output))

        # 詳細プロットの作成（オプション）
        if args.detailed:
            detailed_output = output_dir / "detailed_loss_schedule"
            visualizer.create_detailed_schedule_plot(str(detailed_output))

        print(f"\n✅ 可視化完了! 出力ディレクトリ: {output_dir}")

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        raise

if __name__ == "__main__":
    main()