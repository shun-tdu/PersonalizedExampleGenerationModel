import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path



class TrainingVisualizer:
    """モデルの損失を学習履歴から動的に解析してFigureを保存"""
    def __init__(self, output_dir: str, experiment_id: int):
        self.output_dir = output_dir
        self.experiment_id = experiment_id

        output_dir = Path(self.output_dir)

        if not (output_dir.exists() and output_dir.is_dir()):
            raise FileNotFoundError(f"❌ ディレクトリ「{output_dir}」は存在しません")


    def save_loss_curves(self, history: Dict):
        # lossのhistoryにある損失タイプを取得
        loss_type = self._detect_loss_types(history)
        num_plots = len(loss_type)

        # 1. グループ毎にグラフを作成
        fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(12, 6 * num_plots))

        if num_plots == 1:
            axes = [axes]

        # 2. 各描画領域と損失グループをzimでループ
        for ax, (loss_name, keys_to_plot) in zip(axes, loss_type.items()):
            for key in keys_to_plot:
                loss_values = history[key]
                epochs = range(1, len(loss_values) + 1)

                if 'train' in key:
                    label = f'Train {loss_name}'
                    ax.plot(epochs, loss_values, label=label, color='C0', linestyle='-')
                elif 'val' in key:
                    label = f'Validation {loss_name}'
                    ax.plot(epochs, loss_values, label=label, color='C1', linestyle= '--')

            # 3. 各グラフの装飾
            ax.set_title(f'{loss_name.capitalize()} Loss Curve', fontsize=16)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha = 0.6)

        # 4. グラフ全体のレイアウトを調整し、ファイルに保存
        fig.suptitle(f'All Loss Curves (Experiment {self.experiment_id})', fontsize=20, y=1.02)
        fig.tight_layout(rect=[0, 0, 1, 1])

        save_path = os.path.join(self.output_dir, f'exp_{self.experiment_id}_all_loss_curves.png')

        try:
            fig.savefig(save_path)
            print(f"✅ 統合グラフを保存しました: {save_path}")
        except Exception as e:
            print(f"❌ 統合グラフの保存に失敗しました: {e}")

        # 5. メモリ解放
        plt.close(fig)

    def _detect_loss_types(self, history: Dict) -> Dict[str, list]:
        """プロットする要素を検出"""
        loss_types = {}
        detection_filters = ['total', 'reconstruction', 'kl', 'orthogonal', 'contrastive', 'adversarial', 'style_classification', 'skill_regression']

        # lossの開始
        for key in history.keys():
            if '_loss' in key:
                for detection_filter in detection_filters:
                    if detection_filter in key:
                        loss_types.setdefault(detection_filter, []).append(key)
                        break

        return loss_types

