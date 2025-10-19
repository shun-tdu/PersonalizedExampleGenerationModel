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

        # 2. 各描画領域と損失グループをzipでループ
        for ax, (loss_name, keys_to_plot) in zip(axes, loss_type.items()):
            # CLAUDE_ADDED: 損失名の表示を改善するマッピング
            display_names = {
                'kl_style': 'Style Space KL',
                'kl_skill': 'Skill Space KL',
                'manifold': 'Manifold Formation',
                'style_classification': 'Style Classification',
                'skill_regression': 'Skill Regression',
                'diffusion':'Diffusion Loss'
            }

            for key in keys_to_plot:
                loss_values = history[key]
                epochs = range(1, len(loss_values) + 1)

                # CLAUDE_ADDED: より詳細で明確な凡例ラベル
                if 'train' in key:
                    if loss_name in display_names:
                        label = f'Train {display_names[loss_name]}'
                    else:
                        label = f'Train {loss_name.capitalize()}'
                    ax.plot(epochs, loss_values, label=label, color='C0', linestyle='-')
                elif 'val' in key:
                    if loss_name in display_names:
                        label = f'Validation {display_names[loss_name]}'
                    else:
                        label = f'Validation {loss_name.capitalize()}'
                    ax.plot(epochs, loss_values, label=label, color='C1', linestyle='--')

            # 3. 各グラフの装飾
            # CLAUDE_ADDED: より適切なタイトル表示
            if loss_name in display_names:
                title = f'{display_names[loss_name]} Loss Curve'
            else:
                title = f'{loss_name.capitalize()} Loss Curve'

            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)

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
        # CLAUDE_ADDED: manifold_lossとKL損失の細分化を追加
        detection_filters = [
            'total', 'reconstruction',
            'kl_style', 'kl_skill',  # KL損失を分離
            'orthogonal', 'contrastive', 'adversarial',
            'manifold',  # manifold_loss追加
            'style_classification', 'skill_regression'
            'diffusion'
        ]

        # lossの開始
        for key in history.keys():
            if '_loss' in key:
                # CLAUDE_ADDED: より精密なマッチングでKL損失を分離
                if 'kl_style' in key:
                    loss_types.setdefault('kl_style', []).append(key)
                elif 'kl_skill' in key:
                    loss_types.setdefault('kl_skill', []).append(key)
                elif 'manifold' in key:
                    loss_types.setdefault('manifold', []).append(key)
                else:
                    # 従来の検出ロジック（kl_styleとkl_skill以外）
                    for detection_filter in detection_filters:
                        if detection_filter in key and detection_filter not in ['kl_style', 'kl_skill', 'manifold']:
                            loss_types.setdefault(detection_filter, []).append(key)
                            break

        return loss_types

