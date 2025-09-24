# CLAUDE_ADDED: 軌道重ね合わせ表示用の定性的評価器
from typing import List, Dict, Any, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import random

from .base_evaluator import BaseEvaluator
from .result_manager import EnhancedEvaluationResult


class TrajectoryOverlayEvaluator(BaseEvaluator):
    """元軌道と再構成軌道を重ね合わせて表示する定性的評価器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # 設定パラメータ
        self.n_samples_per_split = self.evaluation_config.get('trajectory_overlay_samples', 6)  # 各データ分割から選択するサンプル数
        self.figure_size = self.evaluation_config.get('trajectory_overlay_figsize', (20, 12))  # 図のサイズ
        self.plot_components = self.evaluation_config.get('trajectory_overlay_components', ['position', 'velocity'])  # プロットする成分
        self.random_seed = self.evaluation_config.get('random_seed', 42)

    def evaluate(self, model: torch.nn.Module, test_data: Dict[str, Any], device: torch.device, result: EnhancedEvaluationResult) -> None:
        """軌道重ね合わせ評価を実行"""

        print("Starting trajectory overlay evaluation...")

        # データ準備
        data_splits = self._prepare_data_splits(test_data)

        # Train、Val、Testの必須分割をチェック
        required_splits = ['train', 'validation', 'test']
        missing_splits = [split for split in required_splits if split not in data_splits or len(data_splits[split]['trajectories']) == 0]

        if missing_splits:
            print(f"Warning: Missing or empty data splits: {missing_splits}")
            print(f"Available splits: {list(data_splits.keys())}")

        # 各データ分割から軌道を再構成（利用可能な分割のみ）
        trajectories_data = {}
        for split_name in required_splits:
            if split_name in data_splits and len(data_splits[split_name]['trajectories']) > 0:
                print(f"Processing {split_name} split with {len(data_splits[split_name]['trajectories'])} trajectories")
                trajectories_data[split_name] = self._reconstruct_trajectories(
                    model, data_splits[split_name], device
                )
            else:
                print(f"Skipping {split_name} split (not available or empty)")

        if len(trajectories_data) == 0:
            print("No trajectory data available for overlay evaluation")
            return

        # 可視化を生成
        overlay_fig = self._create_trajectory_overlay_plot(trajectories_data)

        # 結果に追加
        result.add_visualization(
            name='trajectory_overlay_comparison',
            fig_or_path=overlay_fig,
            description='学習・検証・テストデータの元軌道と再構成軌道の重ね合わせ表示',
            category='trajectory_analysis'
        )

        # 定量的メトリクスも計算
        metrics = self._compute_reconstruction_metrics(trajectories_data)
        result.add_metric('trajectory_overlay_metrics', metrics)

        print("Trajectory overlay evaluation completed.")

    def _prepare_data_splits(self, test_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """データを学習・検証・テスト分割に整理"""

        # 利用可能なデータキーを確認
        available_keys = list(test_data.keys())
        print(f"Available data keys: {available_keys}")

        data_splits = {}

        # 各データ分割を処理
        split_mappings = {
            'train': ['train_trajectories', 'trajectories_train', 'train_data'],
            'validation': ['val_trajectories', 'trajectories_val', 'validation_data', 'val_data'],
            'test': ['test_trajectories', 'trajectories_test', 'test_data', 'trajectories']
        }

        for split_name, possible_keys in split_mappings.items():
            trajectories = None
            subject_ids = None
            skill_scores = None

            # 軌道データを探す
            for key in possible_keys:
                if key in test_data:
                    trajectories = test_data[key]
                    break

            # 対応するメタデータを探す
            if trajectories is not None:
                # 被験者IDを探す
                for key in [f'{split_name}_subject_ids', f'subject_ids_{split_name}', 'subject_ids']:
                    if key in test_data:
                        subject_ids = test_data[key]
                        break

                # スキルスコアを探す
                for key in [f'{split_name}_skill_scores', f'skill_scores_{split_name}', 'skill_scores']:
                    if key in test_data:
                        skill_scores = test_data[key]
                        break

                # データ分割を保存
                if len(trajectories) > 0:
                    data_splits[split_name] = {
                        'trajectories': trajectories,
                        'subject_ids': subject_ids,
                        'skill_scores': skill_scores
                    }

        return data_splits

    def _reconstruct_trajectories(self, model: torch.nn.Module, split_data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """指定されたデータ分割の軌道を再構成"""

        trajectories = split_data['trajectories']
        subject_ids = split_data.get('subject_ids')
        skill_scores = split_data.get('skill_scores')

        # ランダムサンプリング
        random.seed(self.random_seed)
        n_available = len(trajectories)
        n_select = min(self.n_samples_per_split, n_available)

        selected_indices = random.sample(range(n_available), n_select)

        selected_trajectories = [trajectories[i] for i in selected_indices]
        selected_subject_ids = [subject_ids[i] if subject_ids else f"sample_{i}" for i in selected_indices]
        selected_skill_scores = [skill_scores[i] if skill_scores else 0.0 for i in selected_indices]

        # テンソルに変換
        trajectories_tensor = torch.stack([torch.tensor(traj, dtype=torch.float32) for traj in selected_trajectories])
        trajectories_tensor = trajectories_tensor.to(device)

        # モデルで再構成
        model.eval()
        with torch.no_grad():
            # モデルのforward関数を呼び出し
            if hasattr(model, 'forward'):
                outputs = model(trajectories_tensor)
                reconstructed = outputs.get('reconstructed', outputs.get('trajectory', trajectories_tensor))
            else:
                reconstructed = trajectories_tensor

        # CPUに移動
        original_trajectories = trajectories_tensor.cpu().numpy()
        reconstructed_trajectories = reconstructed.cpu().numpy()

        return {
            'original': original_trajectories,
            'reconstructed': reconstructed_trajectories,
            'subject_ids': selected_subject_ids,
            'skill_scores': selected_skill_scores,
            'indices': selected_indices
        }

    def _create_trajectory_overlay_plot(self, trajectories_data: Dict[str, Dict[str, Any]]) -> plt.Figure:
        """軌道重ね合わせプロットを作成"""

        n_splits = len(trajectories_data)
        if n_splits == 0:
            # データがない場合のプレースホルダー
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No trajectory data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Trajectory Overlay Comparison')
            return fig

        # サブプロット設定
        n_components = len(self.plot_components)
        fig = plt.figure(figsize=self.figure_size)
        gs = GridSpec(n_components, n_splits, figure=fig, hspace=0.3, wspace=0.3)

        # カラーマップ設定
        colors = plt.cm.Set3(np.linspace(0, 1, 12))  # 12色のカラーマップ

        for split_idx, (split_name, data) in enumerate(trajectories_data.items()):
            original = data['original']
            reconstructed = data['reconstructed']
            subject_ids = data['subject_ids']
            skill_scores = data['skill_scores']

            for comp_idx, component in enumerate(self.plot_components):
                ax = fig.add_subplot(gs[comp_idx, split_idx])

                # 成分インデックスを決定
                if component == 'position':
                    comp_indices = [0, 1]  # x, y position
                    comp_labels = ['X Position', 'Y Position']
                elif component == 'velocity':
                    comp_indices = [2, 3]  # x, y velocity
                    comp_labels = ['X Velocity', 'Y Velocity']
                elif component == 'acceleration':
                    comp_indices = [4, 5]  # x, y acceleration
                    comp_labels = ['X Acceleration', 'Y Acceleration']
                else:
                    comp_indices = [0, 1]  # デフォルトは位置
                    comp_labels = ['Component 0', 'Component 1']

                # 各サンプルを重ね合わせプロット
                for sample_idx in range(len(original)):
                    color = colors[sample_idx % len(colors)]
                    alpha = 0.7

                    # 元軌道（実線）
                    if len(comp_indices) >= 2:
                        ax.plot(original[sample_idx, :, comp_indices[0]],
                               original[sample_idx, :, comp_indices[1]],
                               color=color, alpha=alpha, linewidth=2,
                               label=f'Original {subject_ids[sample_idx]}' if sample_idx < 3 else '',
                               linestyle='-')

                        # 再構成軌道（点線）
                        ax.plot(reconstructed[sample_idx, :, comp_indices[0]],
                               reconstructed[sample_idx, :, comp_indices[1]],
                               color=color, alpha=alpha, linewidth=2,
                               label=f'Reconstructed {subject_ids[sample_idx]}' if sample_idx < 3 else '',
                               linestyle='--')

                        # スタート・エンド点をマーク
                        ax.scatter(original[sample_idx, 0, comp_indices[0]],
                                 original[sample_idx, 0, comp_indices[1]],
                                 color=color, s=100, marker='o', edgecolor='black', linewidth=1,
                                 zorder=10)
                        ax.scatter(original[sample_idx, -1, comp_indices[0]],
                                 original[sample_idx, -1, comp_indices[1]],
                                 color=color, s=100, marker='s', edgecolor='black', linewidth=1,
                                 zorder=10)

                # グラフ設定
                ax.set_xlabel(comp_labels[0])
                ax.set_ylabel(comp_labels[1])
                ax.set_title(f'{split_name.title()} - {component.title()}')
                ax.grid(True, alpha=0.3)
                ax.axis('equal')

                # 凡例（最初の行のみ）
                if comp_idx == 0 and split_idx == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # 全体タイトル
        fig.suptitle('Trajectory Overlay Comparison: Original (solid) vs Reconstructed (dashed)',
                     fontsize=16, y=0.95)

        # 凡例の説明を追加
        fig.text(0.02, 0.02, 'Circle: Start point, Square: End point', fontsize=10, style='italic')

        plt.tight_layout()
        return fig

    def _compute_reconstruction_metrics(self, trajectories_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """再構成性能の定量的メトリクスを計算"""

        metrics = {}

        for split_name, data in trajectories_data.items():
            original = data['original']
            reconstructed = data['reconstructed']

            # MSE計算
            mse = np.mean((original - reconstructed) ** 2)

            # 成分別MSE
            component_mse = {}
            component_names = ['pos_x', 'pos_y', 'vel_x', 'vel_y', 'acc_x', 'acc_y']
            for i, comp_name in enumerate(component_names[:original.shape[2]]):
                component_mse[comp_name] = np.mean((original[:, :, i] - reconstructed[:, :, i]) ** 2)

            # 軌道長計算（元 vs 再構成）
            def compute_trajectory_length(trajectories):
                lengths = []
                for traj in trajectories:
                    pos = traj[:, :2]  # x, y position
                    diff = np.diff(pos, axis=0)
                    length = np.sum(np.sqrt(np.sum(diff**2, axis=1)))
                    lengths.append(length)
                return np.array(lengths)

            original_lengths = compute_trajectory_length(original)
            reconstructed_lengths = compute_trajectory_length(reconstructed)
            length_error = np.mean(np.abs(original_lengths - reconstructed_lengths) / (original_lengths + 1e-8))

            metrics[f'{split_name}_mse'] = float(mse)
            metrics[f'{split_name}_component_mse'] = component_mse
            metrics[f'{split_name}_length_error'] = float(length_error)
            metrics[f'{split_name}_n_samples'] = len(original)

        return metrics

    def get_required_data(self) -> List[str]:
        """必要なデータ形式を返す"""
        return ['trajectories', 'subject_ids', 'skill_scores']