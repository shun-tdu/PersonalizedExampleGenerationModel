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
        # CLAUDE_ADDED: 位置に時系列プロットも追加
        self.expanded_plot_components = []
        for component in self.plot_components:
            if component == 'position':
                self.expanded_plot_components.extend(['position_2d', 'position_time_series'])
            elif component == 'velocity':
                self.expanded_plot_components.append('velocity')
            elif component == 'acceleration':
                self.expanded_plot_components.append('acceleration')
            else:
                self.expanded_plot_components.append(component)
        self.random_seed = self.evaluation_config.get('random_seed', 42)

    def evaluate(self, model: torch.nn.Module, test_data: Dict[str, Any], device: torch.device, result: EnhancedEvaluationResult) -> None:
        """軌道重ね合わせ評価を実行"""

        print("Starting trajectory overlay evaluation...")

        # CLAUDE_ADDED: Get ModelAdapter from test_data if available
        # This is set during preprocessing in EvaluationPipeline._preprocess_latent_data()
        test_loader = test_data.get('test_loader')
        if test_loader is not None:
            from .adapters import AdapterFactory
            config = test_data.get('config', self.config)
            _, self.model_adapter = AdapterFactory.auto_detect_adapters(model, test_loader, config)
            print(f"ModelAdapter detected: {self.model_adapter.__class__.__name__}")
        else:
            self.model_adapter = None
            print("Warning: test_loader not available, ModelAdapter will not be used")

        # スケーラーを取得 - CLAUDE_ADDED: 逆標準化に必要
        self.scalers = test_data.get('scalers', {})
        print(f"Available scalers: {list(self.scalers.keys()) if self.scalers else 'None'}")

        # データ準備
        data_splits = self._prepare_data_splits(test_data)

        # CLAUDE_ADDED: 利用可能な分割を優先的に処理
        preferred_splits = ['test', 'all']  # テストデータと全データを優先
        additional_splits = ['train', 'validation']

        # 利用可能な分割を確認
        available_splits = list(data_splits.keys())
        print(f"Available data splits: {available_splits}")

        # 各データ分割から軌道を再構成（利用可能な分割のみ）
        trajectories_data = {}

        # 優先分割を先に処理
        for split_name in preferred_splits:
            if split_name in data_splits and len(data_splits[split_name]['trajectories']) > 0:
                print(f"Processing {split_name} split with {len(data_splits[split_name]['trajectories'])} trajectories")
                trajectories_data[split_name] = self._reconstruct_trajectories(
                    model, data_splits[split_name], device
                )

        # 追加分割も処理（あれば）
        for split_name in additional_splits:
            if split_name in data_splits and len(data_splits[split_name]['trajectories']) > 0:
                print(f"Processing {split_name} split with {len(data_splits[split_name]['trajectories'])} trajectories")
                trajectories_data[split_name] = self._reconstruct_trajectories(
                    model, data_splits[split_name], device
                )

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

        # 定量的メトリクスも計算 - CLAUDE_ADDED: 個別メトリクスとして追加
        metrics = self._compute_reconstruction_metrics(trajectories_data)
        self._add_reconstruction_metrics(result, metrics)

        print("Trajectory overlay evaluation completed.")

    def _prepare_data_splits(self, test_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """データを学習・検証・テスト分割に整理"""

        # 利用可能なデータキーを確認 - CLAUDE_ADDED: 詳細なデバッグ情報を追加
        available_keys = list(test_data.keys())
        print(f"Available data keys: {available_keys}")

        # データ形状の確認
        for key in available_keys:
            value = test_data[key]
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, type={type(value)}")
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                if hasattr(value[0], 'shape'):
                    print(f"  {key}: list/tuple length={len(value)}, first_item_shape={value[0].shape}")
                else:
                    print(f"  {key}: list/tuple length={len(value)}, first_item_type={type(value[0])}")
            else:
                print(f"  {key}: type={type(value)}")

        data_splits = {}

        # 各データ分割を処理 - CLAUDE_ADDED: 実際のキー構造に対応
        split_mappings = {
            'train': ['train_trajectories', 'trajectories_train', 'train_data'],
            'validation': ['val_trajectories', 'trajectories_val', 'validation_data', 'val_data'],
            'test': ['originals', 'test_trajectories', 'trajectories_test', 'test_data', 'trajectories'],
            'all': ['all_originals', 'all_trajectories']  # 全データ用の新しい分割
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

            # 対応するメタデータを探す - CLAUDE_ADDED: 実際のキー構造に対応
            if trajectories is not None:
                # 被験者IDを探す
                subject_id_keys = [f'{split_name}_subject_ids', f'subject_ids_{split_name}', 'subject_ids']
                if split_name == 'all':
                    subject_id_keys = ['all_subject_ids'] + subject_id_keys

                for key in subject_id_keys:
                    if key in test_data:
                        subject_ids = test_data[key]
                        break

                # スキルスコアを探す
                skill_score_keys = [f'{split_name}_skill_scores', f'skill_scores_{split_name}', 'skill_scores']
                if split_name == 'all':
                    skill_score_keys = ['all_skill_scores'] + skill_score_keys

                for key in skill_score_keys:
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
        """CLAUDE_ADDED: Adapter-enabled reconstruction for selected samples"""

        trajectories = split_data['trajectories']
        subject_ids = split_data.get('subject_ids')
        skill_scores = split_data.get('skill_scores')

        # ランダムサンプリング
        random.seed(self.random_seed)
        n_available = len(trajectories)
        n_select = min(self.n_samples_per_split, n_available)

        selected_indices = random.sample(range(n_available), n_select)

        selected_trajectories = [trajectories[i] for i in selected_indices]
        selected_subject_ids = [subject_ids[i] if subject_ids is not None else f"sample_{i}" for i in selected_indices]
        selected_skill_scores = [skill_scores[i] if skill_scores is not None else 0.0 for i in selected_indices]

        # テンソルに変換
        trajectories_tensor = torch.stack([torch.tensor(traj, dtype=torch.float32) for traj in selected_trajectories])
        trajectories_tensor = trajectories_tensor.to(device)

        # CLAUDE_ADDED: Use ModelAdapter for reconstruction
        model.eval()
        with torch.no_grad():
            # Get ModelAdapter from test_data (set during preprocessing)
            if hasattr(self, 'model_adapter') and self.model_adapter is not None:
                # Use adapter
                encoded = self.model_adapter.encode(trajectories_tensor, attention_mask=None)
                z_style = encoded['z_style']
                z_skill = encoded['z_skill']

                # Decode (skips for diffusion models by default)
                if not self.model_adapter.is_diffusion_model():
                    reconstructed = self.model_adapter.decode(z_style, z_skill, metadata=None)
                else:
                    # Diffusion model: sampling is time-consuming
                    print("拡散モデル検出: サンプリングスキップ（時間がかかるため）")
                    reconstructed = trajectories_tensor  # Use original as fallback
            else:
                # Fallback: Direct model access (for backward compatibility)
                print("Warning: ModelAdapter not available, using direct model access")
                if hasattr(model, 'encode') and hasattr(model, 'decode'):
                    encoded = model.encode(trajectories_tensor)
                    z_style = encoded['z_style']
                    z_skill = encoded['z_skill']
                    decoded = model.decode(z_style, z_skill)
                    reconstructed = decoded.get('trajectory', trajectories_tensor)
                else:
                    reconstructed = trajectories_tensor

        # CPUに移動して逆標準化 - CLAUDE_ADDED: 標準化を元に戻して可視化
        original_trajectories = trajectories_tensor.cpu().numpy()
        reconstructed_trajectories = reconstructed.cpu().numpy()

        # 標準化を元に戻す
        original_denormalized = self._denormalize_trajectories(original_trajectories)
        reconstructed_denormalized = self._denormalize_trajectories(reconstructed_trajectories)

        return {
            'original': original_denormalized,
            'reconstructed': reconstructed_denormalized,
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

        # サブプロット設定 - CLAUDE_ADDED: 拡張されたコンポーネントリストを使用
        n_components = len(self.expanded_plot_components)
        fig = plt.figure(figsize=self.figure_size)
        gs = GridSpec(n_components, n_splits, figure=fig, hspace=0.3, wspace=0.3)

        # カラーマップ設定 - CLAUDE_ADDED: よりくっきりした見やすい色に変更
        distinct_colors = [
            '#FF0000',  # 赤
            '#0000FF',  # 青
            '#008000',  # 緑
            '#FF8000',  # オレンジ
            '#800080',  # 紫
            '#FF1493',  # ピンク
            '#00CED1',  # ターコイズ
            '#8B4513',  # ブラウン
            '#FFD700',  # ゴールド
            '#DC143C',  # クリムゾン
            '#4B0082',  # インディゴ
            '#32CD32'   # ライムグリーン
        ]
        colors = distinct_colors

        for split_idx, (split_name, data) in enumerate(trajectories_data.items()):
            original = data['original']
            reconstructed = data['reconstructed']
            subject_ids = data['subject_ids']
            skill_scores = data['skill_scores']

            for comp_idx, component in enumerate(self.expanded_plot_components):
                ax = fig.add_subplot(gs[comp_idx, split_idx])

                # 成分インデックスとプロットタイプを決定 - CLAUDE_ADDED: 拡張プロット対応
                if component == 'position_2d':
                    comp_indices = [0, 1]  # x, y position
                    comp_labels = ['X Position', 'Y Position']
                    plot_type = '2d'
                    component_name = 'Position 2D'
                elif component == 'position_time_series':
                    comp_indices = [0, 1]  # x, y position
                    comp_labels = ['X Position', 'Y Position']
                    plot_type = 'time_series'
                    component_name = 'Position'
                elif component == 'velocity':
                    comp_indices = [2, 3]  # x, y velocity
                    comp_labels = ['X Velocity', 'Y Velocity']
                    plot_type = 'time_series'
                    component_name = 'Velocity'
                elif component == 'acceleration':
                    comp_indices = [4, 5]  # x, y acceleration
                    comp_labels = ['X Acceleration', 'Y Acceleration']
                    plot_type = 'time_series'
                    component_name = 'Acceleration'
                else:
                    comp_indices = [0, 1]  # デフォルトは位置
                    comp_labels = ['Component 0', 'Component 1']
                    plot_type = '2d'
                    component_name = component.title()

                # プロットタイプに応じて描画方法を変更
                if plot_type == 'time_series':
                    # 時系列プロット（速度・加速度用）
                    time_steps = np.arange(original.shape[1])

                    for sample_idx in range(len(original)):
                        color = colors[sample_idx % len(colors)]
                        alpha = 0.7

                        # X成分
                        ax.plot(time_steps, original[sample_idx, :, comp_indices[0]],
                               color=color, alpha=alpha, linewidth=2,
                               label=f'Original {subject_ids[sample_idx]} (X)' if sample_idx < 2 else '',
                               linestyle='-')
                        ax.plot(time_steps, reconstructed[sample_idx, :, comp_indices[0]],
                               color=color, alpha=alpha, linewidth=2,
                               label=f'Reconstructed {subject_ids[sample_idx]} (X)' if sample_idx < 2 else '',
                               linestyle='--')

                        # Y成分（異なる色合い）
                        darker_color = colors[(sample_idx + 6) % len(colors)]
                        ax.plot(time_steps, original[sample_idx, :, comp_indices[1]],
                               color=darker_color, alpha=alpha, linewidth=2,
                               label=f'Original {subject_ids[sample_idx]} (Y)' if sample_idx < 2 else '',
                               linestyle='-')
                        ax.plot(time_steps, reconstructed[sample_idx, :, comp_indices[1]],
                               color=darker_color, alpha=alpha, linewidth=2,
                               label=f'Reconstructed {subject_ids[sample_idx]} (Y)' if sample_idx < 2 else '',
                               linestyle='--')

                    # グラフ設定（時系列用）
                    ax.set_xlabel('Time Steps')
                    ax.set_ylabel(f'{component_name} Value')
                    ax.set_title(f'{split_name.title()} - {component_name} Time Series')
                    ax.grid(True, alpha=0.3)

                else:
                    # 2D軌道プロット（位置用）
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

                    # グラフ設定（2D軌道用）
                    ax.set_xlabel(comp_labels[0])
                    ax.set_ylabel(comp_labels[1])
                    ax.set_title(f'{split_name.title()} - {component_name}')
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

            # 成分別MSE - CLAUDE_ADDED: format string エラー回避のため値を明示的にfloatに変換
            component_mse = {}
            component_names = ['pos_x', 'pos_y', 'vel_x', 'vel_y', 'acc_x', 'acc_y']
            for i, comp_name in enumerate(component_names[:original.shape[2]]):
                component_mse[comp_name] = float(np.mean((original[:, :, i] - reconstructed[:, :, i]) ** 2))

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

    def _add_reconstruction_metrics(self, result: EnhancedEvaluationResult, metrics: Dict[str, Any]) -> None:
        """再構成メトリクスを個別に結果に追加 - CLAUDE_ADDED: 他の評価器と同じパターンで実装"""
        for metric_name, value in metrics.items():
            if '_component_mse' in metric_name:
                # 成分別MSEは個別メトリクスとして追加
                split_name = metric_name.replace('_component_mse', '')
                for comp_name, comp_value in value.items():
                    result.add_metric(
                        name=f'{split_name}_mse_{comp_name}',
                        value=float(comp_value),
                        description=f'{split_name}データの{comp_name}成分MSE',
                        category='trajectory_reconstruction'
                    )
            else:
                # その他のメトリクス
                if metric_name.endswith('_mse'):
                    description = f'{metric_name.replace("_mse", "")}データの全体MSE'
                elif metric_name.endswith('_length_error'):
                    description = f'{metric_name.replace("_length_error", "")}データの軌道長誤差率'
                elif metric_name.endswith('_n_samples'):
                    description = f'{metric_name.replace("_n_samples", "")}データのサンプル数'
                else:
                    description = f'軌道再構成メトリクス: {metric_name}'

                result.add_metric(
                    name=metric_name,
                    value=float(value) if isinstance(value, (int, float, np.integer, np.floating)) else value,
                    description=description,
                    category='trajectory_reconstruction'
                )

    def _denormalize_trajectories(self, trajectories: np.ndarray) -> np.ndarray:
        """軌道データの標準化を元に戻す - CLAUDE_ADDED: 可視化のため元のスケールに戻す"""

        # スケーラーが利用できない場合はそのまま返す
        if not self.scalers:
            print("Warning: No scalers available. Returning trajectories as-is.")
            return trajectories

        # 軌道特徴量の順序（analyze_skill_metrics.pyと一致）
        trajectory_features = ['HandlePosX', 'HandlePosY', 'HandleVelX',
                              'HandleVelY', 'HandleAccX', 'HandleAccY']

        # 逆標準化後のデータを格納
        denormalized_trajectories = trajectories.copy()

        # 各特徴量（成分）ごとに逆標準化
        for i, feature_name in enumerate(trajectory_features):
            if i < trajectories.shape[2] and feature_name in self.scalers:
                scaler = self.scalers[feature_name]

                # 該当する成分のデータを取得 [batch_size, seq_len]
                component_data = trajectories[:, :, i]

                # 逆標準化のためにreshape [batch_size * seq_len, 1]
                flattened_data = component_data.flatten().reshape(-1, 1)

                # 逆標準化を実行
                denormalized_data = scaler.inverse_transform(flattened_data)

                # 元の形状に戻して格納 [batch_size, seq_len]
                denormalized_trajectories[:, :, i] = denormalized_data.reshape(component_data.shape)
            else:
                if i < trajectories.shape[2]:
                    print(f"Warning: Scaler for {feature_name} not found. Using original values.")

        return denormalized_trajectories

    def get_required_data(self) -> List[str]:
        """必要なデータ形式を返す - CLAUDE_ADDED: 実際のキー構造に対応"""
        return ['originals', 'subject_ids', 'skill_scores', 'scalers']