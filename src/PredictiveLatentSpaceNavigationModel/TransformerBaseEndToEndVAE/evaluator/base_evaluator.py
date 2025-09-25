import importlib.util
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Protocol

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.core.pylabtools import figsize

from .result_manager import EnhancedEvaluationResult, VisualizationItem
# Evaluator imports moved to _setup_evaluators method to avoid circular imports


class BaseEvaluator(ABC):
    """評価器の基底クラス"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluation_config = config.get('evaluation', {})

    @abstractmethod
    def evaluate(self, model: torch.nn.Module, test_data: Dict[str, Any], device: torch.device, result: EnhancedEvaluationResult) -> None:
        """評価を実行
        
        Args:
            model: 評価対象のモデル
            test_data: テストデータ
            device: デバイス
            result: 共有の評価結果オブジェクト（呼び出し元で作成済み）
        """
        pass

    @abstractmethod
    def get_required_data(self) -> List[str]:
        """必要なデータ形式を返す"""
        pass


class EvaluationPipeline:
    """EnhancedEvaluationResult対応の評価パイプライン管理クラス"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluators = []

        # 設定に基づいて評価器を初期化
        self._setup_evaluators()

    def _setup_evaluators(self):
        """評価器をセットアップ"""
        evaluation_config = self.config.get('evaluation', {})

        # 旧仕様評価器（優先度高）
        if evaluation_config.get('comprehensive_latent_analysis', False):
            from .comprehensive_evaluator import ComprehensiveLatentSpaceEvaluator
            self.evaluators.append(ComprehensiveLatentSpaceEvaluator(self.config))
        if evaluation_config.get('latent_space_analysis', False):
            self.evaluators.append(LatentSpaceEvaluator(self.config))

        # BaseLine評価器
        if evaluation_config.get('trajectory_generation_analysis', False):
            from .base_line_evaluator import TrajectoryGenerationEvaluator
            self.evaluators.append(TrajectoryGenerationEvaluator(self.config))
        if evaluation_config.get('style_skill_orthogonality_analysis', False):
            from .base_line_evaluator import OrthogonalityEvaluator
            self.evaluators.append(OrthogonalityEvaluator(self.config))
        if evaluation_config.get('trajectory_overlay_analysis', False):
            from .trajectory_overlay_evaluator import TrajectoryOverlayEvaluator
            self.evaluators.append(TrajectoryOverlayEvaluator(self.config))

        # スタイル空間評価器
        if evaluation_config.get('visualize_style_space_analysis', False):
            from .style_latent_space_evaluator import VisualizeStyleSpaceEvaluator
            self.evaluators.append(VisualizeStyleSpaceEvaluator(self.config))
        if evaluation_config.get('style_clustering_analysis', False):
            from .style_latent_space_evaluator import StyleClusteringEvaluator
            self.evaluators.append(StyleClusteringEvaluator(self.config))
        if evaluation_config.get('style_classification_analysis', False):
            from .style_latent_space_evaluator import StyleClassificationEvaluator
            self.evaluators.append(StyleClassificationEvaluator(self.config))

        # スキル空間評価器
        if evaluation_config.get('visualize_skill_space_analysis', False):
            from .skill_latent_space_evaluator import VisualizeSkillSpaceEvaluator
            self.evaluators.append(VisualizeSkillSpaceEvaluator(self.config))
        if evaluation_config.get('visualize_KDE', False):
            from .skill_latent_space_evaluator import SkillManifoldAnalysisEvaluator
            self.evaluators.append(SkillManifoldAnalysisEvaluator(self.config))
        if evaluation_config.get('skill_score_regression_analysis', False):
            from .skill_latent_space_evaluator import SkillScoreRegressionEvaluator
            self.evaluators.append(SkillScoreRegressionEvaluator(self.config))
        if evaluation_config.get('skill_latent_dimension_vs_score_analysis', False):
            from .skill_latent_space_evaluator import SkillLatentDimensionVSScoreEvaluator
            self.evaluators.append(SkillLatentDimensionVSScoreEvaluator(self.config))

        # カスタム評価器の動的ロード
        custom_evaluators = evaluation_config.get('custom_evaluators', [])
        for evaluator_config in custom_evaluators:
            evaluator_class = self._load_evaluator_class(evaluator_config)
            if evaluator_class:
                self.evaluators.append(evaluator_class(self.config))

    def _load_evaluator_class(self, evaluator_config: Dict[str, str]) -> Optional[type]:
        """カスタム評価器クラスを動的ロード"""
        try:
            module_name = evaluator_config['module']
            class_name = evaluator_config['class']

            module = importlib.import_module(module_name)
            evaluator_class = getattr(module, class_name)

            return evaluator_class
        except Exception as e:
            print(f"カスタム評価器ロードエラー: {e}")
            return None

    def run_evaluation(self, model: torch.nn.Module, test_data: Dict[str, Any], device: torch.device) -> 'EnhancedEvaluationResult':
        """全ての評価を実行"""
        # EnhancedEvaluationResultの初期化に必要なパラメータ
        experiment_id = test_data.get('experiment_id', 0)
        output_dir = test_data.get('output_dir', 'outputs')

        results = []

        successful_evaluations = 0
        total_evaluations = len(self.evaluators)

        for evaluator in self.evaluators:
            print(f"実行中: {evaluator.__class__.__name__}")

            try:
                # 必要なデータの確認
                required_data = evaluator.get_required_data()
                missing_data = [key for key in required_data if key not in test_data]

                if missing_data:
                    print(f"警告: {evaluator.__class__.__name__} に必要なデータが不足: {missing_data}")
                    continue

                # 評価実行
                result = evaluator.evaluate(model, test_data, device)
                results.append(result)

                successful_evaluations += 1
                print(f"✓ {evaluator.__class__.__name__} 完了")

            except Exception as e:
                print(f"✗ 評価エラー ({evaluator.__class__.__name__}): {e}")
                import traceback
                print(f"詳細: {traceback.format_exc()}")
                continue

        # 評価完了サマリー
        print(f"\n評価完了: {successful_evaluations}/{total_evaluations} 成功")

        # 統合レポート生成
        if successful_evaluations > 0 and results:
            try:
                # 最初の結果オブジェクトを返す
                main_result = results[0] if results else None
                if main_result and hasattr(main_result, 'create_comprehensive_report'):
                    report_path = main_result.create_comprehensive_report()
                    print(f"統合レポート生成: {report_path}")
                return main_result
            except Exception as e:
                print(f"レポート生成エラー: {e}")
                return results[0] if results else None
                
        return None

    def run_unified_evaluation(self, model: torch.nn.Module, test_data: Dict[str, Any], device: torch.device) -> 'EnhancedEvaluationResult':
        """統一されたレポートで全ての評価を実行（共有EnhancedEvaluationResultを使用）"""
        # EnhancedEvaluationResultの初期化
        experiment_id = test_data.get('experiment_id', 0)
        output_dir = test_data.get('output_dir', 'outputs')
        
        shared_result = EnhancedEvaluationResult(experiment_id, output_dir)
        
        successful_evaluations = 0
        total_evaluations = len(self.evaluators)

        for evaluator in self.evaluators:
            print(f"実行中: {evaluator.__class__.__name__}")

            try:
                # 必要なデータの確認
                required_data = evaluator.get_required_data()
                missing_data = [key for key in required_data if key not in test_data]

                if missing_data:
                    print(f"警告: {evaluator.__class__.__name__} に必要なデータが不足: {missing_data}")
                    continue

                # 共有結果オブジェクトを渡して評価実行
                evaluator.evaluate(model, test_data, device, shared_result)

                successful_evaluations += 1
                print(f"✓ {evaluator.__class__.__name__} 完了")

            except Exception as e:
                print(f"✗ 評価エラー ({evaluator.__class__.__name__}): {e}")
                import traceback
                print(f"詳細: {traceback.format_exc()}")
                continue

        # 評価完了サマリー
        print(f"\n統一評価完了: {successful_evaluations}/{total_evaluations} 成功")

        # 統合レポート生成
        if successful_evaluations > 0:
            try:
                report_path = shared_result.create_comprehensive_report()
                print(f"統一レポート生成: {report_path}")
                return shared_result
            except Exception as e:
                print(f"レポート生成エラー: {e}")
                return shared_result
                
        return shared_result

    def get_evaluator_info(self) -> List[Dict[str, str]]:
        """登録されている評価器の情報を取得"""
        evaluator_info = []
        for evaluator in self.evaluators:
            info = {
                'name': evaluator.__class__.__name__,
                'module': evaluator.__class__.__module__,
                'required_data': evaluator.get_required_data(),
                'supports_enhanced': hasattr(evaluator, 'evaluate_enhanced')
            }
            evaluator_info.append(info)
        return evaluator_info

    def validate_test_data(self, test_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """テストデータの妥当性を事前チェック"""
        validation_results = {}

        for evaluator in self.evaluators:
            evaluator_name = evaluator.__class__.__name__
            required_data = evaluator.get_required_data()
            missing_data = [key for key in required_data if key not in test_data]

            validation_results[evaluator_name] = missing_data

        return validation_results

    def run_selective_evaluation(self, model: torch.nn.Module, test_data: Dict[str, Any], device: torch.device,
                                 evaluator_names: List[str] = None) -> 'EnhancedEvaluationResult':
        """指定された評価器のみ実行"""
        if evaluator_names is None:
            return self.run_evaluation(model, test_data, device)

        # 一時的に評価器をフィルタリング
        original_evaluators = self.evaluators
        filtered_evaluators = [
            evaluator for evaluator in self.evaluators
            if evaluator.__class__.__name__ in evaluator_names
        ]

        self.evaluators = filtered_evaluators

        try:
            result = self.run_evaluation(model, test_data, device)
        finally:
            # 元の評価器リストを復元
            self.evaluators = original_evaluators

        return result


class LatentSpaceEvaluator(BaseEvaluator):
    """潜在空間分析評価器"""

    def get_required_data(self) -> List[str]:
        return ['test_loader', 'output_dir']

    def evaluate(self, model: torch.nn.Module, test_data: Dict[str, Any], device: torch.device, result: EnhancedEvaluationResult):
        """潜在空間の包括評価"""
        experiment_id = test_data.get('experiment_id', 0)
        output_dir = test_data['output_dir']
        test_loader = test_data['test_loader']

        # 共有結果オブジェクトが渡されない場合は新規作成
        if result is None:
            result = EnhancedEvaluationResult(experiment_id, output_dir)

        # 潜在変数抽出
        latent_data = self._extract_latent_variables(model, test_loader, device)

        # 1. 再構成性能評価
        recon_metrics = self._evaluate_reconstruction(latent_data)
        for name, value in recon_metrics.items():
            result.add_metric(f'reconstruction_{name}', value, 'RMSEによる再構築精度', 'reconstruction')

        # 2. スタイル分離評価
        style_metrics = self._evaluate_style_separation(latent_data)
        for name, value in style_metrics.items():
            result.add_metric(f'style_{name}', value, 'スタイル分離評価', 'latent_space')

        # 3. スキル表現評価
        skill_metrics = self._evaluate_skill_representation(latent_data, test_data.get('test_df'))
        for name, value in skill_metrics.items():
            result.add_metric(f'skill_{name}', value, 'スキル表現評価', 'latent_space')

        # 4. 可視化生成(PCA)
        pca_fig = self._create_pca_visualization(latent_data)
        result.add_visualization('latent_pca', pca_fig, '潜在空間のPCA可視化', 'latent_space')

        # 4. 可視化生成(t-SNE) - 一時的にコメントアウト
        # tsne_fig = self._create_tsne_visualization(latent_data)
        # result.add_visualization('latent_tsne', tsne_fig, '潜在空間のt-SNE可視化', 'latent_space')

        # HTMLレポート生成
        report_path = result.create_comprehensive_report()


    def _extract_latent_variables(self, model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, Any]:
        """潜在変数を抽出"""
        model.eval()
        all_z_style = []
        all_z_skill = []
        all_subject_ids = []
        all_reconstructions = []
        all_originals = []

        with torch.no_grad():
            for batch_data in test_loader:
                trajectories = batch_data[0].to(device)
                subject_ids = batch_data[1]

                # エンコード
                encoded = model.encode(trajectories)
                z_style = encoded['z_style']
                z_skill = encoded['z_skill']

                # デコード（動的にスキップ接続対応を判断）
                decoded = self._safe_decode(model, z_style, z_skill, encoded)
                reconstructed = decoded['trajectory']

                all_z_style.append(z_style.cpu().numpy())
                all_z_skill.append(z_skill.cpu().numpy())
                all_subject_ids.extend(subject_ids)
                all_reconstructions.append(reconstructed.cpu().numpy())
                all_originals.append(trajectories.cpu().numpy())

        return {
            'z_style': np.vstack(all_z_style),
            'z_skill': np.vstack(all_z_skill),
            'subject_ids': all_subject_ids,
            'reconstructions': np.vstack(all_reconstructions),
            'originals': np.vstack(all_originals)
        }

    def _evaluate_reconstruction(self, latent_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """再構成性能評価"""
        mse = np.mean((latent_data['originals'] - latent_data['reconstructions']) ** 2)
        return {'mse': mse}

    def _evaluate_style_separation(self, latent_data: Dict[str, Any]) -> Dict[str, float]:
        """スタイル分離性能評価"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        z_style = latent_data['z_style']
        subject_ids = latent_data['subject_ids']

        unique_subjects = list(set(subject_ids))
        if len(unique_subjects) < 2:
            return {'separation_ari': 0.0, 'separation_ratio': 0.0}

        # 被験者ラベルを数値に変換
        subject_to_idx = {subj: i for i, subj in enumerate(unique_subjects)}
        true_labels = [subject_to_idx[subj] for subj in subject_ids]

        # K-meansクラスタリング
        kmeans = KMeans(n_clusters=len(unique_subjects), random_state=42)
        pred_labels = kmeans.fit_predict(z_style)

        # ARI計算
        ari = adjusted_rand_score(true_labels, pred_labels)

        # 分離比計算
        centroids = np.array([np.mean(z_style[np.array(subject_ids) == subj], axis=0)
                              for subj in unique_subjects])
        between_var = np.var(centroids.flatten())

        within_vars = []
        for subj in unique_subjects:
            subj_data = z_style[np.array(subject_ids) == subj]
            if len(subj_data) > 1:
                within_vars.append(np.var(subj_data.flatten()))

        within_var = np.mean(within_vars) if within_vars else 0.001
        separation_ratio = between_var / within_var

        return {
            'separation_ari': ari,
            'separation_ratio': separation_ratio
        }

    def _evaluate_skill_representation(self, latent_data: Dict[str, np.ndarray], test_df: Optional[Any]) -> Dict[str, float]:
        """スキル表現評価"""
        if test_df is None:
            return {}

        from scipy.stats import pearsonr

        z_skill = latent_data['z_skill']

        # パフォーマンス指標との相関
        perf_cols = [col for col in test_df.columns if col.startswith('perf_')]
        if not perf_cols:
            return {}

        performance_df = test_df.groupby(['subject_id', 'trial_num']).first()[perf_cols].reset_index()

        # データ長を合わせる
        min_length = min(len(z_skill), len(performance_df))
        z_skill = z_skill[:min_length]
        performance_df = performance_df.iloc[:min_length]

        correlations = {}
        for col in perf_cols:
            metric_name = col.replace('perf_', '')
            metric_values = performance_df[col].values

            if len(set(metric_values)) > 1 and np.std(metric_values) > 1e-6:
                # 最も相関の高い次元を見つける
                max_corr = 0.0
                for dim in range(z_skill.shape[1]):
                    try:
                        corr, _ = pearsonr(z_skill[:, dim], metric_values)
                        if abs(corr) > abs(max_corr):
                            max_corr = corr
                    except:
                        continue
                correlations[f'{metric_name}_correlation'] = max_corr

        return correlations

    def _safe_decode(self, model: torch.nn.Module, z_style: torch.Tensor, z_skill: torch.Tensor, encoded: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """モデルのスキップ接続対応を動的に判断してデコード

        Args:
            model: モデルインスタンス
            z_style: スタイル潜在変数
            z_skill: スキル潜在変数
            encoded: エンコード結果辞書

        Returns:
            デコード結果辞書
        """
        import inspect

        # Method 1: エンコード結果にスキップ接続があるかチェック
        has_skip_connections = 'skip_connections' in encoded and encoded['skip_connections'] is not None

        # Method 2: decodeメソッドの引数をチェック
        decode_signature = inspect.signature(model.decode)
        accepts_skip_connections = len(decode_signature.parameters) >= 4  # self, z_style, z_skill, skip_connections

        # Method 3: 実際に3引数で呼び出してみる（最も確実）
        if has_skip_connections and accepts_skip_connections:
            try:
                # スキップ接続ありでデコード
                skip_connections = encoded['skip_connections']
                return model.decode(z_style, z_skill, skip_connections)
            except (TypeError, RuntimeError) as e:
                # 3引数でエラーが出た場合は2引数にフォールバック
                print(f"スキップ接続付きデコードが失敗、2引数モードにフォールバック: {e}")
                pass

        # フォールバック: 2引数での従来型デコード
        try:
            return model.decode(z_style, z_skill)
        except Exception as e:
            # さらなるフォールバック: forwardメソッドを使用
            print(f"デコードが失敗、forwardメソッドを使用: {e}")
            # モデルのforwardで再構成を取得
            with torch.no_grad():
                # ダミー入力から形状を推定
                dummy_input = torch.zeros(z_style.size(0), model.seq_len if hasattr(model, 'seq_len') else 100,
                                        6, device=z_style.device)  # 一般的な運動データ次元
                forward_result = model(dummy_input)
                return {'trajectory': forward_result.get('reconstructed', dummy_input)}

    def _create_pca_visualization(self, latent_data: Dict[str, np.ndarray]) -> plt.Figure:
        """PCA可視化のFigureを生成"""
        from sklearn.decomposition import PCA

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # スタイル空間PCA
        if latent_data['z_style'].shape[1] >= 2:
            pca = PCA(n_components=2)
            z_style_pca = pca.fit_transform(latent_data['z_style'])

            unique_subjects = list(set(latent_data['subject_ids']))
            subject_colors = [unique_subjects.index(subj) for subj in latent_data['subject_ids']]

            scatter = axes[0].scatter(z_style_pca[:, 0], z_style_pca[:, 1],
                                      c=subject_colors, cmap='tab10', alpha=0.7)
            axes[0].set_title('Style Space PCA')
            axes[0].set_xlabel('PC1')
            axes[0].set_ylabel('PC2')

        # スキル空間PCA
        if latent_data['z_skill'].shape[1] >= 2:
            z_skill_pca = pca.fit_transform(latent_data['z_skill'])
            axes[1].scatter(z_skill_pca[:, 0], z_skill_pca[:, 1],
                            c=subject_colors, cmap='viridis', alpha=0.7)
            axes[1].set_title('Skill Space PCA')
            axes[1].set_xlabel('PC1')
            axes[1].set_ylabel('PC2')

        plt.tight_layout()
        return fig


