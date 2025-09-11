# スタイル潜在空間の評価器
from typing import List, Dict, Any, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


from base_evaluator import BaseEvaluator
from src.PredictiveLatentSpaceNavigationModel.TransformerBaseEndToEndVAE.evaluator import EnhancedEvaluationResult


class VisualizeStyleSpaceEvaluator(BaseEvaluator):
    """スタイル潜在空間の可視化評価"""
    def __init__(self, config:Dict[str, Any]):
        super().__init__(config)
        self.min_samples_for_tsne = 30
        style_components = config.get('evaluation').get('style_component')
        self.n_components = style_components if style_components in [2, 3] else 2

    def evaluate(self, model, test_data, result: EnhancedEvaluationResult) -> None:
        """スタイル潜在空間の可視化評価を実行"""
        experiment_id = test_data.get('experiment_id')
        z_style = test_data.get('z_style')
        subject_ids = test_data.get('subject_ids')
        output_dir = test_data.get('output_dir')

        print("=" * 60)
        print("スタイル潜在空間の可視化評価実行")
        print("=" * 60)

        pca_fig, tsne_fig = self._create_style_latent_space_visualizations(z_style=z_style, subject_ids=subject_ids)

        result.add_visualization("style_pca",pca_fig,
                                 description="スタイル潜在空間のPCA可視化",
                                 category="style_analysis")
        result.add_visualization("style_tsne", tsne_fig,
                                 description="スタイル潜在空間のt-SNE可視化",
                                 category="style_analysis")

    def get_required_data(self) -> List[str]:
        return ['z_style', 'experiment_id']

    def _create_style_latent_space_visualizations(self, z_style, subject_ids, n_components=2) -> Union[Tuple[plt.Figure,plt.Figure], Tuple[plotly.graph_objs.Figure,plotly.graph_objs.Figure]]:
        """包括的可視化生成 - 主成分次元が2のときはMatplot Figure、 3のときはPlotly Figureオブジェクトを返す"""
        print(f"\n🎨 可視化生成中...")

        # カラーマップ準備
        subject_to_idx = {subj: i for i, subj in enumerate(subject_ids)}
        subject_colors = [subject_to_idx[subj] for subj in subject_ids]

        # 主成分次元が2のときはMatplot Figureオブジェクトを返す
        if n_components == 2:
            # 1. スタイル空間PCA
            fig_pca, ax_pca = plt.subplots(figsize=(10, 8))
            if z_style.shape[1] >= 2:
                pca_style = PCA(n_components=2)
                z_style_pca = pca_style.fit_transform(z_style)

                scatter = ax_pca.scatter(z_style_pca[:, 0], z_style_pca[:, 1],
                                             c=subject_colors, cmap='tab10', alpha=0.7, s=30)
                ax_pca.set_title(f'Style Space PCA\n(Explained: {pca_style.explained_variance_ratio_.sum():.3f})')
                ax_pca.set_xlabel('PC1')
                ax_pca.set_ylabel('PC2')

                # 重心プロット
                for i, subject in enumerate(subject_ids):
                    mask = np.array(subject_ids) == subject
                    if np.any(mask):
                        center = np.mean(z_style_pca[mask], axis=0)
                        ax_pca.scatter(center[0], center[1], c='red', s=200, marker='x', linewidth=3)
                        ax_pca.annotate(subject, center, xytext=(5, 5), textcoords='offset points')

            # 2. スタイル空間t-SNE
            fig_tsne, ax_tsne = plt.subplots(figsize=(10, 8))
            if len(z_style) >= self.min_samples_for_tsne:
                try:
                    tsne = TSNE(n_components=2, perplexity=min(30, len(z_style) // 4), random_state=42)
                    z_style_tsne = tsne.fit_transform(z_style)

                    ax_tsne.scatter(z_style_tsne[:, 0], z_style_tsne[:, 1],
                                       c=subject_colors, cmap='tab10', alpha=0.7, s=30)
                    ax_tsne.set_title('Style Space t-SNE')
                except Exception as e:
                    ax_tsne.text(0.5, 0.5, f't-SNE Failed: {str(e)}', transform=ax_tsne.transAxes, ha='center')
            else:
                ax_tsne.text(0.5, 0.5, 'Insufficient samples for t-SNE', transform=ax_tsne.transAxes, ha='center')

            return fig_pca, fig_tsne

        elif n_components == 3:
            fig_pca = None
            # 1. スタイル空間PCA
            if z_style.shape[1] > 3:
                pca_style = PCA(n_components=3)
                z_style_pca = pca_style.fit_transform(z_style)

                df_pca = pd.DataFrame({
                    "PC1": z_style_pca[:, 0],
                    "PC2": z_style_pca[:, 1],
                    "PC3": z_style_pca[:, 2],
                    "subject_ids":subject_colors
                })

                fig_pca = px.scatter_3d(
                    df_pca,
                    x="PC1",
                    y="PC2",
                    z="PC3",
                    color='subject_ids',
                    title= f'Style Space PCA\n(Explained: {pca_style.explained_variance_ratio_.sum():.3f})',
                    hover_data=['subject_ids']
                )
                fig_pca.update_layout(title_x=0.5)
                fig_pca.update_traces(marker=dict(size=4, opacity=0.8))


            # 2. スタイル空間t-SNE
            fig_tsne = None
            if len(z_style) >= self.min_samples_for_tsne:
                try:
                    tsne = TSNE(n_components=3, perplexity=min(30, len(z_style) // 4), random_state=42)
                    z_style_tsne = tsne.fit_transform(z_style)

                    df_tsne = pd.DataFrame({
                        "PC1": z_style_tsne[:, 0],
                        "PC2": z_style_tsne[:, 1],
                        "PC3": z_style_tsne[:, 2],
                        "subject_ids":subject_colors
                    })

                    fig_tsne = px.scatter_3d(
                        df_tsne,
                        x="PC1",
                        y="PC2",
                        z="PC3",
                        color='subject_ids',
                        title=f'Style Space t-SNE(3D)',
                        hover_data=['subject_ids']
                    )
                    fig_tsne.update_layout(title_x=0.5)
                    fig_tsne.update_traces(marker=dict(size=4, opacity=0.8))

                except Exception as e:
                    # エラーが発生した場合は空のFigureを作成し、テキストで通知
                    fig_tsne = plotly.graph_objs.Figure()
                    fig_tsne.add_annotation(text=f"t-SNE Failed: {e}", showarrow=False, font=dict(size=16))
                    fig_tsne.update_layout(title_text="Style Space t-SNE (3D) - Failed")
            else:
                # サンプル数が不足している場合も同様
                fig_tsne = plotly.graph_objs.Figure()
                fig_tsne.add_annotation(text="Insufficient samples for t-SNE", showarrow=False, font=dict(size=16))
                fig_tsne.update_layout(title_text="Style Space t-SNE (3D) - Skipped")

            return fig_pca, fig_tsne
        else:
            raise ValueError(f"主成分分析の主成分次元数が不適切です．2か3を選択してください．")


class StyleClusteringEvaluator(BaseEvaluator):
    """スタイル潜在空間内のクラスタリング性能の評価"""
    def __init__(self, config:Dict[str, Any]):
        pass

    def evaluate(self, model, test_data, result: EnhancedEvaluationResult) -> None:
        """スタイル潜在空間のクラスタリング評価を実行"""
        z_style = test_data.get('z_style')
        subject_ids = test_data.get('subject_ids')

        print("=" * 60)
        print("スタイル潜在空間クラスタリング評価実行")
        print("=" * 60)

        # 被験者情報の準備
        unique_subjects = list(set(subject_ids))
        n_subjects = len(unique_subjects)

        if n_subjects < self.min_subjects_for_clustering:
            print(f"⚠️ 被験者数不足: {n_subjects} < {self.min_subjects_for_clustering}")
            result.add_metric("clustering_status", 0, "被験者数不足", "clustering")
            return

        # 真のクラスタラベル作成
        subject_to_idx = {subj: i for i, subj in enumerate(unique_subjects)}
        true_labels = [subject_to_idx[subj] for subj in subject_ids]

        # K-meansクラスタリング実行
        clustering_results = self._perform_kmeans_clustering(z_style, true_labels, n_subjects)

        # シルエットスコア評価
        silhouette_results = self._perform_silhouette_score_evaluation(z_style, true_labels, n_subjects)

        # 調整ランド指標評価
        ari_results = self._perform_adjusted_rand_index_evaluation(z_style, true_labels, n_subjects)

        # 結果をメトリクスに追加
        self._add_clustering_metrics(result, clustering_results, silhouette_results, ari_results)

        print("✅ スタイル潜在空間クラスタリング評価完了")

    def _perform_kmeans_clustering(self, z_style, true_labels, n_clusters):
        """K-meansクラスタリングを実行"""

        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            predicted_labels = kmeans.fit_predict(z_style)

            print(f"📊 K-means完了: {n_clusters}クラスタ")
            return {
                'predicted_labels': predicted_labels,
                'cluster_centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_,
                'success': True
            }
        except Exception as e:
            print(f"❌ K-meansエラー: {e}")
            return {'success': False, 'error': str(e)}

    def _perform_silhouette_score_evaluation(self, z_style, true_labels, n_clusters):
        """シルエットスコア評価"""
        from sklearn.metrics import silhouette_score, silhouette_samples
        from sklearn.cluster import KMeans

        results = {}

        try:
            # K-meansでクラスタリング
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            predicted_labels = kmeans.fit_predict(z_style)

            # 全体のシルエットスコア
            overall_silhouette = silhouette_score(z_style, predicted_labels)

            # サンプルごとのシルエットスコア
            sample_silhouette = silhouette_samples(z_style, predicted_labels)

            # クラスタごとの平均シルエットスコア
            cluster_silhouettes = {}
            for i in range(n_clusters):
                cluster_mask = predicted_labels == i
                if np.any(cluster_mask):
                    cluster_silhouettes[i] = np.mean(sample_silhouette[cluster_mask])

            # 真のラベルでのシルエットスコア（参考値）
            true_silhouette = silhouette_score(z_style, true_labels) if len(set(true_labels)) > 1 else 0.0

            print(f"🎯 シルエットスコア:")
            print(f"  予測クラスタ: {overall_silhouette:.4f}")
            print(f"  真のラベル: {true_silhouette:.4f}")

            # 判定
            if overall_silhouette > 0.7:
                silhouette_status = "優秀"
            elif overall_silhouette > 0.5:
                silhouette_status = "良好"
            elif overall_silhouette > 0.25:
                silhouette_status = "普通"
            else:
                silhouette_status = "不良"

            print(f"  判定: {silhouette_status}")

            results = {
                'overall_silhouette': overall_silhouette,
                'true_label_silhouette': true_silhouette,
                'cluster_silhouettes': cluster_silhouettes,
                'sample_silhouettes': sample_silhouette,
                'silhouette_status': silhouette_status,
                'success': True
            }

        except Exception as e:
            print(f"❌ シルエットスコア計算エラー: {e}")
            results = {'success': False, 'error': str(e)}

        return results

    def _perform_adjusted_rand_index_evaluation(self, z_style, true_labels, n_clusters):
        """調整ランド指標評価"""
        from sklearn.metrics import adjusted_rand_score
        from sklearn.cluster import KMeans

        results = {}

        try:
            # K-meansでクラスタリング
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            predicted_labels = kmeans.fit_predict(z_style)

            # 調整ランド指標計算
            ari = adjusted_rand_score(true_labels, predicted_labels)

            print(f"🎲 調整ランド指標 (ARI): {ari:.4f}")

            # 判定
            if ari > 0.75:
                ari_status = "優秀"
            elif ari > 0.5:
                ari_status = "良好"
            elif ari > 0.25:
                ari_status = "普通"
            else:
                ari_status = "不良"

            print(f"  判定: {ari_status}")

            # 追加の分析: クラスタ純度
            cluster_purities = self._calculate_cluster_purity(true_labels, predicted_labels, n_clusters)

            results = {
                'ari': ari,
                'ari_status': ari_status,
                'predicted_labels': predicted_labels,
                'cluster_purities': cluster_purities,
                'success': True
            }

        except Exception as e:
            print(f"❌ ARI計算エラー: {e}")
            results = {'success': False, 'error': str(e)}

        return results

    def _calculate_cluster_purity(self, true_labels, predicted_labels, n_clusters):
        """クラスタ純度の計算"""
        purities = {}

        for cluster_id in range(n_clusters):
            cluster_mask = np.array(predicted_labels) == cluster_id
            if np.any(cluster_mask):
                cluster_true_labels = np.array(true_labels)[cluster_mask]
                # 最も多い真のラベルの割合
                unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
                max_count = np.max(counts)
                purity = max_count / len(cluster_true_labels)
                purities[cluster_id] = {
                    'purity': purity,
                    'dominant_label': unique_labels[np.argmax(counts)],
                    'size': len(cluster_true_labels)
                }

        return purities

    def _add_clustering_metrics(self, result, clustering_results, silhouette_results, ari_results):
        """クラスタリングメトリクスを結果に追加"""
        # シルエットスコアメトリクス
        if silhouette_results.get('success', False):
            result.add_metric('style_clustering_silhouette',
                              silhouette_results['overall_silhouette'],
                              'スタイル潜在空間のシルエットスコア', 'clustering')
            result.add_metric('style_clustering_true_silhouette',
                              silhouette_results['true_label_silhouette'],
                              '真のラベルでのシルエットスコア（参考）', 'clustering')

        # ARI メトリクス
        if ari_results.get('success', False):
            result.add_metric('style_clustering_ari',
                              ari_results['ari'],
                              'スタイル潜在空間の調整ランド指標', 'clustering')

        # K-meansメトリクス
        if clustering_results.get('success', False):
            result.add_metric('style_clustering_inertia',
                              clustering_results['inertia'],
                              'K-meansクラスタ内二乗和', 'clustering')

    def get_required_data(self) -> List[str]:
        return ['z_style', 'subject_ids', 'experiment_id']


class StyleClassificationEvaluator(BaseEvaluator):
    """スタイル潜在変数から簡単なSVM,MLPで被験者の分類が可能化を評価"""
    def __init__(self, config:Dict[str, Any]):
        pass

    def evaluate(self, model, test_data, device) -> EnhancedEvaluationResult:
        pass

    def get_required_data(self) -> List[str]:
        pass


