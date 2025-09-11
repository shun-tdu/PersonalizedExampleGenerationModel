# スタイル潜在空間の評価器
from typing import List, Dict, Any, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import ConcatDataset, DataLoader

from base_evaluator import BaseEvaluator
from src.PredictiveLatentSpaceNavigationModel.TransformerBaseEndToEndVAE.evaluator import EnhancedEvaluationResult


class VisualizeStyleSpaceEvaluator(BaseEvaluator):
    """スタイル潜在空間の可視化評価"""
    def __init__(self, config:Dict[str, Any]):
        super().__init__(config)
        self.min_samples_for_tsne = 30

    def evaluate(self, model, test_data, device) -> EnhancedEvaluationResult:
        pass

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

    def evaluate(self, model, test_data, device) -> EnhancedEvaluationResult:
        pass

    def get_required_data(self) -> List[str]:
        pass

class StyleClassificationEvaluator(BaseEvaluator):
    """スタイル潜在変数から簡単なSVM,MLPで被験者の分類が可能化を評価"""
    def __init__(self, config:Dict[str, Any]):
        pass

    def evaluate(self, model, test_data, device) -> EnhancedEvaluationResult:
        pass

    def get_required_data(self) -> List[str]:
        pass


