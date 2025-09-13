# スタイル潜在空間の評価器
from typing import List, Dict, Any, Union, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import plotly
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# CLAUDE_ADDED: 日本語フォント警告を回避するためのmatplotlib設定
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']

from .base_evaluator import BaseEvaluator
from .result_manager import EnhancedEvaluationResult


class VisualizeStyleSpaceEvaluator(BaseEvaluator):
    """スタイル潜在空間の可視化評価"""
    def __init__(self, config:Dict[str, Any]):
        super().__init__(config)
        self.min_samples_for_tsne = 30
        style_components = config.get('evaluation').get('style_component')
        self.n_components = style_components if style_components in [2, 3] else 2

    def evaluate(self, model: torch.nn.Module, test_data: Dict[str, Any], device: torch.device, result: EnhancedEvaluationResult):
        """スタイル潜在空間の可視化評価を実行"""
        z_style = test_data.get('z_style')
        subject_ids = test_data.get('subject_ids')

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
        
        print("✅ スタイル潜在空間可視化評価完了")

    def get_required_data(self) -> List[str]:
        return ['z_style', 'experiment_id']

    def _create_style_latent_space_visualizations(self, z_style: np.ndarray, subject_ids: List[str], n_components: int = 2) -> Union[Tuple[plt.Figure,plt.Figure], Tuple[plotly.graph_objs.Figure,plotly.graph_objs.Figure]]:
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
        super().__init__(config)
        self.min_subjects_for_clustering = 2

    def evaluate(self, model: torch.nn.Module, test_data: Dict[str, Any], device: torch.device, result: EnhancedEvaluationResult):
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
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.test_size = 0.2
        self.cv_folds = 5
        self.random_state = 42
        self.min_samples_per_subject = 3
        self.min_subjects = 2

    def evaluate(self, model: torch.nn.Module, test_data: Dict[str, Any], device: torch.device, result: EnhancedEvaluationResult):
        """スタイル潜在変数からの被験者分類評価を実行"""
        z_style = test_data.get('z_style')
        subject_ids = test_data.get('subject_ids')

        print("=" * 60)
        print("被験者分類評価実行 (MLP & SVM)")
        print("=" * 60)

        # データ妥当性チェック
        unique_subjects = list(set(subject_ids))
        subject_counts = {subj: subject_ids.count(subj) for subj in unique_subjects}
        
        if len(unique_subjects) < self.min_subjects:
            print(f"⚠️ 被験者数不足: {len(unique_subjects)} < {self.min_subjects}")
            result.add_metric("classification_status", 0, "被験者数不足", "style_classification")
            return
            
        insufficient_samples = [subj for subj, count in subject_counts.items() 
                               if count < self.min_samples_per_subject]
        if insufficient_samples:
            print(f"⚠️ 一部被験者のサンプル不足: {insufficient_samples}")
            
        # データ前処理
        X_processed, y_processed, label_mapping = self._preprocess_classification_data(z_style, subject_ids)
        
        # 1. MLP分類評価
        mlp_results = self._evaluate_mlp_classification(X_processed, y_processed, label_mapping)
        
        # 2. SVM分類評価
        svm_results = self._evaluate_svm_classification(X_processed, y_processed, label_mapping)
        
        # 3. ベースライン比較（ランダムフォレスト）
        baseline_results = self._evaluate_baseline_classification(X_processed, y_processed, label_mapping)
        
        # 4. 交差検証による頑健性評価
        cv_results = self._perform_classification_cross_validation(X_processed, y_processed, label_mapping)
        
        # 5. 可視化生成
        visualization_fig = self._create_classification_visualization(
            X_processed, y_processed, label_mapping, mlp_results, svm_results, baseline_results
        )
        
        # 結果をメトリクスに追加
        self._add_classification_metrics(result, mlp_results, svm_results, baseline_results, cv_results)
        
        # 可視化を追加
        result.add_visualization("style_classification_analysis", visualization_fig,
                                description="MLPとSVMによる被験者分類性能分析",
                                category="style_analysis")
        
        print("✅ 被験者分類評価完了")

    def _preprocess_classification_data(self, z_style, subject_ids):
        """分類用データ前処理"""
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        print(f"\n📋 分類データ前処理...")
        
        # 被験者情報
        unique_subjects = list(set(subject_ids))
        subject_counts = {subj: subject_ids.count(subj) for subj in unique_subjects}
        
        print(f"  被験者数: {len(unique_subjects)}")
        for subj, count in subject_counts.items():
            print(f"    {subj}: {count}サンプル")
        
        # ラベルエンコーディング
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(subject_ids)
        label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
        
        # 特徴量の標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(z_style)
        
        # NaN除去
        valid_indices = ~(np.isnan(X_scaled).any(axis=1) | np.isnan(y_encoded))
        X_clean = X_scaled[valid_indices]
        y_clean = y_encoded[valid_indices]
        
        print(f"  前処理後サンプル数: {len(y_clean)}")
        print(f"  特徴量次元: {X_clean.shape[1]}")
        
        return X_clean, y_clean, label_mapping

    def _evaluate_mlp_classification(self, X, y, label_mapping):
        """MLP分類評価"""
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
        
        print(f"\n🧠 MLP分類評価...")
        
        try:
            # 訓練・テスト分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, 
                stratify=y if len(np.unique(y)) > 1 else None
            )
            
            # 複数のMLPアーキテクチャを試す
            mlp_configs = [
                {'hidden_layer_sizes': (50,), 'name': 'MLP-50'},
                {'hidden_layer_sizes': (100,), 'name': 'MLP-100'},
                {'hidden_layer_sizes': (50, 25), 'name': 'MLP-50-25'},
                {'hidden_layer_sizes': (100, 50), 'name': 'MLP-100-50'},
            ]
            
            best_mlp_result = None
            best_accuracy = -1
            
            for config in mlp_configs:
                try:
                    mlp = MLPClassifier(
                        hidden_layer_sizes=config['hidden_layer_sizes'],
                        activation='relu',
                        solver='adam',
                        alpha=0.001,
                        max_iter=1000,
                        random_state=self.random_state,
                        early_stopping=True,
                        validation_fraction=0.1
                    )
                    
                    mlp.fit(X_train, y_train)
                    y_pred = mlp.predict(X_test)
                    
                    # 評価指標
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    # 訓練データでの性能（過学習チェック）
                    y_train_pred = mlp.predict(X_train)
                    train_accuracy = accuracy_score(y_train, y_train_pred)
                    
                    # 混同行列
                    cm = confusion_matrix(y_test, y_pred)
                    
                    result_config = {
                        'name': config['name'],
                        'architecture': config['hidden_layer_sizes'],
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'train_accuracy': train_accuracy,
                        'overfitting': train_accuracy - accuracy,
                        'confusion_matrix': cm,
                        'y_pred': y_pred,
                        'y_test': y_test,
                        'model': mlp,
                        'success': True
                    }
                    
                    print(f"  {config['name']}: Acc={accuracy:.4f}, F1={f1:.4f}, 過学習={train_accuracy-accuracy:.4f}")
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_mlp_result = result_config
                        
                except Exception as e:
                    print(f"  {config['name']}: エラー {e}")
                    continue
            
            if best_mlp_result:
                print(f"  最優秀MLP: {best_mlp_result['name']} (Acc={best_mlp_result['accuracy']:.4f})")
                return best_mlp_result
            else:
                return {'success': False, 'error': 'すべてのMLP構成が失敗'}
                
        except Exception as e:
            print(f"  ❌ MLP評価エラー: {e}")
            return {'success': False, 'error': str(e)}

    def _evaluate_svm_classification(self, X, y, label_mapping):
        """SVM分類評価"""
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        print(f"\n📐 SVM分類評価...")
        
        try:
            # 訓練・テスト分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state,
                stratify=y if len(np.unique(y)) > 1 else None
            )
            
            # SVM カーネルと パラメータ
            svm_configs = [
                {'kernel': 'linear', 'name': 'SVM-Linear'},
                {'kernel': 'rbf', 'name': 'SVM-RBF'},
                {'kernel': 'poly', 'degree': 2, 'name': 'SVM-Poly2'},
            ]
            
            best_svm_result = None
            best_accuracy = -1
            
            for config in svm_configs:
                try:
                    # ハイパーパラメータ探索
                    if config['kernel'] == 'linear':
                        param_grid = {'C': [0.1, 1, 10]}
                    else:
                        param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.1]}
                    
                    base_svm = SVC(kernel=config['kernel'], random_state=self.random_state,
                                  **{k: v for k, v in config.items() if k not in ['kernel', 'name']})
                    
                    # 小規模データセットの場合は簡単な評価
                    if len(X_train) < 50:
                        svm = base_svm
                        if config['kernel'] != 'linear':
                            svm.set_params(C=1, gamma='scale')
                        else:
                            svm.set_params(C=1)
                    else:
                        grid_search = GridSearchCV(base_svm, param_grid, cv=min(3, self.cv_folds), 
                                                 scoring='accuracy', n_jobs=-1)
                        grid_search.fit(X_train, y_train)
                        svm = grid_search.best_estimator_
                    
                    svm.fit(X_train, y_train)
                    y_pred = svm.predict(X_test)
                    
                    # 評価指標
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    # 訓練データでの性能
                    y_train_pred = svm.predict(X_train)
                    train_accuracy = accuracy_score(y_train, y_train_pred)
                    
                    # 混同行列
                    cm = confusion_matrix(y_test, y_pred)
                    
                    result_config = {
                        'name': config['name'],
                        'kernel': config['kernel'],
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'train_accuracy': train_accuracy,
                        'overfitting': train_accuracy - accuracy,
                        'confusion_matrix': cm,
                        'y_pred': y_pred,
                        'y_test': y_test,
                        'model': svm,
                        'success': True
                    }
                    
                    print(f"  {config['name']}: Acc={accuracy:.4f}, F1={f1:.4f}, 過学習={train_accuracy-accuracy:.4f}")
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_svm_result = result_config
                        
                except Exception as e:
                    print(f"  {config['name']}: エラー {e}")
                    continue
            
            if best_svm_result:
                print(f"  最優秀SVM: {best_svm_result['name']} (Acc={best_svm_result['accuracy']:.4f})")
                return best_svm_result
            else:
                return {'success': False, 'error': 'すべてのSVM構成が失敗'}
                
        except Exception as e:
            print(f"  ❌ SVM評価エラー: {e}")
            return {'success': False, 'error': str(e)}

    def _evaluate_baseline_classification(self, X, y, label_mapping):
        """ベースライン分類評価（ランダムフォレスト）"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        print(f"\n🌲 ベースライン分類評価...")
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state,
                stratify=y if len(np.unique(y)) > 1 else None
            )
            
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            print(f"  RandomForest: Acc={accuracy:.4f}, F1={f1:.4f}")
            
            return {
                'name': 'RandomForest',
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm,
                'y_pred': y_pred,
                'y_test': y_test,
                'model': rf,
                'success': True
            }
            
        except Exception as e:
            print(f"  ❌ ベースライン評価エラー: {e}")
            return {'success': False, 'error': str(e)}

    def _perform_classification_cross_validation(self, X, y, label_mapping):
        """分類交差検証による頑健性評価"""
        from sklearn.model_selection import cross_val_score
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        
        print(f"\n🔄 分類交差検証評価...")
        
        cv_results = {}
        models = {
            'MLP': MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=self.random_state),
            'SVM': SVC(kernel='rbf', C=1, gamma='scale', random_state=self.random_state),
            'RandomForest': RandomForestClassifier(n_estimators=50, random_state=self.random_state)
        }
        
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=min(self.cv_folds, len(y)//4), 
                                       scoring='accuracy', n_jobs=-1)
                cv_results[name] = {
                    'mean_accuracy': np.mean(scores),
                    'std_accuracy': np.std(scores),
                    'scores': scores,
                    'success': True
                }
                print(f"  {name}: Acc = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
                
            except Exception as e:
                print(f"  {name}: CV エラー {e}")
                cv_results[name] = {'success': False, 'error': str(e)}
        
        return cv_results

    def _create_classification_visualization(self, X, y, label_mapping, mlp_results, svm_results, baseline_results):
        """分類性能の可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 結果リスト
        results = [
            (mlp_results, 'MLP'),
            (svm_results, 'SVM'), 
            (baseline_results, 'RandomForest')
        ]
        
        # 成功した結果のみフィルタ
        successful_results = [(r, name) for r, name in results if r.get('success', False)]
        
        if not successful_results:
            axes[0, 0].text(0.5, 0.5, 'すべてのモデルが失敗', ha='center', va='center', transform=axes[0, 0].transAxes)
            return fig
        
        # 1. 混同行列（最良モデル）
        best_result = max(successful_results, key=lambda x: x[0].get('accuracy', -1))
        if best_result and 'confusion_matrix' in best_result[0]:
            result, name = best_result
            cm = result['confusion_matrix']
            subjects = [label_mapping[i] for i in range(len(label_mapping))]
            
            im = axes[0, 0].imshow(cm, interpolation='nearest', cmap='Blues')
            axes[0, 0].set_title(f'Confusion Matrix - {name}\n(Acc={result.get("accuracy", 0):.3f})')
            
            # ラベル設定
            tick_marks = np.arange(len(subjects))
            axes[0, 0].set_xticks(tick_marks)
            axes[0, 0].set_yticks(tick_marks)
            axes[0, 0].set_xticklabels(subjects, rotation=45)
            axes[0, 0].set_yticklabels(subjects)
            
            # 数値表示
            thresh = cm.max() / 2.
            for i, j in np.ndindex(cm.shape):
                axes[0, 0].text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black")
            
            axes[0, 0].set_ylabel('True Subject')
            axes[0, 0].set_xlabel('Predicted Subject')
        
        # 2. 性能比較（精度）
        if len(successful_results) > 1:
            names = [name for _, name in successful_results]
            accuracies = [result.get('accuracy', 0) for result, _ in successful_results]
            f1_scores = [result.get('f1', 0) for result, _ in successful_results]
            
            x_pos = np.arange(len(names))
            
            axes[0, 1].bar(x_pos - 0.2, accuracies, 0.4, label='Accuracy', alpha=0.7)
            axes[0, 1].bar(x_pos + 0.2, f1_scores, 0.4, label='F1 Score', alpha=0.7)
            axes[0, 1].set_xlabel('Models')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_title('Performance Comparison')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(names)
            axes[0, 1].legend()
            
            # 数値表示
            for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
                axes[0, 1].text(i-0.2, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
                axes[0, 1].text(i+0.2, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom')
        
        # 3. 特徴量重要度（RandomForestがある場合）
        rf_result = next(((r, name) for r, name in successful_results if name == 'RandomForest'), None)
        if rf_result and rf_result[0].get('success', False):
            model = rf_result[0]['model']
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:10]  # 上位10特徴量
                
                axes[0, 2].bar(range(len(indices)), importances[indices])
                axes[0, 2].set_title('Feature Importance (RandomForest)')
                axes[0, 2].set_xlabel('Feature Index')
                axes[0, 2].set_ylabel('Importance')
                axes[0, 2].set_xticks(range(len(indices)))
                axes[0, 2].set_xticklabels([f'Dim{i+1}' for i in indices], rotation=45)
        
        # 4. 2D可視化（PCA）
        from sklearn.decomposition import PCA
        if X.shape[1] >= 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            subjects = [label_mapping[i] for i in sorted(label_mapping.keys())]
            colors = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
            
            for i, subject in enumerate(subjects):
                mask = y == i
                if np.any(mask):
                    axes[1, 0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                                     c=[colors[i]], label=subject, alpha=0.7, s=30)
            
            axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
            axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
            axes[1, 0].set_title('2D PCA Projection')
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 5. 被験者別サンプル数
        subject_counts = np.bincount(y)
        subjects = [label_mapping[i] for i in range(len(subject_counts))]
        
        axes[1, 1].bar(range(len(subject_counts)), subject_counts)
        axes[1, 1].set_xlabel('Subject')
        axes[1, 1].set_ylabel('Sample Count')
        axes[1, 1].set_title('Samples per Subject')
        axes[1, 1].set_xticks(range(len(subjects)))
        axes[1, 1].set_xticklabels(subjects, rotation=45)
        
        # 数値表示
        for i, count in enumerate(subject_counts):
            axes[1, 1].text(i, count + 0.5, str(count), ha='center', va='bottom')
        
        # 6. 統計サマリー
        axes[1, 2].axis('off')
        summary_text = "分類性能サマリー\n" + "="*20 + "\n"
        
        for result, name in successful_results:
            summary_text += f"{name}:\n"
            summary_text += f"  Accuracy = {result.get('accuracy', 0):.3f}\n"
            summary_text += f"  Precision = {result.get('precision', 0):.3f}\n"
            summary_text += f"  Recall = {result.get('recall', 0):.3f}\n"
            summary_text += f"  F1-Score = {result.get('f1', 0):.3f}\n"
            if 'overfitting' in result:
                summary_text += f"  過学習 = {result.get('overfitting', 0):.3f}\n"
            summary_text += "\n"
        
        summary_text += f"被験者数: {len(label_mapping)}\n"
        summary_text += f"総サンプル数: {len(y)}"
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        return fig

    def _add_classification_metrics(self, result, mlp_results, svm_results, baseline_results, cv_results):
        """分類メトリクスを結果に追加"""
        # MLP結果
        if mlp_results.get('success', False):
            result.add_metric('style_mlp_accuracy', mlp_results.get('accuracy', 0),
                            'MLP分類精度', 'style_classification')
            result.add_metric('style_mlp_f1', mlp_results.get('f1', 0),
                            'MLP F1スコア', 'style_classification')
            result.add_metric('style_mlp_overfitting', mlp_results.get('overfitting', 0),
                            'MLPの過学習度', 'style_classification')
        
        # SVM結果  
        if svm_results.get('success', False):
            result.add_metric('style_svm_accuracy', svm_results.get('accuracy', 0),
                            'SVM分類精度', 'style_classification')
            result.add_metric('style_svm_f1', svm_results.get('f1', 0),
                            'SVM F1スコア', 'style_classification')
        
        # ベースライン結果
        if baseline_results.get('success', False):
            result.add_metric('style_rf_accuracy', baseline_results.get('accuracy', 0),
                            'RandomForest分類精度', 'style_classification')
        
        # 最優秀モデル
        all_results = [mlp_results, svm_results, baseline_results]
        successful = [r for r in all_results if r.get('success', False)]
        if successful:
            best_model = max(successful, key=lambda x: x.get('accuracy', -1))
            result.add_metric('style_best_classification_accuracy', best_model.get('accuracy', 0),
                            '最優秀分類モデルの精度', 'style_classification')
        
        # 交差検証結果
        for name, cv_result in cv_results.items():
            if cv_result.get('success', False):
                result.add_metric(f'style_{name.lower()}_cv_accuracy_mean', cv_result.get('mean_accuracy', 0),
                                f'{name}のCV平均精度', 'style_classification')
                result.add_metric(f'style_{name.lower()}_cv_accuracy_std', cv_result.get('std_accuracy', 0),
                                f'{name}のCV精度標準偏差', 'style_classification')

    def get_required_data(self) -> List[str]:
        return ['z_style', 'subject_ids', 'experiment_id']


