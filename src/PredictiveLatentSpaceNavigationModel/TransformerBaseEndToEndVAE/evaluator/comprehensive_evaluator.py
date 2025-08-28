import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import ConcatDataset, DataLoader

from .base_evaluator import BaseEvaluator
from .result_manager import EnhancedEvaluationResult


class ComprehensiveLatentSpaceEvaluator(BaseEvaluator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_samples_for_tsne = 30
        self.min_subjects_for_clustering = 2

    def get_required_data(self) -> List[str]:
        return ['test_loader', 'train_dataset', 'val_dataset', 'test_dataset', 'output_dir', 'experiment_id']

    def evaluate(self, model, test_data, device) -> EnhancedEvaluationResult:
        """包括的潜在空間評価を実行"""
        experiment_id = test_data.get('experiment_id')
        output_dir = test_data['output_dir']
        train_dataset = test_data['train_dataset']
        val_dataset = test_data['val_dataset']
        test_dataset = test_data['test_dataset']

        train_config = self.config.get('training', {})

        result = EnhancedEvaluationResult(experiment_id, output_dir)

        print("=" * 60)
        print("包括的潜在空間分析開始")
        print("=" * 60)

        # 全データセットを結合
        all_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
        all_dataloader = DataLoader(all_dataset, train_config.get('batch_size', 16))

        # 全データセットから潜在変数を抽出
        extracted_latent_variables = self._extract_latent_variables(model, all_dataloader, device)

        if not extracted_latent_variables:
            result.add_metric('analysis_status', 0, 'データ抽出失敗', 'error')
            return result

        # 詳細分析を実行
        analysis_results = self._perform_detailed_analysis(extracted_latent_variables, output_dir, experiment_id)

        #結果をEnhancedEvaluationResultに格納
        self._populate_evaluation_result(result, analysis_results)

        # HTMLレポート生成
        report_path = result.create_comprehensive_report()

        return result


    def _extract_latent_variables(self, model, all_dataloader, device)->Dict[str, Dict]:
        """全データセットから潜在変数を抽出"""
        model.eval()

        z_style_list = []
        z_skill_list = []
        subject_ids = []
        is_expert_list = []

        result = {}

        with torch.no_grad():
            for batch_data in all_dataloader:
                trajectories = batch_data[0].to(device)
                subj_ids = batch_data[1] if len(batch_data) > 1 else [f"subj_{i}" for i in range(len(trajectories))]
                is_expert = batch_data[2] if len(batch_data) > 2 else torch.zeros(len(trajectories))

                encoded = model.encode(trajectories)
                z_style_list.append(encoded['z_style'].cpu().numpy())
                z_skill_list.append(encoded['z_skill'].cpu().numpy())
                subject_ids.extend(subj_ids)
                is_expert_list.extend(is_expert.cpu().numpy())

        if z_style_list:
            result = {
                'z_style': np.vstack(z_style_list),
                'z_skill': np.vstack(z_skill_list),
                'subject_ids': subject_ids,
                'is_expert': np.array(is_expert_list)
            }

            print(f"  ✅ {len(subject_ids)}サンプル, {len(set(subject_ids))}被験者")

        return result

    def _perform_detailed_analysis(self, data, output_dir, experiment_id) -> Dict[str, Any]:
        """詳細な潜在空間分析を実行"""

        z_style = data['z_style']
        z_skill = data['z_skill']
        subject_ids = data['subject_ids']
        is_expert = data['is_expert']

        # 被験者情報
        unique_subjects = list(set(subject_ids))
        n_subjects = len(unique_subjects)
        subject_to_idx = {subj: i for i, subj in enumerate(unique_subjects)}
        subject_labels = [subject_to_idx[subj] for subj in subject_ids]

        print(f"\n📈 潜在空間統計:")
        print(f"  総サンプル数: {len(z_style)}")
        print(f"  被験者数: {n_subjects}")
        print(f"  スタイル次元: {z_style.shape[1]}")
        print(f"  スキル次元: {z_skill.shape[1]}")

        # 分析実行
        style_analysis = self._analyze_style_separation(z_style, subject_ids, unique_subjects)
        skill_analysis = self._analyze_skill_distribution(z_skill, is_expert)
        clustering_analysis = self._evaluate_clustering_performance(z_style, subject_labels, n_subjects)

        # 可視化生成
        visualization_fig = self._create_comprehensive_visualizations(
            z_style, z_skill, subject_ids, is_expert, unique_subjects,
            output_dir, experiment_id
        )

        # 結果統合
        results = {
            'total_samples': len(z_style),
            'n_subjects': n_subjects,
            'unique_subjects': unique_subjects,
            'style_analysis': style_analysis,
            'skill_analysis': skill_analysis,
            'clustering_analysis': clustering_analysis,
            'visualization_fig': visualization_fig
        }

        self._print_comprehensive_summary(results)

        return results

    def _analyze_style_separation(self, z_style, subject_ids, unique_subjects) -> Dict[str, Any]:
        """スタイル分離の詳細分析"""
        print(f"\n🎨 スタイル分離分析:")

        # 被験者ごとの統計
        subject_stats = {}
        for subject in unique_subjects:
            mask = np.array(subject_ids) == subject
            subject_data = z_style[mask]

            if len(subject_data) > 0:
                stats = {
                    'mean': np.mean(subject_data, axis=0),
                    'std': np.std(subject_data, axis=0),
                    'n_samples': len(subject_data),
                    'centroid': np.mean(subject_data, axis=0)
                }
                subject_stats[subject] = stats
                print(f"  {subject}: {len(subject_data)}サンプル")

        # 被験者間距離計算
        if len(unique_subjects) > 1:
            centroids = np.array([stats['centroid'] for stats in subject_stats.values()])
            inter_distances = pdist(centroids, metric='euclidean')
            distance_matrix = squareform(inter_distances)

            # 被験者内分散の平均
            within_variances = []
            for stats in subject_stats.values():
                if stats['n_samples'] > 1:
                    within_variances.append(np.mean(stats['std'] ** 2))

            avg_within_var = np.mean(within_variances) if within_variances else 0.001
            avg_between_dist = np.mean(inter_distances)

            # 分離指標
            separation_ratio = avg_between_dist / (np.sqrt(avg_within_var) + 1e-8)

            print(f"  平均被験者間距離: {avg_between_dist:.4f}")
            print(f"  平均被験者内分散: {avg_within_var:.4f}")
            print(f"  分離指標: {separation_ratio:.4f}")

            # 判定
            if separation_ratio > 3.0:
                status = "優秀な分離"
            elif separation_ratio > 1.5:
                status = "中程度の分離"
            elif separation_ratio > 0.5:
                status = "弱い分離"
            else:
                status = "分離不十分"

            print(f"  判定: {status}")

            return {
                'separation_ratio': separation_ratio,
                'avg_between_distance': avg_between_dist,
                'avg_within_variance': avg_within_var,
                'distance_matrix': distance_matrix.tolist(),
                'subject_stats': {k: {sk: sv.tolist() if isinstance(sv, np.ndarray) else sv
                                      for sk, sv in v.items()} for k, v in subject_stats.items()},
                'status': status
            }

        return {'status': '分析不可（被験者数不足）'}

    def _analyze_skill_distribution(self, z_skill, is_expert) -> Dict[str, Any]:
        """スキル分布の分析"""
        print(f"\n🏆 スキル分布分析:")

        if len(set(is_expert)) < 2:
            print("  ⚠️ 熟練度ラベルの多様性不足")
            return {'status': 'insufficient_labels'}

        expert_mask = is_expert == 1
        novice_mask = is_expert == 0

        expert_data = z_skill[expert_mask]
        novice_data = z_skill[novice_mask]

        print(f"  熟練者: {len(expert_data)}サンプル")
        print(f"  初心者: {len(novice_data)}サンプル")

        if len(expert_data) > 0 and len(novice_data) > 0:
            expert_centroid = np.mean(expert_data, axis=0)
            novice_centroid = np.mean(novice_data, axis=0)

            skill_separation = np.linalg.norm(expert_centroid - novice_centroid)

            expert_variance = np.mean(np.var(expert_data, axis=0))
            novice_variance = np.mean(np.var(novice_data, axis=0))
            avg_variance = (expert_variance + novice_variance) / 2

            skill_ratio = skill_separation / (np.sqrt(avg_variance) + 1e-8)

            print(f"  熟練度分離距離: {skill_separation:.4f}")
            print(f"  スキル分離比: {skill_ratio:.4f}")

            return {
                'skill_separation': skill_separation,
                'skill_ratio': skill_ratio,
                'expert_centroid': expert_centroid.tolist(),
                'novice_centroid': novice_centroid.tolist()
            }

        return {'status': 'insufficient_data'}

    def _evaluate_clustering_performance(self, z_style, subject_labels, n_subjects) -> Dict[str, Any]:
        """クラスタリング性能評価"""
        print(f"\n🎯 クラスタリング性能評価:")

        if n_subjects < self.min_subjects_for_clustering:
            print("  ⚠️ クラスタリング評価不可（被験者数不足）")
            return {'status': 'insufficient_subjects'}

        try:
            # K-meansクラスタリング
            kmeans = KMeans(n_clusters=n_subjects, random_state=42, n_init=10)
            predicted_labels = kmeans.fit_predict(z_style)

            # 評価指標
            ari = adjusted_rand_score(subject_labels, predicted_labels)
            silhouette = silhouette_score(z_style, predicted_labels)

            print(f"  Adjusted Rand Index: {ari:.4f}")
            print(f"  Silhouette Score: {silhouette:.4f}")

            # 判定
            if ari > 0.7:
                ari_status = "優秀"
            elif ari > 0.3:
                ari_status = "良好"
            elif ari > 0.1:
                ari_status = "普通"
            else:
                ari_status = "不良"

            print(f"  ARI判定: {ari_status}")

            return {
                'ari': ari,
                'silhouette': silhouette,
                'predicted_labels': predicted_labels.tolist(),
                'ari_status': ari_status
            }

        except Exception as e:
            print(f"  クラスタリング評価エラー: {e}")
            return {'status': f'clustering_error: {str(e)}'}

    def _create_comprehensive_visualizations(self, z_style, z_skill, subject_ids, is_expert,
                                             unique_subjects, output_dir, experiment_id):
        """包括的可視化生成 - Figureオブジェクトを返す"""
        print(f"\n🎨 可視化生成中...")


        fig, axes = plt.subplots(2, 3, figsize=(20, 14))

        # カラーマップ準備
        subject_to_idx = {subj: i for i, subj in enumerate(unique_subjects)}
        subject_colors = [subject_to_idx[subj] for subj in subject_ids]

        # 1. スタイル空間PCA
        if z_style.shape[1] >= 2:
            pca_style = PCA(n_components=2)
            z_style_pca = pca_style.fit_transform(z_style)

            scatter = axes[0, 0].scatter(z_style_pca[:, 0], z_style_pca[:, 1],
                                         c=subject_colors, cmap='tab10', alpha=0.7, s=30)
            axes[0, 0].set_title(f'Style Space PCA\n(Explained: {pca_style.explained_variance_ratio_.sum():.3f})')
            axes[0, 0].set_xlabel('PC1')
            axes[0, 0].set_ylabel('PC2')

            # 重心プロット
            for i, subject in enumerate(unique_subjects):
                mask = np.array(subject_ids) == subject
                if np.any(mask):
                    center = np.mean(z_style_pca[mask], axis=0)
                    axes[0, 0].scatter(center[0], center[1], c='red', s=200, marker='x', linewidth=3)
                    axes[0, 0].annotate(subject, center, xytext=(5, 5), textcoords='offset points')

        # 2. スタイル空間t-SNE
        if len(z_style) >= self.min_samples_for_tsne:
            try:
                tsne = TSNE(n_components=2, perplexity=min(30, len(z_style) // 4), random_state=42)
                z_style_tsne = tsne.fit_transform(z_style)

                axes[0, 1].scatter(z_style_tsne[:, 0], z_style_tsne[:, 1],
                                   c=subject_colors, cmap='tab10', alpha=0.7, s=30)
                axes[0, 1].set_title('Style Space t-SNE')
            except Exception as e:
                axes[0, 1].text(0.5, 0.5, f't-SNE Failed: {str(e)}', transform=axes[0, 1].transAxes, ha='center')
        else:
            axes[0, 1].text(0.5, 0.5, 'Insufficient samples for t-SNE', transform=axes[0, 1].transAxes, ha='center')

        # 3. スキル空間PCA
        if z_skill.shape[1] >= 2:
            pca_skill = PCA(n_components=2)
            z_skill_pca = pca_skill.fit_transform(z_skill)

            axes[0, 2].scatter(z_skill_pca[:, 0], z_skill_pca[:, 1],
                               c=is_expert, cmap='RdYlBu', alpha=0.7, s=30)
            axes[0, 2].set_title(f'Skill Space PCA\n(Expert=Blue, Novice=Red)')

        # 4. 被験者間距離行列
        n_subjects = len(unique_subjects)
        distance_matrix = np.zeros((n_subjects, n_subjects))

        for i, subj_i in enumerate(unique_subjects):
            for j, subj_j in enumerate(unique_subjects):
                mask_i = np.array(subject_ids) == subj_i
                mask_j = np.array(subject_ids) == subj_j

                if np.any(mask_i) and np.any(mask_j):
                    center_i = np.mean(z_style[mask_i], axis=0)
                    center_j = np.mean(z_style[mask_j], axis=0)
                    distance_matrix[i, j] = np.linalg.norm(center_i - center_j)

        im = axes[1, 0].imshow(distance_matrix, cmap='viridis')
        axes[1, 0].set_title('Inter-Subject Distance Matrix')
        axes[1, 0].set_xticks(range(n_subjects))
        axes[1, 0].set_yticks(range(n_subjects))
        axes[1, 0].set_xticklabels(unique_subjects, rotation=45)
        axes[1, 0].set_yticklabels(unique_subjects)
        plt.colorbar(im, ax=axes[1, 0])

        # 5. 被験者別分布
        for i, subject in enumerate(unique_subjects):
            mask = np.array(subject_ids) == subject
            if np.any(mask) and z_style.shape[1] >= 2:
                subject_data = z_style[mask]
                axes[1, 1].scatter(subject_data[:, 0], subject_data[:, 1],
                                   label=subject, alpha=0.6, s=20)

        axes[1, 1].set_title('Style Space Distribution by Subject')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 6. 潜在次元の分散
        style_vars = np.var(z_style, axis=0)
        axes[1, 2].bar(range(len(style_vars)), style_vars)
        axes[1, 2].set_title('Style Dimension Variances')
        axes[1, 2].set_xlabel('Dimension')
        axes[1, 2].set_ylabel('Variance')

        plt.tight_layout()
        
        # ファイルに保存
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        save_path = os.path.join(plots_dir, f'comprehensive_latent_analysis_exp{experiment_id}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        print(f"  ✅ 可視化保存: {save_path}")
        print(f"  ✅ 可視化Figure生成完了")

        return fig

    def _print_comprehensive_summary(self, results):
        """包括的サマリー出力"""
        print("\n" + "=" * 60)
        print("📋 包括的潜在空間分析サマリー")
        print("=" * 60)

        print(f"総サンプル数: {results['total_samples']}")
        print(f"被験者数: {results['n_subjects']}")

        # スタイル分離結果
        style_analysis = results.get('style_analysis', {})
        if 'separation_ratio' in style_analysis:
            print(f"スタイル分離指標: {style_analysis['separation_ratio']:.4f}")
            print(f"スタイル分離判定: {style_analysis['status']}")

        # スキル分析結果
        skill_analysis = results.get('skill_analysis', {})
        if 'skill_ratio' in skill_analysis:
            print(f"スキル分離比: {skill_analysis['skill_ratio']:.4f}")

        # クラスタリング結果
        clustering_analysis = results.get('clustering_analysis', {})
        if 'ari' in clustering_analysis:
            print(f"クラスタリング性能 (ARI): {clustering_analysis['ari']:.4f}")
            print(f"ARI判定: {clustering_analysis['ari_status']}")

        print("=" * 60)

    def _populate_evaluation_result(self, result: EnhancedEvaluationResult, analysis_results: Dict[str, Any]):
        """分析結果をEnhancedEvaluationResultに格納"""
        # 基本メトリクス
        result.add_metric('total_samples', analysis_results['total_samples'],
                          '総サンプル数', 'basic')
        result.add_metric('n_subjects', analysis_results['n_subjects'],
                          '被験者数', 'basic')

        # スタイル分離メトリクス
        style_analysis = analysis_results.get('style_analysis', {})
        if 'separation_ratio' in style_analysis:
            result.add_metric('style_separation_ratio', style_analysis['separation_ratio'],
                              'スタイル分離指標', 'style_separation')
            result.add_metric('style_avg_between_distance', style_analysis['avg_between_distance'],
                              '被験者間平均距離', 'style_separation')
            result.add_metric('style_avg_within_variance', style_analysis['avg_within_variance'],
                              '被験者内平均分散', 'style_separation')

        # スキル分析メトリクス
        skill_analysis = analysis_results.get('skill_analysis', {})
        if 'skill_ratio' in skill_analysis:
            result.add_metric('skill_separation_ratio', skill_analysis['skill_ratio'],
                              'スキル分離比', 'skill_analysis')
            result.add_metric('skill_separation_distance', skill_analysis['skill_separation'],
                              '熟練度分離距離', 'skill_analysis')

        # クラスタリング性能メトリクス
        clustering_analysis = analysis_results.get('clustering_analysis', {})
        if 'ari' in clustering_analysis:
            result.add_metric('clustering_ari', clustering_analysis['ari'],
                              'Adjusted Rand Index', 'clustering')
            result.add_metric('clustering_silhouette', clustering_analysis['silhouette'],
                              'Silhouette Score', 'clustering')

        # 可視化(FigureオブジェクトをEnhancedEvaluationResultに渡す)
        if 'visualization_fig' in analysis_results:
            result.add_visualization('comprehensive_latent_analysis',
                                     analysis_results['visualization_fig'],
                                     '包括的潜在空間分析（スタイル・スキル分離、クラスタリング性能を含む6つのプロット）', 
                                     'comprehensive')






