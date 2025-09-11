# スキル潜在空間の評価器
from typing import List, Dict, Any, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from base_evaluator import BaseEvaluator
from src.PredictiveLatentSpaceNavigationModel.TransformerBaseEndToEndVAE.evaluator import EnhancedEvaluationResult


class VisualizeSkillSpaceEvaluator(BaseEvaluator):
    """スキル潜在空間の可視化評価"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_samples_for_tsne = 30
        skill_components = config.get('evaluation', {}).get('skill_component', 2)
        self.n_components = skill_components if skill_components in [2, 3] else 2

    def evaluate(self, model, test_data, result: EnhancedEvaluationResult) -> None:
        """スキル潜在空間の可視化評価を実行"""
        z_skill = test_data.get('z_skill')
        skill_scores = test_data.get('skill_scores')
        subject_ids = test_data.get('subject_ids')

        print("=" * 60)
        print("スキル潜在空間の可視化評価実行")
        print("=" * 60)

        pca_fig, tsne_fig = self._create_skill_latent_space_visualizations(
            z_skill=z_skill, skill_scores=skill_scores, subject_ids=subject_ids, n_components=self.n_components
        )

        result.add_visualization("skill_pca", pca_fig,
                                description="スキル潜在空間のPCA可視化（スキルスコアによる色分け）",
                                category="skill_analysis")
        result.add_visualization("skill_tsne", tsne_fig,
                                description="スキル潜在空間のt-SNE可視化（スキルスコアによる色分け）",
                                category="skill_analysis")

        print("✅ スキル潜在空間可視化評価完了")

    def get_required_data(self) -> List[str]:
        return ['z_skill', 'skill_scores', 'subject_ids', 'experiment_id']

    def _create_skill_latent_space_visualizations(self, z_skill, skill_scores, subject_ids, n_components=2) -> Union[Tuple[plt.Figure, plt.Figure], Tuple[plotly.graph_objs.Figure, plotly.graph_objs.Figure]]:
        """包括的可視化生成 - スキルスコアによる色分け。2Dの場合はMatplotlib、3Dの場合はPlotly Figureオブジェクトを返す"""
        print(f"\n🎯 スキル空間可視化生成中...")

        # スキルスコア正規化（カラーマップ用）
        skill_scores_normalized = (skill_scores - np.min(skill_scores)) / (np.max(skill_scores) - np.min(skill_scores) + 1e-8)

        # 2次元の場合（matplotlib）
        if n_components == 2:
            # 1. スキル空間PCA
            fig_pca, ax_pca = plt.subplots(figsize=(12, 10))
            if z_skill.shape[1] >= 2:
                pca_skill = PCA(n_components=2)
                z_skill_pca = pca_skill.fit_transform(z_skill)

                scatter = ax_pca.scatter(z_skill_pca[:, 0], z_skill_pca[:, 1],
                                        c=skill_scores, cmap='RdYlBu_r', alpha=0.7, s=50,
                                        edgecolors='black', linewidth=0.5)
                ax_pca.set_title(f'Skill Space PCA\n(Color: Skill Score, Explained: {pca_skill.explained_variance_ratio_.sum():.3f})')
                ax_pca.set_xlabel(f'PC1 ({pca_skill.explained_variance_ratio_[0]:.3f})')
                ax_pca.set_ylabel(f'PC2 ({pca_skill.explained_variance_ratio_[1]:.3f})')

                # カラーバー追加
                cbar = plt.colorbar(scatter, ax=ax_pca)
                cbar.set_label('Skill Score', rotation=270, labelpad=20)

                # 統計情報を追加
                ax_pca.text(0.02, 0.98, f'Samples: {len(z_skill)}\nSubjects: {len(set(subject_ids))}\nSkill Range: [{np.min(skill_scores):.3f}, {np.max(skill_scores):.3f}]', 
                           transform=ax_pca.transAxes, verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()

            # 2. スキル空間t-SNE
            fig_tsne, ax_tsne = plt.subplots(figsize=(12, 10))
            if len(z_skill) >= self.min_samples_for_tsne:
                try:
                    tsne = TSNE(n_components=2, perplexity=min(30, len(z_skill) // 4), random_state=42)
                    z_skill_tsne = tsne.fit_transform(z_skill)

                    scatter = ax_tsne.scatter(z_skill_tsne[:, 0], z_skill_tsne[:, 1],
                                            c=skill_scores, cmap='RdYlBu_r', alpha=0.7, s=50,
                                            edgecolors='black', linewidth=0.5)
                    ax_tsne.set_title('Skill Space t-SNE\n(Color: Skill Score)')
                    ax_tsne.set_xlabel('t-SNE 1')
                    ax_tsne.set_ylabel('t-SNE 2')

                    # カラーバー追加
                    cbar = plt.colorbar(scatter, ax=ax_tsne)
                    cbar.set_label('Skill Score', rotation=270, labelpad=20)

                    # 統計情報を追加
                    ax_tsne.text(0.02, 0.98, f'Samples: {len(z_skill)}\nSkill Range: [{np.min(skill_scores):.3f}, {np.max(skill_scores):.3f}]', 
                               transform=ax_tsne.transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                except Exception as e:
                    ax_tsne.text(0.5, 0.5, f't-SNE Failed: {str(e)}', transform=ax_tsne.transAxes, ha='center')
            else:
                ax_tsne.text(0.5, 0.5, 'Insufficient samples for t-SNE', transform=ax_tsne.transAxes, ha='center')

            plt.tight_layout()
            return fig_pca, fig_tsne

        # 3次元の場合（plotly）
        elif n_components == 3:
            fig_pca = None
            # 1. スキル空間PCA
            if z_skill.shape[1] >= 3:
                pca_skill = PCA(n_components=3)
                z_skill_pca = pca_skill.fit_transform(z_skill)

                df_pca = pd.DataFrame({
                    "PC1": z_skill_pca[:, 0],
                    "PC2": z_skill_pca[:, 1],
                    "PC3": z_skill_pca[:, 2],
                    "skill_scores": skill_scores,
                    "subject_ids": subject_ids
                })

                fig_pca = px.scatter_3d(
                    df_pca,
                    x="PC1",
                    y="PC2",
                    z="PC3",
                    color='skill_scores',
                    color_continuous_scale='RdYlBu_r',
                    title=f'Skill Space PCA (3D)\n(Color: Skill Score, Explained: {pca_skill.explained_variance_ratio_.sum():.3f})',
                    hover_data={
                        'subject_ids': True,
                        'skill_scores': ':.3f',
                        'PC1': ':.3f',
                        'PC2': ':.3f', 
                        'PC3': ':.3f'
                    },
                    labels={
                        'PC1': f'PC1 ({pca_skill.explained_variance_ratio_[0]:.3f})',
                        'PC2': f'PC2 ({pca_skill.explained_variance_ratio_[1]:.3f})',
                        'PC3': f'PC3 ({pca_skill.explained_variance_ratio_[2]:.3f})',
                        'skill_scores': 'Skill Score'
                    }
                )
                fig_pca.update_layout(title_x=0.5)
                fig_pca.update_traces(marker=dict(size=4, opacity=0.8, line=dict(width=0.5, color='black')))

            # 2. スキル空間t-SNE
            fig_tsne = None
            if len(z_skill) >= self.min_samples_for_tsne:
                try:
                    tsne = TSNE(n_components=3, perplexity=min(30, len(z_skill) // 4), random_state=42)
                    z_skill_tsne = tsne.fit_transform(z_skill)

                    df_tsne = pd.DataFrame({
                        "t-SNE1": z_skill_tsne[:, 0],
                        "t-SNE2": z_skill_tsne[:, 1],
                        "t-SNE3": z_skill_tsne[:, 2],
                        "skill_scores": skill_scores,
                        "subject_ids": subject_ids
                    })

                    fig_tsne = px.scatter_3d(
                        df_tsne,
                        x="t-SNE1",
                        y="t-SNE2",
                        z="t-SNE3",
                        color='skill_scores',
                        color_continuous_scale='RdYlBu_r',
                        title='Skill Space t-SNE (3D)\n(Color: Skill Score)',
                        hover_data={
                            'subject_ids': True,
                            'skill_scores': ':.3f',
                            't-SNE1': ':.3f',
                            't-SNE2': ':.3f',
                            't-SNE3': ':.3f'
                        },
                        labels={'skill_scores': 'Skill Score'}
                    )
                    fig_tsne.update_layout(title_x=0.5)
                    fig_tsne.update_traces(marker=dict(size=4, opacity=0.8, line=dict(width=0.5, color='black')))

                except Exception as e:
                    # エラーが発生した場合は空のFigureを作成し、テキストで通知
                    fig_tsne = plotly.graph_objs.Figure()
                    fig_tsne.add_annotation(text=f"t-SNE Failed: {e}", showarrow=False, font=dict(size=16))
                    fig_tsne.update_layout(title_text="Skill Space t-SNE (3D) - Failed")
            else:
                # サンプル数が不足している場合も同様
                fig_tsne = plotly.graph_objs.Figure()
                fig_tsne.add_annotation(text="Insufficient samples for t-SNE", showarrow=False, font=dict(size=16))
                fig_tsne.update_layout(title_text="Skill Space t-SNE (3D) - Skipped")

            return fig_pca, fig_tsne
        else:
            raise ValueError(f"主成分分析の主成分次元数が不適切です．2か3を選択してください．")


class SkillScoreRegressionEvaluator(BaseEvaluator):
    """スキル潜在変数から簡単なSVM,MLPでスキルスコアを回帰可能かを評価"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.test_size = 0.2
        self.cv_folds = 5
        self.random_state = 42
        self.min_samples = 20

    def evaluate(self, model, test_data, result: EnhancedEvaluationResult) -> None:
        """スキル潜在変数からのスキルスコア回帰評価を実行"""
        z_skill = test_data.get('z_skill')
        skill_scores = test_data.get('skill_scores')
        subject_ids = test_data.get('subject_ids')

        print("=" * 60)
        print("スキルスコア回帰評価実行 (MLP & SVM)")
        print("=" * 60)

        if len(skill_scores) < self.min_samples:
            print(f"⚠️ サンプル数不足: {len(skill_scores)} < {self.min_samples}")
            result.add_metric("regression_status", 0, "サンプル数不足", "skill_regression")
            return

        # データ前処理
        X_processed, y_processed = self._preprocess_data(z_skill, skill_scores)
        
        # 1. MLP回帰評価
        mlp_results = self._evaluate_mlp_regression(X_processed, y_processed)
        
        # 2. SVM回帰評価
        svm_results = self._evaluate_svm_regression(X_processed, y_processed)
        
        # 3. ベースライン比較（線形回帰）
        baseline_results = self._evaluate_baseline_regression(X_processed, y_processed)
        
        # 4. 交差検証による頑健性評価
        cv_results = self._perform_cross_validation(X_processed, y_processed)
        
        # 5. 可視化生成
        visualization_fig = self._create_regression_visualization(
            X_processed, y_processed, mlp_results, svm_results, baseline_results
        )
        
        # 結果をメトリクスに追加
        self._add_regression_metrics(result, mlp_results, svm_results, baseline_results, cv_results)
        
        # 可視化を追加
        result.add_visualization("skill_regression_analysis", visualization_fig,
                                description="MLPとSVMによるスキルスコア回帰性能分析",
                                category="skill_analysis")
        
        print("✅ スキルスコア回帰評価完了")

    def _preprocess_data(self, z_skill, skill_scores):
        """データ前処理"""
        from sklearn.preprocessing import StandardScaler
        
        print(f"\n📋 データ前処理...")
        print(f"  入力次元: {z_skill.shape[1]}")
        print(f"  サンプル数: {len(skill_scores)}")
        print(f"  スキルスコア範囲: [{np.min(skill_scores):.3f}, {np.max(skill_scores):.3f}]")
        
        # 特徴量の標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(z_skill)
        
        # NaN除去
        valid_indices = ~(np.isnan(X_scaled).any(axis=1) | np.isnan(skill_scores))
        X_clean = X_scaled[valid_indices]
        y_clean = np.array(skill_scores)[valid_indices]
        
        print(f"  前処理後サンプル数: {len(y_clean)}")
        
        return X_clean, y_clean

    def _evaluate_mlp_regression(self, X, y):
        """MLP回帰評価"""
        from sklearn.neural_network import MLPRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        print(f"\n🧠 MLP回帰評価...")
        
        try:
            # 訓練・テスト分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            # 複数のMLPアーキテクチャを試す
            mlp_configs = [
                {'hidden_layer_sizes': (50,), 'name': 'MLP-50'},
                {'hidden_layer_sizes': (100,), 'name': 'MLP-100'},
                {'hidden_layer_sizes': (50, 25), 'name': 'MLP-50-25'},
                {'hidden_layer_sizes': (100, 50), 'name': 'MLP-100-50'},
            ]
            
            best_mlp_result = None
            best_r2 = -np.inf
            
            for config in mlp_configs:
                try:
                    mlp = MLPRegressor(
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
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # 訓練データでの性能
                    y_train_pred = mlp.predict(X_train)
                    train_r2 = r2_score(y_train, y_train_pred)
                    
                    result_config = {
                        'name': config['name'],
                        'architecture': config['hidden_layer_sizes'],
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'train_r2': train_r2,
                        'overfitting': train_r2 - r2,
                        'y_pred': y_pred,
                        'y_test': y_test,
                        'model': mlp,
                        'success': True
                    }
                    
                    print(f"  {config['name']}: R²={r2:.4f}, RMSE={rmse:.4f}, 過学習={train_r2-r2:.4f}")
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_mlp_result = result_config
                        
                except Exception as e:
                    print(f"  {config['name']}: エラー {e}")
                    continue
            
            if best_mlp_result:
                print(f"  最優秀MLP: {best_mlp_result['name']} (R²={best_mlp_result['r2']:.4f})")
                return best_mlp_result
            else:
                return {'success': False, 'error': 'すべてのMLP構成が失敗'}
                
        except Exception as e:
            print(f"  ❌ MLP評価エラー: {e}")
            return {'success': False, 'error': str(e)}

    def _evaluate_svm_regression(self, X, y):
        """SVM回帰評価"""
        from sklearn.svm import SVR
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        print(f"\n📐 SVM回帰評価...")
        
        try:
            # 訓練・テスト分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            # SVM カーネルと パラメータ
            svm_configs = [
                {'kernel': 'linear', 'name': 'SVM-Linear'},
                {'kernel': 'rbf', 'name': 'SVM-RBF'},
                {'kernel': 'poly', 'degree': 2, 'name': 'SVM-Poly2'},
            ]
            
            best_svm_result = None
            best_r2 = -np.inf
            
            for config in svm_configs:
                try:
                    # ハイパーパラメータ探索
                    if config['kernel'] == 'linear':
                        param_grid = {'C': [0.1, 1, 10]}
                    else:
                        param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.1]}
                    
                    base_svm = SVR(kernel=config['kernel'], **{k: v for k, v in config.items() if k not in ['kernel', 'name']})
                    
                    # 小規模データセットの場合は簡単な評価
                    if len(X_train) < 50:
                        svm = base_svm
                        if config['kernel'] != 'linear':
                            svm.set_params(C=1, gamma='scale')
                        else:
                            svm.set_params(C=1)
                    else:
                        grid_search = GridSearchCV(base_svm, param_grid, cv=min(3, self.cv_folds), 
                                                 scoring='r2', n_jobs=-1)
                        grid_search.fit(X_train, y_train)
                        svm = grid_search.best_estimator_
                    
                    svm.fit(X_train, y_train)
                    y_pred = svm.predict(X_test)
                    
                    # 評価指標
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # 訓練データでの性能
                    y_train_pred = svm.predict(X_train)
                    train_r2 = r2_score(y_train, y_train_pred)
                    
                    result_config = {
                        'name': config['name'],
                        'kernel': config['kernel'],
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'train_r2': train_r2,
                        'overfitting': train_r2 - r2,
                        'y_pred': y_pred,
                        'y_test': y_test,
                        'model': svm,
                        'success': True
                    }
                    
                    print(f"  {config['name']}: R²={r2:.4f}, RMSE={rmse:.4f}, 過学習={train_r2-r2:.4f}")
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_svm_result = result_config
                        
                except Exception as e:
                    print(f"  {config['name']}: エラー {e}")
                    continue
            
            if best_svm_result:
                print(f"  最優秀SVM: {best_svm_result['name']} (R²={best_svm_result['r2']:.4f})")
                return best_svm_result
            else:
                return {'success': False, 'error': 'すべてのSVM構成が失敗'}
                
        except Exception as e:
            print(f"  ❌ SVM評価エラー: {e}")
            return {'success': False, 'error': str(e)}

    def _evaluate_baseline_regression(self, X, y):
        """ベースライン回帰評価（線形回帰）"""
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        print(f"\n📏 ベースライン回帰評価...")
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"  線形回帰: R²={r2:.4f}, RMSE={rmse:.4f}")
            
            return {
                'name': 'LinearRegression',
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'y_pred': y_pred,
                'y_test': y_test,
                'model': lr,
                'success': True
            }
            
        except Exception as e:
            print(f"  ❌ ベースライン評価エラー: {e}")
            return {'success': False, 'error': str(e)}

    def _perform_cross_validation(self, X, y):
        """交差検証による頑健性評価"""
        from sklearn.model_selection import cross_val_score
        from sklearn.neural_network import MLPRegressor
        from sklearn.svm import SVR
        from sklearn.linear_model import LinearRegression
        
        print(f"\n🔄 交差検証評価...")
        
        cv_results = {}
        models = {
            'MLP': MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=self.random_state),
            'SVM': SVR(kernel='rbf', C=1, gamma='scale'),
            'Linear': LinearRegression()
        }
        
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=min(self.cv_folds, len(y)//4), 
                                       scoring='r2', n_jobs=-1)
                cv_results[name] = {
                    'mean_r2': np.mean(scores),
                    'std_r2': np.std(scores),
                    'scores': scores,
                    'success': True
                }
                print(f"  {name}: R² = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
                
            except Exception as e:
                print(f"  {name}: CV エラー {e}")
                cv_results[name] = {'success': False, 'error': str(e)}
        
        return cv_results

    def _create_regression_visualization(self, X, y, mlp_results, svm_results, baseline_results):
        """回帰性能の可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 結果リスト
        results = [
            (mlp_results, 'MLP'),
            (svm_results, 'SVM'), 
            (baseline_results, 'Linear')
        ]
        
        # 成功した結果のみフィルタ
        successful_results = [(r, name) for r, name in results if r.get('success', False)]
        
        if not successful_results:
            axes[0, 0].text(0.5, 0.5, 'すべてのモデルが失敗', ha='center', va='center', transform=axes[0, 0].transAxes)
            return fig
        
        # 1. 予測 vs 実測値プロット
        for i, (result, name) in enumerate(successful_results[:3]):
            if 'y_test' in result and 'y_pred' in result:
                y_test = result['y_test']
                y_pred = result['y_pred']
                
                axes[0, i].scatter(y_test, y_pred, alpha=0.6, s=30)
                
                # 理想線
                min_val, max_val = min(np.min(y_test), np.min(y_pred)), max(np.max(y_test), np.max(y_pred))
                axes[0, i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                axes[0, i].set_xlabel('True Skill Score')
                axes[0, i].set_ylabel('Predicted Skill Score')
                axes[0, i].set_title(f'{name}\n(R² = {result.get("r2", 0):.3f})')
                
                # R²とRMSEを表示
                axes[0, i].text(0.05, 0.95, f'R² = {result.get("r2", 0):.3f}\nRMSE = {result.get("rmse", 0):.3f}',
                               transform=axes[0, i].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. 残差プロット（最良モデル）
        best_result = max(successful_results, key=lambda x: x[0].get('r2', -np.inf))
        if best_result and 'y_test' in best_result[0] and 'y_pred' in best_result[0]:
            result, name = best_result
            residuals = result['y_test'] - result['y_pred']
            axes[1, 0].scatter(result['y_pred'], residuals, alpha=0.6, s=30)
            axes[1, 0].axhline(y=0, color='r', linestyle='--')
            axes[1, 0].set_xlabel('Predicted Skill Score')
            axes[1, 0].set_ylabel('Residuals')
            axes[1, 0].set_title(f'Residual Plot - {name}')
        
        # 5. 性能比較バープロット
        if len(successful_results) > 1:
            names = [name for _, name in successful_results]
            r2_scores = [result.get('r2', 0) for result, _ in successful_results]
            rmse_scores = [result.get('rmse', 0) for result, _ in successful_results]
            
            x_pos = np.arange(len(names))
            
            axes[1, 1].bar(x_pos - 0.2, r2_scores, 0.4, label='R²', alpha=0.7)
            axes[1, 1].set_xlabel('Models')
            axes[1, 1].set_ylabel('R² Score')
            axes[1, 1].set_title('Model Comparison (R²)')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(names)
            
            # 数値表示
            for i, score in enumerate(r2_scores):
                axes[1, 1].text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')
        
        # 6. 統計サマリー
        axes[1, 2].axis('off')
        summary_text = "回帰性能サマリー\n" + "="*20 + "\n"
        
        for result, name in successful_results:
            summary_text += f"{name}:\n"
            summary_text += f"  R² = {result.get('r2', 0):.3f}\n"
            summary_text += f"  RMSE = {result.get('rmse', 0):.3f}\n"
            summary_text += f"  MAE = {result.get('mae', 0):.3f}\n"
            if 'overfitting' in result:
                summary_text += f"  過学習 = {result.get('overfitting', 0):.3f}\n"
            summary_text += "\n"
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        return fig

    def _add_regression_metrics(self, result, mlp_results, svm_results, baseline_results, cv_results):
        """回帰メトリクスを結果に追加"""
        # MLP結果
        if mlp_results.get('success', False):
            result.add_metric('skill_mlp_r2', mlp_results.get('r2', 0),
                            'MLP回帰のR²値', 'skill_regression')
            result.add_metric('skill_mlp_rmse', mlp_results.get('rmse', 0),
                            'MLP回帰のRMSE', 'skill_regression')
            result.add_metric('skill_mlp_overfitting', mlp_results.get('overfitting', 0),
                            'MLPの過学習度', 'skill_regression')
        
        # SVM結果  
        if svm_results.get('success', False):
            result.add_metric('skill_svm_r2', svm_results.get('r2', 0),
                            'SVM回帰のR²値', 'skill_regression')
            result.add_metric('skill_svm_rmse', svm_results.get('rmse', 0),
                            'SVM回帰のRMSE', 'skill_regression')
        
        # ベースライン結果
        if baseline_results.get('success', False):
            result.add_metric('skill_linear_r2', baseline_results.get('r2', 0),
                            '線形回帰のR²値', 'skill_regression')
        
        # 最優秀モデル
        all_results = [mlp_results, svm_results, baseline_results]
        successful = [r for r in all_results if r.get('success', False)]
        if successful:
            best_model = max(successful, key=lambda x: x.get('r2', -np.inf))
            result.add_metric('skill_best_regression_r2', best_model.get('r2', 0),
                            '最優秀回帰モデルのR²値', 'skill_regression')
        
        # 交差検証結果
        for name, cv_result in cv_results.items():
            if cv_result.get('success', False):
                result.add_metric(f'skill_{name.lower()}_cv_r2_mean', cv_result.get('mean_r2', 0),
                                f'{name}のCV平均R²値', 'skill_regression')
                result.add_metric(f'skill_{name.lower()}_cv_r2_std', cv_result.get('std_r2', 0),
                                f'{name}のCV R²標準偏差', 'skill_regression')

    def get_required_data(self) -> List[str]:
        return ['z_skill', 'skill_scores', 'experiment_id']


class SkillLatentDimensionVSScoreEvaluator(BaseEvaluator):
    """スキル潜在変数の主次元とスキルスコアが線形な関係にあるかを評価"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_samples = 10
        self.significance_level = 0.05

    def evaluate(self, model, test_data, result: EnhancedEvaluationResult) -> None:
        """スキル潜在次元とスキルスコアの線形関係評価を実行"""
        z_skill = test_data.get('z_skill')
        skill_scores = test_data.get('skill_scores')
        subject_ids = test_data.get('subject_ids')

        print("=" * 60)
        print("スキル潜在次元 vs スキルスコア線形関係評価実行")
        print("=" * 60)

        if len(skill_scores) < self.min_samples:
            print(f"⚠️ サンプル数不足: {len(skill_scores)} < {self.min_samples}")
            result.add_metric("linear_relationship_status", 0, "サンプル数不足", "skill_linearity")
            return

        # 1. PCA分析による主次元特定
        pca_results = self._perform_pca_analysis(z_skill, skill_scores)
        
        # 2. 各次元との相関分析
        correlation_results = self._analyze_dimension_correlations(z_skill, skill_scores)
        
        # 3. 線形回帰分析
        regression_results = self._perform_regression_analysis(z_skill, skill_scores)
        
        # 4. 可視化生成
        visualization_fig = self._create_linearity_visualization(
            z_skill, skill_scores, pca_results, correlation_results, regression_results
        )
        
        # 結果をメトリクスに追加
        self._add_linearity_metrics(result, pca_results, correlation_results, regression_results)
        
        # 可視化を追加
        result.add_visualization("skill_linearity_analysis", visualization_fig,
                                description="スキル潜在次元とスキルスコアの線形関係分析",
                                category="skill_analysis")
        
        print("✅ スキル潜在次元 vs スキルスコア線形関係評価完了")

    def _perform_pca_analysis(self, z_skill, skill_scores):
        """PCA分析でスキル潜在空間の主次元を特定"""
        from sklearn.decomposition import PCA
        from scipy.stats import pearsonr
        
        print(f"\n🔍 PCA分析実行...")
        
        # PCA実行
        pca = PCA()
        z_skill_pca = pca.fit_transform(z_skill)
        
        # 各主成分とスキルスコアの相関
        pc_correlations = []
        pc_p_values = []
        
        n_components = min(z_skill.shape[1], len(skill_scores))
        
        for i in range(n_components):
            try:
                corr, p_value = pearsonr(z_skill_pca[:, i], skill_scores)
                pc_correlations.append(corr)
                pc_p_values.append(p_value)
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"  PC{i+1}: r={corr:.4f}, p={p_value:.4f} {significance}")
                
            except Exception as e:
                print(f"  PC{i+1}: 計算エラー {e}")
                pc_correlations.append(0.0)
                pc_p_values.append(1.0)
        
        # 最も相関の高い主成分
        abs_correlations = [abs(c) for c in pc_correlations]
        best_pc_idx = np.argmax(abs_correlations) if abs_correlations else 0
        best_correlation = pc_correlations[best_pc_idx] if pc_correlations else 0.0
        
        print(f"  最高相関: PC{best_pc_idx+1} (r={best_correlation:.4f})")
        
        return {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'pc_correlations': pc_correlations,
            'pc_p_values': pc_p_values,
            'best_pc_idx': best_pc_idx,
            'best_correlation': best_correlation,
            'pca_components': pca.components_,
            'z_skill_pca': z_skill_pca
        }

    def _analyze_dimension_correlations(self, z_skill, skill_scores):
        """各潜在次元とスキルスコアの相関分析"""
        from scipy.stats import pearsonr, spearmanr
        
        print(f"\n📊 次元別相関分析...")
        
        correlations = []
        p_values = []
        spearman_correlations = []
        spearman_p_values = []
        
        for dim in range(z_skill.shape[1]):
            try:
                # ピアソン相関（線形関係）
                pearson_corr, pearson_p = pearsonr(z_skill[:, dim], skill_scores)
                correlations.append(pearson_corr)
                p_values.append(pearson_p)
                
                # スピアマン相関（単調関係）
                spearman_corr, spearman_p = spearmanr(z_skill[:, dim], skill_scores)
                spearman_correlations.append(spearman_corr)
                spearman_p_values.append(spearman_p)
                
                significance = "***" if pearson_p < 0.001 else "**" if pearson_p < 0.01 else "*" if pearson_p < 0.05 else ""
                print(f"  Dim{dim+1}: Pearson r={pearson_corr:.4f}(p={pearson_p:.4f}){significance}, Spearman ρ={spearman_corr:.4f}")
                
            except Exception as e:
                print(f"  Dim{dim+1}: 計算エラー {e}")
                correlations.append(0.0)
                p_values.append(1.0)
                spearman_correlations.append(0.0)
                spearman_p_values.append(1.0)
        
        # 最も相関の高い次元
        abs_correlations = [abs(c) for c in correlations]
        best_dim_idx = np.argmax(abs_correlations) if abs_correlations else 0
        best_dim_correlation = correlations[best_dim_idx] if correlations else 0.0
        
        print(f"  最高線形相関: Dim{best_dim_idx+1} (r={best_dim_correlation:.4f})")
        
        return {
            'pearson_correlations': correlations,
            'pearson_p_values': p_values,
            'spearman_correlations': spearman_correlations,
            'spearman_p_values': spearman_p_values,
            'best_dim_idx': best_dim_idx,
            'best_dim_correlation': best_dim_correlation
        }

    def _perform_regression_analysis(self, z_skill, skill_scores):
        """線形回帰分析"""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error
        from scipy import stats
        import warnings
        
        print(f"\n📈 線形回帰分析...")
        
        results = {}
        
        try:
            # 単変量回帰（最も相関の高い次元）
            correlations = []
            for dim in range(z_skill.shape[1]):
                try:
                    corr, _ = stats.pearsonr(z_skill[:, dim], skill_scores)
                    correlations.append(abs(corr))
                except:
                    correlations.append(0.0)
            
            best_dim = np.argmax(correlations)
            X_univariate = z_skill[:, best_dim].reshape(-1, 1)
            
            # 単変量線形回帰
            lr_uni = LinearRegression()
            lr_uni.fit(X_univariate, skill_scores)
            y_pred_uni = lr_uni.predict(X_univariate)
            
            r2_uni = r2_score(skill_scores, y_pred_uni)
            mse_uni = mean_squared_error(skill_scores, y_pred_uni)
            
            # 多変量回帰（全次元）
            lr_multi = LinearRegression()
            lr_multi.fit(z_skill, skill_scores)
            y_pred_multi = lr_multi.predict(z_skill)
            
            r2_multi = r2_score(skill_scores, y_pred_multi)
            mse_multi = mean_squared_error(skill_scores, y_pred_multi)
            
            # 統計的検定
            n = len(skill_scores)
            p_uni = z_skill.shape[1]  # 単変量なので実質1
            p_multi = z_skill.shape[1]
            
            # F統計量（多変量回帰の有意性）
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f_stat = (r2_multi / p_multi) / ((1 - r2_multi) / max(1, n - p_multi - 1))
                f_p_value = 1 - stats.f.cdf(f_stat, p_multi, max(1, n - p_multi - 1))
            
            print(f"  単変量回帰 (Dim{best_dim+1}): R²={r2_uni:.4f}, MSE={mse_uni:.4f}")
            print(f"  多変量回帰 (全次元): R²={r2_multi:.4f}, MSE={mse_multi:.4f}")
            print(f"  F検定: F={f_stat:.4f}, p={f_p_value:.4f}")
            
            # 線形性判定
            if r2_uni > 0.7:
                linearity_status = "強い線形関係"
            elif r2_uni > 0.4:
                linearity_status = "中程度の線形関係"
            elif r2_uni > 0.1:
                linearity_status = "弱い線形関係"
            else:
                linearity_status = "線形関係なし"
                
            print(f"  線形性判定: {linearity_status}")
            
            results = {
                'best_dim': best_dim,
                'univariate_r2': r2_uni,
                'univariate_mse': mse_uni,
                'multivariate_r2': r2_multi,
                'multivariate_mse': mse_multi,
                'f_statistic': f_stat,
                'f_p_value': f_p_value,
                'linearity_status': linearity_status,
                'univariate_coef': lr_uni.coef_[0],
                'univariate_intercept': lr_uni.intercept_,
                'y_pred_uni': y_pred_uni,
                'y_pred_multi': y_pred_multi,
                'success': True
            }
            
        except Exception as e:
            print(f"  ❌ 回帰分析エラー: {e}")
            results = {'success': False, 'error': str(e)}
        
        return results

    def _create_linearity_visualization(self, z_skill, skill_scores, pca_results, correlation_results, regression_results):
        """線形関係の可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 主成分とスキルスコアの散布図
        if pca_results and 'z_skill_pca' in pca_results:
            best_pc = pca_results['best_pc_idx']
            pc_data = pca_results['z_skill_pca'][:, best_pc]
            
            axes[0, 0].scatter(pc_data, skill_scores, alpha=0.6, s=30)
            axes[0, 0].set_xlabel(f'PC{best_pc+1}')
            axes[0, 0].set_ylabel('Skill Score')
            axes[0, 0].set_title(f'PC{best_pc+1} vs Skill Score\n(r={pca_results["best_correlation"]:.3f})')
            
            # 回帰直線
            z = np.polyfit(pc_data, skill_scores, 1)
            p = np.poly1d(z)
            axes[0, 0].plot(sorted(pc_data), p(sorted(pc_data)), "r--", alpha=0.8)
        
        # 2. 最も相関の高い潜在次元とスキルスコアの散布図
        if correlation_results and regression_results.get('success', False):
            best_dim = correlation_results['best_dim_idx']
            dim_data = z_skill[:, best_dim]
            
            axes[0, 1].scatter(dim_data, skill_scores, alpha=0.6, s=30)
            axes[0, 1].set_xlabel(f'Latent Dim {best_dim+1}')
            axes[0, 1].set_ylabel('Skill Score')
            axes[0, 1].set_title(f'Best Latent Dim vs Skill Score\n(r={correlation_results["best_dim_correlation"]:.3f})')
            
            # 回帰直線
            if 'y_pred_uni' in regression_results:
                sorted_indices = np.argsort(dim_data)
                axes[0, 1].plot(dim_data[sorted_indices], regression_results['y_pred_uni'][sorted_indices], "r--", alpha=0.8)
        
        # 3. 相関係数ヒートマップ
        if correlation_results:
            correlations = correlation_results['pearson_correlations']
            n_dims = len(correlations)
            corr_matrix = np.array(correlations).reshape(1, -1)
            
            im = axes[0, 2].imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            axes[0, 2].set_xticks(range(n_dims))
            axes[0, 2].set_xticklabels([f'D{i+1}' for i in range(n_dims)])
            axes[0, 2].set_yticks([0])
            axes[0, 2].set_yticklabels(['Skill Score'])
            axes[0, 2].set_title('Dimension-Skill Correlations')
            plt.colorbar(im, ax=axes[0, 2])
        
        # 4. 回帰診断: 残差プロット
        if regression_results.get('success', False) and 'y_pred_uni' in regression_results:
            residuals = skill_scores - regression_results['y_pred_uni']
            axes[1, 0].scatter(regression_results['y_pred_uni'], residuals, alpha=0.6, s=30)
            axes[1, 0].axhline(y=0, color='r', linestyle='--')
            axes[1, 0].set_xlabel('Predicted Skill Score')
            axes[1, 0].set_ylabel('Residuals')
            axes[1, 0].set_title('Residual Plot (Univariate)')
        
        # 5. 予測 vs 実測値
        if regression_results.get('success', False):
            axes[1, 1].scatter(skill_scores, regression_results['y_pred_uni'], alpha=0.6, s=30, label='Univariate')
            if 'y_pred_multi' in regression_results:
                axes[1, 1].scatter(skill_scores, regression_results['y_pred_multi'], alpha=0.6, s=30, label='Multivariate')
            
            # 理想線
            min_score, max_score = np.min(skill_scores), np.max(skill_scores)
            axes[1, 1].plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.8)
            axes[1, 1].set_xlabel('True Skill Score')
            axes[1, 1].set_ylabel('Predicted Skill Score')
            axes[1, 1].set_title('Prediction vs True Values')
            axes[1, 1].legend()
        
        # 6. 統計サマリー
        axes[1, 2].axis('off')
        summary_text = "線形関係分析結果\n" + "="*20 + "\n"
        
        if regression_results.get('success', False):
            summary_text += f"単変量R²: {regression_results.get('univariate_r2', 0):.3f}\n"
            summary_text += f"多変量R²: {regression_results.get('multivariate_r2', 0):.3f}\n"
            summary_text += f"判定: {regression_results.get('linearity_status', 'N/A')}\n\n"
        
        if pca_results:
            summary_text += f"最高PC相関: {pca_results.get('best_correlation', 0):.3f}\n"
        
        if correlation_results:
            summary_text += f"最高次元相関: {correlation_results.get('best_dim_correlation', 0):.3f}\n"
            
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig

    def _add_linearity_metrics(self, result, pca_results, correlation_results, regression_results):
        """線形関係メトリクスを結果に追加"""
        # PCA関連メトリクス
        if pca_results:
            result.add_metric('skill_pc_best_correlation', 
                            abs(pca_results.get('best_correlation', 0)),
                            '主成分との最高相関係数', 'skill_linearity')
        
        # 次元別相関メトリクス
        if correlation_results:
            result.add_metric('skill_dim_best_correlation',
                            abs(correlation_results.get('best_dim_correlation', 0)),
                            '潜在次元との最高相関係数', 'skill_linearity')
            
            # 有意な相関の数
            significant_correlations = sum(1 for p in correlation_results.get('pearson_p_values', []) 
                                        if p < self.significance_level)
            result.add_metric('skill_significant_correlations',
                            significant_correlations,
                            '有意な相関を持つ次元数', 'skill_linearity')
        
        # 回帰分析メトリクス
        if regression_results.get('success', False):
            result.add_metric('skill_univariate_r2',
                            regression_results.get('univariate_r2', 0),
                            '単変量回帰のR²値', 'skill_linearity')
            result.add_metric('skill_multivariate_r2',
                            regression_results.get('multivariate_r2', 0),
                            '多変量回帰のR²値', 'skill_linearity')
            result.add_metric('skill_regression_f_statistic',
                            regression_results.get('f_statistic', 0),
                            '回帰の有意性F統計量', 'skill_linearity')

    def get_required_data(self) -> List[str]:
        return ['z_skill', 'skill_scores', 'experiment_id']