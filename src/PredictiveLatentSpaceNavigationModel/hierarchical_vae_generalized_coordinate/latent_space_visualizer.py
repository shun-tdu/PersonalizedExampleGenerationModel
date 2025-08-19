import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from torch.utils.data import DataLoader
import yaml
import sqlite3
import json
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 一般化座標VAEをインポート
try:
    from models.hierarchical_vae_generalized_coordinate import HierarchicalVAEGeneralizedCoordinate
    from train_hierarchical_vae_generalized_coordinate import GeneralizedCoordinateDataset, create_dataloaders
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("一般化座標VAEモジュールが見つかりません")
    sys.exit(1)


class LatentSpaceVisualizerGeneralizedCoordinate:
    """一般化座標VAE潜在空間可視化クラス"""

    def __init__(self, experiment_id=None, db_path=None, model_path=None, config_path=None,
                 output_dir="latent_visualizations"):
        self.experiment_id = experiment_id
        self.db_path = db_path
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)

        # 実験情報の取得
        if experiment_id is not None and db_path is not None:
            self.experiment_info = self._load_experiment_from_db()
            self.model_path = self.experiment_info['model_path']
            self.config_path = self.experiment_info['config_path']

            # 出力ディレクトリを実験IDベースに設定
            if output_dir == "latent_visualizations":  # デフォルト値の場合
                self.output_dir = os.path.join(output_dir, f"experiment_{experiment_id}")
                os.makedirs(self.output_dir, exist_ok=True)
        else:
            # 従来の方式（直接パス指定）
            self.model_path = model_path
            self.config_path = config_path
            self.experiment_info = None

        # モデルと設定の読み込み
        self.config = self._load_config()
        self.model = self._load_model()

        # カラーパレット設定
        self.color_palette = None

        print(f"一般化座標VAE潜在空間可視化システム初期化完了")
        if self.experiment_id:
            print(f"実験ID: {self.experiment_id}")
            print(f"実験名: {self.experiment_info.get('experiment_name', 'Unknown')}")
        print(f"出力先: {self.output_dir}")
        print(f"使用デバイス: {self.device}")

    def _load_experiment_from_db(self):
        """データベースから実験情報を読み込み"""
        print(f"データベースから実験ID {self.experiment_id} の情報を読み込み中...")

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 実験情報を取得
                cursor.execute("""
                               SELECT experiment_name,
                                      status,
                                      model_path,
                                      best_model_path,
                                      config_path,
                                      config_backup_path,
                                      reconstruction_mse,
                                      style_separation_score,
                                      skill_performance_correlation,
                                      data_path,
                                      description,
                                      tags,
                                      created_at,
                                      end_time
                               FROM hierarchical_experiments
                               WHERE id = ?
                               """, (self.experiment_id,))

                result = cursor.fetchone()

                if result is None:
                    raise ValueError(f"実験ID {self.experiment_id} が見つかりません")

                experiment_info = {
                    'experiment_name': result[0],
                    'status': result[1],
                    'model_path': result[2] or result[3],  # model_pathが無い場合はbest_model_pathを使用
                    'config_path': result[4],
                    'config_backup_path': result[5],
                    'reconstruction_mse': result[6],
                    'style_separation_score': result[7],
                    'skill_performance_correlation': result[8],
                    'data_path': result[9],
                    'description': result[10],
                    'tags': result[11],
                    'created_at': result[12],
                    'end_time': result[13]
                }

                print(f"実験情報読み込み完了:")
                print(f"  実験名: {experiment_info['experiment_name']}")
                print(f"  ステータス: {experiment_info['status']}")
                print(f"  モデルパス: {experiment_info['model_path']}")
                print(f"  設定ファイル: {experiment_info['config_path']}")

                if experiment_info['status'] != 'completed':
                    print(f"警告: 実験ステータスが'completed'ではありません: {experiment_info['status']}")

                return experiment_info

        except sqlite3.Error as e:
            raise RuntimeError(f"データベース読み込みエラー: {e}")

    def _load_config(self):
        """設定ファイル読み込み"""
        config_path = self.config_path
        
        # 設定ファイルが見つからない場合、代替パスを試す
        if not os.path.exists(config_path):
            print(f"設定ファイルが見つかりません: {config_path}")
            
            # バックアップディレクトリを確認
            if self.experiment_info and 'config_backup_path' in self.experiment_info:
                backup_path = self.experiment_info.get('config_backup_path')
                if backup_path and os.path.exists(backup_path):
                    print(f"バックアップ設定ファイルを使用: {backup_path}")
                    config_path = backup_path
                else:
                    # configs_processedディレクトリを検索
                    base_name = os.path.basename(config_path)
                    config_name = os.path.splitext(base_name)[0]
                    
                    # 相対パスで検索
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    processed_dir = os.path.join(current_dir, 'configs_processed')
                    
                    if os.path.exists(processed_dir):
                        for file in os.listdir(processed_dir):
                            if config_name in file and file.endswith('.yaml'):
                                processed_path = os.path.join(processed_dir, file)
                                print(f"処理済み設定ファイルを使用: {processed_path}")
                                config_path = processed_path
                                break
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # データベースから読み込んだ場合、データパスを更新
        if self.experiment_info and 'data_path' in self.experiment_info:
            if self.experiment_info['data_path']:
                config['data']['data_path'] = self.experiment_info['data_path']

        return config

    def _load_model(self):
        """学習済みモデル読み込み"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {self.model_path}")

        model = HierarchicalVAEGeneralizedCoordinate(**self.config['model'])
        model.load_model(self.model_path, self.device)
        model.eval()
        print(f"モデル読み込み完了: {self.model_path}")

        return model

    def extract_all_latent_representations(self, use_all_data=True):
        """全データから潜在表現を抽出"""
        print("=== 潜在表現抽出開始 ===")

        # データローダー作成
        if use_all_data:
            # 全データを使用（train/val/testを統合）
            try:
                master_df = pd.read_parquet(self.config['data']['data_path'])
                print(f"全データ読み込み完了: {len(master_df)}サンプル")

                # 全データでデータセット作成
                dataset = GeneralizedCoordinateDataset(
                    master_df,
                    seq_len=self.config['model']['seq_len']
                )
                dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

            except Exception as e:
                print(f"全データ読み込みエラー: {e}")
                # フォールバック: 通常のデータ分割を使用
                train_loader, val_loader, test_loader, test_df = create_dataloaders(
                    self.config['data']['data_path'],
                    self.config['model']['seq_len'],
                    batch_size=32
                )
                # 全ローダーを結合
                all_data = []
                for loader in [train_loader, val_loader, test_loader]:
                    for batch in loader:
                        all_data.append(batch)

                dataloader = all_data
                master_df = test_df  # メタデータ用
        else:
            # テストデータのみ使用
            _, _, test_loader, master_df = create_dataloaders(
                self.config['data']['data_path'],
                self.config['model']['seq_len'],
                batch_size=32
            )
            dataloader = test_loader

        # 潜在表現抽出
        all_latent_data = {
            'z_style': [],
            'z_skill': [],
            'z_primitive': [],
            'subject_ids': [],
            'trial_nums': [],
            'is_expert': [],
            'batch_indices': []
        }

        self.model.eval()
        batch_idx = 0

        with torch.no_grad():
            if isinstance(dataloader, list):
                # フォールバック用
                for batch in tqdm(dataloader, desc="潜在表現抽出中"):
                    trajectories, subject_ids, is_expert = batch
                    self._process_batch(trajectories, subject_ids, is_expert, all_latent_data, batch_idx)
                    batch_idx += 1
            else:
                # 通常のデータローダー
                for trajectories, subject_ids, is_expert in tqdm(dataloader, desc="潜在表現抽出中"):
                    self._process_batch(trajectories, subject_ids, is_expert, all_latent_data, batch_idx)
                    batch_idx += 1

        # numpy配列に変換
        latent_arrays = {
            'z_style': np.vstack(all_latent_data['z_style']),
            'z_skill': np.vstack(all_latent_data['z_skill']),
            'z_primitive': np.vstack(all_latent_data['z_primitive'])
        }

        # メタデータ
        metadata = {
            'subject_ids': all_latent_data['subject_ids'],
            'is_expert': np.concatenate(all_latent_data['is_expert']),
            'batch_indices': all_latent_data['batch_indices']
        }

        print(f"潜在表現抽出完了:")
        print(f"  スタイル潜在変数: {latent_arrays['z_style'].shape}")
        print(f"  スキル潜在変数: {latent_arrays['z_skill'].shape}")
        print(f"  プリミティブ潜在変数: {latent_arrays['z_primitive'].shape}")
        print(f"  被験者数: {len(set(metadata['subject_ids']))}")

        return latent_arrays, metadata

    def _process_batch(self, trajectories, subject_ids, is_expert, all_latent_data, batch_idx):
        """バッチ処理"""
        trajectories = trajectories.to(self.device)

        # 階層的エンコーディング
        encoded = self.model.encode_hierarchically(trajectories)

        # CPU に移動して保存
        all_latent_data['z_style'].append(encoded['z_style'].cpu().numpy())
        all_latent_data['z_skill'].append(encoded['z_skill'].cpu().numpy())
        all_latent_data['z_primitive'].append(encoded['z_primitive'].cpu().numpy())

        # メタデータ
        all_latent_data['subject_ids'].extend(subject_ids)
        all_latent_data['is_expert'].append(is_expert.cpu().numpy())
        all_latent_data['batch_indices'].extend([batch_idx] * len(subject_ids))

    def setup_colors(self, subject_ids):
        """被験者ごとの色設定"""
        unique_subjects = sorted(list(set(subject_ids)))
        n_subjects = len(unique_subjects)

        # カラーパレット選択
        if n_subjects <= 10:
            palette = sns.color_palette("tab10", n_subjects)
        elif n_subjects <= 20:
            palette = sns.color_palette("tab20", n_subjects)
        else:
            palette = sns.color_palette("husl", n_subjects)

        self.color_palette = {subject: palette[i] for i, subject in enumerate(unique_subjects)}

        print(f"色設定完了: {n_subjects}被験者")
        return self.color_palette

    def visualize_latent_space(self, latent_arrays, metadata, methods=['pca', 'tsne', 'umap']):
        """潜在空間の可視化"""
        print("=== 潜在空間可視化開始 ===")

        # 色設定
        colors = self.setup_colors(metadata['subject_ids'])

        # 各階層の潜在空間を可視化
        for layer_name, latent_data in latent_arrays.items():
            print(f"\n--- {layer_name} 可視化 ---")

            for method in methods:
                print(f"{method.upper()}による次元削減中...")

                try:
                    # 次元削減実行
                    if method == 'pca':
                        reducer = PCA(n_components=2, random_state=42)
                        embedding = reducer.fit_transform(latent_data)
                        variance_ratio = reducer.explained_variance_ratio_
                        title_suffix = f"(PC1: {variance_ratio[0]:.1%}, PC2: {variance_ratio[1]:.1%})"

                    elif method == 'tsne':
                        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_data) // 4))
                        embedding = reducer.fit_transform(latent_data)
                        title_suffix = ""

                    elif method == 'umap':
                        reducer = umap.UMAP(n_components=2, random_state=42)
                        embedding = reducer.fit_transform(latent_data)
                        title_suffix = ""

                    # 可視化実行
                    self._plot_embedding(
                        embedding, metadata, colors,
                        title=f"{layer_name.upper()} - {method.upper()} {title_suffix}",
                        save_name=f"{layer_name}_{method}"
                    )

                except Exception as e:
                    print(f"{method}による{layer_name}の可視化でエラー: {e}")
                    continue

        # 総合比較図も作成
        self._create_comparison_plot(latent_arrays, metadata, colors)

        # 一般化座標VAE特有の分析: スキル軸の可視化
        self._visualize_skill_axes(latent_arrays, metadata, colors)

        print("=== 潜在空間可視化完了 ===")

    def _plot_embedding(self, embedding, metadata, colors, title, save_name):
        """埋め込み結果のプロット"""
        plt.figure(figsize=(12, 8))

        # 被験者ごとにプロット
        for subject_id in set(metadata['subject_ids']):
            mask = np.array(metadata['subject_ids']) == subject_id

            if mask.sum() > 0:
                plt.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=[colors[subject_id]],
                    label=f'Subject {subject_id}',
                    alpha=0.7,
                    s=50
                )

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

        # 凡例（被験者が多い場合は調整）
        if len(colors) <= 15:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存
        save_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"保存完了: {save_path}")

    def _create_comparison_plot(self, latent_arrays, metadata, colors):
        """階層比較図の作成"""
        print("階層比較図作成中...")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        layer_names = ['z_style', 'z_skill', 'z_primitive']
        layer_titles = ['Style (Level 3)', 'Skill (Level 2)', 'Primitive (Level 1)']

        for i, (layer_name, layer_title) in enumerate(zip(layer_names, layer_titles)):
            try:
                # PCAによる次元削減
                pca = PCA(n_components=2, random_state=42)
                embedding = pca.fit_transform(latent_arrays[layer_name])

                # 被験者ごとにプロット
                for subject_id in set(metadata['subject_ids']):
                    mask = np.array(metadata['subject_ids']) == subject_id

                    if mask.sum() > 0:
                        axes[i].scatter(
                            embedding[mask, 0],
                            embedding[mask, 1],
                            c=[colors[subject_id]],
                            label=f'Sub{subject_id}' if i == 0 else "",  # 凡例は最初だけ
                            alpha=0.7,
                            s=30
                        )

                axes[i].set_title(
                    f"{layer_title}\n(PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%})")
                axes[i].set_xlabel('PC1')
                axes[i].set_ylabel('PC2')
                axes[i].grid(True, alpha=0.3)

            except Exception as e:
                print(f"{layer_name}の比較図作成でエラー: {e}")
                axes[i].text(0.5, 0.5, f"Error: {str(e)}", transform=axes[i].transAxes, ha='center')

        # 凡例を右側に配置
        if len(colors) <= 15:
            axes[0].legend(bbox_to_anchor=(-0.1, 1), loc='upper right')

        plt.suptitle('Hierarchical VAE Generalized Coordinate Latent Space Comparison (PCA)', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存
        save_path = os.path.join(self.output_dir, "hierarchical_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"階層比較図保存完了: {save_path}")

    def _visualize_skill_axes(self, latent_arrays, metadata, colors):
        """一般化座標VAE特有のスキル軸可視化"""
        print("スキル軸分析可視化中...")

        try:
            z_skill = latent_arrays['z_skill']
            
            # スキル軸の主成分分析
            pca = PCA(n_components=min(5, z_skill.shape[1]))
            skill_pca = pca.fit_transform(z_skill)

            # スキル軸の可視化
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 主成分1vs2
            for subject_id in set(metadata['subject_ids']):
                mask = np.array(metadata['subject_ids']) == subject_id
                if mask.sum() > 0:
                    axes[0, 0].scatter(
                        skill_pca[mask, 0], skill_pca[mask, 1],
                        c=[colors[subject_id]], alpha=0.7, s=40,
                        label=f'Sub{subject_id}' if len(set(metadata['subject_ids'])) <= 10 else ""
                    )
            
            axes[0, 0].set_title(f"Skill Space PC1 vs PC2\n(Var: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%})")
            axes[0, 0].set_xlabel('PC1')
            axes[0, 0].set_ylabel('PC2')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. 主成分1vs3（3成分以上ある場合）
            if skill_pca.shape[1] >= 3:
                for subject_id in set(metadata['subject_ids']):
                    mask = np.array(metadata['subject_ids']) == subject_id
                    if mask.sum() > 0:
                        axes[0, 1].scatter(
                            skill_pca[mask, 0], skill_pca[mask, 2],
                            c=[colors[subject_id]], alpha=0.7, s=40
                        )
                axes[0, 1].set_title(f"Skill Space PC1 vs PC3\n(Var: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[2]:.1%})")
                axes[0, 1].set_xlabel('PC1')
                axes[0, 1].set_ylabel('PC3')
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, "Not enough components", transform=axes[0, 1].transAxes, ha='center')

            # 3. スキル次元の分散説明率
            axes[1, 0].bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
            axes[1, 0].set_title('Explained Variance Ratio by Skill PC')
            axes[1, 0].set_xlabel('Principal Component')
            axes[1, 0].set_ylabel('Explained Variance Ratio')
            axes[1, 0].grid(True, alpha=0.3)

            # 4. 累積寄与率
            cumulative_var = np.cumsum(pca.explained_variance_ratio_)
            axes[1, 1].plot(range(len(cumulative_var)), cumulative_var, 'bo-')
            axes[1, 1].set_title('Cumulative Explained Variance')
            axes[1, 1].set_xlabel('Principal Component')
            axes[1, 1].set_ylabel('Cumulative Explained Variance')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80%')
            axes[1, 1].axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='90%')
            axes[1, 1].legend()

            if len(set(metadata['subject_ids'])) <= 10:
                axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()

            # 保存
            save_path = os.path.join(self.output_dir, "skill_axes_analysis.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"スキル軸分析図保存完了: {save_path}")

        except Exception as e:
            print(f"スキル軸可視化でエラー: {e}")

    def analyze_cluster_quality(self, latent_arrays, metadata):
        """クラスタ品質の分析"""
        print("=== クラスタ品質分析 ===")

        from sklearn.metrics import silhouette_score, adjusted_rand_score
        from sklearn.cluster import KMeans

        subject_labels = np.array([hash(sid) % 1000 for sid in metadata['subject_ids']])  # 数値ラベル化

        analysis_results = {}

        for layer_name, latent_data in latent_arrays.items():
            print(f"\n--- {layer_name} 分析 ---")

            try:
                # クラスタリング
                n_clusters = len(set(metadata['subject_ids']))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(latent_data)

                # シルエット係数
                silhouette = silhouette_score(latent_data, cluster_labels)

                # 調整ランド指数（被験者ラベルとの一致度）
                ari = adjusted_rand_score(subject_labels, cluster_labels)

                analysis_results[layer_name] = {
                    'silhouette_score': silhouette,
                    'adjusted_rand_index': ari,
                    'n_clusters': n_clusters
                }

                print(f"  シルエット係数: {silhouette:.4f}")
                print(f"  調整ランド指数: {ari:.4f}")

            except Exception as e:
                print(f"  分析エラー: {e}")

        # 結果保存
        results_path = os.path.join(self.output_dir, "cluster_analysis.json")
        
        # numpy型をJSONシリアライゼ可能な型に変換
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return obj
        
        serializable_results = convert_numpy_types(analysis_results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n分析結果保存: {results_path}")
        return analysis_results

    def save_latent_representations(self, latent_arrays, metadata):
        """潜在表現の保存"""
        print("=== 潜在表現データ保存 ===")

        # DataFrameとして保存
        data_for_df = {
            'subject_id': metadata['subject_ids'],
            'is_expert': metadata['is_expert'],
            'batch_idx': metadata['batch_indices']
        }

        # 各階層の潜在変数を追加
        for layer_name, latent_data in latent_arrays.items():
            for i in range(latent_data.shape[1]):
                data_for_df[f'{layer_name}_dim{i}'] = latent_data[:, i]

        df = pd.DataFrame(data_for_df)

        # 保存
        csv_path = os.path.join(self.output_dir, "latent_representations.csv")
        df.to_csv(csv_path, index=False)

        parquet_path = os.path.join(self.output_dir, "latent_representations.parquet")
        df.to_parquet(parquet_path, index=False)

        print(f"CSV保存: {csv_path}")
        print(f"Parquet保存: {parquet_path}")

        # 統計サマリー
        summary_path = os.path.join(self.output_dir, "latent_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("一般化座標VAE潜在表現サマリー\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"データ数: {len(df)}\n")
            f.write(f"被験者数: {df['subject_id'].nunique()}\n")
            f.write(f"エキスパート率: {df['is_expert'].mean():.1%}\n\n")

            for layer_name in ['z_style', 'z_skill', 'z_primitive']:
                layer_cols = [col for col in df.columns if col.startswith(layer_name)]
                f.write(f"{layer_name} ({len(layer_cols)}次元):\n")
                f.write(df[layer_cols].describe().to_string())
                f.write("\n\n")

        print(f"サマリー保存: {summary_path}")

    def run_complete_analysis(self, use_all_data=True, methods=['pca', 'tsne', 'umap']):
        """完全な分析の実行"""
        print("=" * 60)
        print("一般化座標VAE潜在空間完全分析開始")
        print("=" * 60)

        # 1. 潜在表現抽出
        latent_arrays, metadata = self.extract_all_latent_representations(use_all_data)

        # 2. 可視化
        self.visualize_latent_space(latent_arrays, metadata, methods)

        # 3. クラスタ品質分析
        self.analyze_cluster_quality(latent_arrays, metadata)

        # 4. データ保存
        self.save_latent_representations(latent_arrays, metadata)

        # 5. 分析レポート生成
        self._generate_analysis_report(latent_arrays, metadata)

        print("=" * 60)
        print("一般化座標VAE潜在空間完全分析完了")
        print(f"結果保存先: {self.output_dir}")
        print("=" * 60)

    def _generate_analysis_report(self, latent_arrays, metadata):
        """分析レポート生成"""
        report_path = os.path.join(self.output_dir, "analysis_report.md")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 一般化座標VAE潜在空間分析レポート\n\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 実験情報
            if self.experiment_info:
                f.write("## 実験情報\n")
                f.write(f"- 実験ID: {self.experiment_id}\n")
                f.write(f"- 実験名: {self.experiment_info['experiment_name']}\n")
                f.write(f"- ステータス: {self.experiment_info['status']}\n")
                f.write(f"- 説明: {self.experiment_info.get('description', 'なし')}\n")
                f.write(f"- 作成日時: {self.experiment_info.get('created_at', 'Unknown')}\n")
                f.write(f"- 完了日時: {self.experiment_info.get('end_time', 'Unknown')}\n\n")

                # 実験結果
                f.write("## 実験結果\n")
                if self.experiment_info.get('reconstruction_mse'):
                    f.write(f"- 再構成MSE: {self.experiment_info['reconstruction_mse']:.6f}\n")
                if self.experiment_info.get('style_separation_score'):
                    f.write(f"- スタイル分離スコア: {self.experiment_info['style_separation_score']:.4f}\n")
                if self.experiment_info.get('skill_performance_correlation'):
                    f.write(f"- スキル性能相関: {self.experiment_info['skill_performance_correlation']:.4f}\n")
                f.write("\n")

            f.write("## モデル特徴\n")
            f.write("- モデル: 一般化座標VAE (Hierarchical VAE Generalized Coordinate)\n")
            f.write("- 階層構造: 3レベル（スタイル→スキル→プリミティブ）\n")
            f.write("- 特徴: 自由エネルギー原理に基づく予測符号化\n\n")

            f.write("## データ概要\n")
            f.write(f"- 総サンプル数: {len(metadata['subject_ids'])}\n")
            f.write(f"- 被験者数: {len(set(metadata['subject_ids']))}\n")
            f.write(f"- エキスパート比率: {np.mean(metadata['is_expert']):.1%}\n\n")

            f.write("## 潜在空間次元\n")
            for layer_name, latent_data in latent_arrays.items():
                f.write(f"- {layer_name}: {latent_data.shape[1]}次元\n")
            f.write("\n")

            f.write("## 生成ファイル\n")
            f.write("### 可視化画像\n")
            for layer in ['z_style', 'z_skill', 'z_primitive']:
                for method in ['pca', 'tsne', 'umap']:
                    f.write(f"- {layer}_{method}.png\n")
            f.write("- hierarchical_comparison.png\n")
            f.write("- skill_axes_analysis.png (一般化座標VAE特有)\n\n")

            f.write("### データファイル\n")
            f.write("- latent_representations.csv\n")
            f.write("- latent_representations.parquet\n")
            f.write("- cluster_analysis.json\n")
            f.write("- latent_summary.txt\n\n")

            f.write("## 使用方法\n")
            f.write("```python\n")
            f.write("import pandas as pd\n")
            f.write("df = pd.read_parquet('latent_representations.parquet')\n")
            f.write("# 被験者別分析などに活用\n")
            f.write("```\n")

        print(f"分析レポート生成: {report_path}")


def parse_experiment_ids(id_string):
    """実験ID範囲文字列を解析してIDリストを返す
    
    例:
    - "1-5" -> [1, 2, 3, 4, 5]
    - "1,3,5" -> [1, 3, 5]
    - "1-3,7-9" -> [1, 2, 3, 7, 8, 9]
    """
    ids = []
    parts = id_string.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # 範囲指定
            start, end = map(int, part.split('-'))
            ids.extend(range(start, end + 1))
        else:
            # 単一ID
            ids.append(int(part))
    
    return sorted(list(set(ids)))  # 重複削除してソート


def list_available_experiments(db_path):
    """利用可能な実験一覧を表示"""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                           SELECT id,
                                  experiment_name,
                                  status,
                                  reconstruction_mse,
                                  style_separation_score,
                                  created_at,
                                  end_time
                           FROM hierarchical_experiments
                           WHERE status = 'completed'
                           ORDER BY id DESC LIMIT 20
                           """)

            experiments = cursor.fetchall()

            if not experiments:
                print("完了した実験が見つかりません")
                return

            print("\n利用可能な実験一覧（最新20件）:")
            print("=" * 100)
            print(f"{'ID':>3} | {'実験名':25} | {'再構成MSE':12} | {'スタイル分離':12} | {'作成日時':16}")
            print("-" * 100)

            for exp in experiments:
                exp_id, name, status, mse, style_score, created, ended = exp

                mse_str = f"{mse:.6f}" if mse else "---"
                style_str = f"{style_score:.4f}" if style_score else "---"
                created_str = created.split('T')[0] if created else "---"

                print(f"{exp_id:>3} | {name[:25]:25} | {mse_str:12} | {style_str:12} | {created_str:16}")

            print("=" * 100)
            print("\n使用例:")
            print("  単一実験: python latent_space_visualizer.py --experiment_id 1")
            print("  範囲指定: python latent_space_visualizer.py --experiment_ids '1-5'")
            print("  複数指定: python latent_space_visualizer.py --experiment_ids '1,3,5'")
            print("  混合指定: python latent_space_visualizer.py --experiment_ids '1-3,7-9'")

    except sqlite3.Error as e:
        print(f"データベース読み取りエラー: {e}")


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_DB_PATH = os.path.join(SCRIPT_DIR, 'hierarchical_experiments_generalized_coordinate.db')

    parser = argparse.ArgumentParser(description="一般化座標VAE潜在空間可視化システム")

    # 実験ID指定 vs 直接パス指定
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--experiment_id', type=int,
                       help='データベースの実験ID（単一）')
    group.add_argument('--experiment_ids', type=str,
                       help='実験ID範囲指定（例: 1-5, 1,3,5, 1-3,7-9）')
    group.add_argument('--model_path', type=str,
                       help='学習済みモデルのパス（直接指定）')

    parser.add_argument('--db_path', type=str,
                        default=DEFAULT_DB_PATH,
                        help='実験データベースのパス')
    parser.add_argument('--config_path', type=str,
                        help='設定ファイルのパス（model_path使用時のみ必要）')
    # デフォルト出力ディレクトリをoutputs/latent_visualizationsに設定
    default_output_dir = os.path.join(SCRIPT_DIR, 'outputs', 'latent_visualizations')
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                        help='出力ディレクトリ')
    parser.add_argument('--use_all_data', action='store_true', default=True,
                        help='全データを使用（デフォルト: True）')
    parser.add_argument('--methods', nargs='+', default=['pca', 'tsne', 'umap'],
                        choices=['pca', 'tsne', 'umap'],
                        help='使用する次元削減手法')
    parser.add_argument('--list_experiments', action='store_true',
                        help='利用可能な実験一覧を表示')

    args = parser.parse_args()

    # 実験一覧表示
    if args.list_experiments:
        list_available_experiments(args.db_path)
        return

    # 引数検証
    if args.model_path and not args.config_path:
        parser.error("--model_path を使用する場合は --config_path も指定してください")
    
    if not args.list_experiments and not any([args.experiment_id, args.experiment_ids, args.model_path]):
        parser.error("--experiment_id, --experiment_ids, --model_path のいずれかを指定してください")

    # 可視化実行
    if args.experiment_id:
        # 単一実験の解析 - 実験ID別ディレクトリに出力
        exp_output_dir = os.path.join(args.output_dir, f"experiment_{args.experiment_id}")
        visualizer = LatentSpaceVisualizerGeneralizedCoordinate(
            experiment_id=args.experiment_id,
            db_path=args.db_path,
            output_dir=exp_output_dir
        )
        visualizer.run_complete_analysis(
            use_all_data=args.use_all_data,
            methods=args.methods
        )
    elif args.experiment_ids:
        # 複数実験の一括解析
        try:
            experiment_ids = parse_experiment_ids(args.experiment_ids)
            print(f"解析対象実験ID: {experiment_ids}")
            
            # 各実験IDに対して個別のディレクトリで解析
            for exp_id in experiment_ids:
                print(f"\n=== 実験ID {exp_id} の解析開始 ===")
                
                # 実験ID別の出力ディレクトリ
                exp_output_dir = os.path.join(args.output_dir, f"experiment_{exp_id}")
                
                try:
                    visualizer = LatentSpaceVisualizerGeneralizedCoordinate(
                        experiment_id=exp_id,
                        db_path=args.db_path,
                        output_dir=exp_output_dir
                    )
                    visualizer.run_complete_analysis(
                        use_all_data=args.use_all_data,
                        methods=args.methods
                    )
                    print(f"実験ID {exp_id} の解析完了")
                except Exception as e:
                    print(f"実験ID {exp_id} の解析でエラー: {e}")
                    continue
            
            print(f"\n=== 全実験の一括解析完了 ===")
            print(f"処理した実験数: {len(experiment_ids)}")
            
        except ValueError as e:
            print(f"実験ID範囲指定エラー: {e}")
            return
    else:
        # 直接パス指定
        visualizer = LatentSpaceVisualizerGeneralizedCoordinate(
            model_path=args.model_path,
            config_path=args.config_path,
            output_dir=args.output_dir
        )
        visualizer.run_complete_analysis(
            use_all_data=args.use_all_data,
            methods=args.methods
        )


if __name__ == "__main__":
    main()