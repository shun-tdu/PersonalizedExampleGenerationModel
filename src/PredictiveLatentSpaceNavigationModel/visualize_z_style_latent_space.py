# CLAUDE_ADDED: z_style潜在空間可視化スクリプト
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from models.beta_vae import BetaVAE
import sqlite3
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("警告: UMAPがインストールされていません。PCAのみ使用可能です。")
    print("UMAPを使用するには: pip install umap-learn")

def load_and_preprocess_data(data_path: str, target_sequence_length: int = 100):
    """
    データを読み込み、軌道データを固定長に前処理する（train.pyと同じ前処理方式）
    """
    print(f"データを読み込み中: {data_path}")
    df = pd.read_parquet(data_path)
    
    # 軌道データの準備
    trajectory_data = []
    subject_labels = []
    trial_info = []
    
    for subject_id, subject_df in df.groupby('subject_id'):
        print(f"被験者 {subject_id} を処理中...")
        trial_count = 0
        
        for trial_num, trial_df in subject_df.groupby('trial_num'):
            # train.pyと同じ前処理を適用
            # 1. 軌道データを絶対座標として取得
            trajectory_abs = trial_df[['HandlePosX', 'HandlePosY']].values
            
            if len(trajectory_abs) > 0:
                # 2. 差分計算（train.pyと同じ）
                trajectory_diff = np.diff(trajectory_abs, axis=0)
                trajectory_diff = np.insert(trajectory_diff, 0, [0, 0], axis=0)
                
                # 3. 固定長化（パディング/切り捨て）
                if len(trajectory_diff) > target_sequence_length:
                    processed_trajectory = trajectory_diff[:target_sequence_length]
                else:
                    padding = np.zeros((target_sequence_length - len(trajectory_diff), 2))
                    processed_trajectory = np.vstack([trajectory_diff, padding])
                
                trajectory_data.append(processed_trajectory)
                subject_labels.append(subject_id)
                trial_info.append(f"{subject_id}_trial_{trial_num}")
                trial_count += 1
        
        print(f"  - {trial_count}試行を処理")
    
    print(f"総データ数: {len(trajectory_data)}試行, 被験者数: {len(set(subject_labels))}人")
    return np.array(trajectory_data), subject_labels, trial_info

def load_model_config_from_db(db_path: str, experiment_id: int):
    """
    データベースから指定された実験IDのモデル設定を読み込む
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT hidden_dim, style_latent_dim, skill_latent_dim, beta
            FROM experiments WHERE id = ?
        """, (experiment_id,))
        result = cursor.fetchone()
        
        if result is None:
            raise ValueError(f"実験ID {experiment_id} が見つかりません")
        
        hidden_dim, style_latent_dim, skill_latent_dim, beta = result
        
        # 固定値と組み合わせてmodel_configを作成
        model_config = {
            'input_dim': 2,
            'seq_len': 100,
            'n_layers': 2,
            'hidden_dim': int(hidden_dim),
            'style_latent_dim': int(style_latent_dim),
            'skill_latent_dim': int(skill_latent_dim),
            'beta': float(beta)
        }
        
        print(f"実験ID {experiment_id} のモデル設定を読み込みました:")
        for key, value in model_config.items():
            print(f"  {key}: {value}")
        
        return model_config

def load_model(model_path: str, model_config: dict):
    """
    学習済みβ-VAEモデルを読み込む
    """
    print(f"モデルを読み込み中: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    model = BetaVAE(**model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, device

def encode_to_z_style(model, X, device, batch_size: int = 32):
    """
    軌道データをz_style潜在空間にエンコードする
    """
    print("z_style潜在空間へエンコード中...")
    X_tensor = torch.FloatTensor(X)
    z_style_list = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            encoded = model.encode(batch)
            z_style_list.append(encoded['z_style'].cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  処理済み: {i + len(batch)}/{len(X_tensor)}試行")
    
    z_style_all = np.vstack(z_style_list)
    print(f"エンコード完了: {z_style_all.shape}")
    return z_style_all

def visualize_z_style_pca(z_style_all, subject_labels, save_path: str = None):
    """
    z_style潜在空間をPCAで2次元に削減して可視化
    """
    print("PCA次元削減中...")
    # PCAで2次元に次元削減
    pca = PCA(n_components=2)
    z_style_2d = pca.fit_transform(z_style_all)
    
    # データフレーム作成
    viz_df = pd.DataFrame({
        'PC1': z_style_2d[:, 0],
        'PC2': z_style_2d[:, 1],
        'subject_id': subject_labels
    })
    
    # 被験者ごとの色設定
    unique_subjects = sorted(list(set(subject_labels)))
    colors = sns.color_palette("husl", len(unique_subjects))
    
    # プロット作成
    plt.figure(figsize=(12, 8))
    
    for i, subject_id in enumerate(unique_subjects):
        subject_data = viz_df[viz_df['subject_id'] == subject_id]
        plt.scatter(subject_data['PC1'], subject_data['PC2'], 
                   c=[colors[i]], label=f'被験者 {subject_id}', 
                   alpha=0.7, s=50)
    
    plt.xlabel(f'PC1 (寄与率: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 (寄与率: {pca.explained_variance_ratio_[1]:.2%})')
    plt.title('🎨 z_style潜在空間の被験者別分布 (PCA)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 統計情報をテキストで追加
    stats_text = f"""統計情報:
データ数: {len(z_style_all):,}試行
被験者数: {len(unique_subjects)}人
潜在次元: {z_style_all.shape[1]}次元
PC1寄与率: {pca.explained_variance_ratio_[0]:.2%}
PC2寄与率: {pca.explained_variance_ratio_[1]:.2%}
累積寄与率: {pca.explained_variance_ratio_[:2].sum():.2%}"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可視化結果を保存: {save_path}")
    
    plt.show()
    
    # 被験者別統計
    print("\n=== 被験者別z_style潜在空間統計 ===")
    subject_stats = viz_df.groupby('subject_id').agg({
        'PC1': ['mean', 'std', 'count'],
        'PC2': ['mean', 'std', 'count']
    }).round(4)
    print(subject_stats)
    
    return viz_df, pca

def visualize_z_style_umap(z_style_all, subject_labels, save_path: str = None, 
                          n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42):
    """
    z_style潜在空間をUMAPで2次元に削減して可視化
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAPがインストールされていません。pip install umap-learn を実行してください。")
    
    print("UMAP次元削減中...")
    # UMAPで2次元に次元削減
    reducer = umap.UMAP(
        n_components=2, 
        n_neighbors=n_neighbors, 
        min_dist=min_dist, 
        random_state=random_state,
        verbose=True
    )
    z_style_2d = reducer.fit_transform(z_style_all)
    
    # データフレーム作成
    viz_df = pd.DataFrame({
        'UMAP1': z_style_2d[:, 0],
        'UMAP2': z_style_2d[:, 1],
        'subject_id': subject_labels
    })
    
    # 被験者ごとの色設定
    unique_subjects = sorted(list(set(subject_labels)))
    colors = sns.color_palette("husl", len(unique_subjects))
    
    # プロット作成
    plt.figure(figsize=(12, 8))
    
    for i, subject_id in enumerate(unique_subjects):
        subject_data = viz_df[viz_df['subject_id'] == subject_id]
        plt.scatter(subject_data['UMAP1'], subject_data['UMAP2'], 
                   c=[colors[i]], label=f'被験者 {subject_id}', 
                   alpha=0.7, s=50)
    
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('🎨 z_style潜在空間の被験者別分布 (UMAP)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 統計情報とパラメータをテキストで追加
    stats_text = f"""統計情報:
データ数: {len(z_style_all):,}試行
被験者数: {len(unique_subjects)}人
潜在次元: {z_style_all.shape[1]}次元

UMAPパラメータ:
n_neighbors: {n_neighbors}
min_dist: {min_dist}
random_state: {random_state}"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可視化結果を保存: {save_path}")
    
    plt.show()
    
    # 被験者別統計
    print("\n=== 被験者別z_style潜在空間統計 (UMAP) ===")
    subject_stats = viz_df.groupby('subject_id').agg({
        'UMAP1': ['mean', 'std', 'count'],
        'UMAP2': ['mean', 'std', 'count']
    }).round(4)
    print(subject_stats)
    
    return viz_df, reducer

def visualize_z_style_comparison(z_style_all, subject_labels, save_path: str = None):
    """
    PCAとUMAPの比較可視化
    """
    # 被験者ごとの色設定
    unique_subjects = sorted(list(set(subject_labels)))
    colors = sns.color_palette("husl", len(unique_subjects))
    color_map = {subject: colors[i] for i, subject in enumerate(unique_subjects)}
    
    # サブプロット作成
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # PCA
    print("PCA次元削減中...")
    pca = PCA(n_components=2)
    z_style_pca = pca.fit_transform(z_style_all)
    
    for i, subject_id in enumerate(unique_subjects):
        subject_mask = np.array(subject_labels) == subject_id
        axes[0].scatter(z_style_pca[subject_mask, 0], z_style_pca[subject_mask, 1], 
                       c=[colors[i]], label=f'被験者 {subject_id}', alpha=0.7, s=50)
    
    axes[0].set_xlabel(f'PC1 (寄与率: {pca.explained_variance_ratio_[0]:.2%})')
    axes[0].set_ylabel(f'PC2 (寄与率: {pca.explained_variance_ratio_[1]:.2%})')
    axes[0].set_title('PCA')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # UMAP
    if UMAP_AVAILABLE:
        print("UMAP次元削減中...")
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        z_style_umap = reducer.fit_transform(z_style_all)
        
        for i, subject_id in enumerate(unique_subjects):
            subject_mask = np.array(subject_labels) == subject_id
            axes[1].scatter(z_style_umap[subject_mask, 0], z_style_umap[subject_mask, 1], 
                           c=[colors[i]], label=f'被験者 {subject_id}', alpha=0.7, s=50)
        
        axes[1].set_xlabel('UMAP Dimension 1')
        axes[1].set_ylabel('UMAP Dimension 2')
        axes[1].set_title('UMAP')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'UMAPが利用できません\npip install umap-learn', 
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=16)
        axes[1].set_title('UMAP (利用不可)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"比較可視化結果を保存: {save_path}")
    
    plt.show()
    
    return pca, reducer if UMAP_AVAILABLE else None

def main():
    # 設定
    DATA_PATH = "PredictiveLatentSpaceNavigationModel/DataPreprocess/my_data.parquet"
    DB_PATH = "PredictiveLatentSpaceNavigationModel/experiments.db"
    EXPERIMENT_ID = 17  # 使用したい実験ID
    MODEL_PATH = f"PredictiveLatentSpaceNavigationModel/outputs/checkpoints/best_model_exp{EXPERIMENT_ID}.pth"
    
    # 可視化方法の選択
    # 'pca', 'umap', 'comparison' から選択
    VISUALIZATION_METHOD = 'comparison'  # デフォルトは比較表示
    
    # UMAPパラメータ（UMAPを使用する場合）
    UMAP_PARAMS = {
        'n_neighbors': 15,    # 近傍点数 (5-50)
        'min_dist': 0.1,      # 最小距離 (0.0-1.0)
        'random_state': 42
    }
    
    try:
        print("🎨 z_style潜在空間可視化を開始")
        print(f"可視化方法: {VISUALIZATION_METHOD.upper()}")
        print("=" * 50)
        
        # 1. データベースからモデル設定を読み込み
        model_config = load_model_config_from_db(DB_PATH, EXPERIMENT_ID)
        
        # 2. データ読み込みと前処理
        trajectory_data, subject_labels, trial_info = load_and_preprocess_data(
            DATA_PATH, target_sequence_length=model_config['seq_len']
        )
        
        # 3. モデル読み込み
        model, device = load_model(MODEL_PATH, model_config)
        
        # 4. z_styleエンコード
        z_style_all = encode_to_z_style(model, trajectory_data, device)
        
        # 5. 可視化実行
        if VISUALIZATION_METHOD == 'pca':
            save_path = f"PredictiveLatentSpaceNavigationModel/z_style_pca_exp{EXPERIMENT_ID}.png"
            viz_df, reducer = visualize_z_style_pca(z_style_all, subject_labels, save_path)
            
        elif VISUALIZATION_METHOD == 'umap':
            if not UMAP_AVAILABLE:
                print("❌ UMAPが利用できません。PCAで実行します。")
                save_path = f"PredictiveLatentSpaceNavigationModel/z_style_pca_exp{EXPERIMENT_ID}.png"
                viz_df, reducer = visualize_z_style_pca(z_style_all, subject_labels, save_path)
            else:
                save_path = f"PredictiveLatentSpaceNavigationModel/z_style_umap_exp{EXPERIMENT_ID}.png"
                viz_df, reducer = visualize_z_style_umap(
                    z_style_all, subject_labels, save_path, **UMAP_PARAMS
                )
                
        elif VISUALIZATION_METHOD == 'comparison':
            save_path = f"PredictiveLatentSpaceNavigationModel/z_style_comparison_exp{EXPERIMENT_ID}.png"
            pca, reducer = visualize_z_style_comparison(z_style_all, subject_labels, save_path)
            
        else:
            raise ValueError(f"未対応の可視化方法: {VISUALIZATION_METHOD}")
        
        print(f"\n✅ z_style潜在空間可視化が完了しました！")
        if VISUALIZATION_METHOD == 'comparison':
            print(f"比較結果は {save_path} に保存されました。")
        else:
            print(f"結果は {save_path} に保存されました。")
        
        # 使用方法の説明
        print("\n" + "=" * 50)
        print("📖 可視化方法の変更:")
        print("VISUALIZATION_METHOD を以下の値に変更してください:")
        print("  - 'pca': PCAのみ")
        print("  - 'umap': UMAPのみ (要: pip install umap-learn)")
        print("  - 'comparison': PCAとUMAPの比較表示")
        print("\nUMAPパラメータの調整:")
        print(f"  - n_neighbors: {UMAP_PARAMS['n_neighbors']} (5-50, 大きいほど大域構造重視)")
        print(f"  - min_dist: {UMAP_PARAMS['min_dist']} (0.0-1.0, 小さいほどクラスタが密)")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()