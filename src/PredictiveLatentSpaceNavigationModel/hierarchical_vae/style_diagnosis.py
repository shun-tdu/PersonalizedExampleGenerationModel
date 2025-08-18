#!/usr/bin/env python3
"""
スタイル表現品質診断スクリプト
階層型VAEがスタイルの分散表現を獲得できているかを診断
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from models.hierarchical_vae import HierarchicalVAE
    from train_hierarchical_vae import create_dataloaders
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("階層型VAEモジュールのパスを確認してください")
    sys.exit(1)


def diagnose_style_representation(model_path, data_path, config_path=None, output_dir="style_diagnosis"):
    """
    スタイル表現の品質診断メイン関数

    計算内容の詳細:
    1. スタイル次元数: モデルが学習したスタイル潜在空間の次元
    2. 活用次元数: 実際に意味のある変動を示す次元の数
    3. 平均標準偏差: スタイルベクトルがどの程度多様に分布しているか
    4. 被験者分離ARI: クラスタリングで被験者を正しく分離できる度合い
    """

    print("=" * 60)
    print("スタイル表現品質診断")
    print("=" * 60)

    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    # 1. モデル読み込み
    print(f"\n1. モデル読み込み: {model_path}")

    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model = HierarchicalVAE(**config['model'])
    else:
        # デフォルト設定でモデル作成
        model = HierarchicalVAE(
            input_dim=2,
            seq_len=100,
            hidden_dim=128,
            primitive_latent_dim=32,
            skill_latent_dim=16,
            style_latent_dim=8
        )

    try:
        model.load_model(model_path, device)
        print("✅ モデル読み込み成功")
    except Exception as e:
        print(f"❌ モデル読み込み失敗: {e}")
        return None

    # 2. データ読み込み
    print(f"\n2. データ読み込み: {data_path}")

    try:
        _, _, test_loader, test_df = create_dataloaders(
            data_path,
            seq_len=100,
            batch_size=32
        )
        print("✅ データ読み込み成功")
    except Exception as e:
        print(f"❌ データ読み込み失敗: {e}")
        return None

    # 3. スタイル潜在変数抽出
    print(f"\n3. スタイル潜在変数抽出中...")

    all_z_style = []
    all_subjects = []
    all_trials = []

    model.eval()
    with torch.no_grad():
        for trajectories, subject_ids, _ in test_loader:
            trajectories = trajectories.to(device)
            encoded = model.encode_hierarchically(trajectories)

            all_z_style.append(encoded['z_style'].cpu().numpy())
            all_subjects.extend(subject_ids)

    z_style = np.vstack(all_z_style)
    print(f"✅ 抽出完了: {z_style.shape[0]}サンプル, {z_style.shape[1]}次元")

    # 4. 基本統計診断
    print(f"\n4. 基本統計診断")
    print("-" * 30)

    # 4-1. 基本形状
    n_samples, n_dims = z_style.shape
    print(f"スタイル次元数: {n_dims}")
    print(f"サンプル数: {n_samples}")
    print(f"被験者数: {len(set(all_subjects))}")

    # 4-2. 次元の活用度分析
    dim_stds = np.std(z_style, axis=0)
    active_dims = np.sum(dim_stds > 0.01)  # 閾値0.01以上の標準偏差を持つ次元
    utilization_rate = active_dims / n_dims

    print(f"活用次元数: {active_dims}/{n_dims} ({utilization_rate:.1%})")
    print(f"平均標準偏差: {np.mean(dim_stds):.4f}")
    print(f"標準偏差の範囲: {np.min(dim_stds):.4f} - {np.max(dim_stds):.4f}")

    # 活用度の診断
    if utilization_rate < 0.3:
        print("❌ 多くの次元が未活用（潜在空間の崩壊）")
        utilization_status = "CRITICAL"
    elif utilization_rate < 0.7:
        print("⚠️ 一部次元が未活用")
        utilization_status = "FAIR"
    else:
        print("✅ 潜在次元が適切に活用されている")
        utilization_status = "GOOD"

    # 4-3. 分散分析
    total_variance = np.var(z_style)
    print(f"総分散: {total_variance:.4f}")

    # PCA分析
    pca = PCA()
    pca.fit(z_style)
    explained_variance = pca.explained_variance_ratio_

    pc1_variance = explained_variance[0]
    pc2_variance = explained_variance[1] if len(explained_variance) > 1 else 0
    pc3_variance = explained_variance[2] if len(explained_variance) > 2 else 0

    print(f"PC1で説明される分散: {pc1_variance:.1%}")
    print(f"PC1+PC2で説明される分散: {pc1_variance + pc2_variance:.1%}")
    print(f"PC1+PC2+PC3で説明される分散: {pc1_variance + pc2_variance + pc3_variance:.1%}")

    # 分散集中度の診断
    if pc1_variance > 0.8:
        print("❌ 1次元に情報が集中（表現の貧困）")
        variance_status = "CRITICAL"
    elif pc1_variance > 0.6:
        print("⚠️ 少数次元に情報が偏っている")
        variance_status = "FAIR"
    else:
        print("✅ 分散が適切に分布している")
        variance_status = "GOOD"

    # 5. 被験者分離度診断
    print(f"\n5. 被験者分離度診断")
    print("-" * 30)

    unique_subjects = list(set(all_subjects))
    n_unique_subjects = len(unique_subjects)

    if n_unique_subjects < 2:
        print("❌ 被験者が1人のみ（分離診断不可）")
        separation_status = "NO_DATA"
        ari_score = 0
    else:
        # K-meansクラスタリング
        kmeans = KMeans(n_clusters=n_unique_subjects, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(z_style)

        # 真の被験者ラベルを数値化
        subject_to_num = {subj: i for i, subj in enumerate(unique_subjects)}
        true_labels = [subject_to_num[subj] for subj in all_subjects]

        # Adjusted Rand Index (ARI) 計算
        ari_score = adjusted_rand_score(true_labels, cluster_labels)

        print(f"被験者分離ARI: {ari_score:.4f}")
        print(f"  ARI=1.0: 完全分離")
        print(f"  ARI=0.0: ランダム分離")
        print(f"  ARI<0.0: ランダム以下")

        # ARIによる診断
        if ari_score < 0.1:
            print("❌ CRITICAL: スタイル表現が獲得できていません")
            separation_status = "CRITICAL"
        elif ari_score < 0.3:
            print("⚠️ POOR: スタイル表現が不十分です")
            separation_status = "POOR"
        elif ari_score < 0.6:
            print("✅ FAIR: スタイル表現が中程度獲得されています")
            separation_status = "FAIR"
        else:
            print("✅ GOOD: スタイル表現が良好に獲得されています")
            separation_status = "GOOD"

    # 6. 被験者内一貫性分析
    print(f"\n6. 被験者内一貫性分析")
    print("-" * 30)

    intra_subject_distances = []
    inter_subject_distances = []

    for i, subject1 in enumerate(unique_subjects):
        mask1 = np.array(all_subjects) == subject1
        subject1_data = z_style[mask1]

        # 被験者内距離（同一被験者の試行間距離）
        if len(subject1_data) > 1:
            for j in range(len(subject1_data)):
                for k in range(j + 1, len(subject1_data)):
                    dist = np.linalg.norm(subject1_data[j] - subject1_data[k])
                    intra_subject_distances.append(dist)

        # 被験者間距離（異なる被験者間の距離）
        for subject2 in unique_subjects[i + 1:]:
            mask2 = np.array(all_subjects) == subject2
            subject2_data = z_style[mask2]

            for data1 in subject1_data:
                for data2 in subject2_data:
                    dist = np.linalg.norm(data1 - data2)
                    inter_subject_distances.append(dist)

    mean_intra = np.mean(intra_subject_distances) if intra_subject_distances else 0
    mean_inter = np.mean(inter_subject_distances) if inter_subject_distances else 0
    separation_ratio = mean_inter / (mean_intra + 1e-8)

    print(f"被験者内平均距離: {mean_intra:.4f}")
    print(f"被験者間平均距離: {mean_inter:.4f}")
    print(f"分離比率: {separation_ratio:.2f}")
    print(f"  分離比率>2.0: 良好な分離")
    print(f"  分離比率1.5-2.0: 中程度の分離")
    print(f"  分離比率<1.5: 分離不十分")

    # 7. 可視化
    print(f"\n7. 可視化生成中...")
    create_visualizations(z_style, all_subjects, unique_subjects, output_dir)

    # 8. 総合診断
    print(f"\n8. 総合診断")
    print("=" * 30)

    # 各項目のスコア化
    scores = {
        'utilization': utilization_status,
        'variance': variance_status,
        'separation': separation_status
    }

    # 総合判定
    critical_count = sum(1 for status in scores.values() if status == "CRITICAL")
    good_count = sum(1 for status in scores.values() if status == "GOOD")

    if critical_count > 0:
        overall_status = "CRITICAL"
        recommendation = "モデル・データの根本的見直しが必要"
    elif good_count >= 2:
        overall_status = "GOOD"
        recommendation = "次のステージ（階層分離検証）へ進む"
    else:
        overall_status = "FAIR"
        recommendation = "ハイパーパラメータ調整での改善を試行"

    print(f"総合ステータス: {overall_status}")
    print(f"推奨アクション: {recommendation}")

    # 9. 結果保存
    results = {
        'model_path': model_path,
        'data_path': data_path,
        'n_samples': n_samples,
        'n_dims': n_dims,
        'n_subjects': n_unique_subjects,
        'utilization_rate': utilization_rate,
        'active_dims': active_dims,
        'total_variance': total_variance,
        'pc1_variance': pc1_variance,
        'ari_score': ari_score,
        'mean_intra_distance': mean_intra,
        'mean_inter_distance': mean_inter,
        'separation_ratio': separation_ratio,
        'overall_status': overall_status,
        'recommendation': recommendation
    }

    import json
    results_path = os.path.join(output_dir, 'diagnosis_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n診断完了！結果は {output_dir} に保存されました")

    return results


def create_visualizations(z_style, all_subjects, unique_subjects, output_dir):
    """可視化の生成"""

    # PCA可視化
    pca = PCA(n_components=2)
    z_style_pca = pca.fit_transform(z_style)

    plt.figure(figsize=(12, 5))

    # PCAプロット
    plt.subplot(1, 2, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_subjects)))

    for i, subject in enumerate(unique_subjects):
        mask = np.array(all_subjects) == subject
        plt.scatter(z_style_pca[mask, 0], z_style_pca[mask, 1],
                    c=[colors[i]], label=f'Subject {subject}', alpha=0.7)

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title('Style Latent Space (PCA)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # 次元別標準偏差
    plt.subplot(1, 2, 2)
    dim_stds = np.std(z_style, axis=0)
    plt.bar(range(len(dim_stds)), dim_stds)
    plt.axhline(y=0.01, color='r', linestyle='--', label='活用閾値')
    plt.xlabel('Style Dimension')
    plt.ylabel('Standard Deviation')
    plt.title('Dimension Utilization')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'style_diagnosis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 可視化保存完了: {output_dir}/style_diagnosis.png")


def main():
    parser = argparse.ArgumentParser(description="スタイル表現品質診断")
    parser.add_argument('--model_path', type=str, required=True, help='学習済みモデルのパス')
    parser.add_argument('--data_path', type=str, required=True, help='データファイルのパス')
    parser.add_argument('--config_path', type=str, help='設定ファイルのパス')
    parser.add_argument('--output_dir', type=str, default='style_diagnosis', help='出力ディレクトリ')

    args = parser.parse_args()

    results = diagnose_style_representation(
        model_path=args.model_path,
        data_path=args.data_path,
        config_path=args.config_path,
        output_dir=args.output_dir
    )

    if results:
        print(f"\n{'=' * 60}")
        print("診断サマリー")
        print(f"{'=' * 60}")
        print(f"ARI スコア: {results['ari_score']:.4f}")
        print(f"活用次元率: {results['utilization_rate']:.1%}")
        print(f"分離比率: {results['separation_ratio']:.2f}")
        print(f"総合判定: {results['overall_status']}")
        print(f"推奨: {results['recommendation']}")


if __name__ == "__main__":
    main()