# CLAUDE_ADDED
import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# パスを追加
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, current_dir)

# 直接インポート
from hybrid_model import HybridTrajectoryModel
sys.path.insert(0, os.path.join(src_dir, 'DataPreprocess'))
from DataPreprocessForOverFitting import load_processed_data


def generate_trajectories(
    model_path: str,
    data_path: str = '../../data/Datasets/overfitting_dataset.npz',
    num_conditions: int = 10,
    samples_per_condition: int = 3,
    output_dir: str = 'generated_outputs',
    device: str = 'auto'
):
    """
    学習済みハイブリッドモデルで軌道を生成
    
    Args:
        model_path: 学習済みモデルのパス
        data_path: データセットのパス
        num_conditions: 生成に使用する条件数
        samples_per_condition: 各条件あたりのサンプル数
        output_dir: 出力ディレクトリ
        device: デバイス ('auto', 'cuda', 'cpu')
    """
    
    # デバイス設定
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用デバイス: {device}")
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    
    # データ読み込み
    print("データを読み込み中...")
    trajectories, conditions = load_processed_data(data_path)
    
    # モデル読み込み
    print(f"モデルを読み込み中: {model_path}")
    model = HybridTrajectoryModel.load_model(model_path, device=device)
    model.eval()
    
    # テスト条件を準備
    test_conditions = torch.tensor(conditions[:num_conditions], dtype=torch.float32).to(device)
    original_trajectories = trajectories[:num_conditions]
    
    print(f"生成開始: {num_conditions}条件 × {samples_per_condition}サンプル")
    
    # 軌道生成
    with torch.no_grad():
        generated_trajectories = model.generate(
            condition=test_conditions,
            sequence_length=trajectories.shape[1],
            num_samples=samples_per_condition
        )
    
    generated_trajectories = generated_trajectories.cpu().numpy()
    test_conditions_cpu = test_conditions.cpu().numpy()
    
    print(f"生成完了: {generated_trajectories.shape}")
    
    # 結果を保存
    save_results(
        generated_trajectories, 
        original_trajectories, 
        test_conditions_cpu, 
        output_dir,
        samples_per_condition
    )
    
    # 可視化
    visualize_results(
        generated_trajectories, 
        original_trajectories, 
        test_conditions_cpu, 
        output_dir,
        samples_per_condition
    )
    
    return generated_trajectories, original_trajectories, test_conditions_cpu


def save_results(generated, original, conditions, output_dir, samples_per_condition):
    """結果をファイルに保存"""
    
    # 数値データを保存
    np.save(os.path.join(output_dir, 'data', 'generated_trajectories.npy'), generated)
    np.save(os.path.join(output_dir, 'data', 'original_trajectories.npy'), original)
    np.save(os.path.join(output_dir, 'data', 'test_conditions.npy'), conditions)
    
    # メタデータを保存
    metadata = {
        'num_conditions': len(conditions),
        'samples_per_condition': samples_per_condition,
        'total_generated_samples': generated.shape[0],
        'sequence_length': generated.shape[1],
        'feature_dim': generated.shape[2],
        'condition_names': ['動作時間', '終点誤差', 'ジャーク'],
        'generated_shape': generated.shape,
        'original_shape': original.shape,
        'conditions_shape': conditions.shape,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"データを保存: {output_dir}/data/")


def visualize_results(generated, original, conditions, output_dir, samples_per_condition):
    """結果を可視化"""
    
    num_conditions = len(conditions)
    condition_names = ['動作時間', '終点誤差', 'ジャーク']
    
    # 1. 軌道比較プロット
    plt.figure(figsize=(20, 4 * min(5, num_conditions)))
    
    for i in range(min(5, num_conditions)):
        plt.subplot(min(5, num_conditions), 4, i*4 + 1)
        
        # 元の軌道
        plt.plot(original[i, :, 0], original[i, :, 1], 
                'k-', linewidth=3, label='Original', alpha=0.8)
        
        # 生成された軌道
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for j in range(samples_per_condition):
            gen_idx = i * samples_per_condition + j
            color = colors[j % len(colors)]
            plt.plot(generated[gen_idx, :, 0], generated[gen_idx, :, 1],
                    '--', color=color, linewidth=2, alpha=0.7, label=f'Generated {j+1}')
        
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'条件 {i+1}: 軌道比較')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # X軸時系列
        plt.subplot(min(5, num_conditions), 4, i*4 + 2)
        plt.plot(original[i, :, 0], 'k-', linewidth=3, label='Original')
        for j in range(samples_per_condition):
            gen_idx = i * samples_per_condition + j
            color = colors[j % len(colors)]
            plt.plot(generated[gen_idx, :, 0], '--', color=color, linewidth=2, alpha=0.7, label=f'Gen {j+1}')
        plt.xlabel('Time Step')
        plt.ylabel('X Position')
        plt.title(f'条件 {i+1}: X軸時系列')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Y軸時系列
        plt.subplot(min(5, num_conditions), 4, i*4 + 3)
        plt.plot(original[i, :, 1], 'k-', linewidth=3, label='Original')
        for j in range(samples_per_condition):
            gen_idx = i * samples_per_condition + j
            color = colors[j % len(colors)]
            plt.plot(generated[gen_idx, :, 1], '--', color=color, linewidth=2, alpha=0.7, label=f'Gen {j+1}')
        plt.xlabel('Time Step')
        plt.ylabel('Y Position')
        plt.title(f'条件 {i+1}: Y軸時系列')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 条件パラメータ
        plt.subplot(min(5, num_conditions), 4, i*4 + 4)
        condition_values = conditions[i]
        bars = plt.bar(condition_names, condition_values, alpha=0.7, color='skyblue')
        plt.ylabel('値')
        plt.title(f'条件 {i+1}: パラメータ')
        plt.xticks(rotation=45)
        
        # 各バーに値を表示
        for bar, value in zip(bars, condition_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('ハイブリッドモデル生成結果', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'trajectory_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 統計的分析プロット
    plt.figure(figsize=(15, 10))
    
    # 軌道長の分布
    plt.subplot(2, 3, 1)
    original_lengths = [np.sum(np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))) for traj in original]
    generated_lengths = [np.sum(np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))) for traj in generated]
    
    plt.hist(original_lengths, bins=20, alpha=0.7, label='Original', color='black')
    plt.hist(generated_lengths, bins=20, alpha=0.7, label='Generated', color='red')
    plt.xlabel('軌道長')
    plt.ylabel('頻度')
    plt.title('軌道長の分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 最大変位の分布
    plt.subplot(2, 3, 2)
    original_max_disp = [np.max(np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)) for traj in original]
    generated_max_disp = [np.max(np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)) for traj in generated]
    
    plt.hist(original_max_disp, bins=20, alpha=0.7, label='Original', color='black')
    plt.hist(generated_max_disp, bins=20, alpha=0.7, label='Generated', color='red')
    plt.xlabel('最大変位')
    plt.ylabel('頻度')
    plt.title('最大変位の分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 終点位置の分布
    plt.subplot(2, 3, 3)
    plt.scatter(original[:, -1, 0], original[:, -1, 1], alpha=0.7, label='Original', color='black', s=50)
    plt.scatter(generated[:, -1, 0], generated[:, -1, 1], alpha=0.5, label='Generated', color='red', s=30)
    plt.xlabel('終点X座標')
    plt.ylabel('終点Y座標')
    plt.title('終点位置の分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 各条件での生成多様性
    plt.subplot(2, 3, 4)
    diversities = []
    for i in range(num_conditions):
        gen_samples = generated[i*samples_per_condition:(i+1)*samples_per_condition]
        # サンプル間の平均距離を計算
        distances = []
        for j in range(samples_per_condition):
            for k in range(j+1, samples_per_condition):
                dist = np.mean(np.sqrt(np.sum((gen_samples[j] - gen_samples[k])**2, axis=1)))
                distances.append(dist)
        diversities.append(np.mean(distances) if distances else 0)
    
    plt.bar(range(num_conditions), diversities, alpha=0.7, color='green')
    plt.xlabel('条件番号')
    plt.ylabel('生成多様性')
    plt.title('各条件での生成多様性')
    plt.grid(True, alpha=0.3)
    
    # 条件パラメータと多様性の関係
    plt.subplot(2, 3, 5)
    for i, name in enumerate(condition_names):
        plt.scatter(conditions[:num_conditions, i], diversities, alpha=0.7, label=name)
    plt.xlabel('条件パラメータ値')
    plt.ylabel('生成多様性')
    plt.title('条件パラメータと多様性の関係')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # サンプル数の情報
    plt.subplot(2, 3, 6)
    info_text = f"""
生成統計:
- 条件数: {num_conditions}
- 条件あたりサンプル数: {samples_per_condition}
- 総生成サンプル数: {generated.shape[0]}
- 系列長: {generated.shape[1]}
- 平均生成多様性: {np.mean(diversities):.4f}
    """
    plt.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
    plt.axis('off')
    plt.title('生成統計')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'statistical_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可視化結果を保存: {output_dir}/plots/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ハイブリッドモデルでの軌道生成')
    parser.add_argument('--model_path', type=str, required=True,
                       help='学習済みモデルのパス')
    parser.add_argument('--data_path', type=str, 
                       default='../../data/Datasets/overfitting_dataset.npz',
                       help='データセットのパス')
    parser.add_argument('--num_conditions', type=int, default=10,
                       help='生成に使用する条件数')
    parser.add_argument('--samples_per_condition', type=int, default=3,
                       help='各条件あたりのサンプル数')
    parser.add_argument('--output_dir', type=str, default='generated_outputs',
                       help='出力ディレクトリ')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='使用デバイス')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"エラー: モデルファイルが見つかりません: {args.model_path}")
        print("先に学習を実行してください")
        exit(1)
    
    # 軌道生成を実行
    generated, original, conditions = generate_trajectories(
        model_path=args.model_path,
        data_path=args.data_path,
        num_conditions=args.num_conditions,
        samples_per_condition=args.samples_per_condition,
        output_dir=args.output_dir,
        device=args.device
    )
    
    print("\n=== 生成完了 ===")
    print(f"生成された軌道数: {generated.shape[0]}")
    print(f"結果保存先: {args.output_dir}")
    print(f"- データファイル: {args.output_dir}/data/")
    print(f"- 可視化結果: {args.output_dir}/plots/")