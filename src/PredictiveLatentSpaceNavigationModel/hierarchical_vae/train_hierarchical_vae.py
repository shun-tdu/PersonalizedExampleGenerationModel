import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ChainedScheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import sqlite3
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import seaborn as sns
import json

# ===== numpy型のSQLite自動変換を登録 =====
sqlite3.register_adapter(np.float32, float)
sqlite3.register_adapter(np.float64, float)
sqlite3.register_adapter(np.int32, int)
sqlite3.register_adapter(np.int64, int)
sqlite3.register_adapter(np.bool_, bool)

# --- パス設定とインポート ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 階層型VAEをインポート
try:
    from models.hierarchical_vae import HierarchicalVAE
except ImportError:
    print("警告: models.hierarchical_vae のインポートに失敗しました")
    sys.exit(1)

try:
    from DataPreprocess.data_preprocess import calculate_jerk, calculate_path_efficiency, calculate_approach_angle, \
        calculate_sparc
except ImportError:
    print("警告: data_preprocess のインポートに失敗しました")


class SkillAxisAnalyzer:
    """スキル潜在空間での上手さの軸を分析・抽出するクラス"""

    def __init__(self):
        self.skill_improvement_directions = {}
        self.performance_correlations = {}

    def analyze_skill_axes(self, z_skill_data, performance_data):
        """スキル潜在変数とパフォーマンス指標の相関分析"""
        print("=== スキル軸分析開始 ===")

        skill_dim = z_skill_data.shape[1]

        # 各パフォーマンス指標との相関分析
        for metric_name, metric_values in performance_data.items():
            if len(metric_values) == 0:
                continue

            correlations = []

            for dim in range(skill_dim):
                if len(set(metric_values)) > 1 and np.std(metric_values) > 1e-6:
                    try:
                        corr, p_value = pearsonr(z_skill_data[:, dim], metric_values)
                        correlations.append((corr, p_value, dim))
                    except:
                        correlations.append((0.0, 1.0, dim))
                else:
                    correlations.append((0.0, 1.0, dim))

            correlations.sort(key=lambda x: abs(x[0]), reverse=True)
            best_corr, best_p, best_dim = correlations[0]

            self.performance_correlations[metric_name] = {
                'best_dimension': best_dim,
                'correlation': best_corr,
                'p_value': best_p,
                'all_correlations': correlations
            }

            print(f"{metric_name}: 最強相関次元={best_dim}, r={best_corr:.4f}, p={best_p:.4f}")

        # 総合的な上手さ軸の抽出
        self._extract_overall_skill_axis(z_skill_data, performance_data)

        # 個別指標の改善方向
        self._extract_specific_improvement_directions(z_skill_data, performance_data)

    def _extract_overall_skill_axis(self, z_skill_data, performance_data):
        """総合的なスキル向上軸を抽出"""
        print("\n--- 総合スキル軸の抽出 ---")

        normalized_metrics = {}

        for metric_name, values in performance_data.items():
            values = np.array(values)

            if len(values) == 0 or np.std(values) < 1e-6:
                continue

            # 指標の方向性を統一
            if metric_name in ['trial_time', 'trial_error', 'jerk', 'trial_variability']:
                normalized_values = -values
            else:
                normalized_values = values

            # 標準化
            if np.std(normalized_values) > 1e-6:
                normalized_values = (normalized_values - normalized_values.mean()) / normalized_values.std()
                normalized_metrics[metric_name] = normalized_values

        if len(normalized_metrics) == 0:
            print("警告: 有効なパフォーマンス指標がありません")
            return

        # 総合スキルスコア
        overall_skill_score = np.zeros(len(z_skill_data))
        weight_sum = 0

        for metric_name, normalized_values in normalized_metrics.items():
            overall_skill_score += normalized_values
            weight_sum += 1

        if weight_sum > 0:
            overall_skill_score /= weight_sum

        # 線形回帰
        try:
            reg = LinearRegression()
            reg.fit(z_skill_data, overall_skill_score)

            overall_improvement_direction = reg.coef_
            improvement_magnitude = np.linalg.norm(overall_improvement_direction)

            if improvement_magnitude > 1e-6:
                overall_improvement_direction = overall_improvement_direction / improvement_magnitude

                self.skill_improvement_directions['overall'] = {
                    'direction': overall_improvement_direction,
                    'r_squared': reg.score(z_skill_data, overall_skill_score),
                    'coefficients': reg.coef_
                }

                print(f"総合スキル軸: R²={reg.score(z_skill_data, overall_skill_score):.4f}")
            else:
                print("警告: 総合改善方向の計算に失敗")
        except Exception as e:
            print(f"総合スキル軸抽出エラー: {e}")

    def _extract_specific_improvement_directions(self, z_skill_data, performance_data):
        """個別指標の改善方向を抽出"""
        print("\n--- 個別指標改善方向の抽出 ---")

        for metric_name, values in performance_data.items():
            values = np.array(values)

            if len(values) == 0 or np.std(values) < 1e-6:
                continue

            # 方向性を統一
            if metric_name in ['trial_time', 'trial_error', 'jerk', 'trial_variability']:
                target_values = -values
            else:
                target_values = values

            try:
                reg = LinearRegression()
                reg.fit(z_skill_data, target_values)

                improvement_direction = reg.coef_
                improvement_magnitude = np.linalg.norm(improvement_direction)

                if improvement_magnitude > 1e-6:
                    improvement_direction = improvement_direction / improvement_magnitude

                    self.skill_improvement_directions[metric_name] = {
                        'direction': improvement_direction,
                        'r_squared': reg.score(z_skill_data, target_values),
                        'coefficients': reg.coef_
                    }

                    print(f"{metric_name}: R²={reg.score(z_skill_data, target_values):.4f}")
            except Exception as e:
                print(f"{metric_name}の改善方向抽出エラー: {e}")

    def get_improvement_direction(self, metric='overall', confidence_threshold=0.1):
        """指定された指標の改善方向を取得"""
        if metric not in self.skill_improvement_directions:
            print(f"警告: {metric}の改善方向が見つかりません。")
            if 'overall' in self.skill_improvement_directions:
                print("総合方向を使用します。")
                metric = 'overall'
            else:
                raise ValueError("利用可能な改善方向がありません")

        direction_info = self.skill_improvement_directions[metric]

        if direction_info['r_squared'] < confidence_threshold:
            print(f"警告: {metric}のR²={direction_info['r_squared']:.4f}が閾値{confidence_threshold}を下回ります")

        return direction_info['direction']


class TrajectoryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = 100, feature_cols=['HandlePosX', 'HandlePosY']):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.trials = list(df.groupby(['subject_id', 'trial_num']))

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx) -> tuple:
        """
        指定されたインデックスのデータを取得する

        :return: tuple(軌道テンソル, 被験者ID, 熟練度ラベル)
        """
        (subject_id, _), trial_df = self.trials[idx]

        # --- 各試行データに対する前処理 ---
        # 1. 軌道データをNumpy配列に変換
        trajectory_abs = trial_df[self.feature_cols].values

        # 2. 差分計算
        trajectory_diff = np.diff(trajectory_abs, axis=0)
        trajectory_diff = np.insert(trajectory_diff, 0, [0, 0], axis=0)

        # 3. 固定長化 (パディング/切り捨て)
        if len(trajectory_diff) > self.seq_len:
            processed_trajectory = trajectory_diff[:self.seq_len]
        else:
            padding = np.zeros((self.seq_len - len(trajectory_diff), len(self.feature_cols)))
            processed_trajectory = np.vstack([trajectory_diff, padding])

        # 4. ラベルを取得
        is_expert = trial_df['is_expert'].iloc[0]

        # 5. テンソルに変換して返す
        return (
            torch.tensor(processed_trajectory, dtype=torch.float32),
            subject_id,
            torch.tensor(is_expert, dtype=torch.long)
        )


def create_dataloaders(master_data_path: str, seq_len: int, batch_size: int, random_seed: int = 42) -> tuple:
    """データローダーを作成"""
    try:
        master_df = pd.read_parquet(master_data_path)
    except FileNotFoundError:
        print(f"エラー: ファイル '{master_data_path}' が見つかりません。")
        return None, None, None, None

    # --- 被験者ベースでのデータ分割 ---
    np.random.seed(random_seed)
    subject_ids = master_df['subject_id'].unique()
    np.random.shuffle(subject_ids)

    # 例: 4人学習、1人検証、1人テスト
    # 被験者数が少ないので、人数をハードコーディング
    if len(subject_ids) < 3:
        raise ValueError("データセットの分割には最低3人の被験者が必要です。")

    train_ids = subject_ids[:-2]
    val_ids = subject_ids[-2:-1]
    test_ids = subject_ids[-1:]

    train_df = master_df[master_df['subject_id'].isin(train_ids)]
    val_df = master_df[master_df['subject_id'].isin(val_ids)]
    test_df = master_df[master_df['subject_id'].isin(test_ids)]

    print(f"データ分割: 学習用={len(train_ids)}人, 検証用={len(val_ids)}人, テスト用={len(test_ids)}人")

    train_dataset = TrajectoryDataset(train_df, seq_len=seq_len)
    val_dataset = TrajectoryDataset(val_df, seq_len=seq_len)
    test_dataset = TrajectoryDataset(test_df, seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, test_df


def update_db(db_path: str, experiment_id: int, data: dict):
    """階層型VAE実験データベースを更新"""
    try:
        with sqlite3.connect(db_path) as conn:
            set_clause = ", ".join([f"{key} = ?" for key in data.keys()])
            query = f"UPDATE hierarchical_experiments SET {set_clause} WHERE id = ?"
            values = tuple(data.values()) + (experiment_id,)
            conn.execute(query, values)
            conn.commit()
    except Exception as e:
        print(f"!!! DB更新エラー (ID: {experiment_id}): {e} !!!")
        # フォールバック: 古いテーブル名も試す
        try:
            with sqlite3.connect(db_path) as conn:
                set_clause = ", ".join([f"{key} = ?" for key in data.keys()])
                query = f"UPDATE experiments SET {set_clause} WHERE id = ?"
                values = tuple(data.values()) + (experiment_id,)
                conn.execute(query, values)
                conn.commit()
                print(f"フォールバック成功: 古いテーブル名で更新")
        except Exception as e2:
            print(f"フォールバックも失敗: {e2}")


def validate_and_convert_config(config: dict) -> dict:
    """設定ファイルの妥当性検証と型変換"""

    # 型変換が必要な数値パラメータのマッピング
    numeric_conversions = {
        # モデル設定
        'model.input_dim': int,
        'model.seq_len': int,
        'model.hidden_dim': int,
        'model.primitive_latent_dim': int,
        'model.skill_latent_dim': int,
        'model.style_latent_dim': int,
        'model.beta_primitive': float,
        'model.beta_skill': float,
        'model.beta_style': float,
        'model.precision_lr': float,

        # 学習設定
        'training.batch_size': int,
        'training.num_epochs': int,
        'training.lr': float,  # ← ここが重要
        'training.weight_decay': float,
        'training.clip_grad_norm': float,
        'training.warmup_epochs': int,
        'training.scheduler_T_0': int,
        'training.scheduler_T_mult': int,
        'training.scheduler_eta_min': float,
        'training.patience': int,

        # 階層設定
        'hierarchical_settings.primitive_learning_start': float,
        'hierarchical_settings.skill_learning_start': float,
        'hierarchical_settings.style_learning_start': float,
        'hierarchical_settings.prediction_error_weights.level1': float,
        'hierarchical_settings.prediction_error_weights.level2': float,
        'hierarchical_settings.prediction_error_weights.level3': float,
        'hierarchical_settings.exemplar_generation.skill_enhancement_factor': float,
        'hierarchical_settings.exemplar_generation.style_preservation_weight': float,
        'hierarchical_settings.exemplar_generation.max_enhancement_steps': int,
    }

    def get_nested_value(data, key_path):
        """ネストした辞書から値を取得"""
        keys = key_path.split('.')
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def set_nested_value(data, key_path, value):
        """ネストした辞書に値を設定"""
        keys = key_path.split('.')
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    # 数値変換の実行
    for key_path, target_type in numeric_conversions.items():
        current_value = get_nested_value(config, key_path)

        if current_value is not None:
            try:
                # 文字列の場合は数値に変換
                if isinstance(current_value, str):
                    if target_type == float:
                        converted_value = float(current_value)
                    elif target_type == int:
                        converted_value = int(float(current_value))  # 1e3 → 1000.0 → 1000
                    else:
                        converted_value = target_type(current_value)
                elif isinstance(current_value, (int, float)):
                    converted_value = target_type(current_value)
                else:
                    converted_value = current_value

                set_nested_value(config, key_path, converted_value)

                # デバッグ出力
                if str(current_value) != str(converted_value):
                    print(
                        f"型変換: {key_path} = {current_value} ({type(current_value)}) → {converted_value} ({type(converted_value)})")

            except (ValueError, TypeError) as e:
                print(f"警告: {key_path}の型変換に失敗: {current_value} → {target_type.__name__} (エラー: {e})")

    # 必須セクションの検証
    required_sections = {
        'data': ['data_path'],
        'model': ['input_dim', 'seq_len', 'hidden_dim', 'primitive_latent_dim', 'skill_latent_dim', 'style_latent_dim'],
        'training': ['batch_size', 'num_epochs', 'lr'],
        'logging': ['output_dir']
    }

    for section, keys in required_sections.items():
        if section not in config:
            raise ValueError(f"設定ファイルに'{section}'セクションがありません")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"'{section}.{key}'が設定されていません")

    # データパスの存在確認
    data_path = config['data'].get('data_path', '')
    if data_path and not os.path.exists(data_path):
        print(f"警告: データファイル {data_path} が見つかりません")

    return config

def extract_latent_variables_hierarchical(model, test_loader, device):
    """階層型VAEからテストデータの潜在変数を抽出"""
    model.eval()
    all_z_style = []
    all_z_skill = []
    all_z_primitive = []
    all_subject_ids = []
    all_is_expert = []
    all_reconstructions = []
    all_originals = []

    with torch.no_grad():
        for trajectories, subject_ids, is_expert in tqdm(test_loader, desc="階層潜在変数抽出中"):
            trajectories = trajectories.to(device)

            encoded = model.encode_hierarchically(trajectories)
            z_style = encoded['z_style']
            z_skill = encoded['z_skill']
            z_primitive = encoded['z_primitive']

            reconstructed = model.decode_hierarchically(z_style, z_skill, z_primitive)

            all_z_style.append(z_style.cpu().numpy())
            all_z_skill.append(z_skill.cpu().numpy())
            all_z_primitive.append(z_primitive.cpu().numpy())
            all_subject_ids.append(subject_ids)
            all_is_expert.append(is_expert.cpu().numpy())
            all_reconstructions.append(reconstructed.cpu().numpy())
            all_originals.append(trajectories.cpu().numpy())

    return {
        'z_style': np.vstack(all_z_style),
        'z_skill': np.vstack(all_z_skill),
        'z_primitive': np.vstack(all_z_primitive),
        'subject_ids': all_subject_ids,
        'is_expert': np.concatenate(all_is_expert),
        'reconstructions': np.vstack(all_reconstructions),
        'originals': np.vstack(all_originals)
    }


def run_skill_axis_analysis(model, test_loader, test_df, device):
    """スキル軸分析を実行"""
    print("=== スキル軸分析開始 ===")

    model.eval()
    all_z_skill = []
    all_subject_ids = []

    # スキル潜在変数を抽出
    with torch.no_grad():
        for trajectories, subject_ids, is_expert in test_loader:
            trajectories = trajectories.to(device)
            encoded = model.encode_hierarchically(trajectories)
            all_z_skill.append(encoded['z_skill'].cpu().numpy())
            all_subject_ids.extend(subject_ids)

    z_skill_data = np.vstack(all_z_skill)

    # パフォーマンス指標を取得
    perf_cols = [col for col in test_df.columns if col.startswith('perf_')]
    performance_df = test_df.groupby(['subject_id', 'trial_num']).first()[perf_cols].reset_index()

    # データの長さを合わせる
    min_length = min(len(z_skill_data), len(performance_df))
    z_skill_data = z_skill_data[:min_length]
    performance_df = performance_df.iloc[:min_length]

    # パフォーマンス指標辞書を作成
    performance_data = {}
    for col in perf_cols:
        metric_name = col.replace('perf_', '')
        performance_data[metric_name] = performance_df[col].values

    # スキル軸分析を実行
    analyzer = SkillAxisAnalyzer()
    analyzer.analyze_skill_axes(z_skill_data, performance_data)

    print("スキル軸分析完了！")
    return analyzer


def generate_axis_based_exemplars(model, analyzer, test_loader, device, save_path):
    """軸ベース個人最適化お手本生成"""
    print("=== 軸ベース個人最適化お手本生成 ===")

    model.eval()

    # 代表データの取得
    with torch.no_grad():
        for trajectories, subject_ids, is_expert in test_loader:
            trajectories = trajectories.to(device)
            encoded = model.encode_hierarchically(trajectories)

            # 最初のサンプルを代表として使用
            z_style = encoded['z_style'][[0]]
            current_skill = encoded['z_skill'][[0]]
            break

    # 異なる改善ターゲットでお手本を生成
    target_metrics = ['overall', 'trial_time', 'trial_error']
    enhancement_factor = 0.15

    generated_exemplars = {}

    with torch.no_grad():
        # 現在レベル
        current_exemplar = model.decode_hierarchically(z_style, current_skill)
        generated_exemplars['current'] = current_exemplar.cpu().numpy().squeeze()

        # 各指標での改善
        for target in target_metrics:
            try:
                improvement_direction = analyzer.get_improvement_direction(target)
                improvement_direction = torch.tensor(
                    improvement_direction,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                enhanced_skill = current_skill + enhancement_factor * improvement_direction
                enhanced_exemplar = model.decode_hierarchically(z_style, enhanced_skill)
                generated_exemplars[target] = enhanced_exemplar.cpu().numpy().squeeze()

            except Exception as e:
                print(f"{target}での改善生成エラー: {e}")
                # フォールバック: ランダム改善
                skill_noise = torch.randn_like(current_skill) * 0.1
                enhanced_skill = current_skill + enhancement_factor * skill_noise
                enhanced_exemplar = model.decode_hierarchically(z_style, enhanced_skill)
                generated_exemplars[target] = enhanced_exemplar.cpu().numpy().squeeze()

    # 可視化
    n_exemplars = len(generated_exemplars)
    fig, axes = plt.subplots(1, n_exemplars, figsize=(5 * n_exemplars, 5))
    if n_exemplars == 1:
        axes = [axes]

    for i, (target, traj) in enumerate(generated_exemplars.items()):
        cumsum_traj = np.cumsum(traj, axis=0)

        axes[i].plot(cumsum_traj[:, 0], cumsum_traj[:, 1], 'b-', linewidth=2, alpha=0.8)
        axes[i].scatter(cumsum_traj[0, 0], cumsum_traj[0, 1], c='green', s=100, label='Start', zorder=5)
        axes[i].scatter(cumsum_traj[-1, 0], cumsum_traj[-1, 1], c='red', s=100, label='End', zorder=5)

        if target == 'current':
            axes[i].set_title(f'Current Level')
            axes[i].plot(cumsum_traj[:, 0], cumsum_traj[:, 1], 'g-', linewidth=3, alpha=0.7)
        else:
            axes[i].set_title(f'Enhanced for\n{target.replace("_", " ").title()}')

        axes[i].set_xlabel('X Position')
        axes[i].set_ylabel('Y Position')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        axes[i].set_aspect('equal')

    plt.suptitle('Axis-Based Personalized Exemplars\n(Performance-Guided Skill Enhancement)', fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"軸ベース個人最適化お手本生成完了: {save_path}")


def quantitative_evaluation_hierarchical(latent_data: dict, test_df: pd.DataFrame, output_dir: str):
    """階層型VAE用の定量的評価"""
    print("=== 階層型VAE定量的評価開始 ===")

    # 再構成性能
    mse = np.mean((latent_data['originals'] - latent_data['reconstructions']) ** 2)
    print(f"再構成MSE: {mse:.6f}")

    # パフォーマンス指標を抽出
    perf_cols = [col for col in test_df.columns if col.startswith('perf_')]
    if not perf_cols:
        raise ValueError("パフォーマンス指標列が見つかりません。")

    performance_df = test_df.groupby(['subject_id', 'trial_num']).first()[['is_expert'] + perf_cols].reset_index()

    # 潜在変数をDataFrameに変換
    z_style_df = pd.DataFrame(
        latent_data['z_style'],
        columns=[f'z_style_{i}' for i in range(latent_data['z_style'].shape[1])]
    )
    z_skill_df = pd.DataFrame(
        latent_data['z_skill'],
        columns=[f'z_skill_{i}' for i in range(latent_data['z_skill'].shape[1])]
    )
    z_primitive_df = pd.DataFrame(
        latent_data['z_primitive'],
        columns=[f'z_primitive_{i}' for i in range(latent_data['z_primitive'].shape[1])]
    )

    # データを結合
    analysis_df = pd.concat([
        performance_df.reset_index(drop=True),
        z_style_df,
        z_skill_df,
        z_primitive_df
    ], axis=1)

    # 階層別相関分析
    correlations = {'style': {}, 'skill': {}, 'primitive': {}}
    metric_names = ['trial_time', 'trial_error', 'jerk', 'path_efficiency', 'approach_angle', 'sparc',
                    'trial_variability']
    available_metrics = [metric for metric in metric_names if f'perf_{metric}' in analysis_df.columns]

    for hierarchy in ['style', 'skill', 'primitive']:
        z_cols = [col for col in analysis_df.columns if f'z_{hierarchy}' in col]

        for metric_name in available_metrics:
            metric_col = f'perf_{metric_name}'
            correlations[hierarchy][metric_name] = []

            for z_col in z_cols:
                subset = analysis_df[[metric_col, z_col]].dropna()

                if len(subset) > 1 and subset[metric_col].std() > 1e-6 and subset[z_col].std() > 1e-6:
                    corr, p_value = pearsonr(subset[z_col], subset[metric_col])
                else:
                    corr, p_value = 0.0, 1.0

                correlations[hierarchy][metric_name].append((corr, p_value))

    # スタイル分離性評価
    style_ari = 0.0
    if 'subject_id' in analysis_df.columns:
        try:
            z_style_data = analysis_df[[col for col in analysis_df.columns if 'z_style' in col]].values
            n_subjects = len(analysis_df['subject_id'].unique())

            if len(z_style_data) > n_subjects:
                kmeans = KMeans(n_clusters=n_subjects, random_state=42)
                style_clusters = kmeans.fit_predict(z_style_data)
                subject_labels = pd.Categorical(analysis_df['subject_id']).codes
                style_ari = adjusted_rand_score(subject_labels, style_clusters)
        except Exception as e:
            print(f"スタイル分離性評価エラー: {e}")

    eval_results = {
        'reconstruction_mse': mse,
        'hierarchical_correlations': correlations,
        'style_separation_score': style_ari,
        'performance_data': performance_df.to_dict('records')
    }

    print("=" * 20 + " 階層型VAE定量的評価完了 " + "=" * 20)
    return eval_results, analysis_df


def run_hierarchical_evaluation_v2(model, test_loader, test_df, output_dir, device, experiment_id):
    """改良版階層型VAE評価（スキル軸分析統合）"""
    print("\n" + "=" * 50)
    print("改良版階層型VAE評価開始（スキル軸分析付き）")
    print("=" * 50)

    # 1. スキル軸分析
    analyzer = run_skill_axis_analysis(model, test_loader, test_df, device)

    # 2. 階層潜在変数抽出
    latent_data = extract_latent_variables_hierarchical(model, test_loader, device)

    # 3. 定量的評価
    eval_results, analysis_df = quantitative_evaluation_hierarchical(latent_data, test_df, output_dir)

    # 4. 軸ベース個人最適化お手本生成
    exemplar_v2_path = os.path.join(output_dir, 'plots', f'axis_based_exemplars_exp{experiment_id}.png')
    generate_axis_based_exemplars(model, analyzer, test_loader, device, exemplar_v2_path)

    # 5. 結果統合
    eval_results.update({
        'skill_axis_analysis_completed': True,
        'axis_based_exemplars_path': exemplar_v2_path,
        'skill_improvement_directions_available': list(analyzer.skill_improvement_directions.keys()),
        'best_skill_correlations': {k: v['correlation'] for k, v in analyzer.performance_correlations.items()}
    })

    # 6. 結果保存
    eval_results_path = os.path.join(output_dir, 'results', f'hierarchical_evaluation_v2_exp{experiment_id}.json')
    os.makedirs(os.path.dirname(eval_results_path), exist_ok=True)

    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(eval_results)

    with open(eval_results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    eval_results['evaluation_results_path'] = eval_results_path

    print("=" * 50)
    print("改良版階層型VAE評価完了")
    print("=" * 50)

    return eval_results


def setup_directories(output_dir):
    """出力ディレクトリを作成"""
    dirs = [
        output_dir,
        os.path.join(output_dir, 'checkpoints'),
        os.path.join(output_dir, 'plots'),
        os.path.join(output_dir, 'logs'),
        os.path.join(output_dir, 'results')
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def plot_hierarchical_training_curves(history, save_path):
    """階層型VAEの学習曲線をプロット"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 全体的な損失
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_yscale('log')

    # 再構成損失
    axes[0, 1].plot(history['recon_loss'], label='Reconstruction Loss', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_yscale('log')

    # 学習率
    axes[0, 2].plot(history['learning_rates'], color='purple')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].set_title('Learning Rate Schedule')
    axes[0, 2].grid(True)
    axes[0, 2].set_yscale('log')

    # KL損失
    axes[1, 0].plot(history['kl_primitive_loss'], label='KL Primitive', color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Loss - Motor Primitives')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_yscale('log')

    axes[1, 1].plot(history['kl_skill_loss'], label='KL Skill', color='brown')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('KL Divergence')
    axes[1, 1].set_title('KL Loss - Skills')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_yscale('log')

    axes[1, 2].plot(history['kl_style_loss'], label='KL Style', color='pink')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('KL Divergence')
    axes[1, 2].set_title('KL Loss - Individual Styles')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    axes[1, 2].set_yscale('log')

    plt.suptitle('Hierarchical VAE Training Curves', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"階層型VAE学習曲線を保存: {save_path}")


def train_hierarchical_model_v2(config_path: str, experiment_id: int, db_path: str):
    """完全版階層型VAE学習（スキル軸分析統合）"""
    print("=== 階層型VAE学習開始 ===")

    # 1. 設定読み込み
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    try:
        config = validate_and_convert_config(config)
        print("階層型VAE設定ファイルの検証: ✅ 正常")
    except ValueError as e:
        print(f"❌ 設定ファイルエラー: {e}")
        update_db(db_path, experiment_id, {
            'status': 'failed',
            'end_time': datetime.now().isoformat(),
            'notes': f"Config validation error: {str(e)}"
        })
        raise e

    # DBに学習開始を通知
    update_db(db_path, experiment_id, {
        'status': 'running',
        'start_time': datetime.now().isoformat()
    })

    try:
        # 2. セットアップ
        output_dir = config['logging']['output_dir']
        setup_directories(output_dir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"-- 階層型VAE実験ID: {experiment_id} | デバイス: {device} ---")

        # 3. データ準備
        try:
            train_loader, val_loader, test_loader, test_df = create_dataloaders(
                config['data']['data_path'],
                config['model']['seq_len'],
                config['training']['batch_size']
            )
            if train_loader is None:
                raise ValueError("データローダーの作成に失敗しました")
        except Exception as e:
            print(f"データ準備エラー: {e}")
            update_db(db_path, experiment_id, {
                'status': 'failed',
                'end_time': datetime.now().isoformat()
            })
            raise e

        # 4. 階層型VAEモデルの初期化
        model = HierarchicalVAE(**config['model']).to(device)

        # オプティマイザとスケジューラ
        optimizer = optim.AdamW(
            model.parameters(),
            config['training']['lr'],
            weight_decay=config['training'].get('weight_decay', 1e-5)
        )

        main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['training'].get('scheduler_T_0', 15),
            T_mult=config['training'].get('scheduler_T_mult', 2),
            eta_min=config['training'].get('scheduler_eta_min', 1e-6)
        )
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=config['training'].get('warmup_epochs', 10)
        )
        scheduler = ChainedScheduler([warmup_scheduler, main_scheduler])

        # 5. 学習ループ
        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = config['training'].get('patience', 20)
        history = {
            'train_loss': [], 'val_loss': [], 'learning_rates': [],
            'recon_loss': [], 'kl_primitive_loss': [], 'kl_skill_loss': [], 'kl_style_loss': []
        }

        print(f"学習開始: {config['training']['num_epochs']}エポック, patience={patience}")

        for epoch in range(config['training']['num_epochs']):
            model.train()
            model.update_epoch(epoch, config['training']['num_epochs'])

            epoch_losses = {
                'total': [], 'recon': [], 'kl_primitive': [], 'kl_skill': [], 'kl_style': []
            }

            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["training"]["num_epochs"]} [Train]')
            for trajectories, subject_id, expertise in progress_bar:
                trajectories = trajectories.to(device)

                optimizer.zero_grad()
                outputs = model(trajectories)

                loss = outputs['total_loss']
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training'].get('clip_grad_norm', 1.0)
                )
                optimizer.step()

                # 損失記録
                epoch_losses['total'].append(outputs['total_loss'].item())
                epoch_losses['recon'].append(outputs['reconstruct_loss'].item())
                epoch_losses['kl_primitive'].append(outputs['kl_primitive'].item())
                epoch_losses['kl_skill'].append(outputs['kl_skill'].item())
                epoch_losses['kl_style'].append(outputs['kl_style'].item())

                progress_bar.set_postfix({'Loss': np.mean(epoch_losses['total'])})

            # エポック損失記録
            history['train_loss'].append(np.mean(epoch_losses['total']))
            history['recon_loss'].append(np.mean(epoch_losses['recon']))
            history['kl_primitive_loss'].append(np.mean(epoch_losses['kl_primitive']))
            history['kl_skill_loss'].append(np.mean(epoch_losses['kl_skill']))
            history['kl_style_loss'].append(np.mean(epoch_losses['kl_style']))
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])

            # 検証ループ
            model.eval()
            epoch_val_losses = []
            with torch.no_grad():
                for trajectories, subject_id, expertise in val_loader:
                    trajectories = trajectories.to(device)
                    outputs = model(trajectories)
                    epoch_val_losses.append(outputs['total_loss'].item())

            current_val_loss = np.mean(epoch_val_losses)
            history['val_loss'].append(current_val_loss)

            print(f"Epoch {epoch + 1}: Train Loss: {history['train_loss'][-1]:.4f}, "
                  f"Val Loss: {current_val_loss:.4f}")
            print(f"  Recon: {history['recon_loss'][-1]:.4f}, "
                  f"KL(Prim): {history['kl_primitive_loss'][-1]:.4f}, "
                  f"KL(Skill): {history['kl_skill_loss'][-1]:.4f}, "
                  f"KL(Style): {history['kl_style_loss'][-1]:.4f}")

            # 学習率更新
            scheduler.step()

            # ベストモデル保存 & アーリーストップ
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                epochs_no_improve = 0
                best_model_path = os.path.join(output_dir, 'checkpoints',
                                               f'best_hierarchical_model_exp{experiment_id}.pth')
                model.save_model(best_model_path)
                print(f" -> New best model saved! (Loss: {best_val_loss:.4f})")
            else:
                epochs_no_improve += 1

            if epochs_no_improve > patience:
                print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
                break

        # 6. 学習終了後の処理
        final_model_path = os.path.join(output_dir, 'checkpoints', f'final_hierarchical_model_exp{experiment_id}.pth')
        model.save_model(final_model_path)

        plot_path = os.path.join(output_dir, 'plots', f'hierarchical_training_curves_exp{experiment_id}.png')
        plot_hierarchical_training_curves(history, plot_path)

        print("=== 学習完了、評価開始 ===")

        # 7. 改良版評価実行
        try:
            eval_results = run_hierarchical_evaluation_v2(model, test_loader, test_df, output_dir, device,
                                                          experiment_id)

            # 最高相関値を計算
            best_correlation = 0.0
            best_correlation_metric = 'none'
            if 'best_skill_correlations' in eval_results:
                for metric, corr in eval_results['best_skill_correlations'].items():
                    if abs(corr) > abs(best_correlation):
                        best_correlation = corr
                        best_correlation_metric = metric

            # 評価結果をDBに記録
            update_db(db_path, experiment_id, {
                'status': 'completed',
                'end_time': datetime.now().isoformat(),
                'final_total_loss': history['train_loss'][-1],
                'best_val_loss': best_val_loss,
                'final_epoch': len(history['train_loss']),
                'early_stopped': epochs_no_improve > patience,

                # 階層別最終損失
                'final_recon_loss': history['recon_loss'][-1],
                'final_kl_primitive': history['kl_primitive_loss'][-1],
                'final_kl_skill': history['kl_skill_loss'][-1],
                'final_kl_style': history['kl_style_loss'][-1],

                # 評価指標
                'reconstruction_mse': eval_results['reconstruction_mse'],
                'style_separation_score': eval_results.get('style_separation_score', 0.0),
                'skill_performance_correlation': best_correlation,
                'best_skill_correlation_metric': best_correlation_metric,

                # スキル軸分析結果
                'skill_axis_analysis_completed': eval_results.get('skill_axis_analysis_completed', False),
                'skill_improvement_directions_count': len(
                    eval_results.get('skill_improvement_directions_available', [])),
                'axis_based_improvement_enabled': True,

                # ファイルパス
                'model_path': final_model_path,
                'best_model_path': best_model_path if 'best_model_path' in locals() else final_model_path,
                'training_curves_path': plot_path,
                'axis_based_exemplars_path': eval_results.get('axis_based_exemplars_path'),
                'evaluation_results_path': eval_results.get('evaluation_results_path'),

                'notes': f"完全版階層型VAE - MSE: {eval_results['reconstruction_mse']:.6f}, "
                         f"Style ARI: {eval_results.get('style_separation_score', 0.0):.4f}, "
                         f"Best Corr: {best_correlation:.4f} ({best_correlation_metric})"
            })

            print(f"=== 実験ID: {experiment_id} 正常完了 ===")
            print(f"最終結果:")
            print(f"  再構成MSE: {eval_results['reconstruction_mse']:.6f}")
            print(f"  スタイル分離ARI: {eval_results.get('style_separation_score', 0.0):.4f}")
            print(f"  最高スキル相関: {best_correlation:.4f} ({best_correlation_metric})")

        except Exception as eval_error:
            print(f"評価時にエラーが発生: {eval_error}")
            import traceback
            traceback.print_exc()

            update_db(db_path, experiment_id, {
                'status': 'completed',
                'end_time': datetime.now().isoformat(),
                'final_total_loss': history['train_loss'][-1],
                'best_val_loss': best_val_loss,
                'model_path': final_model_path,
                'training_curves_path': plot_path,
                'notes': f"学習完了、但し評価失敗: {str(eval_error)}"
            })

    except Exception as e:
        print(f"!!! 階層型VAE実験ID: {experiment_id} でエラー発生: {e} !!!")
        import traceback
        traceback.print_exc()

        update_db(db_path, experiment_id, {
            'status': 'failed',
            'end_time': datetime.now().isoformat(),
            'notes': f"学習中エラー: {str(e)}"
        })
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="完全版階層型VAE学習システム")
    parser.add_argument('--config', type=str, required=True, help='設定ファイルのパス')
    parser.add_argument('--experiment_id', type=int, required=True, help='実験管理DBのID')
    parser.add_argument('--db_path', type=str, required=True, help='実験管理DBのパス')

    args = parser.parse_args()

    print(f"階層型VAE学習開始:")
    print(f"  設定ファイル: {args.config}")
    print(f"  実験ID: {args.experiment_id}")
    print(f"  データベース: {args.db_path}")

    train_hierarchical_model_v2(config_path=args.config, experiment_id=args.experiment_id, db_path=args.db_path)