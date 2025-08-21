import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ChainedScheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import sqlite3
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import seaborn as sns
import json
import joblib

# scikit-learnの特定のUserWarningを無視する
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

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
from models.beta_vae_generalized_coordinate import BetaVAEGeneralizedCoordinate

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

class GeneralizedCoordinateDataset(Dataset):
    def __init__(self, df: pd.DataFrame, scalers: dict,feature_cols: list, seq_len: int = 100):
        self.seq_len = seq_len
        self.scalers = scalers
        self.feature_cols = feature_cols

        # データを試行（trial）ごとにグループ化してリストに変換
        self.trials = list(df.groupby(['subject_id', 'trial_num']))
        print(f"Dataset initialized with {len(self.trials)} trials")
        print(f"Feature columns: {self.feature_cols}")
        print(f"Available scalers: {list(self.scalers.keys())}")

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx) -> tuple:
        """
        指定されたインデックスのデータを取得する

        :return: tuple(軌道テンソル, 被験者ID, 熟練度ラベル)
        """
        # 1. インデックスに対応する試行データを取得
        (subject_id, _), trial_df = self.trials[idx]

        # 2. 特徴量データをNumpy配列に変換
        features = trial_df[self.feature_cols].values

        # 3. 物理量別にスケーリングを適用
        scaled_features = apply_physics_based_scaling(features, self.scalers, self.feature_cols)

        # 4. 固定長化
        if len(scaled_features) > self.seq_len:
            processed_trajectory = scaled_features[:self.seq_len]
        else:
            # 最後の値を繰り返し
            padding_length = self.seq_len - len(scaled_features)
            last_value = scaled_features[-1:].repeat(padding_length, axis=0)
            processed_trajectory = np.vstack([scaled_features, last_value])

        # 5. ラベルを取得
        is_expert = trial_df['is_expert'].iloc[0]

        # 5. テンソルに変換して返す
        return (
            torch.tensor(processed_trajectory, dtype=torch.float32),
            subject_id,
            torch.tensor(is_expert, dtype=torch.long)
        )


def create_dataloaders(processed_data_dir: str, seq_len: int, batch_size: int, random_seed: int = 42) -> tuple:
    """データローダーを作成"""
    # 必要なファイルのパス
    train_data_path = os.path.join(processed_data_dir, 'train_data.parquet')
    test_data_path = os.path.join(processed_data_dir, 'test_data.parquet')
    scalers_path = os.path.join(processed_data_dir, 'scalers.joblib')
    feature_config_path = os.path.join(processed_data_dir, 'feature_config.joblib')

    try:
        # データとスケーラーを読み込み
        train_val_df = pd.read_parquet(train_data_path)
        test_df = pd.read_parquet(test_data_path)
        scalers = joblib.load(scalers_path)
        feature_config = joblib.load(feature_config_path)
        feature_cols = feature_config['feature_cols']

        print(f"Loaded scalers: {list(scalers.keys())}")
        print(f"Feature columns: {feature_cols}")

    except FileNotFoundError as e:
        print(f"エラー: 必要なファイルが見つかりません。-> {e}")
        return None, None, None, None

    # 学習データと検証データに分割 ---
    train_val_subject_ids = train_val_df['subject_id'].unique()

    if len(train_val_subject_ids) < 2:
        # 最低でも1人が学習、1人が検証に必要
        print("警告: 検証セットを作成するには学習データの被験者が2人以上必要です。検証セットなしで進めます。")
        train_ids = train_val_subject_ids
        val_ids = []
    else:
        train_ids, val_ids = train_test_split(
            train_val_subject_ids,
            test_size=0.25,  # 例えば学習データ内の25%を検証用にする (4人いたら3人学習, 1人検証)
            random_state=random_seed
        )

    train_df = train_val_df[train_val_df['subject_id'].isin(train_ids)]
    val_df = train_val_df[train_val_df['subject_id'].isin(val_ids)]

    print(f"データ分割: 学習用={len(train_ids)}人, 検証用={len(val_ids)}人, テスト用={len(test_df['subject_id'].unique())}人")

    # --- 4. Datasetの作成 (scalerを渡す) ---
    train_dataset = GeneralizedCoordinateDataset(train_df, scalers, feature_cols ,seq_len)
    val_dataset = GeneralizedCoordinateDataset(val_df, scalers, feature_cols, seq_len) if not val_df.empty else None
    test_dataset = GeneralizedCoordinateDataset(test_df, scalers,feature_cols, seq_len)

    # --- 5. DataLoaderの作成 ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, test_df

def apply_physics_based_scaling(data: np.ndarray, scalers: dict, feature_cols: list = None) -> np.ndarray:
    """物理量別スケーリングを適用"""

    # デフォルトの特徴量順序
    if feature_cols is None:
        feature_cols = ['HandlePosDiffX', 'HandlePosDiffY', 'HandleVelX', 'HandleVelY',
                        'HandleAccX', 'HandleAccY']

    if data.shape[1] != len(feature_cols):
        raise ValueError(f"Expected {len(feature_cols)} features, got {data.shape[1]}")

    scaled_data = np.zeros_like(data)

    # 特徴量タイプのマッピング
    feature_type_map = {
        'HandlePosDiffX': 'position_diff',
        'HandlePosDiffY': 'position_diff',
        'HandleVelDiffX': 'velocity_diff',
        'HandleVelDiffY': 'velocity_diff',
        'HandleAccDiffX': 'acceleration_diff',
        'HandleAccDiffY': 'acceleration_diff',
        'HandlePosX': 'position',
        'HandlePosY': 'position',
        'HandleVelX': 'velocity',
        'HandleVelY': 'velocity',
        'HandleAccX': 'acceleration',
        'HandleAccY': 'acceleration',
        'JerkX': 'jerk',
        'JerkY': 'jerk'
    }

    # 各特徴量タイプごとにグループ化
    type_indices = {}
    for i, col in enumerate(feature_cols):
        feature_type = feature_type_map.get(col, 'unknown')
        if feature_type not in type_indices:
            type_indices[feature_type] = []
        type_indices[feature_type].append(i)

    # タイプごとにスケーリング適用
    for feature_type, indices in type_indices.items():
        if feature_type in scalers:
            scaled_data[:, indices] = scalers[feature_type].transform(data[:, indices])
            # print(f"Applied {feature_type} scaling to indices {indices}")
        else:
            print(f"Warning: No scaler found for {feature_type}, using original data")
            scaled_data[:, indices] = data[:, indices]

    return scaled_data

def update_db(db_path: str, experiment_id: int, data: dict):
    """データベースの指定された実験IDのレコードを更新する"""
    try:
        with sqlite3.connect(db_path) as conn:
            set_clause = ", ".join([f"{key} = ?" for key in data.keys()])
            query = f"UPDATE beta_vae_generalized_coordinate_experiments SET {set_clause} WHERE id = ?"
            values = tuple(data.values()) + (experiment_id, )
            conn.execute(query, values)
            conn.commit()
    except Exception as e:
        print(f"!!! DB更新エラー (ID: {experiment_id}): {e} !!!")

def validate_and_convert_config(config: dict) -> dict:
    """設定ファイルの妥当性検証と型変換"""

    # 型変換が必要な数値パラメータのマッピング
    numeric_conversions = {
        # モデル設定
        'model.input_dim': int,
        'model.seq_len': int,
        'model.hidden_dim': int,
        'model.style_latent_dim': int,
        'model.skill_latent_dim': int,
        'model.beta': float,
        'model.n_layers': int,
        'model.contrastive_weight': float,
        'model.use_triplet': bool,

        # 学習設定
        'training.batch_size': int,
        'training.num_epochs': int,
        'training.lr': float,
        'training.weight_decay': float,
        'training.clip_grad_norm': float,
        'training.warmup_epochs': int,
        'training.scheduler_T_0': int,
        'training.scheduler_T_mult': int,
        'training.scheduler_eta_min': float,
        'training.patience': int,
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
        'model': ['input_dim', 'seq_len', 'hidden_dim', 'style_latent_dim', 'skill_latent_dim', 'beta', 'n_layers', 'contrastive_weight', 'use_triplet'],
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
    all_subject_ids = []
    all_is_expert = []
    all_reconstructions = []
    all_originals = []

    with torch.no_grad():
        for trajectories, subject_ids, is_expert in tqdm(test_loader, desc="階層潜在変数抽出中"):
            trajectories = trajectories.to(device)

            encoded = model.encode(trajectories)
            z_style = encoded['z_style']
            z_skill = encoded['z_skill']

            reconstructed = model.decode(z_style, z_skill)

            all_z_style.append(z_style.cpu().numpy())
            all_z_skill.append(z_skill.cpu().numpy())
            all_subject_ids.append(subject_ids)
            all_is_expert.append(is_expert.cpu().numpy())
            all_reconstructions.append(reconstructed.cpu().numpy())
            all_originals.append(trajectories.cpu().numpy())

    return {
        'z_style': np.vstack(all_z_style),
        'z_skill': np.vstack(all_z_skill),
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


def simple_vae_diagnosis(model, test_loader, device, output_dir, experiment_id):
    """シンプルで確実に動作するVAE診断"""
    print("=== シンプルVAE診断開始 ===")

    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    all_z_style = []
    all_z_skill = []
    all_subject_ids = []

    print("データ収集中...")
    with torch.no_grad():
        for batch_idx, (trajectories, subject_ids, is_expert) in enumerate(test_loader):
            trajectories = trajectories.to(device)

            try:
                # エンコード
                encoded = model.encode_hierarchically(trajectories)
                all_z_style.append(encoded['z_style'].cpu().numpy())
                all_z_skill.append(encoded['z_skill'].cpu().numpy())
                all_subject_ids.extend(subject_ids)

                if batch_idx == 0:
                    print(f"バッチ形状確認: z_style={encoded['z_style'].shape}, z_skill={encoded['z_skill'].shape}")

            except Exception as e:
                print(f"バッチ {batch_idx} でエラー: {e}")
                continue

    if not all_z_style:
        print("❌ データ収集に失敗しました")
        return None

    # データ統合
    z_style = np.vstack(all_z_style)
    z_skill = np.vstack(all_z_skill)

    print(f"収集完了:")
    print(f"  z_style: {z_style.shape}")
    print(f"  z_skill: {z_skill.shape}")
    print(f"  被験者ID数: {len(set(all_subject_ids))}")
    print(f"  総サンプル数: {len(all_subject_ids)}")

    # 被験者IDを数値ラベルに変換
    unique_subjects = sorted(list(set(all_subject_ids)))
    subject_to_label = {subj: i for i, subj in enumerate(unique_subjects)}
    subject_labels = [subject_to_label[subj] for subj in all_subject_ids]

    print(f"被験者: {unique_subjects}")

    # === 基本統計 ===
    print("\n=== 基本統計 ===")
    style_mean = np.mean(z_style, axis=0)
    style_std = np.std(z_style, axis=0)
    style_var_total = np.var(z_style.flatten())

    print(f"スタイル潜在変数:")
    print(f"  平均の絶対値: {np.mean(np.abs(style_mean)):.4f}")
    print(f"  標準偏差の平均: {np.mean(style_std):.4f}")
    print(f"  全体分散: {style_var_total:.4f}")
    print(f"  最大値: {np.max(z_style):.4f}")
    print(f"  最小値: {np.min(z_style):.4f}")

    # === 被験者間・被験者内分散分析 ===
    print("\n=== 分散分析 ===")

    # 被験者ごとの平均計算
    subject_means = []
    within_vars = []

    for subj in unique_subjects:
        mask = np.array(all_subject_ids) == subj
        subject_data = z_style[mask]

        subject_mean = np.mean(subject_data, axis=0)
        subject_means.append(subject_mean)

        if len(subject_data) > 1:
            within_var = np.var(subject_data, axis=0).mean()
            within_vars.append(within_var)
            print(f"  {subj}: 試行数={len(subject_data)}, 内分散={within_var:.4f}")

    subject_means = np.array(subject_means)
    between_var = np.var(subject_means, axis=0).mean()
    avg_within_var = np.mean(within_vars) if within_vars else 0.01

    separation_ratio = between_var / avg_within_var

    print(f"\n分散分解:")
    print(f"  被験者間分散: {between_var:.4f}")
    print(f"  平均被験者内分散: {avg_within_var:.4f}")
    print(f"  分離比 (between/within): {separation_ratio:.4f}")

    # 判定
    if separation_ratio < 1.0:
        print("  ❌ 被験者間分離が不十分")
    elif separation_ratio < 1.5:
        print("  ⚠️  被験者間分離が弱い")
    else:
        print("  ✅ 被験者間分離が良好")

    # === 可視化 ===
    print("\n=== 可視化作成中 ===")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. スタイル空間のPCA
    try:
        pca = PCA(n_components=min(2, z_style.shape[1]))
        z_style_pca = pca.fit_transform(z_style)

        scatter = axes[0, 0].scatter(z_style_pca[:, 0], z_style_pca[:, 1],
                                     c=subject_labels, cmap='tab10', alpha=0.7, s=20)
        axes[0, 0].set_title(f'Style PCA\n(Contribution rate: {pca.explained_variance_ratio_.sum():.3f})')
        axes[0, 0].set_xlabel('PC1')
        axes[0, 0].set_ylabel('PC2')

        # 被験者ごとの重心をプロット
        for i, subj in enumerate(unique_subjects):
            mask = np.array(subject_labels) == i
            if np.any(mask):
                center = np.mean(z_style_pca[mask], axis=0)
                axes[0, 0].scatter(center[0], center[1], c='red', s=100, marker='x')
                axes[0, 0].annotate(subj, center, xytext=(5, 5), textcoords='offset points')

        plt.colorbar(scatter, ax=axes[0, 0])

    except Exception as e:
        axes[0, 0].text(0.5, 0.5, f'PCA Error: {str(e)}',
                        transform=axes[0, 0].transAxes, ha='center')
        print(f"PCA エラー: {e}")

    # 2. スタイル空間のt-SNE（データが十分な場合のみ）
    if len(z_style) >= 30:
        try:
            perplexity = min(30, len(z_style) // 4)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=300)
            z_style_tsne = tsne.fit_transform(z_style)

            scatter = axes[0, 1].scatter(z_style_tsne[:, 0], z_style_tsne[:, 1],
                                         c=subject_labels, cmap='tab10', alpha=0.7, s=20)
            axes[0, 1].set_title('Style t-SNE')
            plt.colorbar(scatter, ax=axes[0, 1])

        except Exception as e:
            axes[0, 1].text(0.5, 0.5, f't-SNE Error: {str(e)}',
                            transform=axes[0, 1].transAxes, ha='center')
            print(f"t-SNE Error: {e}")
    else:
        axes[0, 1].text(0.5, 0.5, 'Insufficient number of samples\n(t-SNE)',
                        transform=axes[0, 1].transAxes, ha='center')

    # 3. 次元別分散
    dim_vars = np.var(z_style, axis=0)
    axes[0, 2].bar(range(len(dim_vars)), dim_vars)
    axes[0, 2].set_title('Style dim variance')
    axes[0, 2].set_xlabel('dim')
    axes[0, 2].set_ylabel('variance')

    # 4. 被験者別分布（最初の2次元）
    for i, subj in enumerate(unique_subjects):
        mask = np.array(subject_labels) == i
        if np.any(mask):
            subject_data = z_style[mask]
            axes[1, 0].scatter(subject_data[:, 0], subject_data[:, 1],
                               label=f'{subj}', alpha=0.6, s=20)

    axes[1, 0].set_title('Style Space (dim0 vs 1)')
    axes[1, 0].set_xlabel('Style dim 0')
    axes[1, 0].set_ylabel('Style dim 1')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 5. 被験者間距離行列
    distance_matrix = np.zeros((len(unique_subjects), len(unique_subjects)))
    for i, subj_i in enumerate(unique_subjects):
        for j, subj_j in enumerate(unique_subjects):
            mean_i = subject_means[i]
            mean_j = subject_means[j]
            distance_matrix[i, j] = np.linalg.norm(mean_i - mean_j)

    im = axes[1, 1].imshow(distance_matrix, cmap='viridis')
    axes[1, 1].set_title('Inter-subject distance matrix')
    axes[1, 1].set_xticks(range(len(unique_subjects)))
    axes[1, 1].set_yticks(range(len(unique_subjects)))
    axes[1, 1].set_xticklabels(unique_subjects, rotation=45)
    axes[1, 1].set_yticklabels(unique_subjects)
    plt.colorbar(im, ax=axes[1, 1])

    # 6. 活性化度分布
    activation_magnitude = np.linalg.norm(z_style, axis=1)
    subject_activations = [activation_magnitude[np.array(subject_labels) == i]
                           for i in range(len(unique_subjects))]

    bp = axes[1, 2].boxplot(subject_activations, labels=unique_subjects)
    axes[1, 2].set_title('Activation level by subject')
    axes[1, 2].set_xlabel('subjects')
    axes[1, 2].set_ylabel('||z_style||')
    axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # 保存
    save_path = os.path.join(output_dir, f'vae_simple_diagnosis_exp{experiment_id}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"可視化保存: {save_path}")

    # === 結果サマリー ===
    print("\n" + "=" * 50)
    print("診断結果サマリー")
    print("=" * 50)
    print(f"✅ データ収集: 成功")
    print(f"📊 サンプル数: {len(z_style)}")
    print(f"👥 被験者数: {len(unique_subjects)}")
    print(f"📈 分離比: {separation_ratio:.3f}")
    print(f"🎯 スタイル活性化: {np.mean(style_std):.3f}")
    print(f"📉 全体分散: {style_var_total:.3f}")

    if separation_ratio < 1.0:
        print("\n🚨 問題: 被験者間分離が不十分")
        print("   💡 推奨: beta_styleを0.05以下に下げる")
        print("   💡 推奨: style_learning_startを0.15に早める")

    if np.mean(style_std) < 0.2:
        print("\n⚠️  問題: スタイル潜在変数の活性化不足")
        print("   💡 推奨: beta_styleをさらに下げる")
        print("   💡 推奨: 学習率を上げる")

    return {
        'separation_ratio': separation_ratio,
        'style_activation': np.mean(style_std),
        'style_variance': style_var_total,
        'subject_means': subject_means,
        'distance_matrix': distance_matrix,
        'z_style': z_style,
        'z_skill': z_skill,
        'subject_labels': subject_labels,
        'unique_subjects': unique_subjects
    }

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
    """BetaVAE用の定量的評価"""
    print("=== BetaVAE定量的評価開始 ===")

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

    # データを結合
    analysis_df = pd.concat([
        performance_df.reset_index(drop=True),
        z_style_df,
        z_skill_df,
    ], axis=1)

    # 階層別相関分析
    correlations = {'style': {}, 'skill': {}}
    metric_names = ['trial_time', 'trial_error', 'jerk', 'path_efficiency', 'approach_angle', 'sparc',
                    'trial_variability']
    available_metrics = [metric for metric in metric_names if f'perf_{metric}' in analysis_df.columns]

    for hierarchy in ['style', 'skill']:
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

    print("=" * 20 + " BetaVAE定量的評価完了 " + "=" * 20)
    return eval_results, analysis_df


def run_beta_vae_evaluation(model, test_loader, test_df, output_dir, device, experiment_id):
    """改良版BetaVAE評価（スキル軸分析統合）"""
    print("\n" + "=" * 50)
    print("改良版BetaVAE評価開始（スキル軸分析付き）")
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

    # 5. 潜在空間の可視化
    latent_space_result_path = os.path.join(output_dir, 'latent_space')
    latent_space_result = simple_vae_diagnosis(model, test_loader, device, latent_space_result_path, experiment_id)

    # 6. 結果統合
    eval_results.update({
        'skill_axis_analysis_completed': True,
        'axis_based_exemplars_path': exemplar_v2_path,
        'skill_improvement_directions_available': list(analyzer.skill_improvement_directions.keys()),
        'best_skill_correlations': {k: v['correlation'] for k, v in analyzer.performance_correlations.items()},
        'latent_space_analysis': latent_space_result
    })

    # 7. 結果保存
    eval_results_path = os.path.join(output_dir, 'results', f'beta_vae_evaluation_exp{experiment_id}.json')
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
    print("改良版BetaVAE評価完了")
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
    axes[1, 0].plot(history['contrastive_loss'], label='Contrastive Loss', color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Contrastive')
    axes[1, 0].set_title('Contrastive Loss')
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

    plt.suptitle('Beta VAE Training Curves', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"BetaVAE学習曲線を保存: {save_path}")

def train_beta_vae_generalized_coordinate_model(config_path: str, experiment_id: int, db_path: str):
    """完全版BetaVAE学習（スキル軸分析統合）"""
    print("=== BetaVAE学習開始 ===")

    # 1. 設定読み込み
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    try:
        config = validate_and_convert_config(config)
        print("BetaVAE設定ファイルの検証: ✅ 正常")
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
        print(f"-- BetaVAE実験ID: {experiment_id} | デバイス: {device} ---")

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

        # 4. BetaVAEモデルの初期化
        model = BetaVAEGeneralizedCoordinate(**config['model']).to(device)

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
            'recon_loss': [], 'kl_style_loss': [], 'kl_skill_loss': [], 'contrastive_loss': [],
        }

        print(f"学習開始: {config['training']['num_epochs']}エポック, patience={patience}")

        for epoch in range(config['training']['num_epochs']):
            model.train()
            model.update_epoch(epoch, config['training']['num_epochs'])

            epoch_losses = {
                'total': [], 'recon': [], 'kl_skill': [], 'kl_style': [], 'contrastive_loss': []
            }

            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["training"]["num_epochs"]} [Train]')
            for trajectories, subject_id, expertise in progress_bar:
                trajectories = trajectories.to(device)

                optimizer.zero_grad()
                outputs = model(trajectories, subject_id)

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
                epoch_losses['kl_style'].append(outputs['kl_style'].item())
                epoch_losses['kl_skill'].append(outputs['kl_skill'].item())
                epoch_losses['contrastive_loss'].append(outputs['contrastive_loss'].item())

                progress_bar.set_postfix({'Loss': np.mean(epoch_losses['total'])})

            # エポック損失記録
            history['train_loss'].append(np.mean(epoch_losses['total']))
            history['recon_loss'].append(np.mean(epoch_losses['recon']))
            history['kl_style_loss'].append(np.mean(epoch_losses['kl_style']))
            history['kl_skill_loss'].append(np.mean(epoch_losses['kl_skill']))
            history['contrastive_loss'].append(np.mean(epoch_losses['contrastive_loss']))
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
                  f"KL(Skill): {history['kl_skill_loss'][-1]:.4f}, "
                  f"KL(Style): {history['kl_style_loss'][-1]:.4f}, "
                  f"Contrastive: {history['contrastive_loss'][-1]:.4f}, ")

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
        final_model_path = os.path.join(output_dir, 'checkpoints', f'final_beta_vae_model_exp{experiment_id}.pth')
        model.save_model(final_model_path)

        plot_path = os.path.join(output_dir, 'plots', f'beta_vae_training_curves_exp{experiment_id}.png')
        plot_hierarchical_training_curves(history, plot_path)

        print("=== 学習完了、評価開始 ===")

        # 7. 改良版評価実行
        try:
            eval_results = run_beta_vae_evaluation(model, test_loader, test_df, output_dir, device,
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
                'final_kl_style': history['kl_style_loss'][-1],
                'final_kl_skill': history['kl_skill_loss'][-1],
                'final_contrastive_loss': history['contrastive_loss'][-1],

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
    parser = argparse.ArgumentParser(description="一般化座標型BetaVAE学習システム")
    parser.add_argument('--config', type=str, required=True, help='設定ファイルのパス')
    parser.add_argument('--experiment_id', type=int, required=True, help='実験管理DBのID')
    parser.add_argument('--db_path', type=str, required=True, help='実験管理DBのパス')

    args = parser.parse_args()

    print(f"階層型VAE学習開始:")
    print(f"  設定ファイル: {args.config}")
    print(f"  実験ID: {args.experiment_id}")
    print(f"  データベース: {args.db_path}")

    train_beta_vae_generalized_coordinate_model(config_path=args.config, experiment_id=args.experiment_id, db_path=args.db_path)