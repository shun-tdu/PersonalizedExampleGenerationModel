import os
import sys
import argparse
import warnings
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
    from models.hierarchical_vae_generalized_coordinate import GeneralizedCoordinateHierarchicalVAE
except ImportError:
    print("警告: models.hierarchical_vae のインポートに失敗しました")
    sys.exit(1)

try:
    from DataPreprocess.data_preprocess import calculate_jerk, calculate_path_efficiency, calculate_approach_angle, \
        calculate_sparc
except ImportError:
    print("警告: data_preprocess のインポートに失敗しました")


class SkillAxisAnalyzer:
    """スキル潜在空間での上手さの軸を分析・抽出するクラス（一般化座標対応版）"""

    def __init__(self):
        self.skill_improvement_directions = {}
        self.performance_correlations = {}

    def analyze_skill_axes(self, z_skill_data, performance_data):
        """スキル潜在変数とパフォーマンス指標の相関分析"""
        print("=== 一般化座標VAE スキル軸分析開始 ===")

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

            # 指標の方向性を統一（低い方が良い指標は符号反転）
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
    """
    一般化座標を使用したデータセット
    既に計算済みの速度・加速度データ（100Hz）を活用
    """

    def __init__(self,
                 df: pd.DataFrame,
                 seq_len: int = 100,
                 dt: float = 0.01,  # 100Hz sampling = 0.01s interval
                 use_precomputed: bool = True):
        """
        Args:
            df: 軌道データフレーム（既に速度・加速度計算済み）
            seq_len: シーケンス長
            dt: サンプリング間隔（100Hz = 0.01s）
            use_precomputed: 事前計算済みの速度・加速度を使用するか
        """
        self.seq_len = seq_len
        self.dt = dt
        self.use_precomputed = use_precomputed

        # 必要なカラムの定義
        self.position_cols = ['HandlePosX', 'HandlePosY']
        self.velocity_cols = ['HandleVelX', 'HandleVelY'] if use_precomputed else None
        self.acceleration_cols = ['HandleAccX', 'HandleAccY'] if use_precomputed else None

        # 試行ごとにデータをグループ化
        self.trials = list(df.groupby(['subject_id', 'trial_num']))

        print(f"✅ 一般化座標データセット初期化完了")
        print(f"   - 総試行数: {len(self.trials)}")
        print(f"   - シーケンス長: {seq_len}")
        print(f"   - サンプリング周波数: {1 / dt:.0f}Hz")
        print(f"   - 事前計算済みデータ使用: {use_precomputed}")

    def __len__(self) -> int:
        return len(self.trials)

    def extract_generalized_coordinates(self, trial_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        試行データから一般化座標を抽出

        Returns:
            basic_coords: [seq_len, 8] 基本座標（VAE学習用）
            full_coords: [seq_len, 12] 完全座標（評価用）
        """
        # === 基本座標の取得 ===

        # 1. 位置（0次）
        position = trial_df[self.position_cols].values  # [seq_len, 2]

        # 2. 速度（1次）
        if self.use_precomputed and self.velocity_cols:
            velocity = trial_df[self.velocity_cols].values
        else:
            # 数値微分で計算
            velocity = np.gradient(position, axis=0) / self.dt

        # 3. 加速度（2次）
        if self.use_precomputed and self.acceleration_cols:
            acceleration = trial_df[self.acceleration_cols].values
        else:
            # 数値微分で計算
            acceleration = np.gradient(velocity, axis=0) / self.dt

        # 4. ジャーク（3次）- 常に数値微分で計算
        jerk = np.gradient(acceleration, axis=0) / self.dt

        # === 合成特徴の計算 ===

        # 5. 速度の大きさ
        speed = np.linalg.norm(velocity, axis=1, keepdims=True)

        # 6. 加速度の大きさ
        acceleration_magnitude = np.linalg.norm(acceleration, axis=1, keepdims=True)

        # 7. ジャークの大きさ
        jerk_magnitude = np.linalg.norm(jerk, axis=1, keepdims=True)

        # 8. 曲率
        curvature = self._compute_curvature(position)

        # === 座標の結合 ===

        # 基本座標（VAE学習用）: 8次元
        basic_coords = np.concatenate([
            position,  # 2次元
            velocity,  # 2次元
            acceleration,  # 2次元
            jerk  # 2次元
        ], axis=1)

        # 完全座標（評価用）: 12次元
        full_coords = np.concatenate([
            basic_coords,  # 8次元
            speed,  # 1次元
            acceleration_magnitude,  # 1次元
            jerk_magnitude,  # 1次元
            curvature  # 1次元
        ], axis=1)

        return basic_coords, full_coords

    def _compute_curvature(self, position: np.ndarray) -> np.ndarray:
        """
        軌道の曲率を計算

        Args:
            position: [seq_len, 2] 位置データ

        Returns:
            curvature: [seq_len, 1] 曲率
        """
        if len(position) < 3:
            return np.zeros((len(position), 1))

        # 1次・2次微分
        dx = np.gradient(position[:, 0])
        dy = np.gradient(position[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # 曲率公式: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx ** 2 + dy ** 2) ** (3 / 2)

        # ゼロ除算回避
        denominator = np.where(denominator < 1e-8, 1e-8, denominator)
        curvature = numerator / denominator

        # 異常値を制限
        curvature = np.clip(curvature, 0, 100)

        return curvature.reshape(-1, 1)

    def _pad_or_truncate(self, coords: np.ndarray) -> np.ndarray:
        """
        固定長化処理（パディングまたは切り捨て）

        Args:
            coords: [actual_len, coord_dim] 座標データ

        Returns:
            processed_coords: [seq_len, coord_dim] 固定長座標
        """
        actual_len, coord_dim = coords.shape

        if actual_len > self.seq_len:
            # 切り捨て
            return coords[:self.seq_len]
        elif actual_len < self.seq_len:
            # ゼロパディング
            padding = np.zeros((self.seq_len - actual_len, coord_dim))
            return np.vstack([coords, padding])
        else:
            return coords

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        データを取得

        Returns:
            full_coords: [seq_len, 12] 完全な一般化座標
            subject_id: 被験者ID
            is_expert: 熟練度ラベル
        """
        (subject_id, trial_num), trial_df = self.trials[idx]

        # 一般化座標を抽出
        basic_coords, full_coords = self.extract_generalized_coordinates(trial_df)

        # 固定長化
        full_coords = self._pad_or_truncate(full_coords)

        # 熟練度ラベルを取得
        is_expert = trial_df['is_expert'].iloc[0] if 'is_expert' in trial_df.columns else 0

        return (
            torch.tensor(full_coords, dtype=torch.float32),
            subject_id,
            torch.tensor(is_expert, dtype=torch.long)
        )

class TrajectoryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = 100, feature_cols=['HandlePosX', 'HandlePosY','HandleVelX','HandleVelY','HandleAccX','HandleAccY']):
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


def create_generalized_dataloaders(master_data_path: str,
                                   seq_len: int,
                                   batch_size: int,
                                   use_precomputed: bool = True,
                                   random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    """
    一般化座標データローダーを作成

    Args:
        master_data_path: データファイルパス
        seq_len: シーケンス長
        batch_size: バッチサイズ
        use_precomputed: 事前計算済み速度・加速度を使用
        random_seed: 乱数シード

    Returns:
        train_loader, val_loader, test_loader, test_df
    """
    try:
        master_df = pd.read_parquet(master_data_path)
        print(f"✅ データファイル読み込み完了: {master_data_path}")
        print(f"   - 総レコード数: {len(master_df):,}")
        print(f"   - 被験者数: {master_df['subject_id'].nunique()}")
        print(f"   - 試行数: {master_df.groupby(['subject_id', 'trial_num']).ngroups}")
    except FileNotFoundError:
        print(f"❌ エラー: ファイル '{master_data_path}' が見つかりません。")
        return None, None, None, None
    except Exception as e:
        print(f"❌ エラー: ファイル読み込みに失敗: {e}")
        return None, None, None, None

    # 必要なカラムの確認
    required_cols = ['subject_id', 'trial_num', 'HandlePosX', 'HandlePosY']
    if use_precomputed:
        required_cols.extend(['HandleVelX', 'HandleVelY', 'HandleAccX', 'HandleAccY'])

    missing_cols = [col for col in required_cols if col not in master_df.columns]
    if missing_cols:
        print(f"❌ エラー: 必要なカラムが不足しています: {missing_cols}")
        return None, None, None, None

    # データ品質チェック
    print("\n📊 データ品質チェック:")
    for col in ['HandlePosX', 'HandlePosY']:
        var_val = master_df[col].var()
        print(f"   - {col} 分散: {var_val:.6f}")

    if use_precomputed:
        for col in ['HandleVelX', 'HandleVelY', 'HandleAccX', 'HandleAccY']:
            var_val = master_df[col].var()
            print(f"   - {col} 分散: {var_val:.6f}")

    # 被験者ベースデータ分割
    np.random.seed(random_seed)
    subject_ids = master_df['subject_id'].unique()
    np.random.shuffle(subject_ids)

    if len(subject_ids) < 3:
        raise ValueError("データセットの分割には最低3人の被験者が必要です。")

    # 分割比率: train 60%, val 20%, test 20%
    n_subjects = len(subject_ids)
    n_train = max(1, int(n_subjects * 0.6))
    n_val = max(1, int(n_subjects * 0.2))
    n_test = n_subjects - n_train - n_val

    if n_test < 1:
        n_test = 1
        n_val = n_subjects - n_train - n_test

    train_ids = subject_ids[:n_train]
    val_ids = subject_ids[n_train:n_train + n_val]
    test_ids = subject_ids[n_train + n_val:]

    train_df = master_df[master_df['subject_id'].isin(train_ids)]
    val_df = master_df[master_df['subject_id'].isin(val_ids)]
    test_df = master_df[master_df['subject_id'].isin(test_ids)]

    print(f"\n🔄 データ分割:")
    print(f"   - 学習用: {len(train_ids)}人 ({len(train_df):,}レコード)")
    print(f"   - 検証用: {len(val_ids)}人 ({len(val_df):,}レコード)")
    print(f"   - テスト用: {len(test_ids)}人 ({len(test_df):,}レコード)")

    # データセット作成
    train_dataset = GeneralizedCoordinateDataset(train_df, seq_len=seq_len, use_precomputed=use_precomputed)
    val_dataset = GeneralizedCoordinateDataset(val_df, seq_len=seq_len, use_precomputed=use_precomputed)
    test_dataset = GeneralizedCoordinateDataset(test_df, seq_len=seq_len, use_precomputed=use_precomputed)

    # データローダー作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n📦 データローダー作成完了:")
    print(f"   - 学習バッチ数: {len(train_loader)}")
    print(f"   - 検証バッチ数: {len(val_loader)}")
    print(f"   - テストバッチ数: {len(test_loader)}")
    print(f"   - バッチサイズ: {batch_size}")

    return train_loader, val_loader, test_loader, test_df


def analyze_generalized_coordinates(dataset: GeneralizedCoordinateDataset,
                                    num_samples: int = 5) -> None:
    """
    一般化座標データセットの分析

    Args:
        dataset: 一般化座標データセット
        num_samples: 分析するサンプル数
    """
    print(f"\n🔬 一般化座標分析（サンプル数: {num_samples}）")
    print("=" * 60)

    coord_names = [
        'pos_x', 'pos_y', 'vel_x', 'vel_y', 'acc_x', 'acc_y', 'jerk_x', 'jerk_y',
        'speed', 'acc_magnitude', 'jerk_magnitude', 'curvature'
    ]

    all_coords = []
    subject_info = []

    # サンプルデータを収集
    for i in range(min(num_samples, len(dataset))):
        coords, subject_id, is_expert = dataset[i]
        coords_np = coords.numpy()
        all_coords.append(coords_np)
        subject_info.append((subject_id, is_expert.item()))

    if not all_coords:
        print("❌ 分析用データがありません")
        return

    all_coords = np.stack(all_coords)  # [num_samples, seq_len, 12]

    # 統計情報を計算
    print("📊 各座標の統計情報:")
    print(f"{'座標名':<15} {'平均':<12} {'標準偏差':<12} {'最小値':<12} {'最大値':<12}")
    print("-" * 60)

    for i, name in enumerate(coord_names):
        coord_data = all_coords[:, :, i].flatten()
        coord_data = coord_data[coord_data != 0]  # パディングを除外

        if len(coord_data) > 0:
            mean_val = np.mean(coord_data)
            std_val = np.std(coord_data)
            min_val = np.min(coord_data)
            max_val = np.max(coord_data)

            print(f"{name:<15} {mean_val:<12.6f} {std_val:<12.6f} {min_val:<12.6f} {max_val:<12.6f}")

    # 被験者情報
    print(f"\n👥 被験者情報:")
    for i, (subject_id, is_expert) in enumerate(subject_info):
        expert_status = "熟達者" if is_expert else "初心者"
        print(f"   サンプル {i + 1}: 被験者{subject_id} ({expert_status})")

    # 座標間の相関（基本座標vs合成特徴）
    print(f"\n🔗 基本座標と合成特徴の整合性チェック:")

    # 速度の大きさの整合性
    vel_x = all_coords[:, :, 2].flatten()
    vel_y = all_coords[:, :, 3].flatten()
    speed_computed = np.sqrt(vel_x ** 2 + vel_y ** 2)
    speed_stored = all_coords[:, :, 8].flatten()

    # 非ゼロ要素のみで比較
    mask = (vel_x != 0) | (vel_y != 0)
    if mask.sum() > 0:
        speed_error = np.mean(np.abs(speed_computed[mask] - speed_stored[mask]))
        print(f"   速度大きさ誤差: {speed_error:.8f}")

    # 加速度の大きさの整合性
    acc_x = all_coords[:, :, 4].flatten()
    acc_y = all_coords[:, :, 5].flatten()
    acc_mag_computed = np.sqrt(acc_x ** 2 + acc_y ** 2)
    acc_mag_stored = all_coords[:, :, 9].flatten()

    mask = (acc_x != 0) | (acc_y != 0)
    if mask.sum() > 0:
        acc_error = np.mean(np.abs(acc_mag_computed[mask] - acc_mag_stored[mask]))
        print(f"   加速度大きさ誤差: {acc_error:.8f}")


def update_db(db_path: str, experiment_id: int, data: dict):
    """一般化座標VAE実験データベースを更新"""
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
    """設定ファイルの妥当性検証と型変換（一般化座標VAE用）"""

    # 型変換が必要な数値パラメータのマッピング
    numeric_conversions = {
        # モデル設定
        'model.basic_coord_dim': int,
        'model.derived_coord_dim': int,
        'model.seq_len': int,
        'model.hidden_dim': int,
        'model.primitive_latent_dim': int,
        'model.skill_latent_dim': int,
        'model.style_latent_dim': int,
        'model.beta_primitive': float,
        'model.beta_skill': float,
        'model.beta_style': float,
        'model.physics_weight': float,
        'model.separation_weight': float,

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

        # データ設定
        'data.use_precomputed': bool,
        'data.random_seed': int,

        # 個人最適化設定
        'exemplar_generation.skill_enhancement_factor': float,
        'exemplar_generation.confidence_threshold': float,
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
                    elif target_type == bool:
                        converted_value = str(current_value).lower() in ['true', '1', 'yes']
                    else:
                        converted_value = target_type(current_value)
                elif isinstance(current_value, (int, float, bool)):
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
        'model': ['basic_coord_dim', 'derived_coord_dim', 'seq_len', 'hidden_dim', 'primitive_latent_dim', 'skill_latent_dim', 'style_latent_dim'],
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


def extract_latent_variables_generalized(model, test_loader, device):
    """一般化座標VAEからテストデータの潜在変数を抽出"""
    model.eval()
    all_z_style = []
    all_z_skill = []
    all_z_primitive = []
    all_subject_ids = []
    all_is_expert = []
    all_reconstructions = []
    all_originals = []

    with torch.no_grad():
        for full_coords, subject_ids, is_expert in tqdm(test_loader, desc="一般化座標潜在変数抽出中"):
            full_coords = full_coords.to(device)

            # 基本座標を抽出してエンコード
            basic_coords, _ = model.split_coordinates(full_coords)
            encoded = model.encode_hierarchically(basic_coords)

            z_style = encoded['z_style']
            z_skill = encoded['z_skill']
            z_primitive = encoded['z_primitive']

            # 完全座標で再構成
            outputs = model(full_coords, subject_ids, is_expert)
            reconstructed = outputs['reconstructed']

            all_z_style.append(z_style.cpu().numpy())
            all_z_skill.append(z_skill.cpu().numpy())
            all_z_primitive.append(z_primitive.cpu().numpy())
            all_subject_ids.extend(subject_ids)
            all_is_expert.append(is_expert.cpu().numpy())
            all_reconstructions.append(reconstructed.cpu().numpy())
            all_originals.append(full_coords.cpu().numpy())

    return {
        'z_style': np.vstack(all_z_style),
        'z_skill': np.vstack(all_z_skill),
        'z_primitive': np.vstack(all_z_primitive),
        'subject_ids': all_subject_ids,
        'is_expert': np.concatenate(all_is_expert),
        'reconstructions': np.vstack(all_reconstructions),
        'originals': np.vstack(all_originals)
    }


def run_skill_axis_analysis_generalized(model, test_loader, test_df, device):
    """一般化座標VAE用スキル軸分析を実行"""
    print("=== 一般化座標VAE スキル軸分析開始 ===")

    model.eval()
    all_z_skill = []
    all_subject_ids = []

    # スキル潜在変数を抽出
    with torch.no_grad():
        for full_coords, subject_ids, is_expert in test_loader:
            full_coords = full_coords.to(device)
            basic_coords, _ = model.split_coordinates(full_coords)
            encoded = model.encode_hierarchically(basic_coords)
            all_z_skill.append(encoded['z_skill'].cpu().numpy())
            all_subject_ids.extend(subject_ids)

    z_skill_data = np.vstack(all_z_skill)

    # パフォーマンス指標を取得
    perf_cols = [col for col in test_df.columns if col.startswith('perf_')]
    if perf_cols:
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
    else:
        print("警告: パフォーマンス指標が見つかりません。ダミーデータで分析します。")
        # ダミーデータ生成（デモ用）
        performance_data = {
            'smoothness': np.random.randn(len(z_skill_data)),
            'efficiency': np.random.randn(len(z_skill_data)),
            'accuracy': np.random.randn(len(z_skill_data))
        }

    # スキル軸分析を実行
    analyzer = SkillAxisAnalyzer()
    analyzer.analyze_skill_axes(z_skill_data, performance_data)

    print("一般化座標VAE スキル軸分析完了！")
    return analyzer


def generate_axis_based_exemplars_generalized(model, analyzer, test_loader, device, save_path):
    """一般化座標VAE用軸ベース個人最適化お手本生成"""
    print("=== 一般化座標VAE 軸ベース個人最適化お手本生成 ===")

    model.eval()

    # 代表データの取得
    with torch.no_grad():
        for full_coords, subject_ids, is_expert in test_loader:
            full_coords = full_coords.to(device)

            # 最初のサンプルを代表として使用
            learner_coords = full_coords[[0]]
            break

    # 異なる改善ターゲットでお手本を生成
    target_metrics = ['overall']
    if len(analyzer.skill_improvement_directions) > 1:
        available_metrics = list(analyzer.skill_improvement_directions.keys())
        target_metrics.extend([m for m in available_metrics[:2] if m != 'overall'])

    enhancement_factor = 0.15
    generated_exemplars = {}

    with torch.no_grad():
        # 現在レベル
        current_exemplar = model.generate_personalized_exemplar(learner_coords, skill_enhancement_factor=0.0)
        generated_exemplars['current'] = current_exemplar.cpu().numpy().squeeze()

        # 各指標での改善
        for target in target_metrics:
            if target == 'overall':
                continue

            try:
                # カスタムお手本生成（改善方向指定）
                basic_coords, _ = model.split_coordinates(learner_coords)
                encoded = model.encode_hierarchically(basic_coords)
                learner_style = encoded['z_style']
                learner_skill = encoded['z_skill']

                improvement_direction = analyzer.get_improvement_direction(target)
                improvement_direction = torch.tensor(
                    improvement_direction,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                enhanced_skill = learner_skill + enhancement_factor * improvement_direction
                enhanced_basic = model.decode_hierarchically(learner_style, enhanced_skill)
                enhanced_derived = model.compute_derived_features(enhanced_basic)
                enhanced_exemplar = model.combine_coordinates(enhanced_basic, enhanced_derived)

                generated_exemplars[target] = enhanced_exemplar.cpu().numpy().squeeze()

            except Exception as e:
                print(f"{target}での改善生成エラー: {e}")
                # フォールバック: デフォルト改善
                enhanced_exemplar = model.generate_personalized_exemplar(learner_coords,
                                                                         skill_enhancement_factor=enhancement_factor)
                generated_exemplars[target] = enhanced_exemplar.cpu().numpy().squeeze()

    # 可視化
    n_exemplars = len(generated_exemplars)
    fig, axes = plt.subplots(1, n_exemplars, figsize=(5 * n_exemplars, 5))
    if n_exemplars == 1:
        axes = [axes]

    for i, (target, coords) in enumerate(generated_exemplars.items()):
        # 基本座標から位置軌道を抽出（最初の2次元が位置）
        position_coords = coords[:, :2]

        axes[i].plot(position_coords[:, 0], position_coords[:, 1], 'b-', linewidth=2, alpha=0.8)
        axes[i].scatter(position_coords[0, 0], position_coords[0, 1], c='green', s=100, label='Start', zorder=5)
        axes[i].scatter(position_coords[-1, 0], position_coords[-1, 1], c='red', s=100, label='End', zorder=5)

        if target == 'current':
            axes[i].set_title(f'Current Level')
            axes[i].plot(position_coords[:, 0], position_coords[:, 1], 'g-', linewidth=3, alpha=0.7)
        else:
            axes[i].set_title(f'Enhanced for\n{target.replace("_", " ").title()}')

        axes[i].set_xlabel('X Position (m)')
        axes[i].set_ylabel('Y Position (m)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        axes[i].set_aspect('equal')

    plt.suptitle(
        'Generalized Coordinate VAE: Axis-Based Personalized Exemplars\n(Physics-Consistent Skill Enhancement)',
        fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"一般化座標VAE 軸ベース個人最適化お手本生成完了: {save_path}")


def quantitative_evaluation_generalized(latent_data: dict, test_df: pd.DataFrame, output_dir: str):
    """一般化座標VAE用の定量的評価"""
    print("=== 一般化座標VAE定量的評価開始 ===")

    # 再構成性能
    mse = np.mean((latent_data['originals'] - latent_data['reconstructions']) ** 2)
    print(f"再構成MSE: {mse:.6f}")

    # 物理的整合性評価
    originals = latent_data['originals']
    reconstructions = latent_data['reconstructions']

    # 基本座標 vs 合成特徴の整合性
    basic_orig = originals[:, :, :8]  # 基本座標
    derived_orig = originals[:, :, 8:]  # 合成特徴
    basic_recon = reconstructions[:, :, :8]
    derived_recon = reconstructions[:, :, 8:]

    basic_mse = np.mean((basic_orig - basic_recon) ** 2)
    derived_mse = np.mean((derived_orig - derived_recon) ** 2)

    print(f"基本座標MSE: {basic_mse:.6f}")
    print(f"合成特徴MSE: {derived_mse:.6f}")
    print(f"物理的整合性比: {derived_mse / (basic_mse + 1e-8):.6f}")

    # パフォーマンス指標を抽出
    perf_cols = [col for col in test_df.columns if col.startswith('perf_')]

    if perf_cols:
        performance_df = test_df.groupby(['subject_id', 'trial_num']).first()[['is_expert'] + perf_cols].reset_index()
    else:
        print("警告: パフォーマンス指標列が見つかりません。")
        performance_df = pd.DataFrame({'is_expert': [0, 1] * (len(latent_data['z_style']) // 2)})

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
    min_len = min(len(performance_df), len(z_style_df))
    analysis_df = pd.concat([
        performance_df.reset_index(drop=True).iloc[:min_len],
        z_style_df.iloc[:min_len],
        z_skill_df.iloc[:min_len],
        z_primitive_df.iloc[:min_len]
    ], axis=1)

    # 階層別相関分析
    correlations = {'style': {}, 'skill': {}, 'primitive': {}}

    if perf_cols:
        metric_names = [col.replace('perf_', '') for col in perf_cols]
        available_metrics = [metric for metric in metric_names if f'perf_{metric}' in analysis_df.columns]
    else:
        available_metrics = []

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
            subject_ids = analysis_df['subject_id'].values
            n_subjects = len(np.unique(subject_ids))

            if len(z_style_data) > n_subjects and n_subjects > 1:
                kmeans = KMeans(n_clusters=n_subjects, random_state=42)
                style_clusters = kmeans.fit_predict(z_style_data)
                subject_labels = pd.Categorical(subject_ids).codes
                style_ari = adjusted_rand_score(subject_labels, style_clusters)
        except Exception as e:
            print(f"スタイル分離性評価エラー: {e}")

    eval_results = {
        'reconstruction_mse': mse,
        'basic_coordinate_mse': basic_mse,
        'derived_feature_mse': derived_mse,
        'physics_consistency_ratio': derived_mse / (basic_mse + 1e-8),
        'hierarchical_correlations': correlations,
        'style_separation_score': style_ari,
        'performance_data': performance_df.to_dict('records') if perf_cols else []
    }

    print("=" * 20 + " 一般化座標VAE定量的評価完了 " + "=" * 20)
    return eval_results, analysis_df

def run_generalized_evaluation(model, test_loader, test_df, output_dir, device, experiment_id):
    """改良版一般化座標VAE評価（スキル軸分析統合）"""
    print("\n" + "=" * 50)
    print("改良版一般化座標VAE評価開始（スキル軸分析付き）")
    print("=" * 50)

    # 1. スキル軸分析
    analyzer = run_skill_axis_analysis_generalized(model, test_loader, test_df, device)

    # 2. 階層潜在変数抽出
    latent_data = extract_latent_variables_generalized(model, test_loader, device)

    # 3. 定量的評価
    eval_results, analysis_df = quantitative_evaluation_generalized(latent_data, test_df, output_dir)

    # 4. 軸ベース個人最適化お手本生成
    exemplar_v2_path = os.path.join(output_dir, 'plots', f'generalized_axis_based_exemplars_exp{experiment_id}.png')
    generate_axis_based_exemplars_generalized(model, analyzer, test_loader, device, exemplar_v2_path)

    # 5. 結果統合
    eval_results.update({
        'skill_axis_analysis_completed': True,
        'axis_based_exemplars_path': exemplar_v2_path,
        'skill_improvement_directions_available': list(analyzer.skill_improvement_directions.keys()),
        'best_skill_correlations': {k: v['correlation'] for k, v in analyzer.performance_correlations.items()}
    })

    # 6. 結果保存
    eval_results_path = os.path.join(output_dir, 'results', f'generalized_evaluation_v2_exp{experiment_id}.json')
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
    print("改良版一般化座標VAE評価完了")
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


def plot_generalized_training_curves(history, save_path):
    """一般化座標VAEの学習曲線をプロット"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # 全体的な損失
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_yscale('log')

    # 物理制約損失
    axes[0, 1].plot(history['physics_loss'], label='Physics Loss', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Physics Loss')
    axes[0, 1].set_title('Physics Constraint Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_yscale('log')

    # 基本座標再構成損失
    axes[0, 2].plot(history['basic_reconstruction_loss'], label='Basic Recon Loss', color='orange')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Basic Reconstruction Loss')
    axes[0, 2].set_title('Basic Coordinate Reconstruction')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    axes[0, 2].set_yscale('log')

    # 合成特徴整合性損失
    axes[1, 0].plot(history['derived_consistency_loss'], label='Derived Consistency', color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Derived Feature Consistency')
    axes[1, 0].set_title('Derived Feature Consistency')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_yscale('log')

    # KL損失
    axes[1, 1].plot(history['kl_loss'], label='Total KL Loss', color='brown')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('KL Divergence')
    axes[1, 1].set_title('KL Divergence Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_yscale('log')

    # 分離損失
    axes[1, 2].plot(history['separation_loss'], label='Separation Loss', color='pink')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Style-Skill Separation')
    axes[1, 2].set_title('Style-Skill Separation Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    axes[1, 2].set_yscale('log')

    # 学習率
    axes[2, 0].plot(history.get('learning_rates', []), color='red')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Learning Rate')
    axes[2, 0].set_title('Learning Rate Schedule')
    axes[2, 0].grid(True)
    axes[2, 0].set_yscale('log')

    # 損失比較
    if len(history['train_loss']) > 1:
        axes[2, 1].plot(history['basic_reconstruction_loss'], label='Basic Recon', alpha=0.7)
        axes[2, 1].plot(history['derived_consistency_loss'], label='Derived Consistency', alpha=0.7)
        axes[2, 1].plot(history['physics_loss'], label='Physics', alpha=0.7)
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Loss Components')
        axes[2, 1].set_title('Loss Component Comparison')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        axes[2, 1].set_yscale('log')

    # 物理的整合性比
    if 'physics_consistency_ratio' in history:
        axes[2, 2].plot(history['physics_consistency_ratio'], label='Consistency Ratio', color='navy')
        axes[2, 2].set_xlabel('Epoch')
        axes[2, 2].set_ylabel('Derived/Basic Loss Ratio')
        axes[2, 2].set_title('Physics Consistency Ratio')
        axes[2, 2].legend()
        axes[2, 2].grid(True)

    plt.suptitle('Generalized Coordinate Hierarchical VAE Training Curves', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"一般化座標VAE学習曲線を保存: {save_path}")


def train_generalized_model(config_path: str, experiment_id: int, db_path: str):
    """完全版一般化座標VAE学習（スキル軸分析統合）"""
    print("=== 一般化座標VAE学習開始 ===")

    # 1. 設定読み込み
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    try:
        config = validate_and_convert_config(config)
        print("一般化座標VAE設定ファイルの検証: ✅ 正常")
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
        print(f"-- 一般化座標VAE実験ID: {experiment_id} | デバイス: {device} ---")

        # 3. データ準備
        try:
            train_loader, val_loader, test_loader, test_df = create_generalized_dataloaders(
                master_data_path=config['data']['data_path'],
                seq_len=config['model']['seq_len'],
                batch_size=config['training']['batch_size'],
                use_precomputed=config['data'].get('use_precomputed', True),
                random_seed=config['data'].get('random_seed', 42)
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

        # 4. 一般化座標VAEモデルの初期化
        model_config = {
            'basic_coord_dim': config['model']['basic_coord_dim'],
            'derived_coord_dim': config['model']['derived_coord_dim'],
            'seq_len': config['model']['seq_len'],
            'hidden_dim': config['model']['hidden_dim'],
            'latent_dims': {
                'primitive': config['model']['primitive_latent_dim'],
                'skill': config['model']['skill_latent_dim'],
                'style': config['model']['style_latent_dim']
            },
            'beta_weights': {
                'primitive': config['model'].get('beta_primitive', 1.0),
                'skill': config['model'].get('beta_skill', 2.0),
                'style': config['model'].get('beta_style', 4.0)
            },
            'physics_weight': config['model'].get('physics_weight', 0.1),
            'separation_weight': config['model'].get('separation_weight', 0.5)
        }

        model = GeneralizedCoordinateHierarchicalVAE(**model_config).to(device)

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
            'physics_loss': [], 'basic_reconstruction_loss': [], 'derived_consistency_loss': [],
            'kl_loss': [], 'separation_loss': [], 'physics_consistency_ratio': []
        }

        print(f"学習開始: {config['training']['num_epochs']}エポック, patience={patience}")

        for epoch in range(config['training']['num_epochs']):
            model.train()

            epoch_losses = {
                'total': [], 'physics': [], 'basic_recon': [], 'derived_consistency': [],
                'kl': [], 'separation': []
            }

            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["training"]["num_epochs"]} [Train]')
            for full_coords, subject_ids, skill_levels in progress_bar:
                full_coords = full_coords.to(device)
                skill_levels = skill_levels.to(device)

                optimizer.zero_grad()
                outputs = model(full_coords, subject_ids, skill_levels)

                loss = outputs['total_loss']
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training'].get('clip_grad_norm', 1.0)
                )
                optimizer.step()

                # 損失記録
                individual_losses = outputs['individual_losses']
                epoch_losses['total'].append(outputs['total_loss'].item())

                # 物理制約損失
                physics_loss = (
                        individual_losses['physics_physics_consistency'] +
                        individual_losses['physics_smoothness'] +
                        individual_losses['physics_energy_conservation']
                ).item()
                epoch_losses['physics'].append(physics_loss)

                # 基本座標再構成損失
                basic_coords, _ = model.split_coordinates(full_coords)
                basic_recon_loss = torch.nn.functional.mse_loss(
                    outputs['reconstructed_basic'], basic_coords
                ).item()
                epoch_losses['basic_recon'].append(basic_recon_loss)

                # 合成特徴整合性損失
                _, target_derived = model.split_coordinates(full_coords)
                derived_consistency_loss = torch.nn.functional.mse_loss(
                    outputs['reconstructed_derived'], target_derived
                ).item()
                epoch_losses['derived_consistency'].append(derived_consistency_loss)

                # KL損失
                kl_loss = (
                        individual_losses['kl_primitive'] +
                        individual_losses['kl_skill'] +
                        individual_losses['kl_style']
                ).item()
                epoch_losses['kl'].append(kl_loss)

                # 分離損失
                sep_loss = 0.0
                for key in ['separation_style_clustering', 'separation_skill_structure', 'separation_orthogonality']:
                    if key in individual_losses:
                        sep_loss += individual_losses[key].item()
                epoch_losses['separation'].append(sep_loss)

                progress_bar.set_postfix({'Loss': np.mean(epoch_losses['total'])})

            # エポック損失記録
            history['train_loss'].append(np.mean(epoch_losses['total']))
            history['physics_loss'].append(np.mean(epoch_losses['physics']))
            history['basic_reconstruction_loss'].append(np.mean(epoch_losses['basic_recon']))
            history['derived_consistency_loss'].append(np.mean(epoch_losses['derived_consistency']))
            history['kl_loss'].append(np.mean(epoch_losses['kl']))
            history['separation_loss'].append(np.mean(epoch_losses['separation']))
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])

            # 物理的整合性比
            basic_loss_avg = np.mean(epoch_losses['basic_recon'])
            derived_loss_avg = np.mean(epoch_losses['derived_consistency'])
            consistency_ratio = derived_loss_avg / (basic_loss_avg + 1e-8)
            history['physics_consistency_ratio'].append(consistency_ratio)

            # 検証ループ
            model.eval()
            epoch_val_losses = []
            with torch.no_grad():
                for full_coords, subject_ids, skill_levels in val_loader:
                    full_coords = full_coords.to(device)
                    skill_levels = skill_levels.to(device)
                    outputs = model(full_coords, subject_ids, skill_levels)
                    epoch_val_losses.append(outputs['total_loss'].item())

            current_val_loss = np.mean(epoch_val_losses)
            history['val_loss'].append(current_val_loss)

            print(f"Epoch {epoch + 1}: Train Loss: {history['train_loss'][-1]:.4f}, "
                  f"Val Loss: {current_val_loss:.4f}")
            print(f"  Physics: {history['physics_loss'][-1]:.4f}, "
                  f"Basic Recon: {history['basic_reconstruction_loss'][-1]:.4f}, "
                  f"Derived Consistency: {history['derived_consistency_loss'][-1]:.6f}")
            print(f"  KL: {history['kl_loss'][-1]:.4f}, "
                  f"Separation: {history['separation_loss'][-1]:.4f}, "
                  f"Consistency Ratio: {consistency_ratio:.6f}")

            # 学習率更新
            scheduler.step()

            # ベストモデル保存 & アーリーストップ
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                epochs_no_improve = 0
                best_model_path = os.path.join(output_dir, 'checkpoints',
                                               f'best_generalized_model_exp{experiment_id}.pth')
                model.save_model(best_model_path)
                print(f" -> New best model saved! (Loss: {best_val_loss:.4f})")
            else:
                epochs_no_improve += 1

            if epochs_no_improve > patience:
                print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
                break

        # 6. 学習終了後の処理
        final_model_path = os.path.join(output_dir, 'checkpoints', f'final_generalized_model_exp{experiment_id}.pth')
        model.save_model(final_model_path)

        plot_path = os.path.join(output_dir, 'plots', f'generalized_training_curves_exp{experiment_id}.png')
        plot_generalized_training_curves(history, plot_path)

        print("=== 学習完了、評価開始 ===")

        # 7. 改良版評価実行
        try:
            eval_results = run_generalized_evaluation(model, test_loader, test_df, output_dir, device, experiment_id)

            # 最高相関値を計算
            best_correlation = 0.0
            best_correlation_metric = 'none'
            if 'best_skill_correlations' in eval_results and eval_results['best_skill_correlations']:
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

                # 一般化座標VAE特有の指標
                'final_physics_loss': history['physics_loss'][-1],
                'final_basic_reconstruction_loss': history['basic_reconstruction_loss'][-1],
                'final_derived_consistency_loss': history['derived_consistency_loss'][-1],
                'final_physics_consistency_ratio': history['physics_consistency_ratio'][-1],

                # 評価指標
                'reconstruction_mse': eval_results['reconstruction_mse'],
                'basic_coordinate_mse': eval_results['basic_coordinate_mse'],
                'derived_feature_mse': eval_results['derived_feature_mse'],
                'physics_consistency_ratio': eval_results['physics_consistency_ratio'],
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

                'notes': f"一般化座標VAE - MSE: {eval_results['reconstruction_mse']:.6f}, "
                         f"Physics Ratio: {eval_results['physics_consistency_ratio']:.6f}, "
                         f"Style ARI: {eval_results.get('style_separation_score', 0.0):.4f}, "
                         f"Best Corr: {best_correlation:.4f} ({best_correlation_metric})"
            })

            print(f"=== 実験ID: {experiment_id} 正常完了 ===")
            print(f"最終結果:")
            print(f"  再構成MSE: {eval_results['reconstruction_mse']:.6f}")
            print(f"  物理的整合性比: {eval_results['physics_consistency_ratio']:.6f}")
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
        print(f"!!! 一般化座標VAE実験ID: {experiment_id} でエラー発生: {e} !!!")
        import traceback
        traceback.print_exc()

        update_db(db_path, experiment_id, {
            'status': 'failed',
            'end_time': datetime.now().isoformat(),
            'notes': f"学習中エラー: {str(e)}"
        })
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="完全版一般化座標VAE学習システム")
    parser.add_argument('--config', type=str, required=True, help='設定ファイルのパス')
    parser.add_argument('--experiment_id', type=int, required=True, help='実験管理DBのID')
    parser.add_argument('--db_path', type=str, required=True, help='実験管理DBのパス')

    args = parser.parse_args()

    print(f"階層型VAE学習開始:")
    print(f"  設定ファイル: {args.config}")
    print(f"  実験ID: {args.experiment_id}")
    print(f"  データベース: {args.db_path}")

    train_generalized_model(config_path=args.config, experiment_id=args.experiment_id, db_path=args.db_path)