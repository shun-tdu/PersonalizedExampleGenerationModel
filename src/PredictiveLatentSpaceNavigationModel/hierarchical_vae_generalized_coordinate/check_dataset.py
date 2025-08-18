import os
import sys
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import sqlite3
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# ===== numpy型のSQLite自動変換を登録 =====
sqlite3.register_adapter(np.float32, float)
sqlite3.register_adapter(np.float64, float)
sqlite3.register_adapter(np.int32, int)
sqlite3.register_adapter(np.int64, int)
sqlite3.register_adapter(np.bool_, bool)


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


if __name__ == "__main__":
    data_path = "PredictiveLatentSpaceNavigationModel/DataPreprocess/my_data.parquet"
    seq_len = 100
    batch_size = 16

    print("🚀 一般化座標データセット テスト")
    print("=" * 50)

    # データローダー作成
    train_loader, val_loader, test_loader, test_df = create_generalized_dataloaders(
        master_data_path=data_path,
        seq_len=seq_len,
        batch_size=batch_size,
        use_precomputed=True,  # 事前計算済みデータを使用
        random_seed=42
    )

    if train_loader is None:
        print("❌ データローダーの作成に失敗しました")
        exit(1)

    # サンプルバッチの確認
    print("\n🧪 サンプルバッチテスト:")
    for i, (coords, subject_ids, is_expert) in enumerate(train_loader):
        print(f"   バッチ {i + 1}:")
        print(f"     - 座標shape: {coords.shape}")
        print(f"     - 被験者ID: {subject_ids}")
        print(f"     - 熟練度: {is_expert}")

        if i >= 2:  # 最初の3バッチのみテスト
            break

    # データセット分析
    analyze_generalized_coordinates(train_loader.dataset, num_samples=10)

    print("\n✅ 一般化座標データセット テスト完了！")