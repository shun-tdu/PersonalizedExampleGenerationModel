import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib # スケーラーの保存に使用

# ----------------------------------------
# 1. データ読み込み
# ----------------------------------------
def load_rawdata(data_dir: str) -> pd.DataFrame | None:
    """指定されたディレクトリから全てのCSVを読み込み、一つのDataFrameに結合する。"""
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not all_files:
        print(f"エラー: ディレクトリ '{data_dir}' にCSVファイルが見つかりません。")
        return None

    df_list = []
    for filename in all_files:
        try:
            basename = os.path.basename(filename)
            parts = basename.replace('.csv', '').split('_')
            subject_id = parts[0]
            block_num = int(parts[1].replace('Block', ''))

            df = pd.read_csv(filename)
            # カラム名を統一し、情報を付与
            df = df.rename(columns={'SubjectId': 'subject_id', 'CurrentTrial': 'trial_num', 'Block': 'block'})
            df['subject_id'] = subject_id
            df['block'] = block_num
            df_list.append(df)
        except Exception as e:
            print(f"ファイル {filename} の読み込み中にエラー: {e}")

    if not df_list:
        return None
    return pd.concat(df_list, ignore_index=True)


# ----------------------------------------
# 2. パフォーマンス指標の計算ヘルパー
# ----------------------------------------
def calculate_jerk(acceleration: np.ndarray, dt: float) -> float:
    if len(acceleration) < 2: return np.nan
    jerk = np.gradient(acceleration, dt, axis=0)
    return np.mean(np.linalg.norm(jerk, axis=1))


def calculate_path_efficiency(trajectory: np.ndarray) -> float:
    start_point, end_point = trajectory[0], trajectory[-1]
    straight_distance = np.linalg.norm(end_point - start_point)
    total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
    if total_distance < 1e-6: return 1.0
    return straight_distance / total_distance


def calculate_approach_angle(trajectory: np.ndarray, n_points: int = 5) -> float:
    if len(trajectory) < n_points: return np.nan
    vec = trajectory[-1] - trajectory[-n_points]
    return np.arctan2(vec[1], vec[0]) * 180 / np.pi


def calculate_sparc(velocity: np.ndarray, dt: float) -> float:
    if len(velocity) < 2: return np.nan
    velocity_magnitude = np.linalg.norm(velocity, axis=1)
    return 1.0 / (np.std(velocity_magnitude) + 1e-6)


# ----------------------------------------
# 3. パフォーマンス指標DataFrameの作成
# ----------------------------------------
def create_performance_dataframe(raw_df: pd.DataFrame, target_sequence_length: int = 100) -> pd.DataFrame:
    """生データから、試行ごとのパフォーマンス指標DataFrameを作成する。"""
    # 'TRIAL_RUNNING'状態のデータのみを抽出
    df = raw_df[raw_df['TrialState'] == 'TRIAL_RUNNING'].copy()
    if df.empty:
        print("エラー: 'TRIAL_RUNNING'状態のデータが見つかりません。")
        return pd.DataFrame()

    # --- 試行間のばらつきを先に計算 ---
    all_trial_variability = {}
    for subject_id, subject_df in df.groupby('subject_id'):
        resampled_trajs = []
        for _, trial_df in subject_df.groupby('trial_num'):
            traj_points = trial_df[['HandlePosX', 'HandlePosY']].values
            original_indices = np.linspace(0, 1, num=len(traj_points))
            target_indices = np.linspace(0, 1, num=target_sequence_length)
            resampled_x = np.interp(target_indices, original_indices, traj_points[:, 0])
            resampled_y = np.interp(target_indices, original_indices, traj_points[:, 1])
            resampled_trajs.append(np.column_stack([resampled_x, resampled_y]))
        if resampled_trajs:
            variability = np.mean(np.std(np.array(resampled_trajs), axis=0))
            all_trial_variability[subject_id] = variability

    # --- 各試行のパフォーマンスを計算 ---
    def calculate_metrics_for_trial(group: pd.DataFrame) -> pd.Series:
        trajectory = group[['HandlePosX', 'HandlePosY']].values
        velocity = group[['HandleVelX', 'HandleVelY']].values
        acceleration = group[['HandleAccX', 'HandleAccY']].values
        target_end_pos = group[['TargetEndPosX', 'TargetEndPosY']].iloc[-1].values
        time_stamps = group['Timestamp'].values

        dt = np.mean(np.diff(time_stamps))
        if np.isnan(dt) or dt == 0: dt = 0.01

        return pd.Series({
            'block': group['block'].iloc[0],
            'trial_time': time_stamps[-1] - time_stamps[0],
            'trial_error': np.linalg.norm(trajectory[-1] - target_end_pos),
            'jerk': calculate_jerk(acceleration, dt),
            'path_efficiency': calculate_path_efficiency(trajectory),
            'approach_angle': calculate_approach_angle(trajectory),
            'sparc': calculate_sparc(velocity, dt),
            'trial_variability': all_trial_variability.get(group.name[0], np.nan)
        })

    performance_df = df.groupby(['subject_id', 'trial_num']).apply(calculate_metrics_for_trial).reset_index()
    return performance_df


# ----------------------------------------
# 4. スキルスコアの計算とグループ分け
# ----------------------------------------
def calculate_skill_scores(perf_df: pd.DataFrame) -> pd.Series:
    """パフォーマンス指標DFから、被験者ごとの総合スキルスコアを計算する。"""
    # 被験者ごとにパフォーマンス指標の平均を計算
    subject_metrics = perf_df.groupby('subject_id').mean()

    # Z-score化
    metrics_zscored = subject_metrics.drop(columns=['block', 'trial_num']).apply(lambda x: (x - x.mean()) / x.std())

    # 方向性の統一 (値が低い方が良い指標の符号を反転)
    metrics_zscored['trial_time'] *= -1
    metrics_zscored['trial_error'] *= -1
    metrics_zscored['jerk'] *= -1
    metrics_zscored['trial_variability'] *= -1
    # path_efficiency, sparc, approach_angleは高い方が良いのでそのまま

    # 総合スキルスコアを算出
    skill_scores = metrics_zscored.sum(axis=1)
    return skill_scores


def classify_by_median_split(skill_scores: pd.Series) -> pd.Series:
    """スキルスコアを中央値で「熟達者(1)」と「初心者(0)」に分類する。"""
    median_score = skill_scores.median()
    return (skill_scores > median_score).astype(int).rename('is_expert')


def main():
    TARGET_SEQ_LEN = 100
    RAWDATA_DIR = '../../../data/RawDatas/'
    PROCESSED_DATA_DIR = 'PredictiveLatentSpaceNavigationModel/DataPreprocess/ForGeneralizedCoordinate'
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)  # ディレクトリがなければ作成

    # --- データ処理の実行 ---
    # 1. 生データ読み込み
    raw_df = load_rawdata(RAWDATA_DIR)
    if raw_df is None: return

    # 2. パフォーマンス指標の計算
    performance_df = create_performance_dataframe(raw_df, target_sequence_length=TARGET_SEQ_LEN)
    if performance_df.empty: return

    # 3. スキルスコアの計算
    skill_scores = calculate_skill_scores(performance_df)

    # 4. 熟達者/初心者の分類
    expertise_labels = classify_by_median_split(skill_scores)

    # 5. 熟練度の情報を含めたデータフレーム
    filtered_df = raw_df[raw_df['TrialState'] == 'TRIAL_RUNNING'].copy()
    filtered_df['is_expert'] = filtered_df['subject_id'].map(expertise_labels)

    # 6. パフォーマンス指標を軌道データに結合
    # performance_dfの列名にprefixを追加してコンフリクトを避ける
    performance_df_renamed = performance_df.add_prefix('perf_').rename(columns={'perf_subject_id': 'subject_id', 'perf_trial_num': 'trial_num'})
    master_df = filtered_df.merge(
        performance_df_renamed,
        on=['subject_id', 'trial_num'],
        how='left'
    )

    # 7. ジャークを計算して列に追加
    def calculate_jerk_for_df(group):
        acc_data = group[['HandleAccX', 'HandleAccY']].values
        jerk_data = np.diff(acc_data, axis=0, prepend=np.array([[0,0]]))
        group['JerkX'] = jerk_data[:, 0]
        group['JerkY'] = jerk_data[:, 1]
        return group

    master_df = master_df.groupby(['subject_id', 'trial_num'], group_keys=False).apply(calculate_jerk_for_df)

    # 8. 訓練データとテストデータに分割
    subject_ids = master_df['subject_id'].unique()
    train_subjects, test_subjects = train_test_split(subject_ids, test_size=0.2, random_state=42)

    train_df = master_df[master_df['subject_id'].isin(train_subjects)]
    test_df = master_df[master_df['subject_id'].isin(test_subjects)]

    print(f"訓練データ被験者数: {len(train_subjects)}, テストデータ被験者数: {len(test_subjects)}")

    # ### NEW STEP 3: スケーラーを訓練データで学習 ###
    # モデルの入力となる特徴量を定義
    feature_cols = [
        'HandlePosX', 'HandlePosY',
        'HandleVelX', 'HandleVelY',
        'HandleAccX', 'HandleAccY',
        'JerkX', 'JerkY'
    ]

    scaler = StandardScaler()
    # 訓練データから特徴量部分だけを抽出してfit
    scaler.fit(train_df[feature_cols])
    print("StandardScalerを訓練データで学習しました。✅")

    # ### NEW STEP 4: 学習済みスケーラーとデータを保存 ###
    # スケーラーを保存
    scaler_path = os.path.join(PROCESSED_DATA_DIR, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"スケーラーを {scaler_path} に保存しました。")

    # 訓練データとテストデータをそれぞれ保存
    train_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, 'train_data.parquet'))
    test_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, 'test_data.parquet'))
    print("訓練データとテストデータをParquet形式で保存しました。")


if __name__ == "__main__":
    main()
