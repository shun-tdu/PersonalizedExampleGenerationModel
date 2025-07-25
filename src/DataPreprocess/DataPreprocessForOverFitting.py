import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import glob

# --- 設定項目 ---
DATA_DIR = '../../data/RawDatas/'  # Unityが出力したログファイルが保存されているディレクトリ
OUTPUT_FILE = '../../data/Datasets/overfitting_dataset.npz'  # 過学習テスト用のデータセットファイル名
TRAJECTORY_LENGTH = 101  # 正規化後の軌道の長さ


# calculate_jerk, process_all_logs, normalize_trajectory 関数は以前のものと同じでOK
# (ここでは簡略化のため省略)

def calculate_jerk(acc_x, acc_y, timestamps):
    """加速度データからジャークの二乗積分を計算する"""
    if len(acc_x) < 2:
        return float('inf')

    dt = np.diff(timestamps)
    # ゼロ除算を避ける
    dt[dt == 0] = 1e-6

    # 加速度の微分がジャーク
    jx = np.diff(acc_x) / dt
    jy = np.diff(acc_y) / dt

    # ジャークの大きさの二乗を時間積分（近似）
    jerk_magnitude_sq = jx ** 2 + jy ** 2
    integrated_jerk = np.sum(jerk_magnitude_sq) * np.mean(dt)

    return integrated_jerk


def process_all_logs(data_dir):
    """指定されたディレクトリ内の全てのCSVを読み込み、一つのDataFrameにまとめる"""
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not all_files:
        print(f"エラー: ディレクトリ '{data_dir}' にCSVファイルが見つかりません。")
        return None

    df_list = []
    for filename in all_files:
        try:
            # ファイル名から被験者IDとブロック番号を抽出
            basename = os.path.basename(filename)
            parts = basename.replace('.csv', '').split('_')
            subject_id = parts[0]
            block_num = int(parts[1].replace('Block', ''))

            df = pd.read_csv(filename)
            df['SubjectID'] = subject_id
            df['Block'] = block_num
            df_list.append(df)
        except Exception as e:
            print(f"ファイル {filename} の読み込み中にエラー: {e}")

    if not df_list:
        return None

    return pd.concat(df_list, ignore_index=True)


def analyze_trials(df):
    """試行ごとのパフォーマンス指標を計算する"""
    trial_metrics = []
    for name, group in df.groupby(['SubjectID', 'Block', 'CurrentTrial']):
        trial_data = group[group['TrialState'] == 'TRIAL_RUNNING'].copy()
        if len(trial_data) < 5: continue

        movement_time = (trial_data['Timestamp'].iloc[-1] - trial_data['Timestamp'].iloc[0]) / 1e7
        actual_end_pos = trial_data[['HandlePosX', 'HandlePosY']].iloc[-1].values
        target_start_pos = trial_data[['TargetStartPosX', 'TargetStartPosY']].iloc[-1].values
        target_end_pos = trial_data[['TargetEndPosX', 'TargetEndPosY']].iloc[-1].values
        endpoint_error = np.linalg.norm(actual_end_pos - target_end_pos)

        acc_x, acc_y, ts = trial_data['HandleAccX'].values, trial_data['HandleAccY'].values, trial_data[
                                                                                                 'Timestamp'].values / 1e7
        jerk = calculate_jerk(acc_x, acc_y, ts)

        trial_metrics.append({
            'SubjectID': name[0],
            'Block': name[1],
            'Trial': name[2],
            'MovementTime': movement_time,
            'EndpointError': endpoint_error,
            'Jerk': jerk,
            'TargetStartPosX': target_start_pos[0],
            'TargetStartPosY': target_start_pos[1],
            'TargetEndPosX': target_end_pos[0],
            'TargetEndPosY': target_end_pos[1],
        })
    return pd.DataFrame(trial_metrics)


def normalize_trajectory(trial_data, length):
    """軌道データを指定された長さに正規化（リサンプリング）し、-1から1に正規化する"""
    x = trial_data['HandlePosX'].values
    y = trial_data['HandlePosY'].values
    timestamps = trial_data['Timestamp'].values / 1e7  # タイムスタンプを秒に変換

    # 始点を保存
    start_pos = np.array([x[0], y[0]])

    # 始点を(0,0)に移動させる
    x_norm = x - start_pos[0]
    y_norm = y - start_pos[1]

    # 実際の時間軸に基づいてリサンプリング
    original_time = timestamps - timestamps[0]  # 開始時刻を0にする
    total_time = original_time[-1]
    resampled_time = np.linspace(0, total_time, num=length)

    # 時間軸に沿った線形補間でリサンプリング
    resampled_x = np.interp(resampled_time, original_time, x_norm)
    resampled_y = np.interp(resampled_time, original_time, y_norm)

    # -1から1に正規化
    trajectory = np.stack([resampled_x, resampled_y], axis=1)  # (length, 2)

    # 全体の最大値と最小値を取得
    max_val = np.max(np.abs(trajectory))
    if max_val == 0:
        max_val = 1.0

    trajectory = trajectory / max_val  # -1から1に正規化

    return trajectory, start_pos, max_val


def main():
    print("--- 過学習用データセットの処理を開始します ---")

    # 1. 全てのログファイルを読み込む
    full_df = process_all_logs(DATA_DIR)
    if full_df is None: return
    print(f"全ログファイルの読み込み完了。")

    # 2. 試行ごとのパフォーマンスを分析
    metrics_df = analyze_trials(full_df)
    print(f"パフォーマンス分析完了。合計 {len(metrics_df)} 試行。")

    # ▼▼▼ 修正点 ▼▼▼
    # 3. 「質の高い試行」のフィルタリングをせず、全ての有効な試行をそのまま使用する
    valid_trials = metrics_df.dropna()  # 計算に失敗した試行（infなど）があれば除外
    print(f"フィルタリングをスキップ。合計 {len(valid_trials)} の有効な試行をデータセットとして使用します。")
    # ▲▲▲ 修正完了 ▲▲▲

    # 4. データセットを作成
    trajectories = []
    conditions = []

    for _, trial_info in valid_trials.iterrows():
        trial_data = full_df[
            (full_df['SubjectID'] == trial_info['SubjectID']) &
            (full_df['Block'] == trial_info['Block']) &
            (full_df['CurrentTrial'] == trial_info['Trial']) &
            (full_df['TrialState'] == 'TRIAL_RUNNING')
            ]

        if len(trial_data) < 5: continue

        # 正規化された軌道を保存
        norm_trajectory, start_pos, scale_val = normalize_trajectory(trial_data, TRAJECTORY_LENGTH)
        trajectories.append(norm_trajectory)

        # 元のゴール位置を取得
        original_goal_pos = np.array([trial_info['TargetEndPosX'], trial_info['TargetEndPosY']])
        normalized_goal_pos = (original_goal_pos - start_pos) / scale_val

        # 被験者の特性ベクトル（コンディション）
        condition_vector = np.array([
            trial_info['MovementTime'],
            trial_info['EndpointError'],
            trial_info['Jerk'],
            normalized_goal_pos[0],
            normalized_goal_pos[1]
        ])
        conditions.append(condition_vector)

    # 条件ベクトルをNumpy配列に変換
    conditions = np.array(conditions)

    # ---- 特徴量ごとに異なる前処理を適用 ----
    # 1. パフォーマンス指標
    scaler = StandardScaler()
    conditions_processed = scaler.fit_transform(conditions)


    # 5. ファイルに保存
    np.savez_compressed(
        OUTPUT_FILE,
        trajectories=np.array(trajectories),
        conditions=conditions_processed,
        condition_scaler_mean=scaler.mean_,
        condition_scaler_scale=scaler.scale_
    )

    print(f"--- データセットの作成完了 ---")
    print(f"軌道データ数: {len(trajectories)}")
    print(f"ファイル '{OUTPUT_FILE}' に保存しました。")


def load_processed_data(data_path):
    """
    処理済みのデータセットを読み込む
    
    Args:
        data_path: データファイルのパス (.npz)
        
    Returns:
        trajectories: 軌道データ [num_samples, sequence_length, 2]
        conditions: 条件データ（最初の3つの特徴量のみ）[num_samples, 3]
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"データファイルが見つかりません: {data_path}")
    
    data = np.load(data_path)
    trajectories = data['trajectories']
    conditions_all = data['conditions']
    
    # 全ての特徴量を使用（MovementTime, EndpointError, Jerk, GoalX, GoalY）
    conditions = conditions_all
    
    print(f"データを読み込みました:")
    print(f"  軌道データ: {trajectories.shape}")  
    print(f"  条件データ: {conditions.shape}")
    print(f"  軌道データ範囲: [{trajectories.min():.3f}, {trajectories.max():.3f}]")
    print(f"  条件データ範囲: [{conditions.min():.3f}, {conditions.max():.3f}]")
    
    return trajectories, conditions


if __name__ == '__main__':
    main()
