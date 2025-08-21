import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib # スケーラーの保存に使用


def analyze_contradiction(df):
    """矛盾の原因を詳細分析"""
    print("🔍 矛盾の原因分析")
    print("=" * 50)

    subjects = df['subject_id'].unique()

    # 1. 分離スコア計算の詳細確認
    print("1. 分離スコア計算の検証:")

    feature_groups = {
        'Position': ['HandlePosDiffX', 'HandlePosDiffY'],
        'Velocity': ['HandleVelDiffX', 'HandleVelDiffY'],
        'Acceleration': ['HandleAccDiffX', 'HandleAccDiffY']
    }

    for group_name, features in feature_groups.items():
        if all(f in df.columns for f in features):
            print(f"\n{group_name}グループ:")

            # 被験者ごとの実際の数値を確認
            subject_stats = {}
            for subject in subjects:
                subject_data = df[df['subject_id'] == subject][features].values
                if len(subject_data) > 0:
                    stats = {
                        'mean': np.mean(subject_data, axis=0),
                        'std': np.std(subject_data, axis=0),
                        'median': np.median(subject_data, axis=0),
                        'data_points': len(subject_data)
                    }
                    subject_stats[subject] = stats
                    print(f"  {subject}: 平均={stats['mean']}, データ点数={stats['data_points']}")

            # 被験者間の実際の差を計算
            if len(subject_stats) > 1:
                means = np.array([stats['mean'] for stats in subject_stats.values()])
                print(f"  被験者間の平均値範囲:")
                for dim in range(means.shape[1]):
                    dim_values = means[:, dim]
                    print(
                        f"    次元{dim}: {np.min(dim_values):.6f} ~ {np.max(dim_values):.6f} (範囲: {np.ptp(dim_values):.6f})")

    # 2. 運動特性での個人差の具体的数値
    print(f"\n2. 運動特性での個人差の具体値:")

    def calculate_movement_features(subject_df):
        """被験者の運動特性を数値化"""
        features = []

        for trial_num, trial_df in subject_df.groupby('trial_num'):
            if len(trial_df) > 10:
                # 直線性
                start_pos = trial_df[['HandlePosX', 'HandlePosY']].iloc[0].values
                end_pos = trial_df[['HandlePosX', 'HandlePosY']].iloc[-1].values
                actual_path = trial_df[['HandlePosX', 'HandlePosY']].values

                straight_distance = np.linalg.norm(end_pos - start_pos)
                path_length = np.sum([np.linalg.norm(actual_path[i + 1] - actual_path[i])
                                      for i in range(len(actual_path) - 1)])
                linearity = straight_distance / path_length if path_length > 0 else 0

                # 速度特性
                vel_x = trial_df['HandleVelX'].values
                vel_y = trial_df['HandleVelY'].values
                speed = np.sqrt(vel_x ** 2 + vel_y ** 2)

                peak_time_ratio = np.argmax(speed) / len(speed) if len(speed) > 0 else 0.5

                features.append({
                    'linearity': linearity,
                    'peak_time_ratio': peak_time_ratio,
                    'max_speed': np.max(speed),
                    'avg_speed': np.mean(speed)
                })

        return features

    # 各被験者の運動特性
    subject_movement_features = {}
    for subject in subjects:
        subject_df = df[df['subject_id'] == subject]
        features = calculate_movement_features(subject_df)
        if features:
            # 平均値を計算
            avg_features = {}
            for key in features[0].keys():
                values = [f[key] for f in features]
                avg_features[key] = np.mean(values)
            subject_movement_features[subject] = avg_features

    # 運動特性での被験者間差を定量化
    print(f"被験者別運動特性:")
    for subject, features in subject_movement_features.items():
        print(f"  {subject}: {features}")

    # 運動特性での分離スコアを計算
    if len(subject_movement_features) > 1:
        print(f"\n運動特性での分離度:")

        for feature_name in ['linearity', 'peak_time_ratio', 'max_speed', 'avg_speed']:
            values = [features[feature_name] for features in subject_movement_features.values()]

            if len(values) > 1:
                between_var = np.var(values)

                # 被験者内分散を概算（全体分散から被験者間分散を引く）
                all_values = []
                for subject in subjects:
                    subject_df = df[df['subject_id'] == subject]
                    feature_values = calculate_movement_features(subject_df)
                    if feature_values:
                        subject_feature_values = [f[feature_name] for f in feature_values]
                        all_values.extend(subject_feature_values)

                if all_values:
                    total_var = np.var(all_values)
                    within_var = max(total_var - between_var, 0.001)  # 最小値で制限
                    separation_score = between_var / within_var

                    print(f"  {feature_name}: 分離スコア = {separation_score:.4f}")

                    if separation_score > 1.0:
                        print(f"    🟢 この特徴では個人差あり！")
                    else:
                        print(f"    🔴 この特徴でも個人差少ない")


def reconcile_contradiction(df):
    """矛盾を解決するための統合分析"""
    print(f"\n🔄 矛盾の解決策分析")
    print("=" * 50)

    print("仮説1: 特徴抽出レベルの違い")
    print("- 差分値での分析 → 個人差検出困難")
    print("- 高次特徴での分析 → 個人差検出可能")

    print(f"\n仮説2: スケールの問題")
    print("- 微細な差が大きなノイズに埋もれている")
    print("- 適切な正規化により個人差が浮上する可能性")

    print(f"\n仮説3: 非線形パターンの存在")
    print("- 線形分析では捉えられない個人差")
    print("- VAEのような非線形モデルで検出可能")

    # 実際に非線形パターンを探索
    subjects = df['subject_id'].unique()

    # 各被験者の「運動シグネチャー」を作成
    print(f"\n🎯 統合的運動シグネチャー分析:")

    subject_signatures = {}

    for subject in subjects:
        subject_df = df[df['subject_id'] == subject]

        # 複数の特徴を統合
        signature_features = []

        for trial_num, trial_df in subject_df.groupby('trial_num'):
            if len(trial_df) > 20:
                # 1. 軌道特徴
                positions = trial_df[['HandlePosX', 'HandlePosY']].values
                velocities = trial_df[['HandleVelX', 'HandleVelY']].values

                # 複合特徴を計算
                composite_features = {
                    # 時空間特徴
                    'path_curvature': calculate_path_curvature(positions),
                    'velocity_smoothness': calculate_velocity_smoothness(velocities),
                    'acceleration_jerk': calculate_jerk_metric(trial_df),

                    # 動的特徴
                    'movement_rhythm': calculate_movement_rhythm(velocities),
                    'force_modulation': calculate_force_modulation(trial_df),
                }

                signature_features.append(composite_features)

        if signature_features:
            # 平均シグネチャーを計算
            avg_signature = {}
            for key in signature_features[0].keys():
                values = [f[key] for f in signature_features if not np.isnan(f[key])]
                if values:
                    avg_signature[key] = np.mean(values)
                else:
                    avg_signature[key] = 0.0

            subject_signatures[subject] = avg_signature

    # シグネチャーでの個人差を分析
    if len(subject_signatures) > 1:
        print(f"統合シグネチャーでの個人差:")

        for subject, signature in subject_signatures.items():
            print(f"  {subject}: {signature}")

        # シグネチャー間距離
        subjects_list = list(subject_signatures.keys())
        signatures_matrix = np.array([list(sig.values()) for sig in subject_signatures.values()])

        if signatures_matrix.shape[0] > 1 and signatures_matrix.shape[1] > 0:
            from scipy.spatial.distance import pdist, squareform

            distances = pdist(signatures_matrix, metric='euclidean')
            distance_matrix = squareform(distances)

            print(f"\n被験者間シグネチャー距離:")
            for i, subj1 in enumerate(subjects_list):
                for j, subj2 in enumerate(subjects_list):
                    if i < j:
                        print(f"  {subj1} vs {subj2}: {distance_matrix[i, j]:.4f}")

            avg_distance = np.mean(distances)
            print(f"平均距離: {avg_distance:.4f}")

            if avg_distance > 0.5:
                print("🟢 統合シグネチャーで明確な個人差あり！")
            elif avg_distance > 0.1:
                print("🟡 統合シグネチャーで中程度の個人差")
            else:
                print("🔴 統合シグネチャーでも個人差少ない")


# ヘルパー関数群
def calculate_path_curvature(positions):
    """軌道の曲率を計算"""
    if len(positions) < 3:
        return 0.0

    # 単純な曲率計算
    curvatures = []
    for i in range(1, len(positions) - 1):
        p1, p2, p3 = positions[i - 1], positions[i], positions[i + 1]

        # 3点から曲率を計算
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        if a > 0 and b > 0 and c > 0:
            s = (a + b + c) / 2
            area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
            curvature = 4 * area / (a * b * c) if (a * b * c) > 0 else 0
            curvatures.append(curvature)

    return np.mean(curvatures) if curvatures else 0.0


def calculate_velocity_smoothness(velocities):
    """速度の滑らかさを計算"""
    if len(velocities) < 2:
        return 0.0

    vel_magnitude = np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)
    velocity_changes = np.abs(np.diff(vel_magnitude))
    return 1.0 / (1.0 + np.std(velocity_changes))


def calculate_jerk_metric(trial_df):
    """ジャーク指標を計算"""
    if len(trial_df) < 3:
        return 0.0

    acc_x = trial_df['HandleAccX'].values
    acc_y = trial_df['HandleAccY'].values

    jerk_x = np.diff(acc_x)
    jerk_y = np.diff(acc_y)
    jerk_magnitude = np.sqrt(jerk_x ** 2 + jerk_y ** 2)

    return np.mean(jerk_magnitude)


def calculate_movement_rhythm(velocities):
    """運動リズムを計算"""
    if len(velocities) < 10:
        return 0.0

    vel_magnitude = np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)

    # FFTで周波数成分を分析
    try:
        fft = np.fft.fft(vel_magnitude)
        freqs = np.fft.fftfreq(len(vel_magnitude))

        # 主要周波数成分
        dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft) // 2])) + 1
        dominant_freq = freqs[dominant_freq_idx]

        return abs(dominant_freq)
    except:
        return 0.0


def calculate_force_modulation(trial_df):
    """力調節パターンを計算"""
    if len(trial_df) < 5:
        return 0.0

    acc_x = trial_df['HandleAccX'].values
    acc_y = trial_df['HandleAccY'].values
    force_estimate = np.sqrt(acc_x ** 2 + acc_y ** 2)

    # 力の変調指標
    force_variability = np.std(force_estimate) / (np.mean(force_estimate) + 1e-6)
    return force_variability


# 実行関数
def resolve_contradiction_analysis(df):
    """矛盾の解決分析を実行"""
    print("🎯 矛盾解決分析")
    print("=" * 60)

    analyze_contradiction(df)
    reconcile_contradiction(df)

    print(f"\n💡 結論:")
    print(f"差分レベルでは個人差が検出困難だが、")
    print(f"高次特徴や統合シグネチャーでは個人差が存在する可能性。")
    print(f"VAEのような非線形モデルによる潜在パターン発見が有望。")

# def analyze_movement_constraints(df):
#     """Point-to-Point動作の制約を定量化"""
#     print("=== Point-to-Point動作制約分析 ===")
#
#     # 1. 軌道の直線性分析
#     print("1. 軌道の直線性:")
#     subjects = df['subject_id'].unique()
#
#     for subject in subjects:
#         subject_df = df[df['subject_id'] == subject]
#         linearity_scores = []
#
#         for trial_num, trial_df in subject_df.groupby('trial_num'):
#             if len(trial_df) > 10:
#                 # 開始点と終了点
#                 start_pos = trial_df[['HandlePosX', 'HandlePosY']].iloc[0].values
#                 end_pos = trial_df[['HandlePosX', 'HandlePosY']].iloc[-1].values
#
#                 # 実際の軌道
#                 actual_path = trial_df[['HandlePosX', 'HandlePosY']].values
#
#                 # 直線距離
#                 straight_distance = np.linalg.norm(end_pos - start_pos)
#
#                 # 実際の経路長
#                 path_length = np.sum([np.linalg.norm(actual_path[i + 1] - actual_path[i])
#                                       for i in range(len(actual_path) - 1)])
#
#                 # 直線性スコア (1に近いほど直線的)
#                 if path_length > 0:
#                     linearity = straight_distance / path_length
#                     linearity_scores.append(linearity)
#
#         if linearity_scores:
#             avg_linearity = np.mean(linearity_scores)
#             print(f"  {subject}: 平均直線性 = {avg_linearity:.4f}")
#
#             if avg_linearity > 0.95:
#                 print(f"    🔴 ほぼ直線的 - 個人差が出にくい")
#             elif avg_linearity > 0.90:
#                 print(f"    🟡 やや直線的")
#             else:
#                 print(f"    🟢 曲線的 - 個人差の可能性あり")
#
#
# def analyze_movement_phases(df):
#     """運動の位相分析"""
#     print("\n2. 運動位相の一様性:")
#
#     subjects = df['subject_id'].unique()
#
#     for subject in subjects:
#         subject_df = df[df['subject_id'] == subject]
#         phase_patterns = []
#
#         for trial_num, trial_df in subject_df.groupby('trial_num'):
#             if len(trial_df) > 20:
#                 # 速度プロファイル
#                 vel_x = trial_df['HandleVelX'].values
#                 vel_y = trial_df['HandleVelY'].values
#                 speed = np.sqrt(vel_x ** 2 + vel_y ** 2)
#
#                 # 運動位相の特定
#                 # 1. 加速期間の割合
#                 acceleration_phase = np.sum(np.diff(speed) > 0) / len(speed)
#
#                 # 2. 最大速度到達時点（正規化）
#                 if len(speed) > 0:
#                     peak_time_ratio = np.argmax(speed) / len(speed)
#                 else:
#                     peak_time_ratio = 0.5
#
#                 # 3. 速度プロファイルの対称性
#                 mid_point = len(speed) // 2
#                 first_half = speed[:mid_point]
#                 second_half = speed[mid_point:mid_point + len(first_half)][::-1]  # 反転
#
#                 if len(first_half) > 0 and len(second_half) > 0:
#                     symmetry = np.corrcoef(first_half, second_half)[0, 1]
#                     if np.isnan(symmetry):
#                         symmetry = 0
#                 else:
#                     symmetry = 0
#
#                 phase_patterns.append({
#                     'acceleration_ratio': acceleration_phase,
#                     'peak_time_ratio': peak_time_ratio,
#                     'velocity_symmetry': symmetry
#                 })
#
#         if phase_patterns:
#             avg_accel_ratio = np.mean([p['acceleration_ratio'] for p in phase_patterns])
#             avg_peak_time = np.mean([p['peak_time_ratio'] for p in phase_patterns])
#             avg_symmetry = np.mean([p['velocity_symmetry'] for p in phase_patterns])
#
#             print(f"  {subject}:")
#             print(f"    加速期間割合: {avg_accel_ratio:.3f}")
#             print(f"    速度ピーク時点: {avg_peak_time:.3f}")
#             print(f"    速度対称性: {avg_symmetry:.3f}")
#
#             # ベルシェイプ（典型的なpoint-to-point）の判定
#             is_typical_ptp = (0.4 < avg_peak_time < 0.6 and avg_symmetry > 0.7)
#             if is_typical_ptp:
#                 print(f"    🔴 典型的なベルシェイプ速度プロファイル")
#             else:
#                 print(f"    🟢 非典型的な速度パターン - 個性あり")
#
#
# def analyze_optimal_control_convergence(df):
#     """最適制御への収束度分析"""
#     print("\n3. 最適制御パターンへの収束:")
#
#     subjects = df['subject_id'].unique()
#
#     # 理論的最適軌道との比較
#     for subject in subjects:
#         subject_df = df[df['subject_id'] == subject]
#         optimality_scores = []
#
#         for trial_num, trial_df in subject_df.groupby('trial_num'):
#             if len(trial_df) > 15:
#                 # 実際の軌道
#                 positions = trial_df[['HandlePosX', 'HandlePosY']].values
#                 velocities = trial_df[['HandleVelX', 'HandleVelY']].values
#
#                 # 最適制御指標
#                 # 1. 躍度最小化（滑らかさ）
#                 jerks = []
#                 for i in range(2, len(velocities)):
#                     jerk = np.linalg.norm(velocities[i] - 2 * velocities[i - 1] + velocities[i - 2])
#                     jerks.append(jerk)
#
#                 if jerks:
#                     jerk_score = 1.0 / (1.0 + np.mean(jerks))  # 躍度が小さいほど高スコア
#                 else:
#                     jerk_score = 0
#
#                 # 2. エネルギー効率
#                 energy = np.sum([np.linalg.norm(v) ** 2 for v in velocities])
#                 path_length = np.sum([np.linalg.norm(positions[i + 1] - positions[i])
#                                       for i in range(len(positions) - 1)])
#
#                 if path_length > 0:
#                     energy_efficiency = 1.0 / (1.0 + energy / path_length)
#                 else:
#                     energy_efficiency = 0
#
#                 optimality_scores.append({
#                     'jerk_optimality': jerk_score,
#                     'energy_efficiency': energy_efficiency
#                 })
#
#         if optimality_scores:
#             avg_jerk_opt = np.mean([s['jerk_optimality'] for s in optimality_scores])
#             avg_energy_eff = np.mean([s['energy_efficiency'] for s in optimality_scores])
#
#             print(f"  {subject}:")
#             print(f"    躍度最適性: {avg_jerk_opt:.4f}")
#             print(f"    エネルギー効率: {avg_energy_eff:.4f}")
#
#             # 最適制御への収束度
#             overall_optimality = (avg_jerk_opt + avg_energy_eff) / 2
#             if overall_optimality > 0.8:
#                 print(f"    🔴 高度に最適化 - 個人差少ない")
#             elif overall_optimality > 0.6:
#                 print(f"    🟡 中程度に最適化")
#             else:
#                 print(f"    🟢 最適化不十分 - 個人差の可能性")
#
#
# def recommend_alternative_tasks():
#     """個人差が現れやすい代替タスクの提案"""
#     print("\n" + "=" * 50)
#     print("個人差が現れやすい代替タスク提案")
#     print("=" * 50)
#
#     alternative_tasks = [
#         {
#             "タスク": "自由軌道描画",
#             "説明": "特定の形状を自由に描く（円、8の字など）",
#             "個人差要因": "描画スタイル、速度変調、軌道選択",
#             "期待される分離スコア": "> 2.0"
#         },
#         {
#             "タスク": "リズミカル運動",
#             "説明": "メトロノームに合わせた反復運動",
#             "個人差要因": "リズム同期パターン、位相関係",
#             "期待される分離スコア": "> 1.5"
#         },
#         {
#             "タスク": "障害物回避",
#             "説明": "動的障害物を避けながらの到達運動",
#             "個人差要因": "回避戦略、予測的制御",
#             "期待される分離スコア": "> 3.0"
#         },
#         {
#             "タスク": "マルチターゲット",
#             "説明": "複数目標への連続到達",
#             "個人差要因": "軌道計画、動作シーケンス",
#             "期待される分離スコア": "> 2.5"
#         },
#         {
#             "タスク": "力制御課題",
#             "説明": "特定の力パターンでの操作",
#             "個人差要因": "力の調節パターン、グリップ戦略",
#             "期待される分離スコア": "> 2.0"
#         }
#     ]
#
#     for i, task in enumerate(alternative_tasks, 1):
#         print(f"{i}. {task['タスク']}")
#         print(f"   説明: {task['説明']}")
#         print(f"   個人差要因: {task['個人差要因']}")
#         print(f"   期待分離スコア: {task['期待される分離スコア']}")
#         print()
#
#
# # 使用方法
# def comprehensive_ptp_analysis(df):
#     """Point-to-Point動作の包括的分析"""
#     print("🎯 Point-to-Point動作制約分析")
#     print("=" * 60)
#
#     analyze_movement_constraints(df)
#     analyze_movement_phases(df)
#     analyze_optimal_control_convergence(df)
#     recommend_alternative_tasks()
#
#     print("\n💡 結論:")
#     print("Point-to-Point動作は本質的に制約が強く、")
#     print("個人差（スタイル）が現れにくい運動タスクです。")
#     print("より自由度の高いタスクでの検証を推奨します。")

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

def add_diffs(group):
    """各種物理量を差分表示変換"""
    pos_data = group[['HandlePosX', 'HandlePosY']].values
    pos_diffs = np.diff(pos_data, axis=0, prepend=pos_data[0:1])

    vel_data = group[['HandleVelX', 'HandleVelY']].values
    vel_diffs = np.diff(vel_data, axis=0, prepend=vel_data[0:1])

    acc_data = group[['HandleAccX', 'HandleAccY']].values
    acc_diffs = np.diff(acc_data, axis=0, prepend=acc_data[0:1])

    group = group.copy()
    group['HandlePosDiffX'] = pos_diffs[:, 0]
    group['HandlePosDiffY'] = pos_diffs[:, 1]

    group['HandleVelDiffX'] = vel_diffs[:, 0]
    group['HandleVelDiffY'] = vel_diffs[:, 1]

    group['HandleAccDiffX'] = acc_diffs[:, 0]
    group['HandleAccDiffY'] = acc_diffs[:, 1]

    return group


def prepare_scaler(feature_df:pd.DataFrame):
    """特徴量タイプごとに個別スケーリング"""

    # 利用可能なカラムを確認
    available_cols = feature_df.columns.tolist()
    print(f"Available columns: {available_cols}")

    scalers = {}

    # 位置
    position_cols = ['HandlePosX', 'HandlePosY']
    if all(col in available_cols for col in position_cols):
        scalers['position'] = StandardScaler().fit(feature_df[position_cols])
        print(f"Position scaler created:{position_cols}")

    # 位置差分
    position_diff_cols = ['HandlePosDiffX', 'HandlePosDiffY']
    if all(col in available_cols for col in position_diff_cols):
        scalers['position_diff'] = StandardScaler().fit(feature_df[position_diff_cols])
        print(f"Position diff scaler created:{position_diff_cols}")

    # 速度
    velocity_cols = ['HandleVelX', 'HandleVelY']
    if all(col in available_cols for col in velocity_cols):
        scalers['velocity'] = StandardScaler().fit(feature_df[velocity_cols])
        print(f"Velocity scaler created:{velocity_cols}")

    # 速度差分
    velocity_diff_cols = ['HandleVelDiffX', 'HandleVelDiffY']
    if all(col in available_cols for col in velocity_diff_cols):
        scalers['velocity_diff'] = StandardScaler().fit(feature_df[velocity_diff_cols])
        print(f"Velocity diff scaler created:{velocity_diff_cols}")

    # 加速度
    acceleration_cols = ['HandleAccX', 'HandleAccY']
    if all(col in available_cols for col in acceleration_cols):
        scalers['acceleration'] = StandardScaler().fit(feature_df[acceleration_cols])
        print(f"acceleration scaler created:{acceleration_cols}")

    # 加速度差分
    acceleration_diff_cols = ['HandleAccDiffX', 'HandleAccDiffY']
    if all(col in available_cols for col in acceleration_diff_cols):
        scalers['acceleration_diff'] = StandardScaler().fit(feature_df[acceleration_diff_cols])
        print(f"acceleration diff scaler created:{acceleration_diff_cols}")

     # 将来的なジャーク（オプション）
    jerk_cols = ['JerkX', 'JerkY']
    if all(col in available_cols for col in jerk_cols):
        scalers['jerk'] = StandardScaler().fit(feature_df[jerk_cols])
        print(f"Jerk scaler created: {jerk_cols}")

    return scalers


def analyze_style_hierarchy(df):
    """スタイル階層の定量的分析"""

    # 被験者ごとの各物理量の特徴を抽出
    subjects = df['subject_id'].unique()

    # 各物理量での被験者間分離度を計算
    separation_scores = {}

    feature_groups = {
        'Position': ['HandlePosDiffX', 'HandlePosDiffY'],
        'Velocity': ['HandleVelDiffX', 'HandleVelDiffY'],
        'Acceleration': ['HandleAccDiffX', 'HandleAccDiffY']
    }

    for group_name, features in feature_groups.items():
        if all(f in df.columns for f in features):
            # 被験者ごとの平均を計算
            subject_means = []
            for subject in subjects:
                subject_data = df[df['subject_id'] == subject][features].mean().values
                subject_means.append(subject_data)

            subject_means = np.array(subject_means)

            # 被験者間分散 / 被験者内分散
            between_var = np.var(subject_means.mean(axis=1))

            within_vars = []
            for subject in subjects:
                subject_data = df[df['subject_id'] == subject][features].values
                if len(subject_data) > 1:
                    within_var = np.var(subject_data.mean(axis=1))
                    within_vars.append(within_var)

            avg_within_var = np.mean(within_vars) if within_vars else 0.01
            separation_score = between_var / (avg_within_var + 1e-8)

            separation_scores[group_name] = separation_score

            print(f"{group_name}:")
            print(f"  被験者間分散: {between_var:.6f}")
            print(f"  平均被験者内分散: {avg_within_var:.6f}")
            print(f"  分離スコア: {separation_score:.4f}")

    return separation_scores


def analyze_style_hierarchy(df):
    """スタイル階層の定量的分析"""
    print("\n" + "=" * 50)
    print("スタイル階層分析開始")
    print("=" * 50)

    # 被験者ごとの各物理量の特徴を抽出
    subjects = df['subject_id'].unique()
    print(f"分析対象被験者数: {len(subjects)}")
    print(f"被験者ID: {list(subjects)}")

    # 各物理量での被験者間分離度を計算
    separation_scores = {}

    feature_groups = {
        'Position': ['HandlePosX', 'HandlePosY'],
        'Velocity': ['HandleVelX', 'HandleVelY'],
        'Acceleration': ['HandleAccX', 'HandleAccY']
    }

    print(f"\n分析対象物理量グループ:")
    for group_name, features in feature_groups.items():
        available = all(f in df.columns for f in features)
        print(f"  {group_name}: {features} -> {'✅' if available else '❌'}")

    for group_name, features in feature_groups.items():
        if all(f in df.columns for f in features):
            # 被験者ごとの平均を計算
            subject_means = []
            subject_within_vars = []

            for subject in subjects:
                subject_data = df[df['subject_id'] == subject][features].values
                if len(subject_data) > 0:
                    subject_mean = np.mean(subject_data, axis=0)
                    subject_means.append(subject_mean)

                    # 被験者内分散も計算
                    if len(subject_data) > 1:
                        within_var = np.var(subject_data, axis=0).mean()
                        subject_within_vars.append(within_var)

            if len(subject_means) > 1:
                subject_means = np.array(subject_means)

                # 被験者間分散：各被験者の平均値の分散
                between_var = np.var(subject_means, axis=0).mean()

                # 平均被験者内分散
                avg_within_var = np.mean(subject_within_vars) if subject_within_vars else 0.01

                # 分離スコア：被験者間分散 / 被験者内分散
                separation_score = between_var / (avg_within_var + 1e-8)

                separation_scores[group_name] = separation_score

                print(f"\n{group_name}:")
                print(f"  被験者間分散: {between_var:.6f}")
                print(f"  平均被験者内分散: {avg_within_var:.6f}")
                print(f"  分離スコア: {separation_score:.4f}")

                # 分離度の評価
                if separation_score > 2.0:
                    print(f"  🟢 高い分離度 - スタイル情報が豊富")
                elif separation_score > 1.0:
                    print(f"  🟡 中程度の分離度 - 一部スタイル情報あり")
                else:
                    print(f"  🔴 低い分離度 - スタイル情報が少ない")

    print(f"\n" + "=" * 50)
    print("スタイル階層分析結果")
    print("=" * 50)

    # 分離スコアでソート
    sorted_scores = sorted(separation_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (group, score) in enumerate(sorted_scores, 1):
        print(f"{i}. {group}: {score:.4f}")

    if sorted_scores:
        best_group, best_score = sorted_scores[0]
        print(f"\n🎯 最も有効なスタイル情報: {best_group} (分離スコア: {best_score:.4f})")

        if best_score > 2.0:
            print("✅ 被験者クラスタリングに十分な情報があります")
        elif best_score > 1.0:
            print("⚠️  被験者クラスタリングは困難ですが可能性があります")
        else:
            print("❌ 被験者クラスタリングは非常に困難です")

    return separation_scores

def main():
    TARGET_SEQ_LEN = 100
    RAWDATA_DIR = '../../../data/RawDatas/'
    PROCESSED_DATA_DIR = 'PredictiveLatentSpaceNavigationModel/DataPreprocess/ForGeneralizedCoordinate'
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

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

    # 7. 訓練データとテストデータに分割
    subject_ids = master_df['subject_id'].unique()
    train_subjects, test_subjects = train_test_split(subject_ids, test_size=0.2, random_state=42)

    train_df = master_df[master_df['subject_id'].isin(train_subjects)]
    test_df = master_df[master_df['subject_id'].isin(test_subjects)]

    print(f"訓練データ被験者数: {len(train_subjects)}, テストデータ被験者数: {len(test_subjects)}")

    # 8. 差分表示の計算
    train_df = train_df.groupby(['subject_id', 'trial_num'], group_keys=False).apply(add_diffs)
    test_df = test_df.groupby(['subject_id', 'trial_num'], group_keys=False).apply(add_diffs)
    print("差分表示の計算完了 ✅ ")

    print("スタイル階層分析実行中...")
    # separation_score = analyze_style_hierarchy(master_df)
    # comprehensive_ptp_analysis(master_df)
    resolve_contradiction_analysis(train_df)
    # print(train_df.columns)

    # 8. スケーラの用意
    feature_cols = [
        'HandlePosDiffX', 'HandlePosDiffY',
        'HandleVelDiffX', 'HandleVelDiffY',
        # 'HandleAccDiffX', 'HandleAccDiffY',
    ]

    scalers = prepare_scaler(train_df[feature_cols])
    print("StandardScalerを訓練データで学習しました。✅")

    # 9. スケーラと特徴量定義を保存
    scaler_path = os.path.join(PROCESSED_DATA_DIR, 'scalers.joblib')
    feature_config_path = os.path.join(PROCESSED_DATA_DIR, 'feature_config.joblib')

    joblib.dump(scalers, scaler_path)
    joblib.dump({'feature_cols': feature_cols}, feature_config_path)

    print(f"スケーラーを {scaler_path} に保存しました。")
    print(f"特徴量設定を {feature_config_path} に保存しました。")

    # 訓練データとテストデータをそれぞれ保存
    train_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, 'train_data.parquet'))
    test_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, 'test_data.parquet'))
    print("訓練データとテストデータをParquet形式で保存しました。")



if __name__ == "__main__":
    main()
