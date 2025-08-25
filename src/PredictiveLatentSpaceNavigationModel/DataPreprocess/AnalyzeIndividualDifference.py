import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import joblib # スケーラーの保存に使用
plt.switch_backend('Agg')

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

# ======== ANOVAのための高次特徴量計算 ========
def calculate_path_curvature(positions):
    """軌道の平均曲率を計算"""
    if len(positions) < 3:
        return np.nan

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


def calculate_path_linearity(positions):
    """軌道の直線性を計算"""
    if len(positions) < 2:
        return np.nan

    start_point = positions[0]
    end_point = positions[-1]
    straight_distance = np.linalg.norm(end_point - start_point)

    actual_distance = np.sum([np.linalg.norm(positions[i + 1] - positions[i])
                              for i in range(len(positions) - 1)])

    return straight_distance / actual_distance if actual_distance > 0 else 0


def calculate_path_efficiency_hf(positions):
    """軌道効率性（高次版）"""
    return calculate_path_linearity(positions)


def calculate_velocity_smoothness(velocities):
    """速度の滑らかさを計算"""
    if len(velocities) < 2:
        return np.nan

    vel_magnitude = np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)
    velocity_changes = np.abs(np.diff(vel_magnitude))

    # 正規化された滑らかさ指標
    smoothness = 1.0 / (1.0 + np.std(velocity_changes))
    return smoothness


def calculate_velocity_symmetry(velocities):
    """速度プロファイルの対称性"""
    if len(velocities) < 4:
        return np.nan

    vel_magnitude = np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)

    # 前半と後半の相関を計算
    mid_point = len(vel_magnitude) // 2
    first_half = vel_magnitude[:mid_point]
    second_half = vel_magnitude[mid_point:mid_point + len(first_half)][::-1]

    if len(first_half) > 1 and len(second_half) > 1:
        correlation = np.corrcoef(first_half, second_half)[0, 1]
        return correlation if not np.isnan(correlation) else 0
    return 0


def calculate_peak_velocity_timing(velocities):
    """最大速度到達タイミング（正規化）"""
    if len(velocities) < 2:
        return np.nan

    vel_magnitude = np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)
    peak_time_ratio = np.argmax(vel_magnitude) / len(vel_magnitude)

    return peak_time_ratio


def calculate_jerk_metric(trial_df):
    """ジャーク指標を計算"""
    if len(trial_df) < 3:
        return np.nan

    acc_x = trial_df['HandleAccX'].values
    acc_y = trial_df['HandleAccY'].values

    jerk_x = np.diff(acc_x)
    jerk_y = np.diff(acc_y)
    jerk_magnitude = np.sqrt(jerk_x ** 2 + jerk_y ** 2)

    return np.mean(jerk_magnitude)


def calculate_acceleration_smoothness(trial_df):
    """加速度の滑らかさ"""
    if len(trial_df) < 3:
        return np.nan

    acc_x = trial_df['HandleAccX'].values
    acc_y = trial_df['HandleAccY'].values
    acc_magnitude = np.sqrt(acc_x ** 2 + acc_y ** 2)

    acc_changes = np.abs(np.diff(acc_magnitude))
    return 1.0 / (1.0 + np.std(acc_changes))


def calculate_movement_rhythm(velocities):
    """運動リズム（主要周波数成分）"""
    if len(velocities) < 10:
        return np.nan

    vel_magnitude = np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)

    try:
        # FFTで周波数解析
        fft = np.fft.fft(vel_magnitude)
        freqs = np.fft.fftfreq(len(vel_magnitude))

        # 主要周波数成分
        dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft) // 2])) + 1
        dominant_freq = abs(freqs[dominant_freq_idx])

        return dominant_freq
    except:
        return np.nan


def calculate_force_modulation(trial_df):
    """力調節パターン"""
    if len(trial_df) < 5:
        return np.nan

    acc_x = trial_df['HandleAccX'].values
    acc_y = trial_df['HandleAccY'].values
    force_estimate = np.sqrt(acc_x ** 2 + acc_y ** 2)

    # 力の変調指標
    force_variability = np.std(force_estimate) / (np.mean(force_estimate) + 1e-6)
    return force_variability


def calculate_temporal_consistency(trial_df):
    """時間的一貫性"""
    if len(trial_df) < 10:
        return np.nan

    timestamps = trial_df['Timestamp'].values
    time_intervals = np.diff(timestamps)

    # 時間間隔の一貫性
    consistency = 1.0 / (1.0 + np.std(time_intervals) / np.mean(time_intervals))
    return consistency


def calculate_movement_efficiency_index(trial_df):
    """総合運動効率指標"""
    if len(trial_df) < 5:
        return np.nan

    # 複数指標の統合
    positions = trial_df[['HandlePosX', 'HandlePosY']].values
    velocities = trial_df[['HandleVelX', 'HandleVelY']].values

    linearity = calculate_path_linearity(positions)
    smoothness = calculate_velocity_smoothness(velocities)
    jerk = calculate_jerk_metric(trial_df)

    # 正規化して統合（ジャークは逆数）
    efficiency = (linearity + smoothness) / (1 + jerk)
    return efficiency


def calculate_control_stability(trial_df):
    """制御安定性指標"""
    if len(trial_df) < 5:
        return np.nan

    acc_x = trial_df['HandleAccX'].values
    acc_y = trial_df['HandleAccY'].values

    # 加速度の標準偏差（制御の安定性の逆指標）
    acc_std = np.std(np.sqrt(acc_x ** 2 + acc_y ** 2))
    stability = 1.0 / (1.0 + acc_std)

    return stability

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


# ---- モデルへの入力データの個人差ANOVA解析 ----
def perform_comprehensive_anova_analysis(df, output_dir='./anova_output'):
    """
    各物理量について被験者間の個人差をANOVAで分析

    Parameters:
    df: DataFrame - 処理済みデータフレーム
    """

    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 出力ディレクトリ: {output_dir}")

    print("🔬 ANOVA分析による被験者間個人差検定")
    print("=" * 70)

    # 分析対象の物理量
    physical_quantities = [
        'HandlePosX', 'HandlePosY',
        'HandleVelX', 'HandleVelY',
        'HandleAccX', 'HandleAccY',
        'HandlePosDiffX', 'HandlePosDiffY',
        'HandleVelDiffX', 'HandleVelDiffY',
        'HandleAccDiffX', 'HandleAccDiffY'
    ]

    # 利用可能な物理量のみを抽出
    available_quantities = [col for col in physical_quantities if col in df.columns]
    print(f"分析対象物理量: {len(available_quantities)}個")
    print(f"対象: {available_quantities}")

    # 被験者数とデータ点数の確認
    subjects = df['subject_id'].unique()
    print(f"\n被験者数: {len(subjects)}")
    print(f"総データ点数: {len(df)}")

    # 被験者ごとのデータ点数
    subject_counts = df['subject_id'].value_counts()
    print(f"被験者ごとのデータ点数:")
    for subject, count in subject_counts.items():
        print(f"  {subject}: {count}点")

    # ANOVA結果を格納するリスト
    anova_results = []

    print(f"\n" + "=" * 70)
    print("ANOVA分析結果")
    print("=" * 70)

    for quantity in available_quantities:
        print(f"\n📊 {quantity} の分析")
        print("-" * 50)

        # 欠損値を除去
        clean_data = df[['subject_id', quantity]].dropna()

        if len(clean_data) == 0:
            print(f"⚠️  {quantity}: データが不足しています")
            continue

        # 被験者ごとのデータを準備
        subject_data = []
        subject_stats = {}

        for subject in subjects:
            subject_values = clean_data[clean_data['subject_id'] == subject][quantity].values
            if len(subject_values) > 0:
                subject_data.append(subject_values)
                subject_stats[subject] = {
                    'mean': np.mean(subject_values),
                    'std': np.std(subject_values),
                    'n': len(subject_values)
                }

        # 被験者が2人以上いる場合のみANOVAを実行
        if len(subject_data) < 2:
            print(f"⚠️  {quantity}: ANOVA実行には最低2被験者必要")
            continue

        try:
            # 1. scipy.stats.f_onewayによるANOVA
            f_statistic, p_value = f_oneway(*subject_data)

            # 2. statsmodelsによる詳細なANOVA
            formula = f'{quantity} ~ C(subject_id)'
            model = ols(formula, data=clean_data).fit()
            anova_table = anova_lm(model, typ=2)

            # 3. 効果量（eta-squared）の計算
            ss_between = anova_table.loc['C(subject_id)', 'sum_sq']
            ss_total = anova_table['sum_sq'].sum()
            eta_squared = ss_between / ss_total

            # 4. 結果の表示
            print(f"F統計量: {f_statistic:.4f}")
            print(f"p値: {p_value:.6f}")
            print(f"効果量 (η²): {eta_squared:.4f}")

            # 統計的有意性の判定
            alpha = 0.05
            if p_value < alpha:
                print(f"🟢 統計的有意 (p < {alpha}) - 被験者間に個人差あり")
                significance = "有意"
            else:
                print(f"🔴 統計的非有意 (p ≥ {alpha}) - 被験者間に個人差なし")
                significance = "非有意"

            # 効果量の解釈
            if eta_squared >= 0.14:
                effect_size = "大"
            elif eta_squared >= 0.06:
                effect_size = "中"
            elif eta_squared >= 0.01:
                effect_size = "小"
            else:
                effect_size = "無視できる"

            print(f"効果量の大きさ: {effect_size}")

            # 被験者別統計の表示
            print(f"\n被験者別統計:")
            for subject, stats in subject_stats.items():
                print(f"  {subject}: 平均={stats['mean']:.6f}, "
                      f"標準偏差={stats['std']:.6f}, N={stats['n']}")

            # 結果を記録
            anova_results.append({
                'quantity': quantity,
                'f_statistic': f_statistic,
                'p_value': p_value,
                'eta_squared': eta_squared,
                'significance': significance,
                'effect_size': effect_size,
                'n_subjects': len(subject_data),
                'total_n': len(clean_data)
            })

        except Exception as e:
            print(f"⚠️  {quantity}: ANOVA分析中にエラー - {str(e)}")

    # 総合結果の表示
    print(f"\n" + "=" * 70)
    print("総合分析結果")
    print("=" * 70)

    if anova_results:
        results_df = pd.DataFrame(anova_results)

        # 有意な物理量
        significant_quantities = results_df[results_df['significance'] == '有意']['quantity'].tolist()
        non_significant_quantities = results_df[results_df['significance'] == '非有意']['quantity'].tolist()

        print(f"統計的有意な物理量 ({len(significant_quantities)}個):")
        if significant_quantities:
            for qty in significant_quantities:
                result = results_df[results_df['quantity'] == qty].iloc[0]
                print(f"  🟢 {qty}: p={result['p_value']:.6f}, η²={result['eta_squared']:.4f} ({result['effect_size']})")
        else:
            print("  なし")

        print(f"\n統計的非有意な物理量 ({len(non_significant_quantities)}個):")
        if non_significant_quantities:
            for qty in non_significant_quantities:
                result = results_df[results_df['quantity'] == qty].iloc[0]
                print(f"  🔴 {qty}: p={result['p_value']:.6f}, η²={result['eta_squared']:.4f}")
        else:
            print("  なし")

        # p値による順位付け
        print(f"\n🏆 個人差の強い順ランキング (p値の昇順):")
        sorted_results = results_df.sort_values('p_value')
        for i, (_, result) in enumerate(sorted_results.iterrows(), 1):
            status = "🟢" if result['significance'] == '有意' else "🔴"
            print(f"  {i}位: {status} {result['quantity']} "
                  f"(p={result['p_value']:.6f}, η²={result['eta_squared']:.4f})")

        return results_df

    else:
        print("分析可能なデータがありませんでした。")
        return None


def create_anova_visualization(df, anova_results_df, output_dir='./anova_output'):
    """
    ANOVA結果の可視化
    """
    if anova_results_df is None or len(anova_results_df) == 0:
        print("可視化するデータがありません。")
        return

    # 図のセットアップ
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Visualization of ANOVA analysis results', fontsize=16)

    # 1. p値のバープロット
    ax1 = axes[0, 0]
    quantities = anova_results_df['quantity']
    p_values = anova_results_df['p_value']
    colors = ['green' if p < 0.05 else 'red' for p in p_values]

    bars1 = ax1.bar(range(len(quantities)), p_values, color=colors, alpha=0.7)
    ax1.axhline(y=0.05, color='black', linestyle='--', alpha=0.5, label='α=0.05')
    ax1.set_xlabel('Physical Quantity')
    ax1.set_ylabel('p value')
    ax1.set_title('ANOVA p value (Green:Significant, Red:Non Significant)')
    ax1.set_xticks(range(len(quantities)))
    ax1.set_xticklabels(quantities, rotation=45, ha='right')
    ax1.legend()
    ax1.set_yscale('log')

    # 2. 効果量のバープロット
    ax2 = axes[0, 1]
    eta_squared = anova_results_df['eta_squared']

    bars2 = ax2.bar(range(len(quantities)), eta_squared, color='blue', alpha=0.7)
    ax2.set_xlabel('Physical Quantity')
    ax2.set_ylabel('Effectiveness (η²)')
    ax2.set_title('ANOVA Effectiveness')
    ax2.set_xticks(range(len(quantities)))
    ax2.set_xticklabels(quantities, rotation=45, ha='right')

    # 効果量の解釈線を追加
    ax2.axhline(y=0.01, color='gray', linestyle=':', alpha=0.5, label='Small Effect')
    ax2.axhline(y=0.06, color='orange', linestyle=':', alpha=0.5, label='Medium Effect')
    ax2.axhline(y=0.14, color='red', linestyle=':', alpha=0.5, label='Highly Effect')
    ax2.legend()

    # 3. F統計量のバープロット
    ax3 = axes[1, 0]
    f_statistics = anova_results_df['f_statistic']

    bars3 = ax3.bar(range(len(quantities)), f_statistics, color='purple', alpha=0.7)
    ax3.set_xlabel('Physical Quantity')
    ax3.set_ylabel('F Statistical Quantity')
    ax3.set_title('ANOVA F Statistical Quantity')
    ax3.set_xticks(range(len(quantities)))
    ax3.set_xticklabels(quantities, rotation=45, ha='right')

    # 4. 散布図：p値 vs 効果量
    ax4 = axes[1, 1]
    colors_scatter = ['green' if p < 0.05 else 'red' for p in p_values]
    scatter = ax4.scatter(eta_squared, -np.log10(p_values), c=colors_scatter, alpha=0.7, s=100)

    ax4.set_xlabel('Effectiveness (η²)')
    ax4.set_ylabel('-log₁₀(p値)')
    ax4.set_title('Effectiveness vs Statistical Significance')
    ax4.axhline(y=-np.log10(0.05), color='black', linestyle='--', alpha=0.5, label='α=0.05')
    ax4.axvline(x=0.06, color='orange', linestyle=':', alpha=0.5, label='Medium Effect')

    # 各点にラベルを追加
    for i, quantity in enumerate(quantities):
        ax4.annotate(quantity, (eta_squared.iloc[i], -np.log10(p_values.iloc[i])),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax4.legend()

    plt.tight_layout()

    # プロットを保存
    plot_path = os.path.join(output_dir, 'anova_analysis_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 可視化結果を保存: {plot_path}")
    plt.close()  # メモリリークを防ぐためプロットを閉じる


def perform_post_hoc_analysis(df, significant_quantities, output_dir='./anova_output'):
    """
    事後分析：有意差が見つかった物理量について詳細分析
    """
    if not significant_quantities:
        print("事後分析対象がありません。")
        return

    print(f"\n" + "=" * 70)
    print("事後分析 (Post-hoc Analysis)")
    print("=" * 70)

    from scipy.stats import tukey_hsd

    for quantity in significant_quantities:
        print(f"\n📈 {quantity} の事後分析")
        print("-" * 40)

        clean_data = df[['subject_id', quantity]].dropna()
        subjects = clean_data['subject_id'].unique()

        # Tukey's HSD検定
        try:
            # 被験者ごとのデータを準備
            subject_groups = []
            subject_names = []

            for subject in subjects:
                subject_values = clean_data[clean_data['subject_id'] == subject][quantity].values
                if len(subject_values) > 0:
                    subject_groups.append(subject_values)
                    subject_names.append(subject)

            if len(subject_groups) >= 2:
                # Tukey HSD検定を実行
                tukey_result = tukey_hsd(*subject_groups)

                print(f"Tukey HSD検定結果:")
                print(f"統計量: {tukey_result.statistic}")
                print(f"p値行列:")

                # p値行列を見やすく表示
                n_subjects = len(subject_names)
                for i in range(n_subjects):
                    for j in range(i + 1, n_subjects):
                        p_val = tukey_result.pvalue[i, j] if hasattr(tukey_result, 'pvalue') else 'N/A'
                        print(f"  {subject_names[i]} vs {subject_names[j]}: p={p_val}")

        except Exception as e:
            print(f"Tukey HSD検定でエラー: {str(e)}")

            # 代替として、全ペア間のt検定を実行
            print(f"代替分析：全ペア間t検定")
            from scipy.stats import ttest_ind

            subjects_list = list(subjects)
            for i in range(len(subjects_list)):
                for j in range(i + 1, len(subjects_list)):
                    subj1_data = clean_data[clean_data['subject_id'] == subjects_list[i]][quantity].values
                    subj2_data = clean_data[clean_data['subject_id'] == subjects_list[j]][quantity].values

                    if len(subj1_data) > 1 and len(subj2_data) > 1:
                        t_stat, p_val = ttest_ind(subj1_data, subj2_data)
                        significance = "🟢" if p_val < 0.05 else "🔴"
                        print(f"  {subjects_list[i]} vs {subjects_list[j]}: "
                              f"t={t_stat:.4f}, p={p_val:.6f} {significance}")


def create_individual_quantity_plots(df, anova_results_df, output_dir='./anova_output'):
    """
    各物理量について個別の箱ひげ図を作成
    """
    if anova_results_df is None or len(anova_results_df) == 0:
        return

    print(f"\n📈 個別物理量プロット作成中...")

    # 有意な物理量と非有意な物理量を分けて処理
    significant_quantities = anova_results_df[anova_results_df['significance'] == '有意']['quantity'].tolist()
    non_significant_quantities = anova_results_df[anova_results_df['significance'] == '非有意']['quantity'].tolist()

    all_quantities = anova_results_df['quantity'].tolist()

    # 各物理量について箱ひげ図を作成
    n_quantities = len(all_quantities)
    n_cols = 4
    n_rows = (n_quantities + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Inter-subject comparisons for each physical variable (box plots)', fontsize=16)

    for i, quantity in enumerate(all_quantities):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # データの準備
        clean_data = df[['subject_id', quantity]].dropna()

        if len(clean_data) > 0:
            # 箱ひげ図の作成
            subjects = clean_data['subject_id'].unique()
            box_data = [clean_data[clean_data['subject_id'] == subject][quantity].values
                        for subject in subjects]

            box_plot = ax.boxplot(box_data, labels=subjects, patch_artist=True)

            # 有意性に応じて色を変更
            if quantity in significant_quantities:
                color = 'lightgreen'
                ax.set_title(f'{quantity}\n(Significant: p<0.05)', fontweight='bold', color='green')
            else:
                color = 'lightcoral'
                ax.set_title(f'{quantity}\n(Non Significant: p≥0.05)', color='red')

            # 箱の色を設定
            for patch in box_plot['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_xlabel('Subject ID')
        ax.set_ylabel(quantity)
        ax.tick_params(axis='x', rotation=45)

    # 使用しない軸を非表示
    for i in range(len(all_quantities), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    # プロットを保存
    boxplot_path = os.path.join(output_dir, 'individual_quantity_boxplots.png')
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    print(f"📊 個別箱ひげ図を保存: {boxplot_path}")
    plt.close()


def create_summary_report(anova_results_df, output_dir='./anova_output'):
    """
    分析結果の詳細レポートをテキストファイルで作成
    """
    if anova_results_df is None:
        return

    report_path = os.path.join(output_dir, 'anova_analysis_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("ANOVA分析による被験者間個人差検定 - 詳細レポート\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"分析日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析対象物理量数: {len(anova_results_df)}\n\n")

        # 全体概要
        significant_count = len(anova_results_df[anova_results_df['significance'] == '有意'])
        f.write("=" * 40 + "\n")
        f.write("全体概要\n")
        f.write("=" * 40 + "\n")
        f.write(f"統計的有意な物理量: {significant_count}/{len(anova_results_df)} 個\n")
        f.write(f"有意率: {significant_count / len(anova_results_df) * 100:.1f}%\n\n")

        # 詳細結果
        f.write("=" * 40 + "\n")
        f.write("詳細結果（p値昇順）\n")
        f.write("=" * 40 + "\n")

        sorted_results = anova_results_df.sort_values('p_value')
        for i, (_, result) in enumerate(sorted_results.iterrows(), 1):
            f.write(f"\n{i:2d}. {result['quantity']}\n")
            f.write(f"    F統計量: {result['f_statistic']:.4f}\n")
            f.write(f"    p値:     {result['p_value']:.6f}\n")
            f.write(f"    効果量:   {result['eta_squared']:.4f} ({result['effect_size']})\n")
            f.write(f"    判定:     {result['significance']}\n")
            f.write(f"    被験者数: {result['n_subjects']}\n")
            f.write(f"    総データ数: {result['total_n']}\n")

        # 統計的有意な物理量の詳細
        significant_df = anova_results_df[anova_results_df['significance'] == '有意']
        if len(significant_df) > 0:
            f.write("\n" + "=" * 40 + "\n")
            f.write("統計的有意な物理量（個人差あり）\n")
            f.write("=" * 40 + "\n")
            for _, result in significant_df.iterrows():
                f.write(f"• {result['quantity']}: p={result['p_value']:.6f}, η²={result['eta_squared']:.4f}\n")

        # 統計的非有意な物理量
        non_significant_df = anova_results_df[anova_results_df['significance'] == '非有意']
        if len(non_significant_df) > 0:
            f.write("\n" + "=" * 40 + "\n")
            f.write("統計的非有意な物理量（個人差なし）\n")
            f.write("=" * 40 + "\n")
            for _, result in non_significant_df.iterrows():
                f.write(f"• {result['quantity']}: p={result['p_value']:.6f}, η²={result['eta_squared']:.4f}\n")

        # 結論
        f.write("\n" + "=" * 40 + "\n")
        f.write("結論\n")
        f.write("=" * 40 + "\n")
        if significant_count > 0:
            f.write(f"分析の結果、{significant_count}個の物理量で統計的に有意な被験者間差が検出されました。\n")
            f.write("これらの物理量では個人差（運動スタイルの違い）が存在することが示唆されます。\n")
        else:
            f.write("分析の結果、いずれの物理量においても統計的に有意な被験者間差は検出されませんでした。\n")
            f.write("Point-to-Pointタスクでは個人差が現れにくい可能性があります。\n")

    print(f"📝 詳細レポートを保存: {report_path}")


def compare_scaling_effects(original_results, scaled_results, output_dir):
    """
    スケーリング前後でのANOVA結果を比較分析
    """
    if original_results is None or scaled_results is None:
        print("比較に必要なデータが不足しています。")
        return

    print("\n" + "=" * 70)
    print("🔄 スケーリング前後の比較分析")
    print("=" * 70)

    # 共通の物理量を抽出
    common_quantities = set(original_results['quantity']) & set(scaled_results['quantity'])

    comparison_data = []

    for quantity in common_quantities:
        orig = original_results[original_results['quantity'] == quantity].iloc[0]
        scaled = scaled_results[scaled_results['quantity'] == quantity].iloc[0]

        # p値の変化
        p_change = scaled['p_value'] / orig['p_value'] if orig['p_value'] > 0 else float('inf')

        # 効果量の変化
        eta_change = scaled['eta_squared'] / orig['eta_squared'] if orig['eta_squared'] > 0 else float('inf')

        # 有意性の変化
        sig_change = f"{orig['significance']} → {scaled['significance']}"

        comparison_data.append({
            'quantity': quantity,
            'original_p': orig['p_value'],
            'scaled_p': scaled['p_value'],
            'p_ratio': p_change,
            'original_eta2': orig['eta_squared'],
            'scaled_eta2': scaled['eta_squared'],
            'eta2_ratio': eta_change,
            'significance_change': sig_change
        })

        print(f"\n📊 {quantity}:")
        print(f"  p値: {orig['p_value']:.6f} → {scaled['p_value']:.6f} ({p_change:.2f}倍)")
        print(f"  η²: {orig['eta_squared']:.6f} → {scaled['eta_squared']:.6f} ({eta_change:.2f}倍)")
        print(f"  有意性: {sig_change}")

    # 比較結果をDataFrameとして保存
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(output_dir, 'scaling_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n💾 比較結果を保存: {comparison_path}")

    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # p値の比較
    ax1 = axes[0]
    x_pos = range(len(comparison_df))
    ax1.bar([x - 0.2 for x in x_pos], comparison_df['original_p'], width=0.4, label='Original', alpha=0.7)
    ax1.bar([x + 0.2 for x in x_pos], comparison_df['scaled_p'], width=0.4, label='Scaled', alpha=0.7)
    ax1.set_xlabel('Physical Quantity')
    ax1.set_ylabel('p value')
    ax1.set_title('Comparison of p-values before and after scaling')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(comparison_df['quantity'], rotation=45, ha='right')
    ax1.set_yscale('log')
    ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
    ax1.legend()

    # 効果量の比較
    ax2 = axes[1]
    ax2.bar([x - 0.2 for x in x_pos], comparison_df['original_eta2'], width=0.4, label='Original', alpha=0.7)
    ax2.bar([x + 0.2 for x in x_pos], comparison_df['scaled_eta2'], width=0.4, label='Scaled', alpha=0.7)
    ax2.set_xlabel('Physical Quantity')
    ax2.set_ylabel('Effectiveness (η²)')
    ax2.set_title('Comparison of p-values before and after scaling')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(comparison_df['quantity'], rotation=45, ha='right')
    ax2.legend()

    plt.tight_layout()
    comparison_plot_path = os.path.join(output_dir, 'scaling_comparison_plot.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 比較プロットを保存: {comparison_plot_path}")
    plt.close()


def main_anova_analysis(df, output_dir='./anova_output'):
    """
    メイン関数：包括的なANOVA分析を実行

    Parameters:
    df: DataFrame - 分析対象データ
    output_dir: str - 出力ディレクトリパス
    """
    print("🚀 ANOVA分析開始")
    print(f"📁 出力先: {output_dir}")

    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # 1. メインのANOVA分析
    anova_results_df = perform_comprehensive_anova_analysis(df, output_dir)

    if anova_results_df is not None:
        # 2. 可視化
        print(f"\n📊 結果の可視化...")
        create_anova_visualization(df, anova_results_df, output_dir)

        # 3. 個別物理量のプロット作成
        create_individual_quantity_plots(df, anova_results_df, output_dir)

        # 4. 有意な物理量について事後分析
        significant_quantities = anova_results_df[anova_results_df['significance'] == '有意']['quantity'].tolist()
        if significant_quantities:
            perform_post_hoc_analysis(df, significant_quantities, output_dir)

        # 5. 結果の保存
        csv_path = os.path.join(output_dir, 'anova_results.csv')
        anova_results_df.to_csv(csv_path, index=False)
        print(f"💾 分析結果CSVを保存: {csv_path}")

        # 6. 詳細レポートの作成
        create_summary_report(anova_results_df, output_dir)

        # 7. 完了メッセージ
        print(f"\n✅ ANOVA分析完了！")
        print(f"📁 すべての結果は '{output_dir}' に保存されました")
        print(f"📊 生成されたファイル:")
        print(f"  - anova_results.csv: 分析結果データ")
        print(f"  - anova_analysis_plots.png: 総合可視化")
        print(f"  - individual_quantity_boxplots.png: 個別箱ひげ図")
        print(f"  - anova_analysis_report.txt: 詳細レポート")

        return anova_results_df

    else:
        print("ANOVA分析を完了できませんでした。")
        return None


def extract_high_level_features(df):
    """
    各試行から高次特徴量を抽出してDataFrameを作成

    Returns:
    pd.DataFrame: 被験者・試行ごとの高次特徴量
    """
    print("🔍 高次特徴量の抽出開始")
    print("=" * 50)

    high_level_features = []

    # 被験者・試行ごとにグループ化
    for (subject_id, trial_num), trial_df in df.groupby(['subject_id', 'trial_num']):
        if len(trial_df) < 20:  # 最小データ点数の確保
            continue

        try:
            # 基本データの準備
            positions = trial_df[['HandlePosX', 'HandlePosY']].values
            velocities = trial_df[['HandleVelX', 'HandleVelY']].values

            # 高次特徴量を計算
            features = {
                'subject_id': subject_id,
                'trial_num': trial_num,

                # 1. 軌道特徴
                'path_curvature': calculate_path_curvature(positions),
                'path_linearity': calculate_path_linearity(positions),
                'path_efficiency': calculate_path_efficiency_hf(positions),

                # 2. 速度特徴
                'velocity_smoothness': calculate_velocity_smoothness(velocities),
                'velocity_symmetry': calculate_velocity_symmetry(velocities),
                'peak_velocity_timing': calculate_peak_velocity_timing(velocities),

                # 3. 加速度・ジャーク特徴
                'jerk_metric': calculate_jerk_metric(trial_df),
                'acceleration_smoothness': calculate_acceleration_smoothness(trial_df),

                # 4. 動的特徴
                'movement_rhythm': calculate_movement_rhythm(velocities),
                'force_modulation': calculate_force_modulation(trial_df),
                'temporal_consistency': calculate_temporal_consistency(trial_df),

                # 5. 統合的特徴
                'movement_efficiency_index': calculate_movement_efficiency_index(trial_df),
                'control_stability': calculate_control_stability(trial_df),
            }

            high_level_features.append(features)

        except Exception as e:
            print(f"⚠️ {subject_id}-{trial_num}: 特徴抽出エラー - {str(e)}")
            continue

    # DataFrameに変換
    features_df = pd.DataFrame(high_level_features)

    print(f"✅ 抽出完了: {len(features_df)}試行, {len(features_df.columns) - 2}特徴量")
    print(f"被験者別試行数:")
    for subject in features_df['subject_id'].unique():
        count = len(features_df[features_df['subject_id'] == subject])
        print(f"  {subject}: {count}試行")

    return features_df

def perform_high_level_anova(features_df, output_dir='./high_level_anova_results'):
    """
    高次特徴量でのANOVA分析を実行
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n🔬 高次特徴量ANOVA分析開始")
    print("=" * 60)

    # 分析対象特徴量（subject_idとtrial_numを除く）
    feature_columns = [col for col in features_df.columns
                       if col not in ['subject_id', 'trial_num']]

    print(f"分析対象特徴量: {len(feature_columns)}個")
    print(f"特徴量: {feature_columns}")

    # 被験者情報
    subjects = features_df['subject_id'].unique()
    print(f"\n被験者数: {len(subjects)}")
    print(f"総試行数: {len(features_df)}")

    subject_counts = features_df['subject_id'].value_counts()
    print(f"被験者別試行数:")
    for subject, count in subject_counts.items():
        print(f"  {subject}: {count}試行")

    # ANOVA結果格納
    anova_results = []

    print(f"\n" + "=" * 60)
    print("高次特徴量ANOVA分析結果")
    print("=" * 60)

    for feature in feature_columns:
        print(f"\n📊 {feature} の分析")
        print("-" * 40)

        # 欠損値を除去
        clean_data = features_df[['subject_id', feature]].dropna()

        if len(clean_data) == 0:
            print(f"⚠️ {feature}: データが不足")
            continue

        # 被験者ごとのデータ準備
        subject_data = []
        subject_stats = {}

        for subject in subjects:
            subject_values = clean_data[clean_data['subject_id'] == subject][feature].values
            if len(subject_values) > 0:
                subject_data.append(subject_values)
                subject_stats[subject] = {
                    'mean': np.mean(subject_values),
                    'std': np.std(subject_values),
                    'n': len(subject_values)
                }

        if len(subject_data) < 2:
            print(f"⚠️ {feature}: 被験者数不足")
            continue

        try:
            # ANOVA実行
            f_statistic, p_value = f_oneway(*subject_data)

            # 詳細ANOVA
            formula = f'{feature} ~ C(subject_id)'
            model = ols(formula, data=clean_data).fit()
            anova_table = anova_lm(model, typ=2)

            # 効果量計算
            ss_between = anova_table.loc['C(subject_id)', 'sum_sq']
            ss_total = anova_table['sum_sq'].sum()
            eta_squared = ss_between / ss_total

            # 結果表示
            print(f"F統計量: {f_statistic:.4f}")
            print(f"p値: {p_value:.6f}")
            print(f"効果量 (η²): {eta_squared:.4f}")

            # 有意性判定
            alpha = 0.05
            if p_value < alpha:
                print(f"🟢 統計的有意 (p < {alpha}) - 被験者間に個人差あり")
                significance = "有意"
            else:
                print(f"🔴 統計的非有意 (p ≥ {alpha}) - 被験者間に個人差なし")
                significance = "非有意"

            # 効果量解釈
            if eta_squared >= 0.14:
                effect_size = "大"
            elif eta_squared >= 0.06:
                effect_size = "中"
            elif eta_squared >= 0.01:
                effect_size = "小"
            else:
                effect_size = "無視できる"

            print(f"効果量の大きさ: {effect_size}")

            # 被験者別統計
            print(f"\n被験者別統計:")
            for subject, stats in subject_stats.items():
                print(f"  {subject}: 平均={stats['mean']:.4f}, "
                      f"標準偏差={stats['std']:.4f}, N={stats['n']}")

            # 結果記録
            anova_results.append({
                'feature': feature,
                'f_statistic': f_statistic,
                'p_value': p_value,
                'eta_squared': eta_squared,
                'significance': significance,
                'effect_size': effect_size,
                'n_subjects': len(subject_data),
                'total_n': len(clean_data)
            })

        except Exception as e:
            print(f"⚠️ {feature}: ANOVA分析エラー - {str(e)}")

    # 結果まとめ
    if anova_results:
        results_df = pd.DataFrame(anova_results)

        # 統計サマリー
        print(f"\n" + "=" * 60)
        print("高次特徴量分析結果サマリー")
        print("=" * 60)

        significant_features = results_df[results_df['significance'] == '有意']['feature'].tolist()
        non_significant_features = results_df[results_df['significance'] == '非有意']['feature'].tolist()

        print(f"統計的有意な特徴量 ({len(significant_features)}個):")
        if significant_features:
            for feat in significant_features:
                result = results_df[results_df['feature'] == feat].iloc[0]
                print(
                    f"  🟢 {feat}: p={result['p_value']:.6f}, η²={result['eta_squared']:.4f} ({result['effect_size']})")
        else:
            print("  なし")

        print(f"\n統計的非有意な特徴量 ({len(non_significant_features)}個):")
        if non_significant_features:
            for feat in non_significant_features:
                result = results_df[results_df['feature'] == feat].iloc[0]
                print(f"  🔴 {feat}: p={result['p_value']:.6f}, η²={result['eta_squared']:.4f}")
        else:
            print("  なし")

        # ランキング
        print(f"\n🏆 個人差の強い順ランキング (p値昇順):")
        sorted_results = results_df.sort_values('p_value')
        for i, (_, result) in enumerate(sorted_results.iterrows(), 1):
            status = "🟢" if result['significance'] == '有意' else "🔴"
            print(f"  {i}位: {status} {result['feature']} "
                  f"(p={result['p_value']:.6f}, η²={result['eta_squared']:.4f})")

        return results_df, features_df
    else:
        print("分析可能なデータがありませんでした。")
        return None, None


def create_high_level_visualization(results_df, features_df, output_dir='./high_level_anova_results'):
    """
    高次特徴量ANOVA結果の可視化
    """
    if results_df is None or len(results_df) == 0:
        print("可視化するデータがありません。")
        return

    print(f"\n📊 高次特徴量結果の可視化...")

    # 1. 総合結果プロット
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('High-Level Features ANOVA Analysis Results', fontsize=16)

    # p値プロット
    ax1 = axes[0, 0]
    features = results_df['feature']
    p_values = results_df['p_value']
    colors = ['green' if p < 0.05 else 'red' for p in p_values]

    bars1 = ax1.bar(range(len(features)), p_values, color=colors, alpha=0.7)
    ax1.axhline(y=0.05, color='black', linestyle='--', alpha=0.5, label='α=0.05')
    ax1.set_xlabel('High-Level Features')
    ax1.set_ylabel('p value')
    ax1.set_title('ANOVA p-values (Green:Significant, Red:Non-Significant)')
    ax1.set_xticks(range(len(features)))
    ax1.set_xticklabels(features, rotation=45, ha='right')
    ax1.legend()
    ax1.set_yscale('log')

    # 効果量プロット
    ax2 = axes[0, 1]
    eta_squared = results_df['eta_squared']

    bars2 = ax2.bar(range(len(features)), eta_squared, color='blue', alpha=0.7)
    ax2.set_xlabel('High-Level Features')
    ax2.set_ylabel('Effect Size (η²)')
    ax2.set_title('ANOVA Effect Sizes')
    ax2.set_xticks(range(len(features)))
    ax2.set_xticklabels(features, rotation=45, ha='right')

    # 効果量基準線
    ax2.axhline(y=0.01, color='gray', linestyle=':', alpha=0.5, label='Small Effect')
    ax2.axhline(y=0.06, color='orange', linestyle=':', alpha=0.5, label='Medium Effect')
    ax2.axhline(y=0.14, color='red', linestyle=':', alpha=0.5, label='Large Effect')
    ax2.legend()

    # F統計量プロット
    ax3 = axes[1, 0]
    f_statistics = results_df['f_statistic']

    bars3 = ax3.bar(range(len(features)), f_statistics, color='purple', alpha=0.7)
    ax3.set_xlabel('High-Level Features')
    ax3.set_ylabel('F-Statistic')
    ax3.set_title('ANOVA F-Statistics')
    ax3.set_xticks(range(len(features)))
    ax3.set_xticklabels(features, rotation=45, ha='right')

    # 散布図
    ax4 = axes[1, 1]
    colors_scatter = ['green' if p < 0.05 else 'red' for p in p_values]
    ax4.scatter(eta_squared, -np.log10(p_values), c=colors_scatter, alpha=0.7, s=100)

    ax4.set_xlabel('Effect Size (η²)')
    ax4.set_ylabel('-log₁₀(p-value)')
    ax4.set_title('Effect Size vs Statistical Significance')
    ax4.axhline(y=-np.log10(0.05), color='black', linestyle='--', alpha=0.5, label='α=0.05')
    ax4.axvline(x=0.06, color='orange', linestyle=':', alpha=0.5, label='Medium Effect')

    # ラベル追加
    for i, feature in enumerate(features):
        ax4.annotate(feature, (eta_squared.iloc[i], -np.log10(p_values.iloc[i])),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax4.legend()

    plt.tight_layout()

    # 保存
    plot_path = os.path.join(output_dir, 'high_level_features_anova_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 総合可視化を保存: {plot_path}")
    plt.close()

    # 2. 有意な特徴量のボックスプロット
    significant_features = results_df[results_df['significance'] == '有意']['feature'].tolist()

    if significant_features:
        n_features = len(significant_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_features > 1 else [axes]

        fig.suptitle('Significant High-Level Features - Individual Differences', fontsize=16)

        for i, feature in enumerate(significant_features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            # データ準備
            clean_data = features_df[['subject_id', feature]].dropna()

            if len(clean_data) > 0:
                subjects = clean_data['subject_id'].unique()
                box_data = [clean_data[clean_data['subject_id'] == subject][feature].values
                            for subject in subjects]

                box_plot = ax.boxplot(box_data, labels=subjects, patch_artist=True)

                # 有意な特徴量なので緑色
                for patch in box_plot['boxes']:
                    patch.set_facecolor('lightgreen')
                    patch.set_alpha(0.7)

                result = results_df[results_df['feature'] == feature].iloc[0]
                ax.set_title(f'{feature}\n(p={result["p_value"]:.4f}, η²={result["eta_squared"]:.4f})',
                             fontweight='bold', color='green')

            ax.set_xlabel('Subject ID')
            ax.set_ylabel(feature)
            ax.tick_params(axis='x', rotation=45)

        # 未使用軸を非表示
        for i in range(len(significant_features), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                if col < len(axes):
                    axes[col].set_visible(False)

        plt.tight_layout()

        boxplot_path = os.path.join(output_dir, 'significant_features_boxplots.png')
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        print(f"📊 有意特徴量ボックスプロットを保存: {boxplot_path}")
        plt.close()


def save_high_level_results(results_df, features_df, output_dir='./high_level_anova_results'):
    """
    高次特徴量分析結果を保存
    """
    if results_df is not None:
        # ANOVA結果
        results_path = os.path.join(output_dir, 'high_level_features_anova_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"💾 ANOVA結果を保存: {results_path}")

        # 特徴量データ
        features_path = os.path.join(output_dir, 'high_level_features_data.csv')
        features_df.to_csv(features_path, index=False)
        print(f"💾 特徴量データを保存: {features_path}")

        # レポート作成
        report_path = os.path.join(output_dir, 'high_level_features_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("高次特徴量による被験者間個人差分析 - 詳細レポート\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"分析日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"分析対象特徴量数: {len(results_df)}\n\n")

            significant_count = len(results_df[results_df['significance'] == '有意'])
            f.write("=" * 40 + "\n")
            f.write("分析結果概要\n")
            f.write("=" * 40 + "\n")
            f.write(f"統計的有意な特徴量: {significant_count}/{len(results_df)} 個\n")
            f.write(f"有意率: {significant_count / len(results_df) * 100:.1f}%\n\n")

            # 詳細結果
            sorted_results = results_df.sort_values('p_value')
            f.write("詳細結果（p値昇順）:\n")
            f.write("-" * 40 + "\n")
            for i, (_, result) in enumerate(sorted_results.iterrows(), 1):
                f.write(f"{i:2d}. {result['feature']}\n")
                f.write(f"    F統計量: {result['f_statistic']:.4f}\n")
                f.write(f"    p値: {result['p_value']:.6f}\n")
                f.write(f"    効果量: {result['eta_squared']:.4f} ({result['effect_size']})\n")
                f.write(f"    判定: {result['significance']}\n\n")

        print(f"📝 詳細レポートを保存: {report_path}")

def main_high_level_analysis(df, output_dir='./high_level_anova_results'):
    """
    高次特徴量分析のメイン実行関数
    """
    print("🚀 高次特徴量による個人差分析開始")
    print("=" * 70)

    # 1. 高次特徴量抽出
    features_df = extract_high_level_features(df)

    if features_df.empty:
        print("❌ 高次特徴量の抽出に失敗しました")
        return None

    # 2. ANOVA分析実行
    results_df, features_df = perform_high_level_anova(features_df, output_dir)

    if results_df is not None:
        # 3. 可視化
        create_high_level_visualization(results_df, features_df, output_dir)

        # 4. 結果保存
        save_high_level_results(results_df, features_df, output_dir)

        # 5. 低レベル特徴量との比較分析
        compare_with_low_level_features(results_df, output_dir)

        # 6. 完了メッセージ
        print(f"\n✅ 高次特徴量分析完了！")
        print(f"📁 結果は '{output_dir}' に保存されました")
        print(f"📊 生成されたファイル:")
        print(f"  - high_level_features_anova_results.csv: ANOVA分析結果")
        print(f"  - high_level_features_data.csv: 抽出された特徴量データ")
        print(f"  - high_level_features_anova_plots.png: 総合可視化")
        print(f"  - significant_features_boxplots.png: 有意特徴量ボックスプロット")
        print(f"  - high_level_features_report.txt: 詳細レポート")
        print(f"  - comparison_with_low_level.png: 低レベル特徴量との比較")

        return results_df, features_df

    else:
        print("❌ 高次特徴量分析を完了できませんでした")
        return None, None


def compare_with_low_level_features(high_level_results, output_dir):
    """
    高次特徴量と低レベル特徴量の分析結果を比較
    """
    print(f"\n📊 低レベル特徴量との比較分析...")

    # 低レベル特徴量の典型的な結果（あなたの既存分析から）
    low_level_results = {
        'HandlePosX': {'p_value': 0.000000, 'eta_squared': 0.0019, 'effect_size': '無視できる'},
        'HandlePosY': {'p_value': 0.000000, 'eta_squared': 0.0008, 'effect_size': '無視できる'},
        'HandleVelX': {'p_value': 0.000000, 'eta_squared': 0.0012, 'effect_size': '無視できる'},
        'HandleVelY': {'p_value': 0.000000, 'eta_squared': 0.0036, 'effect_size': '無視できる'},
        'HandleAccX': {'p_value': 0.999917, 'eta_squared': 0.0000, 'effect_size': '無視できる'},
        'HandleAccY': {'p_value': 0.999173, 'eta_squared': 0.0000, 'effect_size': '無視できる'},
    }

    # 比較プロット作成
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Low-Level vs High-Level Features: Individual Differences Analysis', fontsize=16)

    # 効果量比較
    ax1 = axes[0]

    # 低レベル特徴量
    low_level_names = list(low_level_results.keys())
    low_level_eta2 = [low_level_results[name]['eta_squared'] for name in low_level_names]

    # 高次特徴量
    high_level_names = high_level_results['feature'].tolist()
    high_level_eta2 = high_level_results['eta_squared'].tolist()

    # プロット
    x_low = range(len(low_level_names))
    x_high = range(len(low_level_names), len(low_level_names) + len(high_level_names))

    bars1 = ax1.bar(x_low, low_level_eta2, color='lightcoral', alpha=0.7,
                    label='Low-Level Features')
    bars2 = ax1.bar(x_high, high_level_eta2, color='lightgreen', alpha=0.7,
                    label='High-Level Features')

    ax1.set_xlabel('Features')
    ax1.set_ylabel('Effect Size (η²)')
    ax1.set_title('Effect Size Comparison: Low-Level vs High-Level Features')
    ax1.set_xticks(list(x_low) + list(x_high))
    ax1.set_xticklabels(low_level_names + high_level_names, rotation=45, ha='right')
    ax1.legend()

    # 基準線
    ax1.axhline(y=0.01, color='gray', linestyle=':', alpha=0.5, label='Small Effect')
    ax1.axhline(y=0.06, color='orange', linestyle=':', alpha=0.5, label='Medium Effect')
    ax1.axhline(y=0.14, color='red', linestyle=':', alpha=0.5, label='Large Effect')

    # 有意特徴量数の比較
    ax2 = axes[1]

    # 低レベル特徴量の有意数
    low_level_significant = sum(1 for result in low_level_results.values()
                                if result['p_value'] < 0.05)
    low_level_total = len(low_level_results)

    # 高次特徴量の有意数
    high_level_significant = len(high_level_results[high_level_results['significance'] == '有意'])
    high_level_total = len(high_level_results)

    categories = ['Low-Level\nFeatures', 'High-Level\nFeatures']
    significant_counts = [low_level_significant, high_level_significant]
    total_counts = [low_level_total, high_level_total]
    non_significant_counts = [total_counts[i] - significant_counts[i] for i in range(2)]

    # 積み上げバープロット
    bars1 = ax2.bar(categories, significant_counts, color='green', alpha=0.7,
                    label='Significant (p<0.05)')
    bars2 = ax2.bar(categories, non_significant_counts, bottom=significant_counts,
                    color='red', alpha=0.7, label='Non-Significant (p≥0.05)')

    ax2.set_ylabel('Number of Features')
    ax2.set_title('Statistical Significance: Feature Type Comparison')
    ax2.legend()

    # パーセンテージ表示
    for i, (cat, sig, total) in enumerate(zip(categories, significant_counts, total_counts)):
        percentage = sig / total * 100 if total > 0 else 0
        ax2.text(i, sig / 2, f'{sig}/{total}\n({percentage:.1f}%)',
                 ha='center', va='center', fontweight='bold')

    plt.tight_layout()

    # 保存
    comparison_path = os.path.join(output_dir, 'comparison_with_low_level.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"📊 比較分析プロットを保存: {comparison_path}")
    plt.close()

    # 比較サマリー出力
    print(f"\n📈 比較分析結果:")
    print(f"低レベル特徴量: {low_level_significant}/{low_level_total} "
          f"({low_level_significant / low_level_total * 100:.1f}%) が有意")
    print(f"高次特徴量: {high_level_significant}/{high_level_total} "
          f"({high_level_significant / high_level_total * 100:.1f}%) が有意")

    # 効果量の改善度
    low_level_max_eta2 = max(low_level_eta2) if low_level_eta2 else 0
    high_level_max_eta2 = max(high_level_eta2) if high_level_eta2 else 0

    print(f"\n効果量の改善:")
    print(f"低レベル特徴量最大η²: {low_level_max_eta2:.4f}")
    print(f"高次特徴量最大η²: {high_level_max_eta2:.4f}")
    if low_level_max_eta2 > 0:
        improvement_ratio = high_level_max_eta2 / low_level_max_eta2
        print(f"改善倍率: {improvement_ratio:.2f}倍")


def run_complete_analysis(train_df):
    """
    既存の低レベル分析と高次特徴量分析を両方実行
    """
    print("🎯 完全個人差分析実行")
    print("=" * 70)

    # 1. 既存の低レベルANOVA分析（あなたの既存コード）
    print("\n1️⃣ 低レベル特徴量ANOVA分析...")
    # ここで既存のmain_anova_analysis()を実行

    # 2. 高次特徴量ANOVA分析
    print("\n2️⃣ 高次特徴量ANOVA分析...")
    high_level_results, high_level_features = main_high_level_analysis(
        train_df,
        output_dir='./anova_results/high_level_features'
    )

    # 3. 統合レポート作成
    create_integrated_report(high_level_results, high_level_features)

    return high_level_results, high_level_features


def create_integrated_report(high_level_results, high_level_features):
    """
    低レベル＋高次特徴量の統合レポート作成
    """
    if high_level_results is None:
        return

    print(f"\n📝 統合レポート作成...")

    report_path = './anova_results/integrated_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Point-to-Pointタスクにおける個人差分析 - 統合レポート\n")
        f.write("=" * 80 + "\n\n")

        f.write("【研究の目的】\n")
        f.write("低レベル運動データ（位置・速度・加速度）と高次特徴量（軌道パターン・\n")
        f.write("制御特性など）における被験者間個人差の比較分析\n\n")

        f.write("【主要な発見】\n")
        f.write("1. 低レベル特徴量: 統計的有意だが効果量は無視できるレベル（η² < 0.004）\n")
        f.write("2. 高次特徴量: より大きな個人差を検出（詳細は以下参照）\n")
        f.write("3. Point-to-Pointタスクでは制御レベルが低いほど個人差が減少\n\n")

        # 高次特徴量の結果詳細
        significant_hl = high_level_results[high_level_results['significance'] == '有意']
        f.write("【高次特徴量での有意な個人差】\n")
        if len(significant_hl) > 0:
            for _, result in significant_hl.iterrows():
                f.write(f"• {result['feature']}: ")
                f.write(f"p={result['p_value']:.6f}, η²={result['eta_squared']:.4f}\n")
        else:
            f.write("統計的有意な個人差は検出されませんでした。\n")

        f.write(f"\n【結論】\n")
        f.write("Point-to-Pointタスクにおいて、低レベルの運動データでは実用的な\n")
        f.write("個人差は存在しないが、高次の運動制御特徴では個人差が存在する\n")
        f.write("可能性がある。この結果は運動制御の階層的性質を示唆している。\n")

    print(f"📄 統合レポートを保存: {report_path}")

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
        'HandlePosX', 'HandlePosY',
        'HandleVelX', 'HandleVelY',
        'HandleAccX', 'HandleAccY',
    ]

    scalers = prepare_scaler(train_df[feature_cols])
    print("StandardScalerを訓練データで学習しました。✅")

    # スケーリング後のデータを作成
    scaled_train_df = train_df.copy()

    # 各特徴量グループをスケーリング
    if 'position' in scalers:
        pos_cols = ['HandlePosX', 'HandlePosY']
        scaled_train_df[pos_cols] = scalers['position'].transform(train_df[pos_cols])

    if 'velocity' in scalers:
        vel_cols = ['HandleVelX', 'HandleVelY']
        scaled_train_df[vel_cols] = scalers['velocity'].transform(train_df[vel_cols])

    if 'acceleration' in scalers:
        acc_cols = ['HandleAccX', 'HandleAccY']
        scaled_train_df[acc_cols] = scalers['acceleration'].transform(train_df[acc_cols])

    # # 差分もANOVA解析対象に追加
    # diff_feature_cols = [
    #     'HandlePosDiffX', 'HandlePosDiffY',
    #     'HandleVelDiffX', 'HandleVelDiffY',
    #     'HandleAccDiffX', 'HandleAccDiffY'
    # ]

    # ANOVAで個人差の分析
    ANOVA_RESULT_PATH = "PredictiveLatentSpaceNavigationModel/DataPreprocess/AnovaResults/"
    # 1. スケーリング前の分析
    print("=" * 60)
    print("🔍 スケーリング前データでのANOVA分析")
    print("=" * 60)
    anova_cols_original = ['subject_id'] + feature_cols
    anova_data_original = train_df[anova_cols_original].copy()
    anova_result_original = main_anova_analysis(
        anova_data_original,
        os.path.join(ANOVA_RESULT_PATH, 'original_scale')
    )

    # 2. スケーリング後の分析
    print("=" * 60)
    print("🔍 スケーリング後データでのANOVA分析")
    print("=" * 60)
    anova_cols_scaled = ['subject_id'] + feature_cols
    anova_data_scaled = scaled_train_df[anova_cols_scaled].copy()
    anova_result_scaled = main_anova_analysis(
        anova_data_scaled,
        os.path.join(ANOVA_RESULT_PATH, 'scaled')
    )

    # 3. 比較分析
    compare_scaling_effects(anova_result_original, anova_result_scaled, ANOVA_RESULT_PATH)

    # 4. 高次特徴量のANOVA解析
    high_level_results, high_level_features = main_high_level_analysis(
        train_df,
        os.path.join(ANOVA_RESULT_PATH, 'HighLevelFeatures')
    )

if __name__ == "__main__":
    main()
