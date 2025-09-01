import pandas as pd
import numpy as np
from typing import  Dict, List, Tuple, Optional
from pathlib import  Path
from scipy.interpolate import interp1d, UnivariateSpline

import yaml


class DataPreprocessConfigLoader:
    def __init__(self, config_dir: str):
        # コンフィグファイルの存在確認
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise FileNotFoundError(f"データ前処理用のコンフィグが見つかりません")

        # 設定ファイルの読み込み
        with self.config_dir.open() as f:
            config = yaml.safe_load(f)

        is_valid, error_list = self.validate_config(config)
        if not is_valid:
            error_details = "\n - ".join(error_list)
            raise ValueError(f"設定ファイルに以下の問題があります: \n - {error_details}")
        else:
            self.config =config

    def validate_config(self, config: Dict) -> Tuple[bool, List[str]]:
        """設定を検証し、成功/失敗とエラーリストを返す"""

        all_errors = []
        all_errors.extend(self._validate_paths(config))
        all_errors.extend(self._validate_preprocessing(config))
        all_errors.extend(self._validate_analysis_flags(config))

        return len(all_errors) == 0, all_errors

    def _validate_paths(self, config: Dict) -> List[str]:
        """パス系設定の検証"""
        errors = []

        data_section = config.get('data')
        if data_section is None:
            errors.append("設定ファイルに必須セクション 'data' がありません")
            return errors

        raw_data_dir_section = data_section.get('raw_data_dir')
        if  raw_data_dir_section is None:
            errors.append("設定ファイルに必須セクション 'raw_data_dir' がありません。")
            return errors
        else:
            raw_data_dir = Path(raw_data_dir_section)
            if not raw_data_dir.exists():
                errors.append(f"raw_data_dirに指定されたディレクトリ '{raw_data_dir}' が存在しません")

        return errors

    def _validate_preprocessing(self, config:Dict) -> List[str]:
        """前処理設定の検証"""
        errors = []

        pre_process_section = config.get('pre_process')
        if pre_process_section is None:
            errors.append("設定ファイルに必須セクション 'pre_process' がありません")
            return errors

        target_seq_len_section = pre_process_section.get('target_seq_len')
        if target_seq_len_section is None:
            errors.append("設定ファイルに必須セクション 'target_seq_len' がありません")
            return errors

        interpolate_method_section = pre_process_section.get('interpolate_method')
        if interpolate_method_section is None:
            errors.append("設定ファイルに必須セクション 'interpolate_method' がありません")
            return errors
        else:
            if interpolate_method_section not in ['linear', 'spline']:
                errors.append("interpolate_methodの値が不適切です。['linear', 'spline'] から選択してください")

        return errors

    def _validate_analysis_flags(self, config:Dict) -> List[str]:
        """分析フラグの検証"""
        errors = []

        analysis_section = config.get('analysis')
        if analysis_section is None:
            errors.append("設定ファイルに必須セクション 'analysis' がありません")
            return errors

        anova_skill_metrics_section = analysis_section.get('anova_skill_metrics')
        if anova_skill_metrics_section is None:
            errors.append("設定ファイルに必須セクション 'anova_skill_metrics' がありません")

        factorize_skill_metrics_section = analysis_section.get('factorize_skill_metrics')
        if factorize_skill_metrics_section is None:
            errors.append("設定ファイルに必須セクション 'factorize_skill_metrics' がありません")

        return errors

    def get_config(self) -> Optional[Dict] :
        """検証済み設定を取得"""
        if self.config is not None:
            return self.config.copy()
        else:
            return None

    def get_data_paths(self) -> Tuple[str, str]:
        """データパスを取得"""
        return (self.config['data']['raw_data_dir'],
                self.config['data']['output_dir'])


class TrajectoryDataLoader:
    """軌道データの読み込み・前処理を担当"""
    def __init__(self, config:Dict):
        """検証済み設定を受け取り"""
        self.config = config
        self.raw_data_dir = Path(config['data']['raw_data_dir'])
        self.raw_data_df = self._load_raw_data()
        self.preprocessed_data_df = self._preprocess_trajectories(self.raw_data_df)

    def _load_raw_data(self) -> Optional[pd.DataFrame]:
        """指定されたディレクトリ内の全てのCSVを読み込み、一つのDataFrameに結合する"""
        all_files = list(self.raw_data_dir.glob('*.csv'))

        if not all_files:
            print(f"エラー:ディレクトリ {str(self.raw_data_dir)} にcsvファイルがありません．")
            return None

        df_list = []
        for file_path in all_files:
            try:
                parts = file_path.stem.split('_')
                subject_id = parts[0]
                block_num = int(parts[1].replace('Block', ''))

                df = pd.read_csv(file_path)

                # カラム名を統一し、情報を付与
                df = df.rename(columns={'SubjectId': 'subject_id', 'CurrentTrial': 'trial_num', 'Block': 'block'})
                df['subject_id'] = subject_id
                df['block'] = block_num
                df_list.append(df)
            except Exception as e:
                print(f"ファイル {file_path} の読み込み中にエラー: {e}")

        if not df_list:
            return None

        concatenated_df = pd.concat(df_list, ignore_index=True)

        return concatenated_df[concatenated_df['TrialState'] == 'TRIAL_RUNNING'].copy()

    def _preprocess_trajectories(self, data: pd.DataFrame) -> pd.DataFrame:
        """軌道の前処理(補完、長さ調整)"""
        target_seq_len = self.config['pre_process']['target_seq_len']
        method = self.config['pre_process']['interpolate_method']

        processed_trajectories = [] #全トライアルを蓄積するリスト

        for subject_id, subject_df in data.groupby('subject_id'):
            for trial_num, trial_df in subject_df.groupby('trial_num'):
                try:
                    traj_positions = trial_df[['HandlePosX','HandlePosY']].values
                    traj_velocities = trial_df[['HandleVelX','HandleVelY']].values
                    traj_acceleration = trial_df[['HandleAccX','HandleAccY']].values

                    # 変換先の時間軸の作成
                    original_length = len(traj_positions)
                    original_time = np.linspace(0, 1, original_length)
                    target_time = np.linspace(0,1, target_seq_len)

                    # 最小データポイント検証
                    if original_length < 2:
                        print(f"データ不足: 被験者{subject_id}, トライアル{trial_num} ({original_length}点)")
                        continue

                    interp_pos_x = None
                    interp_pos_y = None
                    interp_vel_x = None
                    interp_vel_y = None
                    interp_acc_x = None
                    interp_acc_y = None

                    if method == 'linear':
                        # 線形補間関数の作成
                        interp_pos_x = interp1d(original_time, traj_positions[:, 0], kind ='linear')
                        interp_pos_y = interp1d(original_time, traj_positions[:, 1], kind ='linear')
                        interp_vel_x = interp1d(original_time, traj_velocities[:, 0], kind='linear')
                        interp_vel_y = interp1d(original_time, traj_velocities[:, 1], kind='linear')
                        interp_acc_x = interp1d(original_time, traj_acceleration[:, 0], kind='linear')
                        interp_acc_y = interp1d(original_time, traj_acceleration[:, 1], kind='linear')
                    elif method == 'spline':
                        # スプライン補間関数の作成
                        interp_pos_x = UnivariateSpline(original_time, traj_positions[:, 0], s=0)
                        interp_pos_y = UnivariateSpline(original_time, traj_positions[:, 1], s=0)
                        interp_vel_x = UnivariateSpline(original_time, traj_velocities[:, 0], s=0)
                        interp_vel_y = UnivariateSpline(original_time, traj_velocities[:, 1], s=0)
                        interp_acc_x = UnivariateSpline(original_time, traj_acceleration[:, 0], s=0)
                        interp_acc_y = UnivariateSpline(original_time, traj_acceleration[:, 1], s=0)
                    else:
                        print(f"未対応の補完方法: {method}")
                        continue

                    # 新しい時間軸で補完
                    resampled_pos_x = interp_pos_x(target_time)
                    resampled_pos_y = interp_pos_y(target_time)
                    resampled_vel_x = interp_vel_x(target_time)
                    resampled_vel_y = interp_vel_y(target_time)
                    resampled_acc_x = interp_acc_x(target_time)
                    resampled_acc_y = interp_acc_y(target_time)

                    # リサンプル後のデータをDataFrameに変換
                    trajectory_df = pd.DataFrame({
                        'subject_id': subject_id,
                        'trial_num': trial_num,
                        'block': trial_df['block'].iloc[0],
                        'time_step':range(target_seq_len),
                        'HandlePosX': resampled_pos_x,
                        'HandlePosY': resampled_pos_y,
                        'HandleVelX': resampled_vel_x,
                        'HandleVelY': resampled_vel_y,
                        'HandleAccX': resampled_acc_x,
                        'HandleAccY': resampled_acc_y,
                        'TargetEndPosX': trial_df['TargetEndPosX'].iloc[0],
                        'TargetEndPosY': trial_df['TargetEndPosY'].iloc[0],
                        'original_length': original_length
                    })

                    processed_trajectories.append(trajectory_df)

                except ValueError as e:
                    print(f"補完エラー: 被験者{subject_id}, トライアル{trial_num}: {e}")
                    continue

        # 全トライアルを結合
        if processed_trajectories:
            return pd.concat(processed_trajectories, ignore_index=True)
        else:
            return pd.DataFrame()   #空のDataFrameを返す

    def get_subjects(self) -> List[str]:
        """被験者IDリストを取得"""
        return self.df['subject_id'].unique().to_list()

    def get_raw_data(self) -> Optional[pd.DataFrame]:
        if self.raw_data_df:
            return self.raw_data_df
        else:
            return None

    def get_preprocessed_data(self) -> Optional[pd.DataFrame]:
        if self.preprocessed_data_df:
            return self.preprocessed_data_df
        else:
            return None


class SkillMetricCalculator:
    """各種スキル指標の計算を担当"""
    def __init__(self, config: Dict):
        """設定を受け取り初期化"""
        self.config = config
        self.target_seq_len = config['pre_process']['target_seq_len']

    
    def calculate_skill_metrics(self, preprocessed_data: pd.DataFrame) -> pd.DataFrame:
        """前処理済みデータから各種スキル指標を計算して標準化"""
        skill_metrics = []
        
        for (subject_id, trial_num), trial_group in preprocessed_data.groupby(['subject_id', 'trial_num']):
            try:
                # 各トライアルから軌道曲率、速度滑らかさ、加速度滑らかさ、ジャーク、制御安定性、時間的一貫性、動作時間、終点誤差を計算
                curvature = self._calculate_curvature(trial_group)
                velocity_smoothness = self._calculate_velocity_smoothness(trial_group)
                acceleration_smoothness = self._calculate_acceleration_smoothness(trial_group)
                jerk_score = self._calculate_jerk(trial_group)
                control_stability = self._calculate_control_stability(trial_group)
                temporal_consistency = self._calculate_temporal_consistency(trial_group)
                trial_time = self._calculate_trial_time(trial_group)
                endpoint_error = self._calculate_endpoint_error(trial_group)

                # 結果をまとめる
                metrics = {
                    'subject_id': subject_id,
                    'trial_num': trial_num,
                    'block': trial_group['block'].iloc[0],
                    'curvature': curvature,
                    'velocity_smoothness': velocity_smoothness,
                    'acceleration_smoothness': acceleration_smoothness,
                    'jerk_score': jerk_score,
                    'control_stability': control_stability,
                    'temporal_consistency': temporal_consistency,
                    'trial_time': trial_time,
                    'endpoint_error': endpoint_error,
                }
                skill_metrics.append(metrics)
                
            except Exception as e:
                print(f"スキル指標計算エラー: 被験者{subject_id}, トライアル{trial_num}: {e}")
                continue
        
        skill_metrics_df =  pd.DataFrame(skill_metrics)

        # Z-score標準化の実行
        if self.config.get('analysis', {}).get('standardize_metrics', True):
            skill_metrics_df = self._standardize_metrics(skill_metrics_df)

        return skill_metrics_df

    def _standardize_metrics(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """スキル指標を標準化する"""
        standardized_df = metrics_df.copy()

        # 通知列のみを標準化対象とする
        metric_directions = {'curvature': False,
                              'velocity_smoothness': True,
                              'acceleration_smoothness': True,
                              'jerk_score': False,
                              'control_stability': True,
                              'temporal_consistency': True,
                              'trial_time': False,
                              'endpoint_error': False,
                              }

        for col in metric_directions.keys():
            if col in standardized_df.columns:
                values = standardized_df[col].dropna()
                if len(values) > 1:
                    mean_val = values.mean()
                    std_val = values.std()
                    if std_val > 0:
                        z_scores = (standardized_df[col] - mean_val) / std_val
                        # 低い値が良い指標は符号を反転
                        if not metric_directions[col]:
                            z_scores = -z_scores
                        standardized_df[col] = z_scores

        return standardized_df


    def _calculate_curvature(self, trial_data: pd.DataFrame):
        """軌道の曲率を計算"""
        positions = trial_data[['HandlePosX','HandlePosY']].values

        if len(positions) < 3:
            return np.nan

        curvatures = []
        for i in range(1,len(positions) - 1):
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

    def _calculate_velocity_smoothness(self, trial_data: pd.DataFrame):
        """速度の滑らかさを計算"""
        velocities = trial_data[['HandleVelX','HandleVelY']].values

        if len(velocities) < 2:
            return np.nan

        vel_magnitude = np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)
        velocity_changes = np.abs(np.diff(vel_magnitude))

        # 正規化された滑らかさ指標
        smoothness = 1.0 / (1.0 + np.std(velocity_changes))
        return smoothness

    def _calculate_acceleration_smoothness(self, trial_data: pd.DataFrame):
        """加速度の滑らかさ"""

        if len(trial_data) < 3:
            return np.nan

        acc_x = trial_data['HandleAccX'].values
        acc_y = trial_data['HandleAccY'].values
        acc_magnitude = np.sqrt(acc_x ** 2 + acc_y ** 2)

        acc_changes = np.abs(np.diff(acc_magnitude))
        return 1.0 / (1.0 + np.std(acc_changes))

    def _calculate_control_stability(self, trial_data: pd.DataFrame):
        """制御安定性指標"""
        if len(trial_data) < 5:
            return np.nan

        acc_x = trial_data['HandleAccX'].values
        acc_y = trial_data['HandleAccY'].values

        # 加速度の標準偏差（制御の安定性の逆指標）
        acc_std = np.std(np.sqrt(acc_x ** 2 + acc_y ** 2))
        stability = 1.0 / (1.0 + acc_std)

        return stability

    def _calculate_temporal_consistency(self, trial_data: pd.DataFrame):
        """時間的一貫性"""
        if len(trial_data) < 10:
            return np.nan

        timestamps = trial_data['Timestamp'].values
        time_intervals = np.diff(timestamps)

        # 時間間隔の一貫性
        consistency = 1.0 / (1.0 + np.std(time_intervals) / np.mean(time_intervals))
        return consistency

    def _calculate_trial_time(self, trial_data: pd.DataFrame) -> float:
        """動作時間の計算 - 軌道の長さから推定"""
        time_stamps = trial_data['Timestamp'].values

        return time_stamps[-1] - time_stamps[0]
    
    def _calculate_endpoint_error(self, trial_data: pd.DataFrame) -> float:
        """終点誤差の計算 - 目標位置からの距離"""
        # 最後の位置を終点として使用
        final_x = trial_data['HandlePosX'].iloc[-1]
        final_y = trial_data['HandlePosY'].iloc[-1]

        target_x = trial_data['TargetEndPosX'].iloc[-1]
        target_y = trial_data['TargetEndPosY'].iloc[-1]
        
        endpoint_error = np.sqrt((final_x - target_x)**2 + (final_y - target_y)**2)
        return endpoint_error
    
    def _calculate_jerk(self, trial_data: pd.DataFrame) -> float:
        """ジャーク（加速度の変化率）の計算"""
        acc_x = trial_data['HandleAccX'].values
        acc_y = trial_data['HandleAccY'].values
        
        # 加速度の微分でジャークを計算
        jerk_x = np.diff(acc_x)
        jerk_y = np.diff(acc_y)
        
        # ジャークの大きさを計算し、平均を取る
        jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2)
        jerk_score = np.mean(jerk_magnitude)
        
        return jerk_score

class SkillAnalyzer:
    """ANOVA,因子解析などの統計解析を担当"""
    pass

class DatasetBuilder:
    """データセット生成クラス"""
    # todo コンフィグの nameごとにユニークな名前のディレクトリを作成する nameが同じ場合でも必ずユニークにする
    pass

if __name__ == '__main__':
    config_dir = 'data_preprocess_default_config.yaml'

    try:
        # コンフィグの読み込み
        config_loader = DataPreprocessConfigLoader(config_dir)
        config = config_loader.get_config()

        # 生データの読みこみ
        trajectory_loader = TrajectoryDataLoader(config)

        # スキル指標の計算
        skill_metrics_calculator = SkillMetricCalculator(config)
        skill_metrics_df = skill_metrics_calculator.calculate_skill_metrics(trajectory_loader.preprocessed_data_df)

        # 因子分析

    except Exception as e:
        print(f"エラーが発生しました: {e}")

