import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import  Dict, List, Tuple, Optional
from pathlib import  Path
from scipy.interpolate import interp1d, UnivariateSpline
from scipy import stats
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

import yaml
import datetime


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

class OutputManager:
    """ファイルの出力を管理する"""
    def __init__(self, validated_config: Dict):
        self.config = validated_config
        self.base_output_dir = self.config['data']['output_dir']
        self.process_name = self.config['information']['name']

        self._skill_analyzer_output_dir, self._dataset_builder_output_path = self._make_unique_output_paths()


    def _make_unique_output_paths(self) -> tuple[Path, Path]:
        """ユニークな出力ディレクトリパスを作成"""
        base_path = Path(self.base_output_dir)
        validated_process_name = self.process_name.replace(' ', '_')
        time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        dir_name = f"{validated_process_name}_{time_stamp}"

        skill_analyzer_result_dir = base_path / dir_name / 'skill_analyze_result'
        data_set_builder_output_dir = base_path / dir_name / 'dataset'

        return skill_analyzer_result_dir, data_set_builder_output_dir

    @property
    def skill_analyzer_output_dir_path(self):
        return self._skill_analyzer_output_dir

    @property
    def dataset_builder_output_dir_path(self):
        return self._dataset_builder_output_path


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
    def __init__(self, config: Dict, output_manager: OutputManager):
        """設定を受け取り初期化"""
        self.config = config
        self.output_manager = output_manager
        self.analysis_config = config.get('analysis', {})
    
    def analyze_skill_metrics(self, skill_metrics_df: pd.DataFrame) -> Dict:
        """スキル指標の統計解析を実行"""
        results = {}
        
        if self.analysis_config.get('anova_skill_metrics', False):
            print("ANOVA解析を実行中...")
            results['anova'] = self._perform_anova_analysis(skill_metrics_df)

            # ANOVA解析結果の可視化
            self._save_anova_plots(skill_metrics_df, results['anova'])

        if self.analysis_config.get('factorize_skill_metrics', False):
            print("因子分析を実行中...")
            results['factor_analysis'] = self._perform_factor_analysis(skill_metrics_df)

            # 因子分析結果の可視化
            self._save_factor_analysis_plots(skill_metrics_df, results['factor_analysis'])

        return results
    
    def _perform_anova_analysis(self, skill_metrics_df: pd.DataFrame) -> Dict:
        """ANOVA解析の実行"""
        anova_results = {}
        
        # 数値スキル指標列を取得
        skill_columns = ['curvature', 'velocity_smoothness', 'acceleration_smoothness',
                        'jerk_score', 'control_stability', 'temporal_consistency', 
                        'trial_time', 'endpoint_error']
        
        for skill_metric in skill_columns:
            if skill_metric in skill_metrics_df.columns:
                try:
                    # 被験者間のスキル指標差異を一元配置分散分析で検定
                    anova_result = self._one_way_anova(skill_metrics_df, skill_metric)
                    anova_results[skill_metric] = anova_result
                    
                except Exception as e:
                    print(f"ANOVA解析エラー ({skill_metric}): {e}")
                    anova_results[skill_metric] = {'error': str(e)}
        
        return anova_results
    
    def _one_way_anova(self, data: pd.DataFrame, metric_name: str) -> Dict:
        """一元配置分散分析の実行"""
        # 被験者ごとのデータをグループ化
        groups = []
        subject_ids = []
        
        for subject_id, subject_data in data.groupby('subject_id'):
            metric_values = subject_data[metric_name].dropna()
            if len(metric_values) > 0:
                groups.append(metric_values.values)
                subject_ids.append(subject_id)
        
        if len(groups) < 2:
            return {'error': '分析に十分なグループ数がありません'}
        
        # ANOVA検定の実行
        f_statistic, p_value = stats.f_oneway(*groups)
        
        # 効果量（eta squared）の計算
        total_variance = np.var(np.concatenate(groups))
        within_variance = np.mean([np.var(group) for group in groups])
        eta_squared = (total_variance - within_variance) / total_variance
        
        return {
            'f_statistic': f_statistic,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'significant': p_value < 0.05,
            'groups_count': len(groups),
            'subject_ids': subject_ids
        }
    
    def _perform_factor_analysis(self, skill_metrics_df: pd.DataFrame) -> Dict:
        """因子分析の実行"""
        try:
            # 数値スキル指標のみを抽出
            skill_columns = ['curvature', 'velocity_smoothness', 'acceleration_smoothness',
                            'jerk_score', 'control_stability', 'temporal_consistency', 
                            'trial_time', 'endpoint_error']
            
            # 欠損値を除去してデータを準備
            analysis_data = skill_metrics_df[skill_columns].dropna()
            
            if len(analysis_data) < 10:
                return {'error': '因子分析に十分なサンプル数がありません'}
            
            # 標準化（念のため再度実行）
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(analysis_data)
            
            # 因子数の決定（固有値>1の基準）
            correlation_matrix = np.corrcoef(scaled_data.T)
            eigenvalues = np.linalg.eigvals(correlation_matrix)
            n_factors = max(1, np.sum(eigenvalues > 1))
            
            # 因子分析の実行
            fa = FactorAnalysis(n_components=n_factors, random_state=42)
            fa.fit(scaled_data)
            
            # 因子負荷量の計算
            factor_loadings = fa.components_.T
            
            # 因子得点の計算
            factor_scores = fa.transform(scaled_data)
            
            return {
                'n_factors': n_factors,
                'factor_loadings': factor_loadings,
                'factor_scores': factor_scores,
                'explained_variance': eigenvalues[:n_factors],
                'skill_columns': skill_columns,
                'sample_size': len(analysis_data)
            }
            
        except Exception as e:
            return {'error': f'因子分析実行エラー: {str(e)}'}

    def _save_anova_plots(self, skill_metrics_df: pd.DataFrame, anova_results: Dict):
        output_dir = self.output_manager.skill_analyzer_output_dir_path
        output_dir.mkdir(parents=True, exist_ok=True)

        # 有意差のある指標と無い指標で分けて表示
        significant_metrics = []
        non_significant_metrics = []

        metrics = []
        eta_values = []
        p_values = []

        for metric, result  in anova_results.items():
            if 'error' not in result:
                if result.get('significant', False):
                    significant_metrics.append(metric)
                else:
                    non_significant_metrics.append(metric)

                metrics.append(metric)
                eta_values.append(result['eta_squared'])
                p_values.append(result['p_value'])

        # 有意差ありの指標の箱ひげ図をプロット
        if significant_metrics:
            self._create_boxplot_grid(skill_metrics_df, significant_metrics, anova_results, output_dir / 'significant_results.png', "有意差のあるスキル指標(被験者間比較)")

        # 有意差なしの指標の箱ひげ図をプロット
        if non_significant_metrics:
            self._create_boxplot_grid(skill_metrics_df, non_significant_metrics, anova_results, output_dir / 'non_significant_results.png',
                                      "有意差のないスキル指標(被験者間比較)")

        # 各指標の効果量、p値をプロット
        if metrics:
            self._create_bar_plot(metrics, eta_values, p_values, output_dir/ 'eta_result.png', "効果量の比較(全評価指標)")


    def _create_boxplot_grid(self, data: pd.DataFrame, metrics: List[str], anova_results: Dict, save_path: Path, title: str):
        """複数指標の箱ひげ図をグリッド表示"""
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize = (5 * n_cols, 4 * n_rows))

        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten() if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # 箱ひげ図の作成
            sns.boxplot(data=data, x='subject_id', y=metric, ax=ax)
            ax.set_title(f'{metric}\n(p={anova_results[metric]["p_value"]:.3f})',
                         fontsize=12)
            ax.set_xlabel('被験者ID')
            ax.set_ylabel('スキル指標値')
            ax.tick_params(axis='x', rotation=45)

            # 余った軸を非表示
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(title, fontsize=16, y=1.02)
        fig.tight_layout()

        try:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ ANOVA箱ひげ図を保存: {save_path}")
        except Exception as e:
            print(f"❌ グラフ保存エラー: {e}")

        plt.close(fig)

    def _create_bar_plot(self, metrics: List, eta_values: List, p_values:List, save_path: Path, title: str):
        """効果量とp値の棒グラフをプロット"""
        data = {
            'metrics': metrics,
            'eta': eta_values,
            'p_value': p_values
        }
        df = pd.DataFrame(data)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (12, 5))

        # 効果量のプロット
        sns.barplot(data=df, x='metrics', y='eta', ax=axes[0])
        axes[0].set_title('Effectiveness')

        # p値のプロット
        sns.barplot(data=df, x='metrics', y='p_value', ax=axes[1])
        axes[1].set_title('p-value')
        axes[1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
        axes[1].legend()

        fig.suptitle(title, fontsize=16, y=1.02)
        fig.tight_layout()

        try:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ ANOVA結果効果量グラフを保存: {save_path}")
        except Exception as e:
            print(f"❌ グラフ保存エラー: {e}")

        plt.close(fig)

    def _save_factor_analysis_plots(self, skill_metrics_df: pd.DataFrame, factor_analysis_results: Dict):
        """因子分析結果の可視化"""
        output_dir = self.output_manager.skill_analyzer_output_dir_path
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 因子負荷量ヒートマップ
        self._create_factor_loading_heatmap(factor_analysis_results, output_dir / 'factor_loadings_heatmap.png')

        # 2. 因子特典散布図(最初の2因子)
        if factor_analysis_results['n_factors'] >= 2:
            self._create_factor_score_scatter(skill_metrics_df, factor_analysis_results,
                                              output_dir / 'factor_scores_scatter.png')
        # 3. 因子寄与率棒グラフ
        self._create_factor_variance_plot(factor_analysis_results, output_dir/ 'factor_variance_explained.png')

    def _create_factor_loading_heatmap(self, factor_analysis_results: Dict, save_path: Path):
        """因子負荷量のヒートマップ作成"""
        loadings = factor_analysis_results['factor_loadings']
        skill_columns = factor_analysis_results['skill_columns']
        n_factors = factor_analysis_results['n_factors']

        # DataFrameに変換
        loading_df = pd.DataFrame(
            loadings,
            index=skill_columns,
            columns=[f'Factor {i+1}' for i in range(n_factors)]
        )

        fig, axes = plt.subplots(figsize = (8, 6))
        sns.heatmap(loading_df, annot=True, cmap='RdBu_r', center=0,
                    fmt ='.2f', ax=axes, cbar_kws={'label': 'Factor Loading'})
        axes.set_title('Factor Loading Matrix', fontsise=14)
        axes.set_xlabel('Factor')
        axes.set_ylabel('Skill Metrics')

        try:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 因子負荷量ヒートマップを保存: {save_path}")
        except Exception as e:
            print(f"❌ ヒートマップ保存エラー: {e}")

        plt.close(fig)


    def _create_factor_score_scatter(self, skill_metrics_df: pd.DataFrame, factor_analysis_results: Dict, save_path: Path):
        """因子特典の散布図作成(第一因子 vs 第二因子)"""
        factor_scores = factor_analysis_results['factor_scores']

        # 被験者情報を追加
        analysis_data = skill_metrics_df[factor_analysis_results['skill_columns']].dropna()
        subject_info = skill_metrics_df.loc[analysis_data.index, ['subject_id', 'block']]

        fig, axes = plt.subplots(figsize=(10, 8))

        # 被験者ごとに色分けして散布図を作成
        unique_subjects = subject_info['subject_id'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_subjects)))

        for i, subject in enumerate(unique_subjects):
            mask = subject_info['subject_id'] == subject
            if np.sum(mask) > 0:
                subject_scores = factor_scores[mask]
                axes.scatter(subject_scores[:, 0], subject_scores[:, 1],
                           c=[colors[i]], label=f'Subject {subject}', alpha=0.7, s=50)

        axes.set_xlabel('Factor 1 score')
        axes.set_ylabel('Factor 2 score')
        axes.set_title('Factor Scatter Plot（Per subjects）', fontsize=14)
        axes.grid(True, alpha=0.3)
        axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 軸の交点に線を追加
        axes.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes.axvline(x=0, color='k', linestyle='-', alpha=0.3)

        try:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 因子得点散布図を保存: {save_path}")
        except Exception as e:
            print(f"❌ 散布図保存エラー: {e}")

        plt.close(fig)

    def _create_factor_variance_plot(self, factor_analysis_results: Dict, save_path: Path):
        """因子の寄与率（説明分散）棒グラフ"""
        eigenvalues = factor_analysis_results['explained_variance']
        n_factors = factor_analysis_results['n_factors']

        # 寄与率の計算
        total_variance = np.sum(eigenvalues)
        contribution_ratios = eigenvalues / total_variance * 100
        cumulative_ratios = np.cumsum(contribution_ratios)

        factor_names = [f'Factor {i + 1}' for i in range(n_factors)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 寄与率棒グラフ
        bars = ax1.bar(factor_names, contribution_ratios, alpha=0.7, color='steelblue')
        ax1.set_title('Explained Variance', fontsize=12)
        ax1.set_ylabel('Explained Variance (%)')
        ax1.set_xlabel('Factor')

        # 値をバーの上に表示
        for bar, ratio in zip(bars, contribution_ratios):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{ratio:.1f}%', ha='center', va='bottom')

        # 累積寄与率折れ線グラフ
        ax2.plot(factor_names, cumulative_ratios, marker='o', color='red', linewidth=2)
        ax2.set_title(' Cumulative Explained Variance ', fontsize=12)
        ax2.set_ylabel(' Cumulative Explained Variance  (%)')
        ax2.set_xlabel('Factor')
        ax2.grid(True, alpha=0.3)

        # 値を点の上に表示
        for i, ratio in enumerate(cumulative_ratios):
            ax2.text(i, ratio + 2, f'{ratio:.1f}%', ha='center', va='bottom')

        fig.suptitle('Factor Analysis：Explained Variance', fontsize=14, y=1.02)
        fig.tight_layout()

        try:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 因子寄与率グラフを保存: {save_path}")
        except Exception as e:
            print(f"❌ 寄与率グラフ保存エラー: {e}")

        plt.close(fig)



class DatasetBuilder:
    """データセット生成クラス"""
    # todo コンフィグの nameごとにユニークな名前のディレクトリを作成する nameが同じ場合でも必ずユニークにする
    pass





if __name__ == '__main__':
    config_dir = 'data_preprocess_default_config.yaml'

    try:
        # コンフィグの読み込み
        config_loader = DataPreprocessConfigLoader(config_dir)
        validated_config = config_loader.get_config()

        # 出力ディレクトリマネージャー
        output_manager = OutputManager(validated_config)

        # 生データの読みこみ
        trajectory_loader = TrajectoryDataLoader(validated_config)

        # スキル指標の計算
        skill_metrics_calculator = SkillMetricCalculator(validated_config)
        skill_metrics_df = skill_metrics_calculator.calculate_skill_metrics(trajectory_loader.preprocessed_data_df)

        # ANOVA and 因子分析
        skill_analyzer = SkillAnalyzer(validated_config, output_manager)
        skill_analyzer.analyze_skill_metrics(skill_metrics_df)

    except Exception as e:
        print(f"エラーが発生しました: {e}")

