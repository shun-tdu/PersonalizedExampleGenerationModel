import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import  Dict, List, Tuple, Optional
from pathlib import  Path
from scipy.interpolate import interp1d, UnivariateSpline
from scipy import stats
# from sklearn.decomposition import FactorAnalysis
try:
    from factor_analyzer import FactorAnalyzer
    FACTOR_ANALYZER_AVAILABLE = True
except ImportError:
    FACTOR_ANALYZER_AVAILABLE = False
    FactorAnalyzer = None

from sklearn.preprocessing import StandardScaler

import yaml
import datetime
import joblib

# CLAUDE_ADDED: Academic paper formatting settings
# Set Times New Roman font for academic papers
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Liberation Serif']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 15
# Set tick direction to inward for academic papers
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def anonymize_subjects(subject_ids):
    """Convert subject names to anonymous labels (Subject1, Subject2, etc.)"""
    unique_subjects = sorted(list(set(subject_ids)))
    return {subj: f"Subject{i+1}" for i, subj in enumerate(unique_subjects)}

def save_academic_figure(fig, save_path):
    """Save figure in both PDF and PNG formats for academic use"""
    save_path = Path(save_path)
    pdf_path = save_path.with_suffix('.pdf')
    png_path = save_path.with_suffix('.png')

    # Save PDF (vector format, scalable, preferred for academic papers)
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300,
               facecolor='white', edgecolor='none')

    # Save PNG (raster format, 300 DPI for print quality)
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300,
               facecolor='white', edgecolor='none')

    print(f"✅ Saved: {pdf_path.name} and {png_path.name}")


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

        self._skill_analyzer_output_dir, self._dataset_builder_output_path, self._skill_score_calculator_output_path = self._make_unique_output_paths()


    def _make_unique_output_paths(self) -> tuple[Path, Path, Path]:
        """ユニークな出力ディレクトリパスを作成"""
        base_path = Path(self.base_output_dir)
        validated_process_name = self.process_name.replace(' ', '_')
        time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        dir_name = f"{validated_process_name}_{time_stamp}"

        skill_analyzer_result_dir = base_path / dir_name / 'skill_analyze_result'
        data_set_builder_output_dir = base_path / dir_name / 'dataset'
        skill_score_calculator_output_dir =base_path/ dir_name /'skill_score_result'

        return skill_analyzer_result_dir, data_set_builder_output_dir, skill_score_calculator_output_dir

    @property
    def skill_analyzer_output_dir_path(self):
        return self._skill_analyzer_output_dir

    @property
    def dataset_builder_output_dir_path(self):
        return self._dataset_builder_output_path

    @property
    def skill_score_calculator_output_dir_path(self):
        return self._skill_score_calculator_output_path


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

                # CLAUDE_ADDED: カラム名を統一し、情報を付与 (デバッグログ追加)
                print(f"Processing file: {file_path.name}, original columns: {df.columns.tolist()}")
                
                # CurrentTrialが存在する場合のみリネーム
                rename_dict = {}
                if 'SubjectId' in df.columns:
                    rename_dict['SubjectId'] = 'subject_id'
                if 'CurrentTrial' in df.columns:
                    rename_dict['CurrentTrial'] = 'trial_num'
                
                if rename_dict:
                    df = df.rename(columns=rename_dict)
                    print(f"Renamed columns: {rename_dict}")
                
                df['subject_id'] = subject_id
                df['block'] = block_num
                
                # CLAUDE_ADDED: データ処理後のカラムと基本統計を確認
                print(f"Final columns: {df.columns.tolist()}")
                if 'trial_num' in df.columns:
                    print(f"trial_num unique values: {df['trial_num'].unique()}")
                print(f"Data shape for {subject_id} Block{block_num}: {df.shape}")
                
                df_list.append(df)
            except Exception as e:
                print(f"ファイル {file_path} の読み込み中にエラー: {e}")

        if not df_list:
            return None

        concatenated_df = pd.concat(df_list, ignore_index=True)
        
        # CLAUDE_ADDED: 結合後の確認
        print(f"Concatenated data shape: {concatenated_df.shape}")
        trial_running_df = concatenated_df[concatenated_df['TrialState'] == 'TRIAL_RUNNING'].copy()
        print(f"TRIAL_RUNNING data shape: {trial_running_df.shape}")
        
        return trial_running_df

    def _preprocess_trajectories(self, data: pd.DataFrame) -> pd.DataFrame:
        """軌道の前処理(補完、長さ調整)"""
        target_seq_len = self.config['pre_process']['target_seq_len']
        method = self.config['pre_process']['interpolate_method']

        print(f"CLAUDE_DEBUG: Starting preprocessing with target_seq_len={target_seq_len}")
        
        processed_trajectories = [] #全トライアルを蓄積するリスト
        
        # CLAUDE_ADDED: 前処理前の統計確認
        print(f"CLAUDE_DEBUG: Input data shape: {data.shape}")
        print(f"CLAUDE_DEBUG: Available columns: {data.columns.tolist()}")
        
        for subject_id, subject_df in data.groupby('subject_id'):
            print(f"CLAUDE_DEBUG: Processing subject {subject_id}, trials: {len(subject_df.groupby(['trial_num', 'block']))}")
            
            for (trial_num, block_num), trial_df in subject_df.groupby(['trial_num', 'block']): # CLAUDE_ADDED: blockもグループ化に含める
                try:
                    traj_positions = trial_df[['HandlePosX','HandlePosY']].values
                    traj_velocities = trial_df[['HandleVelX','HandleVelY']].values
                    traj_acceleration = trial_df[['HandleAccX','HandleAccY']].values

                    # 変換先の時間軸の作成
                    original_length = len(traj_positions)
                    original_time = np.linspace(0, 1, original_length)
                    target_time = np.linspace(0,1, target_seq_len)
                    
                    # CLAUDE_ADDED: 詳細ログ
                    print(f"CLAUDE_DEBUG: Subject {subject_id}, Trial {trial_num}, Block {block_num}: {original_length} -> {target_seq_len}")

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

                    # 元のTimestamp情報を取得して新しい時間軸を作成
                    original_start_time = trial_df['Timestamp'].iloc[0]
                    original_end_time = trial_df['Timestamp'].iloc[-1]
                    resampled_timestamps = np.linspace(original_start_time, original_end_time, target_seq_len)
                    
                    # リサンプル後のデータをDataFrameに変換
                    trajectory_df = pd.DataFrame({
                        'subject_id': subject_id,
                        'trial_num': trial_num,
                        'block': block_num, # CLAUDE_ADDED: グループ化から取得した正しいblock番号を使用
                        'time_step':range(target_seq_len),
                        'Timestamp': resampled_timestamps,
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
        if self.raw_data_df is not None:
            return self.raw_data_df
        else:
            return None

    def get_preprocessed_data(self, block: int) -> Optional[pd.DataFrame]:
        """指定したブロックの前処理済みDataFrameを返す"""
        if self.preprocessed_data_df is not None:
            if block== 0:
                filtered_df = self.preprocessed_data_df
            else:
                filtered_df = self.preprocessed_data_df[self.preprocessed_data_df['block'] == block]

            return filtered_df
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
        # 入力データの検証
        if preprocessed_data is None or len(preprocessed_data) == 0:
            print("⚠️ スキル指標計算: 前処理データが空です")
            return pd.DataFrame()  # 空のDataFrameを返す
        
        skill_metrics = []
        
        for (subject_id, trial_num, block), trial_group in preprocessed_data.groupby(['subject_id', 'trial_num', 'block']):
            try:
                # 各トライアルから軌道曲率、速度滑らかさ、加速度滑らかさ、ジャーク、制御安定性、時間的一貫性、動作時間、終点誤差を計算
                curvature = self.calculate_curvature(trial_group)
                velocity_smoothness = self.calculate_velocity_smoothness(trial_group)
                acceleration_smoothness = self.calculate_acceleration_smoothness(trial_group)
                jerk_score = self.calculate_jerk(trial_group)
                control_stability = self.calculate_control_stability(trial_group)
                temporal_consistency = self.calculate_temporal_consistency(trial_group)
                trial_time = self.calculate_trial_time(trial_group)
                endpoint_error = self.calculate_endpoint_error(trial_group)

                # 結果をまとめる
                metrics = {
                    'subject_id': subject_id,
                    'trial_num': trial_num,
                    'block': block,
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
        # if self.config.get('analysis', {}).get('standardize_metrics', True):
        #     skill_metrics_df = self._standardize_metrics(skill_metrics_df)

        return skill_metrics_df

    @staticmethod
    def standardize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
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

    @staticmethod
    def calculate_curvature(trial_data: pd.DataFrame):
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

    @staticmethod
    def calculate_velocity_smoothness(trial_data: pd.DataFrame):
        """速度の滑らかさを計算"""
        velocities = trial_data[['HandleVelX','HandleVelY']].values

        if len(velocities) < 2:
            return np.nan

        vel_magnitude = np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)
        velocity_changes = np.abs(np.diff(vel_magnitude))

        # 正規化された滑らかさ指標
        smoothness = 1.0 / (1.0 + np.std(velocity_changes))
        return smoothness

    @staticmethod
    def calculate_acceleration_smoothness(trial_data: pd.DataFrame):
        """加速度の滑らかさ"""

        if len(trial_data) < 3:
            return np.nan

        acc_x = trial_data['HandleAccX'].values
        acc_y = trial_data['HandleAccY'].values
        acc_magnitude = np.sqrt(acc_x ** 2 + acc_y ** 2)

        acc_changes = np.abs(np.diff(acc_magnitude))
        return 1.0 / (1.0 + np.std(acc_changes))

    @staticmethod
    def calculate_control_stability(trial_data: pd.DataFrame):
        """制御安定性指標"""
        if len(trial_data) < 5:
            return np.nan

        acc_x = trial_data['HandleAccX'].values
        acc_y = trial_data['HandleAccY'].values

        # 加速度の標準偏差（制御の安定性の逆指標）
        acc_std = np.std(np.sqrt(acc_x ** 2 + acc_y ** 2))
        stability = 1.0 / (1.0 + acc_std)

        return stability

    @staticmethod
    def calculate_temporal_consistency(trial_data: pd.DataFrame):
        """時間的一貫性"""
        if len(trial_data) < 10:
            return np.nan

        timestamps = trial_data['Timestamp'].values
        time_intervals = np.diff(timestamps)

        # 時間間隔の一貫性
        consistency = 1.0 / (1.0 + np.std(time_intervals) / np.mean(time_intervals))
        return consistency

    @staticmethod
    def calculate_trial_time(trial_data: pd.DataFrame) -> float:
        """動作時間の計算 - 軌道の長さから推定"""
        time_stamps = trial_data['Timestamp'].values

        return time_stamps[-1] - time_stamps[0]

    @staticmethod
    def calculate_endpoint_error(trial_data: pd.DataFrame) -> float:
        """終点誤差の計算 - 目標位置からの距離"""
        # 最後の位置を終点として使用
        final_x = trial_data['HandlePosX'].iloc[-1]
        final_y = trial_data['HandlePosY'].iloc[-1]

        target_x = trial_data['TargetEndPosX'].iloc[-1]
        target_y = trial_data['TargetEndPosY'].iloc[-1]
        
        endpoint_error = np.sqrt((final_x - target_x)**2 + (final_y - target_y)**2)
        return endpoint_error

    @staticmethod
    def calculate_jerk(trial_data: pd.DataFrame) -> float:
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

        # スキルスコア計算用のスケーラと因子分析オブジェクト
        self.used_standard_scaler = None
        self.used_factor_analysis = None
    
    def analyze_skill_metrics(self, skill_metrics_df: pd.DataFrame, rotation: str = 'promax') -> Dict:
        """スキル指標の統計解析を実行"""
        results = {}
        
        if self.analysis_config.get('anova_skill_metrics', False):
            print("ANOVA解析を実行中...")
            results['anova'] = self._perform_anova_analysis(skill_metrics_df)

            # ANOVA解析結果の可視化
            self._save_anova_plots(skill_metrics_df, results['anova'])

        if self.analysis_config.get('factorize_skill_metrics', False):
            print("因子分析を実行中...")
            results['factor_analysis'] = self._perform_factor_analysis(skill_metrics_df, rotation)

            if 'error' not in results['factor_analysis']:
                self._save_factor_analysis_plots(skill_metrics_df, results['factor_analysis'])
            else:
                print(f"⚠️ 因子分析がスキップされました: {results['factor_analysis']['error']}")

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
    
    def _perform_factor_analysis(self, skill_metrics_df: pd.DataFrame, rotation: str) -> Dict:
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
            if not FACTOR_ANALYZER_AVAILABLE:
                return {'error': 'factor_analyzer ライブラリがインストールされていません'}
            
            fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
            fa.fit(scaled_data)
            
            # 因子負荷量の計算
            # factor_loadings = fa.components_.T
            factor_loadings = fa.loadings_
            
            # 因子得点の計算
            factor_scores = fa.transform(scaled_data)

            # スケーラと因子分析オブジェクトの保存
            self.used_standard_scaler = scaler
            self.used_factor_analysis = fa

            result = {
                'n_factors': n_factors,
                'factor_loadings': factor_loadings,
                'factor_scores': factor_scores,
                'explained_variance': eigenvalues[:n_factors],
                'skill_columns': skill_columns,
                'sample_size': len(analysis_data)
            }

            if rotation == 'promax':
                factor_correlation = fa.phi_
                result['factor_correlations'] = factor_correlation

            return result
            
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
            self._create_boxplot_grid(skill_metrics_df, significant_metrics, anova_results, output_dir / 'significant_results.png', "Significant Skill Metrics")

        # 有意差なしの指標の箱ひげ図をプロット
        if non_significant_metrics:
            self._create_boxplot_grid(skill_metrics_df, non_significant_metrics, anova_results, output_dir / 'non_significant_results.png',
                                      "Non Significant Skill Metrics")

        # 各指標の効果量、p値をプロット
        if metrics:
            self._create_bar_plot(metrics, eta_values, p_values, output_dir/ 'eta_result.png', "Comparison Effectiveness")

    def _create_boxplot_grid(self, data: pd.DataFrame, metrics: List[str], anova_results: Dict, save_path: Path, title: str):
        """複数指標の箱ひげ図をグリッド表示"""
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        # 8.5cm = 3.35 inches width, adjust height based on number of rows
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize = (3.35, 2.5 * n_rows))

        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten() if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # CLAUDE_ADDED: 被験者名を匿名化
            data_copy = data.copy()
            subject_mapping = anonymize_subjects(data_copy['subject_id'])
            data_copy['subject_id'] = data_copy['subject_id'].map(subject_mapping)

            # 箱ひげ図の作成
            sns.boxplot(data=data_copy, x='subject_id', y=metric, ax=ax)
            ax.set_title(f'{metric}\n(p={anova_results[metric]["p_value"]:.3f})',
                         fontsize=12)
            ax.set_xlabel('Subject ID')
            ax.set_ylabel('Skill Metrics')
            ax.tick_params(axis='x', rotation=45)

            # 余った軸を非表示
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(title, fontsize=16, y=1.02)
        fig.tight_layout()

        try:
            # CLAUDE_ADDED: Use academic figure saving function
            save_academic_figure(fig, save_path)
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

        # 8.5cm = 3.35 inches width for dual subplot
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (3.35, 2.0))

        # 効果量のプロット
        sns.barplot(data=df, x='metrics', y='eta', ax=axes[0])
        axes[0].set_title('Effectiveness')
        axes[0].tick_params(axis='x', rotation=45, labelsize=6)

        # p値のプロット
        sns.barplot(data=df, x='metrics', y='p_value', ax=axes[1])
        axes[1].set_title('p-value')
        axes[1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
        axes[1].legend()
        axes[1].tick_params(axis='x', rotation=45, labelsize=6)

        fig.suptitle(title, fontsize=16, y=1.02)
        fig.tight_layout()

        try:
            # CLAUDE_ADDED: Use academic figure saving function
            save_academic_figure(fig, save_path)
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

        # 4. promax 回転の場合は因子相関行列をプロット
        self._create_factor_correlation_heatmap(factor_analysis_results, output_dir / 'factor_correlation_heatmap.png')

    def analyze_block_wise_factor_analysis(self, skill_metrics_df: pd.DataFrame, rotation: str = 'promax'):
        """ブロック毎の因子分析を実行し、2x2プロットを作成"""
        print("ブロック毎の因子分析を実行中...")

        output_dir = self.output_manager.skill_analyzer_output_dir_path
        output_dir.mkdir(parents=True, exist_ok=True)

        # ブロック1-4の因子分析結果を格納
        block_results = {}

        for block in range(1, 5):
            print(f"ブロック{block}の因子分析を実行中...")
            block_data = skill_metrics_df[skill_metrics_df['block'] == block]

            if len(block_data) < 10:
                print(f"⚠️ ブロック{block}: データが不足しています (サンプル数: {len(block_data)})")
                continue

            # ブロック毎の因子分析を実行
            factor_result = self._perform_factor_analysis(block_data, rotation)

            if 'error' not in factor_result:
                # ブロック情報を追加
                factor_result['block'] = block
                factor_result['block_data'] = block_data
                block_results[block] = factor_result
                print(f"✅ ブロック{block}の因子分析完了")
            else:
                print(f"❌ ブロック{block}の因子分析エラー: {factor_result['error']}")

        if len(block_results) < 4:
            print(f"⚠️ 4ブロック全ての因子分析が完了していません。完了数: {len(block_results)}")

        # 2x2プロットを作成
        if block_results:
            self._create_block_wise_2x2_plots(block_results, output_dir / 'block_wise_factor_analysis.png')

        return block_results

    def _create_block_wise_2x2_plots(self, block_results: Dict, save_path: Path):
        """ブロック毎の因子分析結果を2x2プロットで表示"""
        print("ブロック毎2x2プロットを作成中...")

        # 全被験者の統一マッピングを作成
        all_subjects = set()
        for block_data in block_results.values():
            all_subjects.update(block_data['block_data']['subject_id'].unique())
        global_subject_mapping = anonymize_subjects(list(all_subjects))

        # 2x2サブプロット作成 (8.5cm = 3.35 inches width)
        fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.35))
        axes = axes.flatten()

        for idx, block in enumerate(range(1, 5)):
            ax = axes[idx]

            if block not in block_results:
                ax.text(0.5, 0.5, f'Block {block}\nNo Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=2)
                ax.set_title(f'Block {block}', fontsize=4)
                continue

            result = block_results[block]
            block_data = result['block_data']

            # 因子得点が2因子以上ある場合のみ散布図を描画
            if result['n_factors'] >= 2:
                factor_scores = result['factor_scores']

                # 被験者情報を取得
                analysis_data = block_data[result['skill_columns']].dropna()
                subject_info = block_data.loc[analysis_data.index, ['subject_id']]

                # 被験者ごとに色分けして散布図を作成
                unique_subjects = subject_info['subject_id'].unique()
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_subjects)))

                for i, subject in enumerate(unique_subjects):
                    mask = subject_info['subject_id'] == subject
                    if np.sum(mask) > 0:
                        subject_scores = factor_scores[mask]
                        ax.scatter(subject_scores[:, 0], subject_scores[:, 1],
                                 c=[colors[i]], label=global_subject_mapping[subject],
                                 alpha=0.7, s=1)

                # 軸の設定
                ax.set_xlabel('Factor 1 Score',fontsize=8)
                ax.set_ylabel('Factor 2 Score',fontsize=8)
                # ax.grid(True, alpha=0.3, linewidth=0.2)

                ax.tick_params(axis='both', labelsize=6)

                # 軸の交点に線を追加
                # ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                # ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

                # 左上の散布図（Block 1）にのみ凡例を追加
                if block == 1:
                    ax.legend(loc='upper left',
                             fontsize=4, frameon=True, fancybox=True,
                             shadow=True, framealpha=0.8, markerscale=2)
            else:
                ax.text(0.5, 0.5, f'Block {block}\nInsufficient Factors\n({result["n_factors"]} factor)',
                       ha='center', va='center', transform=ax.transAxes, fontsize=1)
                ax.set_title(f'Block {block}', fontsize=6)

        plt.tight_layout()

        # 学術論文用の保存
        save_academic_figure(fig, save_path)
        plt.close(fig)

        # 追加：ブロック毎の因子負荷量ヒートマップも2x2で作成
        self._create_block_wise_heatmaps(block_results, save_path.parent / 'block_wise_factor_loadings.png')

    def _create_block_wise_heatmaps(self, block_results: Dict, save_path: Path):
        """ブロック毎の因子負荷量ヒートマップを2x2で表示"""
        print("ブロック毎ヒートマップを作成中...")

        # 2x2サブプロット作成 (8.5cm = 3.35 inches width)
        fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.35))
        axes = axes.flatten()

        for idx, block in enumerate(range(1, 5)):
            ax = axes[idx]

            if block not in block_results:
                ax.text(0.5, 0.5, f'Block {block}\nNo Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=2)
                ax.set_title(f'Block {block}', fontsize=4)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            result = block_results[block]

            # 因子負荷量のヒートマップ作成
            loadings = result['factor_loadings']
            skill_columns = result['skill_columns']
            n_factors = result['n_factors']

            # DataFrameに変換
            loading_df = pd.DataFrame(
                loadings,
                index=skill_columns,
                columns=[f'F{i+1}' for i in range(n_factors)]
            )

            # ヒートマップを描画
            heatmap = sns.heatmap(loading_df, annot=True, cmap='RdBu_r', center=0,
                       fmt='.2f', ax=ax, cbar=True,
                       cbar_kws={'shrink': 0.8}, annot_kws={'fontsize': 4})

            # カラーバーのフォントサイズを調整
            if heatmap.collections:
                colorbar = heatmap.collections[0].colorbar
                if colorbar:
                    colorbar.ax.tick_params(labelsize=4)

            ax.set_title(f'Block {block}', fontsize=6)
            ax.set_xlabel('Factor', fontsize=6)
            ax.set_ylabel('Skill Metrics', fontsize=6)
            # 目盛り線を細くし、スキル指標名を斜めに回転
            ax.tick_params(axis='both', labelsize=4, width=0.5, length=2)
            ax.tick_params(axis='y', rotation=45)

        # plt.suptitle('Block-wise Factor Analysis: Factor Loading Matrix', fontsize=18, y=0.98)
        plt.tight_layout()

        # 学術論文用の保存
        save_academic_figure(fig, save_path)
        plt.close(fig)

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

        # 8.5cm = 3.35 inches width for factor loading heatmap
        fig, axes = plt.subplots(figsize = (3.35, 2.5))
        sns.heatmap(loading_df, annot=True, cmap='RdBu_r', center=0,
                    fmt ='.2f', ax=axes, cbar_kws={'label': 'Factor Loading'})
        axes.set_title('Factor Loading Matrix', fontsize=14)
        axes.set_xlabel('Factor')
        axes.set_ylabel('Skill Metrics')

        try:
            # CLAUDE_ADDED: Use academic figure saving function
            save_academic_figure(fig, save_path)
        except Exception as e:
            print(f"❌ ヒートマップ保存エラー: {e}")

        plt.close(fig)

    def _create_factor_score_scatter(self, skill_metrics_df: pd.DataFrame, factor_analysis_results: Dict, save_path: Path):
        """因子特典の散布図作成(第一因子 vs 第二因子)"""
        factor_scores = factor_analysis_results['factor_scores']

        # 被験者情報を追加
        analysis_data = skill_metrics_df[factor_analysis_results['skill_columns']].dropna()
        subject_info = skill_metrics_df.loc[analysis_data.index, ['subject_id', 'block']]

        # CLAUDE_ADDED: 被験者名の匿名化
        subject_mapping = anonymize_subjects(subject_info['subject_id'])

        # 8.5cm = 3.35 inches width for factor score scatter plot
        fig, axes = plt.subplots(figsize=(3.35, 2.8))

        # 被験者ごとに色分けして散布図を作成
        unique_subjects = subject_info['subject_id'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_subjects)))

        for i, subject in enumerate(unique_subjects):
            mask = subject_info['subject_id'] == subject
            if np.sum(mask) > 0:
                subject_scores = factor_scores[mask]
                # CLAUDE_ADDED: 匿名化された被験者名を使用
                axes.scatter(subject_scores[:, 0], subject_scores[:, 1],
                           c=[colors[i]], label=subject_mapping[subject], alpha=0.7, s=50)

        axes.set_xlabel('Factor 1 score')
        axes.set_ylabel('Factor 2 score')
        axes.set_title('Factor Scatter Plot（Per subjects）', fontsize=14)
        axes.grid(True, alpha=0.3)
        axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 軸の交点に線を追加
        axes.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes.axvline(x=0, color='k', linestyle='-', alpha=0.3)

        # CLAUDE_ADDED: 学術論文用の保存
        save_academic_figure(fig, save_path)
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

        # 8.5cm = 3.35 inches width for factor variance plot (dual subplot)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.35, 2.5))

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

        # CLAUDE_ADDED: 学術論文用の保存
        save_academic_figure(fig, save_path)
        plt.close(fig)

    def _create_factor_correlation_heatmap(self, factor_analysis_results: Dict, save_path: Path):
        """因子相関行列のヒートマップ作成"""

        # 因子相関行列の存在確認
        fa_correlation = factor_analysis_results.get('factor_correlations')
        if fa_correlation is None:
            print("💡 因子間相関行列は斜交回転（promaxなど）の場合のみプロットされます。")
            return

        # 8.5cm = 3.35 inches width for factor correlation heatmap
        fig = plt.figure(figsize=(3.35, 2.8))
        sns.heatmap(fa_correlation,
                    annot=True,
                    fmt='.2f',
                    cmap='RdBu_r',
                    vmin=-1,
                    vmax=1)
        plt.title('Factor Correlation Matrix')
        plt.xlabel('Factor')
        plt.ylabel('Factor')

        # CLAUDE_ADDED: 学術論文用の保存
        save_academic_figure(fig, save_path)
        plt.close()


    @property
    def factorize_artifact(self):
        if self.used_standard_scaler is not None and self.used_factor_analysis is not None:
            return self.used_standard_scaler, self.used_factor_analysis
        else:
            print("学習済みStandard ScalerとFactor Analysisオブジェクトが存在しません")
            return None


class SkillScoreCalculator:
    """スキルスコアを計算する"""
    def __init__(self, config: Dict, output_manager: OutputManager):
        self.config = config
        self.output = output_manager
        self.factor_weights = np.array([-0.565, 0.245, -0.19])

    def calc_skill_score(self, skill_metrics_df: pd.DataFrame, expert_scaler: StandardScaler, expert_fa) -> pd.DataFrame:
        """スキルスコアを計算する"""
        plot_data_list = []

        skill_columns = ['curvature', 'velocity_smoothness', 'acceleration_smoothness',
                         'jerk_score', 'control_stability', 'temporal_consistency',
                         'trial_time', 'endpoint_error']

        for subject_id, subject_df in skill_metrics_df.groupby('subject_id'):
            sorted_trials = subject_df.sort_values(by=['block', 'trial_num']).reset_index()

            for i, trial_row in sorted_trials.iterrows():
                try:
                    # CLAUDE_ADDED: 1トライアル分のスキル指標をDataFrameとして抽出（特徴量名を保持）
                    trial_metrics_df = trial_row[skill_columns].to_frame().T

                    # CLAUDE_ADDED: データ型をチェックして数値型に変換
                    # 非数値データが含まれている場合の対処
                    for col in skill_columns:
                        trial_metrics_df[col] = pd.to_numeric(trial_metrics_df[col], errors='coerce')

                    # 欠損値確認
                    if trial_metrics_df.isna().any().any():
                        continue

                    # CLAUDE_ADDED: 学習済みスケーラで標準化（DataFrame形式で渡して特徴量名を保持）
                    scaled_metrics = expert_scaler.transform(trial_metrics_df)

                    # 学習済みFAモデルで因子得点を計算
                    factor_scores = expert_fa.transform(scaled_metrics)

                    # 因子得点を重み付けして単一のスキルスコアに合算
                    skill_score = np.dot(factor_scores[0], self.factor_weights)

                    plot_data_list.append({
                        'subject_id': subject_id,
                        'trial_order': i + 1,
                        'block': trial_row['block'],
                        'trial_num_in_block': trial_row['trial_num'],
                        'skill_score': skill_score
                    })
                except Exception as e:
                    print(f"スコア計算エラー: 被験者 {subject_id}, trial_index {i}: {e}")

        # リストから最終的なDataFrameを作成
        skill_score_df = pd.DataFrame(plot_data_list)

        # スキルスコアの推移をプロット
        if skill_score_df is not None:
            self._save_skill_score_plots(skill_score_df)

        return skill_score_df

    # CLAUDE_ADDED: 因子スコアを含むスキルスコア計算メソッド
    def calc_skill_score_with_factors(self, skill_metrics_df: pd.DataFrame, expert_scaler: StandardScaler, expert_fa) -> pd.DataFrame:
        """スキルスコアと因子スコアを計算する"""
        plot_data_list = []

        skill_columns = ['curvature', 'velocity_smoothness', 'acceleration_smoothness',
                         'jerk_score', 'control_stability', 'temporal_consistency',
                         'trial_time', 'endpoint_error']

        n_factors = expert_fa.n_factors

        for subject_id, subject_df in skill_metrics_df.groupby('subject_id'):
            sorted_trials = subject_df.sort_values(by=['block', 'trial_num']).reset_index()

            for i, trial_row in sorted_trials.iterrows():
                try:
                    # 1トライアル分のスキル指標をDataFrameとして抽出（特徴量名を保持）
                    trial_metrics_df = trial_row[skill_columns].to_frame().T

                    # データ型をチェックして数値型に変換
                    for col in skill_columns:
                        trial_metrics_df[col] = pd.to_numeric(trial_metrics_df[col], errors='coerce')

                    # 欠損値確認
                    if trial_metrics_df.isna().any().any():
                        continue

                    # 学習済みスケーラで標準化（DataFrame形式で渡して特徴量名を保持）
                    scaled_metrics = expert_scaler.transform(trial_metrics_df)

                    # 学習済みFAモデルで因子得点を計算
                    factor_scores = expert_fa.transform(scaled_metrics)

                    # 因子得点を重み付けして単一のスキルスコアに合算
                    skill_score = np.dot(factor_scores[0], self.factor_weights)

                    # CLAUDE_ADDED: 因子スコアを個別に保存
                    data_dict = {
                        'subject_id': subject_id,
                        'trial_order': i + 1,
                        'block': trial_row['block'],
                        'trial_num_in_block': trial_row['trial_num'],
                        'skill_score': skill_score
                    }

                    # 各因子スコアを個別のカラムとして追加
                    for f_idx in range(n_factors):
                        data_dict[f'factor_{f_idx+1}_score'] = factor_scores[0][f_idx]

                    plot_data_list.append(data_dict)

                except Exception as e:
                    print(f"スコア計算エラー: 被験者 {subject_id}, trial_index {i}: {e}")

        # リストから最終的なDataFrameを作成
        skill_score_df = pd.DataFrame(plot_data_list)

        # スキルスコアの推移をプロット
        if skill_score_df is not None:
            self._save_skill_score_plots(skill_score_df)

        print(f"✅ 計算完了: skill_score + {n_factors}個の因子スコア")
        return skill_score_df

    def calculate_stable_skill_score(self, skill_metrics_df: pd.DataFrame,expert_scaler: StandardScaler, expert_fa, window_size=10):
        """安定したスキルスコア計算"""
        stable_scores = []

        skill_columns = ['curvature', 'velocity_smoothness', 'acceleration_smoothness',
                         'jerk_score', 'control_stability', 'temporal_consistency',
                         'trial_time', 'endpoint_error']

        for subject_id, subject_df in skill_metrics_df.groupby('subject_id'):
            sorted_trials = subject_df.sort_values(by=['block', 'trial_num']).reset_index()

            for i, trial_row in sorted_trials.iterrows():
                try:
                    trial_metrics_df = trial_row[skill_columns].to_frame().T

                    # 非数値データが含まれている場合の対処
                    for col in skill_columns:
                        trial_metrics_df[col] = pd.to_numeric(trial_metrics_df[col], errors='coerce')

                    # 欠損値確認
                    if trial_metrics_df.isna().any().any():
                        continue

                    # CLAUDE_ADDED: 学習済みスケーラで標準化（DataFrame形式で渡して特徴量名を保持）
                    scaled_metrics = expert_scaler.transform(trial_metrics_df)

                    # 学習済みFAモデルで因子得点を計算
                    factor_scores = expert_fa.transform(scaled_metrics)

                    # 因子得点を重み付けして単一のスキルスコアに合算
                    skill_score = np.dot(factor_scores[0], self.factor_weights)

                    stable_scores.append({
                        'subject_id': subject_id,
                        'trial_order': i + 1,
                        'block': trial_row['block'],
                        'trial_num_in_block': trial_row['trial_num'],
                        'skill_score': skill_score
                    })

                except Exception as e:
                    print(f"スコア計算エラー: 被験者 {subject_id}, trial_index {i}: {e}")

        skill_score_df = pd.DataFrame(stable_scores)

        skill_score_df['smoothed_skill_score'] = skill_score_df.groupby('subject_id')['skill_score'].transform(lambda x: x.rolling(window_size, min_periods=1).mean())

        # スキルスコアの推移をプロット
        if skill_score_df is not None:
            self._save_skill_score_plots(skill_score_df)

        return pd.DataFrame(stable_scores)



    def _save_skill_score_plots(self, skill_scores: pd.DataFrame):
        """被験者ごとにスキルスコアの推移をプロットする"""
        self.output.skill_score_calculator_output_dir_path.mkdir(parents=True, exist_ok=True)

        save_path = self.output.skill_score_calculator_output_dir_path / 'skill_score_transition'

        # CLAUDE_ADDED: 被験者名の匿名化
        anonymized_scores = skill_scores.copy()
        subject_mapping = anonymize_subjects(skill_scores['subject_id'])
        anonymized_scores['subject_id'] = anonymized_scores['subject_id'].map(subject_mapping)

        # 12cm = 4.72 inches width for skill score plot
        fig = plt.figure(figsize=(4.72, 2.8))

        # CLAUDE_ADDED: smoothed_skill_scoreが存在するかチェックして適切なカラムを選択
        y_column = 'smoothed_skill_score' if 'smoothed_skill_score' in skill_scores.columns else 'skill_score'
        plot_title = 'Smoothed Skill Score Improvement' if y_column == 'smoothed_skill_score' else 'Skill Score Improvement'

        sns.lineplot(
            data=anonymized_scores,
            x='trial_order',
            y=y_column,
            hue='subject_id',
        )

        plt.xlabel('Trial Order [-]', fontsize=9)
        plt.ylabel('Calculated Skill Score [-]', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Subject ID', fontsize=5, title_fontsize=6, loc='best', ncol=2)
        plt.tick_params(axis='both', labelsize=8)

        plt.tight_layout()

        # CLAUDE_ADDED: 学術論文用の保存
        save_academic_figure(fig, save_path)
        plt.close()

        # CLAUDE_ADDED: 標準化したスキルスコアのプロット
        self._save_standardized_skill_score_plot(skill_scores, subject_mapping, y_column)

    def _save_standardized_skill_score_plot(self, skill_scores: pd.DataFrame, subject_mapping: Dict, original_y_column: str):
        """CLAUDE_ADDED: StandardScalerで標準化したスキルスコアをプロットする"""
        # 標準化処理: StandardScaler (DatasetBuilderと同じ処理)
        standardized_scores = skill_scores.copy()

        # 使用するカラム
        score_column = original_y_column

        # StandardScalerで標準化 (全データに対して)
        scaler = StandardScaler()
        skill_values = standardized_scores[score_column].values.reshape(-1, 1)
        standardized_values = scaler.fit_transform(skill_values)

        standardized_column_name = f'standardized_{score_column}'
        standardized_scores[standardized_column_name] = standardized_values.flatten()

        # 被験者名を匿名化
        standardized_scores['subject_id'] = standardized_scores['subject_id'].map(subject_mapping)

        # プロットの保存パス
        save_path = self.output.skill_score_calculator_output_dir_path / 'standardized_skill_score_transition'

        # 12cm = 4.72 inches width for skill score plot
        fig = plt.figure(figsize=(4.72, 2.8))

        sns.lineplot(
            data=standardized_scores,
            x='trial_order',
            y=standardized_column_name,
            hue='subject_id',
        )

        plt.xlabel('Trial Order [-]', fontsize=9)
        plt.ylabel('Standardized Skill Score (Z-score) [-]', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Subject ID', fontsize=5, title_fontsize=6, loc='best', ncol=2)
        plt.tick_params(axis='both', labelsize=8)

        plt.tight_layout()

        # 学術論文用の保存
        save_academic_figure(fig, save_path)
        plt.close()


class DatasetBuilder:
    """データセット生成クラス - 前処理とファイル保存を担当"""
    
    def __init__(self, config: Dict, output_manager: OutputManager):
        self.config = config
        self.output_manager = output_manager
        self.target_seq_len = config['pre_process']['target_seq_len']
        
    def build_skill_trajectory_dataset(self,
                                     skill_metrics_df: pd.DataFrame,
                                     preprocessed_trajectory_df: pd.DataFrame,
                                     trained_scaler: StandardScaler,
                                     trained_fa) -> str:
        """スキルスコア付き軌道データセットを構築し、保存する"""

        print("スキルスコア付き軌道データセットを構築中...")

        # CLAUDE_ADDED: 出力ディレクトリの作成
        dataset_output_dir = self.output_manager.dataset_builder_output_dir_path
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        # 1. スキルスコアと因子スコアを計算
        skill_score_calculator = SkillScoreCalculator(self.config, self.output_manager)
        # CLAUDE_ADDED: コンフィグに基づいて因子スコアを含めるかどうかを切り替え
        use_factor_scores = self.config.get('analysis', {}).get('use_factor_scores', True)

        if use_factor_scores:
            print("💡 因子スコアを含めてデータセットを作成します")
            skill_score_df = skill_score_calculator.calc_skill_score_with_factors(
                skill_metrics_df, trained_scaler, trained_fa
            )
        else:
            print("💡 skill_scoreのみでデータセットを作成します")
            skill_score_df = skill_score_calculator.calc_skill_score(
                skill_metrics_df, trained_scaler, trained_fa
            )

        # 2. 軌道データとスキルスコアを結合
        merged_df = self._merge_trajectory_and_skill_data(
            preprocessed_trajectory_df, skill_score_df
        )

        # 3. トレーニング/テストデータに分割
        train_df, test_df = self._split_train_test(merged_df)

        # 4. 特徴量のスケーリング
        scaled_train_df, scaled_test_df, scalers, feature_config = self._scale_features(
            train_df, test_df
        )

        # 5. データとメタデータを保存
        self._save_dataset_files(
            scaled_train_df, scaled_test_df, scalers, feature_config, dataset_output_dir
        )

        print(f"✅ データセットが保存されました: {dataset_output_dir}")
        return str(dataset_output_dir)
    
    def _merge_trajectory_and_skill_data(self, trajectory_df: pd.DataFrame,
                                       skill_score_df: pd.DataFrame) -> pd.DataFrame:
        """軌道データとスキルスコアデータを結合"""

        # CLAUDE_ADDED: デバッグ情報を追加
        print(f"CLAUDE_DEBUG: Trajectory data shape: {trajectory_df.shape}")
        print(f"CLAUDE_DEBUG: Skill score data shape: {skill_score_df.shape}")
        print(f"CLAUDE_DEBUG: Skill score data columns: {skill_score_df.columns.tolist()}")

        # CLAUDE_ADDED: 各試行のデータ長を確認
        if 'time_step' in trajectory_df.columns:
            trial_lengths = trajectory_df.groupby(['subject_id', 'trial_num', 'block']).size()
            print(f"CLAUDE_DEBUG: Sample trajectory lengths: {trial_lengths.head(5)}")
            print(f"CLAUDE_DEBUG: Min length: {trial_lengths.min()}, Max length: {trial_lengths.max()}")

        # CLAUDE_ADDED: スコアカラムを検出（skill_score + factor_i_score）
        score_columns = ['skill_score']
        factor_score_cols = [col for col in skill_score_df.columns if col.startswith('factor_') and col.endswith('_score')]
        score_columns.extend(factor_score_cols)
        print(f"CLAUDE_DEBUG: Detected score columns to merge: {score_columns}")

        # CLAUDE_ADDED: スキルスコアデータをトライアル単位で結合
        merged_data = []
        successful_merges = 0
        failed_merges = 0

        for _, skill_row in skill_score_df.iterrows():
            subject_id = skill_row['subject_id']
            block = skill_row['block']
            trial_num = skill_row['trial_num_in_block']

            # 該当する軌道データを取得
            trajectory_subset = trajectory_df[
                (trajectory_df['subject_id'] == subject_id) &
                (trajectory_df['block'] == block) &
                (trajectory_df['trial_num'] == trial_num)
            ].copy()

            if not trajectory_subset.empty:
                # CLAUDE_ADDED: 結合前に軌道データの長さをチェック
                print(f"CLAUDE_DEBUG: Merging {subject_id} trial {trial_num} block {block}: length = {len(trajectory_subset)}")

                # CLAUDE_ADDED: 全スコアカラムを全タイムステップに追加
                for score_col in score_columns:
                    if score_col in skill_row:
                        trajectory_subset[score_col] = skill_row[score_col]

                merged_data.append(trajectory_subset)
                successful_merges += 1
            else:
                failed_merges += 1

        print(f"CLAUDE_DEBUG: Successful merges: {successful_merges}, Failed merges: {failed_merges}")

        if merged_data:
            merged_df = pd.concat(merged_data, ignore_index=True)
            print(f"CLAUDE_DEBUG: Final merged data shape: {merged_df.shape}")
            print(f"CLAUDE_DEBUG: Final merged data columns: {merged_df.columns.tolist()}")
            return merged_df
        else:
            return pd.DataFrame()
    
    def _split_train_test(self, merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """被験者レベルでトレーニング/テストに分割"""
        
        subjects = merged_df['subject_id'].unique()
        test_size = self.config.get('data', {}).get('test_split', 0.2)
        random_seed = self.config.get('data', {}).get('random_seed', 42)
        
        from sklearn.model_selection import train_test_split
        train_subjects, test_subjects = train_test_split(
            subjects, test_size=test_size, random_state=random_seed
        )
        
        train_df = merged_df[merged_df['subject_id'].isin(train_subjects)]
        test_df = merged_df[merged_df['subject_id'].isin(test_subjects)]
        
        print(f"データ分割: 学習用被験者={len(train_subjects)}人, テスト用被験者={len(test_subjects)}人")
        
        return train_df, test_df
    
    def _scale_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """特徴量のスケーリング"""

        # CLAUDE_ADDED: スケーリング対象の特徴量を定義
        trajectory_features = ['HandlePosX', 'HandlePosY', 'HandleVelX',
                             'HandleVelY', 'HandleAccX', 'HandleAccY']

        scalers = {}
        scaled_train_df = train_df.copy()
        scaled_test_df = test_df.copy()

        # 特徴量ごとにスケーリング
        for feature in trajectory_features:
            if feature in train_df.columns:
                scaler = StandardScaler()

                # 学習データでフィット
                train_values = train_df[feature].values.reshape(-1, 1)
                scaled_train_values = scaler.fit_transform(train_values)
                scaled_train_df[feature] = scaled_train_values.flatten()

                # テストデータに適用
                test_values = test_df[feature].values.reshape(-1, 1)
                scaled_test_values = scaler.transform(test_values)
                scaled_test_df[feature] = scaled_test_values.flatten()

                scalers[feature] = scaler

        # CLAUDE_ADDED: スコアカラム（skill_score + factor_i_score）を検出してスケーリング
        score_columns = []

        # スキルスコアも正規化
        if 'skill_score' in train_df.columns:
            skill_scaler = StandardScaler()
            train_skill = train_df['skill_score'].values.reshape(-1, 1)
            scaled_train_df['skill_score'] = skill_scaler.fit_transform(train_skill).flatten()

            test_skill = test_df['skill_score'].values.reshape(-1, 1)
            scaled_test_df['skill_score'] = skill_scaler.transform(test_skill).flatten()

            scalers['skill_score'] = skill_scaler
            score_columns.append('skill_score')

        # CLAUDE_ADDED: 因子スコアも正規化
        factor_score_cols = [col for col in train_df.columns if col.startswith('factor_') and col.endswith('_score')]
        for factor_col in factor_score_cols:
            factor_scaler = StandardScaler()
            train_factor = train_df[factor_col].values.reshape(-1, 1)
            scaled_train_df[factor_col] = factor_scaler.fit_transform(train_factor).flatten()

            test_factor = test_df[factor_col].values.reshape(-1, 1)
            scaled_test_df[factor_col] = factor_scaler.transform(test_factor).flatten()

            scalers[factor_col] = factor_scaler
            score_columns.append(factor_col)

        print(f"CLAUDE_DEBUG: Scaled score columns: {score_columns}")

        # CLAUDE_ADDED: 特徴量設定（因子スコアも含める）
        feature_config = {
            'feature_cols': trajectory_features + score_columns,
            'trajectory_features': trajectory_features,
            'score_columns': score_columns,  # CLAUDE_ADDED: スコアカラムのリストも保存
            'target_seq_len': self.target_seq_len
        }

        return scaled_train_df, scaled_test_df, scalers, feature_config
    
    def _save_dataset_files(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                          scalers: Dict, feature_config: Dict, output_dir: Path):
        """データセットファイルを保存"""
        
        # CLAUDE_ADDED: データフレームを保存
        train_df.to_parquet(output_dir / 'train_data.parquet', index=False)
        test_df.to_parquet(output_dir / 'test_data.parquet', index=False)
        
        # スケーラーと設定を保存
        joblib.dump(scalers, output_dir / 'scalers.joblib')
        joblib.dump(feature_config, output_dir / 'feature_config.joblib')
        
        # データセット情報を保存
        dataset_info = {
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'train_subjects': train_df['subject_id'].nunique(),
            'test_subjects': test_df['subject_id'].nunique(),
            'feature_columns': feature_config['feature_cols'],
            'target_seq_len': self.target_seq_len,
            'created_at': datetime.datetime.now().isoformat()
        }
        
        import json
        with open(output_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
            
        print(f"保存されたファイル:")
        print(f"  - train_data.parquet: {len(train_df):,} サンプル") 
        print(f"  - test_data.parquet: {len(test_df):,} サンプル")
        print(f"  - scalers.joblib: {len(scalers)} スケーラー")
        print(f"  - feature_config.joblib: 設定情報")
        print(f"  - dataset_info.json: メタデータ")



if __name__ == '__main__':
    config_dir = 'PredictiveLatentSpaceNavigationModel/DataPreprocess/data_preprocess_default_config.yaml'

    try:
        # コンフィグの読み込み
        config_loader = DataPreprocessConfigLoader(config_dir)
        validated_config = config_loader.get_config()

        # 出力ディレクトリマネージャー
        output_manager = OutputManager(validated_config)

        # 生データの読みこみ
        trajectory_loader = TrajectoryDataLoader(validated_config)

        print("============ 熟達したデータで因子分析を実行 (block_num = 4) ============")

        block_num = 4

        # スキル指標の計算
        skill_metrics_calculator = SkillMetricCalculator(validated_config)
        preprocessed_data = trajectory_loader.get_preprocessed_data(block_num)
        skill_metrics_df = skill_metrics_calculator.calculate_skill_metrics(preprocessed_data)

        print(skill_metrics_df['block'].unique())
        print(skill_metrics_df['trial_num'].unique())

        # ANOVA and 因子分析
        skill_analyzer = SkillAnalyzer(validated_config, output_manager)
        skill_analyzer.analyze_skill_metrics(skill_metrics_df, 'promax')

        # 学習済みスケーラと因子分析オブジェクトを取得
        trained_scaler, trained_fa = skill_analyzer.factorize_artifact

        print("============ 全データでブロック毎因子分析を実行 ============")
        # 全データを取得してブロック毎の因子分析を実行
        all_preprocess_data_for_blockwise = trajectory_loader.get_preprocessed_data(0)
        all_skill_metrics_df_for_blockwise = skill_metrics_calculator.calculate_skill_metrics(all_preprocess_data_for_blockwise)

        # ブロック毎の因子分析を実行
        skill_analyzer.analyze_block_wise_factor_analysis(all_skill_metrics_df_for_blockwise, 'promax')

        print("============ 全データに対してスキルスコアを計算 (block_num = 0) ============")
        block_num = 0

        # スキル指標を全データで計算
        all_preprocess_data = trajectory_loader.get_preprocessed_data(block_num)
        all_skill_metrics_df = skill_metrics_calculator.calculate_skill_metrics(all_preprocess_data)

        print(all_skill_metrics_df['block'].unique())
        print(all_skill_metrics_df['trial_num'].unique())

        if validated_config.get('data').get('make_dataset'):
            print("============ データセットを構築して保存 ============")

            dataset_builder = DatasetBuilder(validated_config, output_manager)
            dataset_path = dataset_builder.build_skill_trajectory_dataset(
                all_skill_metrics_df,
                all_preprocess_data,
                trained_scaler,
                trained_fa
            )
            print(f"🎉 データセット作成完了: {dataset_path}")

        # オプション: スキルスコア推移も別途計算・保存
        skill_score_calculator = SkillScoreCalculator(validated_config, output_manager)
        skill_score_calculator.calculate_stable_skill_score(all_skill_metrics_df, trained_scaler, trained_fa, 5)

    except Exception as e:
        print(f"エラーが発生しました: {e}")

