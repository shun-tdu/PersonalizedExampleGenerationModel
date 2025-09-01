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
        """前処理済みデータから各種スキル指標を計算"""
        skill_metrics = []
        
        for (subject_id, trial_num), trial_group in preprocessed_data.groupby(['subject_id', 'trial_num']):
            try:
                # TODO(human) - スキル指標の計算実装
                # 各トライアルから動作時間、終点誤差、ジャークを計算
                trial_time = self._calculate_trial_time(trial_group)
                endpoint_error = self._calculate_endpoint_error(trial_group)
                jerk_score = self._calculate_jerk(trial_group)
                
                # 結果をまとめる
                metrics = {
                    'subject_id': subject_id,
                    'trial_num': trial_num,
                    'block': trial_group['block'].iloc[0],
                    'trial_time': trial_time,
                    'endpoint_error': endpoint_error,
                    'jerk_score': jerk_score
                }
                skill_metrics.append(metrics)
                
            except Exception as e:
                print(f"スキル指標計算エラー: 被験者{subject_id}, トライアル{trial_num}: {e}")
                continue
        
        return pd.DataFrame(skill_metrics)
    
    def _calculate_trial_time(self, trial_data: pd.DataFrame) -> float:
        """動作時間の計算 - 軌道の長さから推定"""
        # 実装例：正規化された時間軸を実際の時間に変換
        # この実装では target_seq_len が実際の時間ステップ数を表すと仮定
        return len(trial_data) / self.target_seq_len
    
    def _calculate_endpoint_error(self, trial_data: pd.DataFrame) -> float:
        """終点誤差の計算 - 目標位置からの距離"""
        # 最後の位置を終点として使用
        final_x = trial_data['HandlePosX'].iloc[-1]
        final_y = trial_data['HandlePosY'].iloc[-1]
        
        # TODO: 目標位置は設定から取得するか、データに含めるべき
        # ここでは原点(0,0)を目標として仮定
        target_x, target_y = 0.0, 0.0
        
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

        # 生データの読みこみ
        trajectory_loader = TrajectoryDataLoader(config_loader.get_config())

    except Exception as e:
        print(f"エラーが発生しました: {e}")

