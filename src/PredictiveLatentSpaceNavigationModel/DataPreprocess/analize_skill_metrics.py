import pandas as pd
import numpy as np
from typing import  Dict, List, Tuple, Optional
from pathlib import  Path

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
            if interpolate_method_section not in ['linear', 'spline', 'polynomial']:
                errors.append("interpolate_methodの値が不適切です。['linear', 'spline', 'polynomial'] から選択してください")

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

    def get_config(self) -> Optional[Dict, None] :
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

    pass

class SkillMetricCalculator:
    """各種スキル指標の計算を担当"""
    pass

class SkillAnalyzer:
    """ANOVA,因子解析などの統計解析を担当"""
    pass

class DatasetBuilder:
    """データセット生成クラス"""
    # todo コンフィグの nameごとにユニークな名前のディレクトリを作成する nameが同じ場合でも必ずユニークにする
    pass

if __name__ == '__main__':

    pass
