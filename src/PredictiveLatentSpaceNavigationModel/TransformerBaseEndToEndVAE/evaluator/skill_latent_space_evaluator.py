# スキル潜在空間の評価器
from typing import List, Dict, Any

from base_evaluator import BaseEvaluator
from src.PredictiveLatentSpaceNavigationModel.TransformerBaseEndToEndVAE.evaluator import EnhancedEvaluationResult


class VisualizeSkillSpaceEvaluator(BaseEvaluator):
    """スキル潜在空間の可視化評価"""
    def __init__(self, config:Dict[str, Any]):
        pass

    def evaluate(self, model, test_data, device) -> EnhancedEvaluationResult:
        pass

    def get_required_data(self) -> List[str]:
        pass


class SkillScoreRegressionEvaluator(BaseEvaluator):
    """スキル潜在変数から簡単なSVM,MLPでスキルスコアを回帰可能かを評価"""
    def __init__(self, config:Dict[str, Any]):
        pass

    def evaluate(self, model, test_data, device) -> EnhancedEvaluationResult:
        pass

    def get_required_data(self) -> List[str]:
        pass

class SkillLatentDimensionVSScoreEvaluator(BaseEvaluator):
    """スキル潜在変数の主次元とスキルスコアのが線形な関係にあるかを評価"""
    def __init__(self, config:Dict[str, Any]):
        pass

    def evaluate(self, model, test_data, device) -> EnhancedEvaluationResult:
        pass

    def get_required_data(self) -> List[str]:
        pass