# スタイル潜在空間の評価器
from typing import List, Dict, Any

from base_evaluator import BaseEvaluator
from src.PredictiveLatentSpaceNavigationModel.TransformerBaseEndToEndVAE.evaluator import EnhancedEvaluationResult


class VisualizeStyleSpaceEvaluator(BaseEvaluator):
    """スタイル潜在空間の可視化評価"""
    def __init__(self, config:Dict[str, Any]):
        pass

    def evaluate(self, model, test_data, device) -> EnhancedEvaluationResult:
        pass

    def get_required_data(self) -> List[str]:
        pass


class StyleClusteringEvaluator(BaseEvaluator):
    """スタイル潜在空間内のクラスタリング性能の評価"""
    def __init__(self, config:Dict[str, Any]):
        pass

    def evaluate(self, model, test_data, device) -> EnhancedEvaluationResult:
        pass

    def get_required_data(self) -> List[str]:
        pass

class StyleClassificationEvaluator(BaseEvaluator):
    """スタイル潜在変数から簡単なSVM,MLPで被験者の分類が可能化を評価"""
    def __init__(self, config:Dict[str, Any]):
        pass

    def evaluate(self, model, test_data, device) -> EnhancedEvaluationResult:
        pass

    def get_required_data(self) -> List[str]:
        pass


