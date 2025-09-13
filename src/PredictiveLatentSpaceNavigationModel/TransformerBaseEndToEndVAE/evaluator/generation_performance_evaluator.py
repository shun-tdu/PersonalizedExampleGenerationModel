# 軌道生成性能の評価器
from typing import List, Dict, Any
import torch

from .base_evaluator import BaseEvaluator
from .result_manager import EnhancedEvaluationResult

class StyleTransferEvaluator(BaseEvaluator):
    """スタイル潜在変数を固定してスキル潜在変数を変化させたときにスタイルを維持したまま、スキルスコアが変化するかを評価"""
    def __init__(self, config: Dict[str, Any]):
        pass

    def evaluate(self, model: torch.nn.Module, test_data: Dict[str, Any], device: torch.device, result: EnhancedEvaluationResult = None) -> EnhancedEvaluationResult:
        pass

    def get_required_data(self) -> List[str]:
        pass

class SkillTransferEvaluator(BaseEvaluator):
    """スキル潜在変数を固定してスタイル潜在変数を変化させたときにスキルを維持したまま、スタイルスコアが変化するかを評価"""
    def __init__(self, config: Dict[str, Any]):
        pass

    def evaluate(self, model: torch.nn.Module, test_data: Dict[str, Any], device: torch.device, result: EnhancedEvaluationResult = None) -> EnhancedEvaluationResult:
        pass

    def get_required_data(self) -> List[str]:
        pass

