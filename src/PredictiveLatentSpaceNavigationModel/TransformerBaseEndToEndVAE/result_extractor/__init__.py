# CLAUDE_ADDED
"""
Result Extractor Package for Academic Paper Generation
学会論文用結果抽出パッケージ
"""

from .svg_evaluator import AcademicSVGEvaluator
from .evaluation_runner import AcademicEvaluationRunner

__all__ = [
    'AcademicSVGEvaluator',
    'AcademicEvaluationRunner'
]