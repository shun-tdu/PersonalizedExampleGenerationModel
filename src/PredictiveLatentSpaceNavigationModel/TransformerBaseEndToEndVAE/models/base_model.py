from abc import ABC, abstractmethod
from typing import  Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

class BaseExperimentModel(nn.Module, ABC):
    """実験用システム用のベースモデルクラス"""

    def __init__(self, **kwargs):
        super().__init__()
        self.model_config = kwargs

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """順伝播で損失も含めた結果を返す"""
        pass

    def get_loss_dict(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """損失辞書を標準形式で返す"""
        loss_dict = {}
        for key, value in outputs.items():
            if 'loss' in key.lower():
                loss_dict[key] = float(value.item() if torch.is_tensor(value) else value)
        return loss_dict

    def get_evaluation_dict(self, test_data) -> Dict[str, Any]:
        """標準指標を返す(オプション)"""
        return {}

    def save_model(self, path: str):
        """モデル保存"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.model_config
        }, path)

    @classmethod
    def load_model(cls, path: str, **kwargs):
        """モデル読み込み"""
        checkpoint = torch.load(path)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
