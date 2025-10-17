from abc import ABC, abstractmethod
from typing import  Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

class BaseExperimentModel(nn.Module, ABC):
    """実験用システム用のベースモデルクラス"""

    def __init__(self, **kwargs):
        super().__init__()
        self.model_config = kwargs
        self.loss_scheduler = None
        self.optimizer_type = None
        self.lr = None
        self.weight_decay = None

    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """順伝播で損失も含めた結果を返す"""
        pass

    @abstractmethod
    def configure_optimizers(self, training_config :Dict[str, Any]) -> Any:
        """
        モデルに必要なOptimizerとSchedulerを返す
        単一optimizer、または複数optimizerのタプルを返すことができる
        """
        self.optimizer_type = training_config.get('optimizer', 'AdamW')
        self.lr = training_config.get('learning_rate', training_config.get('lr', 1e-3))
        self.weight_decay = training_config.get('weight_decay', 1e-5)


    @abstractmethod
    def training_step(self, batch:Any, optimizers:Any, device:torch.device) -> Dict[str, torch.Tensor]:
        """
        1バッチ分の学習処理を行い、ログ用の損失辞書を返す
        """
        pass

    @abstractmethod
    def validation_step(self, batch:Any, device:torch.device) -> Dict[str, torch.Tensor]:
        """
        1バッチ分の検証処理を行い、ログ用の損失辞書を返す
        """
        pass

    def on_epoch_start(self, epoch: int):
        """
        エポック開始時に呼び出されるコールバック
        スケジューラが存在すれば、その状態を更新する
        """
        if self.loss_scheduler is not None:
            self.loss_scheduler.step(epoch)


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

