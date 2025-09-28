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

    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """順伝播で損失も含めた結果を返す"""
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

    # CLAUDE_ADDED: 汎用的なチェックポイント読み込み機能
    def load_pretrained_weights(self, checkpoint_configs: Dict[str, Any]):
        """
        事前学習済み重みを読み込む汎用機能

        Args:
            checkpoint_configs: チェックポイント設定辞書
                例: {
                    'encoder_checkpoint': '/path/to/encoder.pth',
                    'partial_checkpoint': {
                        'path': '/path/to/partial.pth',
                        'strict': False,
                        'prefix_map': {'old_prefix.': 'new_prefix.'}
                    }
                }
        """
        for config_name, config_value in checkpoint_configs.items():
            if config_value is None:
                continue

            try:
                if isinstance(config_value, str):
                    # 単純なパス指定の場合
                    self._load_simple_checkpoint(config_value, config_name)
                elif isinstance(config_value, dict):
                    # 詳細設定がある場合
                    self._load_detailed_checkpoint(config_value, config_name)
                else:
                    print(f"Warning: Invalid checkpoint config for {config_name}: {config_value}")

            except Exception as e:
                print(f"Warning: Failed to load checkpoint {config_name}: {e}")

    def _load_simple_checkpoint(self, checkpoint_path: str, config_name: str):
        """単純なチェックポイント読み込み"""
        if not torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        else:
            checkpoint = torch.load(checkpoint_path)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        self.load_state_dict(state_dict, strict=False)
        print(f"✓ Loaded checkpoint: {config_name} from {checkpoint_path}")

    def _load_detailed_checkpoint(self, config: Dict[str, Any], config_name: str):
        """詳細設定付きチェックポイント読み込み"""
        checkpoint_path = config.get('path')
        if not checkpoint_path:
            raise ValueError(f"No 'path' specified in detailed config for {config_name}")

        strict = config.get('strict', False)
        prefix_map = config.get('prefix_map', {})
        target_keys = config.get('target_keys', None)  # 特定のキーのみ読み込み

        if not torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        else:
            checkpoint = torch.load(checkpoint_path)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # プレフィックス変換
        if prefix_map:
            new_state_dict = {}
            for old_key, value in state_dict.items():
                new_key = old_key
                for old_prefix, new_prefix in prefix_map.items():
                    if old_key.startswith(old_prefix):
                        new_key = new_prefix + old_key[len(old_prefix):]
                        break
                new_state_dict[new_key] = value
            state_dict = new_state_dict

        # 特定キーのみ読み込み
        if target_keys:
            filtered_state_dict = {k: v for k, v in state_dict.items()
                                 if any(k.startswith(prefix) for prefix in target_keys)}
            state_dict = filtered_state_dict

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)

        if missing_keys:
            print(f"Missing keys in {config_name}: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Unexpected keys in {config_name}: {unexpected_keys[:5]}...")

        print(f"✓ Loaded detailed checkpoint: {config_name} from {checkpoint_path} (strict={strict})")

    def get_checkpoint_requirements(self) -> Dict[str, Any]:
        """
        このモデルが必要とするチェックポイント設定を返す
        サブクラスでオーバーライド可能

        Returns:
            Dict: 必要なチェックポイント設定の仕様
        """
        return {}
