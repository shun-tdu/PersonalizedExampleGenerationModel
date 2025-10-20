# -*- coding: utf-8 -*-
"""
CLAUDE_ADDED: ModelAdapter implementations
Standardizes encode/decode operations for different model types
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from .types import DecodeMetadata


class ModelAdapter(ABC):
    """Base class to standardize model operations"""

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config

    @abstractmethod
    def encode(self, trajectory: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Encode trajectory to latent variables"""
        pass

    @abstractmethod
    def decode(self, z_style: torch.Tensor, z_skill: torch.Tensor,
               metadata: Optional[DecodeMetadata] = None) -> torch.Tensor:
        """Decode latent variables to trajectory [B, seq_len, features]"""
        pass

    @abstractmethod
    def is_diffusion_model(self) -> bool:
        """Check if model is diffusion-based"""
        pass

    @abstractmethod
    def get_model_type(self) -> str:
        """Get model type identifier"""
        pass


class StandardVAEAdapter(ModelAdapter):
    """Adapter for standard VAE models"""

    def encode(self, trajectory: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        return self.model.encode(trajectory)

    def decode(self, z_style: torch.Tensor, z_skill: torch.Tensor,
               metadata: Optional[DecodeMetadata] = None) -> torch.Tensor:
        decoded = self.model.decode(z_style, z_skill)
        return decoded['trajectory']

    def is_diffusion_model(self) -> bool:
        return False

    def get_model_type(self) -> str:
        return 'standard_vae'


class PatchedVAEAdapter(ModelAdapter):
    """Adapter for patched VAE models - converts patches to continuous trajectory"""

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        data_config = config.get('data', {})
        patch_config = data_config.get('patch', {})
        self.patch_size = patch_config.get('size', getattr(model, 'patch_size', 20))
        self.patch_step = patch_config.get('step', self.patch_size)
        self._original_seq_len_cache = {}  # CLAUDE_ADDED: バッチごとの元のシーケンス長をキャッシュ
        print(f"PatchedVAEAdapter: patch_size={self.patch_size}, patch_step={self.patch_step}")

        # CLAUDE_ADDED: デバッグ用に往復変換をテスト
        self._test_roundtrip_conversion()

    def encode(self, trajectory: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # CLAUDE_ADDED: 入力形式を自動判定してパッチ変換
        original_seq_len = None
        if len(trajectory.shape) == 3:  # [B, seq_len, features] -> 連続軌道形式
            # CLAUDE_ADDED: 元のシーケンス長を記録
            original_seq_len = trajectory.shape[1]

            # 連続軌道をパッチに変換
            trajectory_patches, patch_attention_mask = self._trajectory_to_patches(trajectory)
            # attention_maskが提供されていない場合はパッチ変換時のものを使用
            if attention_mask is None:
                attention_mask = patch_attention_mask
        elif len(trajectory.shape) == 4:  # [B, num_patches, patch_size, features] -> 既にパッチ形式
            trajectory_patches = trajectory
        else:
            raise ValueError(f"Unexpected trajectory shape: {trajectory.shape}")

        encoded = self.model.encode(trajectory_patches, attention_mask)

        # CLAUDE_ADDED: 元のシーケンス長をエンコード結果に追加
        if original_seq_len is not None:
            encoded['original_seq_len'] = original_seq_len

        return encoded

    def decode(self, z_style: torch.Tensor, z_skill: torch.Tensor,
               metadata: Optional[DecodeMetadata] = None) -> torch.Tensor:
        num_patches = None
        attention_mask = None
        original_seq_len = None  # CLAUDE_ADDED: encode時に記録された元のシーケンス長

        if metadata is not None:
            num_patches = metadata.get('num_patches')
            attention_mask = metadata.get('attention_mask')
            original_seq_len = metadata.get('original_seq_len')  # CLAUDE_ADDED: 取得

            if num_patches is None and 'original_shape' in metadata:
                original_shape = metadata['original_shape']
                # CLAUDE_ADDED: original_shapeの次元数で処理を分岐
                if len(original_shape) == 4:
                    # [B, num_patches, patch_size, features] -> パッチ形式
                    num_patches = original_shape[1]
                elif len(original_shape) == 3:
                    # [B, seq_len, features] -> 連続軌道形式
                    # original_seq_lenが指定されていればそれを使用、なければoriginal_shapeから計算
                    if original_seq_len is None:
                        original_seq_len = original_shape[1]
                    # 必要なパッチ数を計算
                    num_patches = (original_seq_len - self.patch_size) // self.patch_step + 1

        if num_patches is None:
            num_patches = 100

        decoded = self.model.decode(z_style, z_skill, num_patches)
        patches = decoded['trajectory']
        trajectory = self._patches_to_trajectory(patches, attention_mask)

        # CLAUDE_ADDED: 元の連続軌道長にトリミング（3D入力の場合）
        if original_seq_len is not None and trajectory.shape[1] != original_seq_len:
            # 長すぎる場合はトリミング
            if trajectory.shape[1] > original_seq_len:
                trajectory = trajectory[:, :original_seq_len, :]
            # 短い場合はゼロパディング
            else:
                pad_len = original_seq_len - trajectory.shape[1]
                padding = torch.zeros(trajectory.shape[0], pad_len, trajectory.shape[2], device=trajectory.device)
                trajectory = torch.cat([trajectory, padding], dim=1)

        return trajectory

    def _trajectory_to_patches(self, trajectory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        CLAUDE_ADDED: Convert continuous trajectory to patches
        :param trajectory: [B, seq_len, features]
        :return: (patches [B, num_patches, patch_size, features], attention_mask [B, num_patches])
        """
        B, seq_len, F = trajectory.shape

        # パッチ数を計算
        num_patches = (seq_len - self.patch_size) // self.patch_step + 1

        # パッチテンソルとマスクを初期化
        patches = torch.zeros(B, num_patches, self.patch_size, F, device=trajectory.device)
        attention_mask = torch.zeros(B, num_patches, dtype=torch.bool, device=trajectory.device)

        # パッチを切り出し
        for i in range(num_patches):
            start = i * self.patch_step
            end = start + self.patch_size

            if end <= seq_len:
                patches[:, i, :, :] = trajectory[:, start:end, :]
                attention_mask[:, i] = False  # 有効なパッチ
            else:
                # 最後のパッチが不完全な場合はパディング
                remaining = seq_len - start
                patches[:, i, :remaining, :] = trajectory[:, start:, :]
                patches[:, i, remaining:, :] = 0  # ゼロパディング
                attention_mask[:, i] = True  # パディング部分をマスク

        return patches, attention_mask

    def _patches_to_trajectory(self, patches: torch.Tensor,
                               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert patches to continuous trajectory (overlap averaged)"""
        B, S, P, F = patches.shape
        seq_len = (S - 1) * self.patch_step + P
        trajectory = torch.zeros(B, seq_len, F, device=patches.device)
        counts = torch.zeros(B, seq_len, 1, device=patches.device)

        for i in range(S):
            if attention_mask is not None:
                is_padding = attention_mask[:, i]
                if is_padding.all():
                    break
            else:
                is_padding = torch.zeros(B, dtype=torch.bool, device=patches.device)

            start = i * self.patch_step
            end = start + P
            valid_mask = (~is_padding).float().view(B, 1, 1)
            trajectory[:, start:end, :] += patches[:, i, :, :] * valid_mask
            counts[:, start:end, :] += valid_mask

        trajectory = trajectory / counts.clamp(min=1)
        return trajectory

    def is_diffusion_model(self) -> bool:
        return False

    def get_model_type(self) -> str:
        return 'patched_vae'

    def _test_roundtrip_conversion(self) -> None:
        """CLAUDE_ADDED: Test patch roundtrip conversion (trajectory -> patches -> trajectory)"""
        print("\n" + "="*60)
        print("Testing patch roundtrip conversion...")
        print("="*60)

        # テストパラメータ
        batch_size = 2
        seq_len = 210
        features = 6

        # テスト用の連続軌道を作成（線形増加パターン）
        test_trajectory = torch.arange(batch_size * seq_len * features, dtype=torch.float32)
        test_trajectory = test_trajectory.reshape(batch_size, seq_len, features)

        print(f"Original trajectory shape: {test_trajectory.shape}")
        print(f"Patch size: {self.patch_size}, Patch step: {self.patch_step}")

        # 連続軌道 -> パッチ
        patches, attention_mask = self._trajectory_to_patches(test_trajectory)
        print(f"Patches shape: {patches.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        print(f"Number of masked patches: {attention_mask.sum().item()}")

        # パッチ -> 連続軌道
        reconstructed_trajectory = self._patches_to_trajectory(patches, attention_mask)
        print(f"Reconstructed trajectory shape: {reconstructed_trajectory.shape}")

        # 誤差を計算
        if test_trajectory.shape == reconstructed_trajectory.shape:
            diff = test_trajectory - reconstructed_trajectory
            rmse = torch.sqrt(torch.mean(diff ** 2)).item()
            max_error = torch.max(torch.abs(diff)).item()

            print(f"\nRoundtrip conversion error:")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  Max absolute error: {max_error:.6f}")

            # サンプル値の比較（最初の10フレーム、最初の特徴量）
            print(f"\nSample comparison (first 10 frames, first feature):")
            print(f"  Original:      {test_trajectory[0, :10, 0].cpu().numpy()}")
            print(f"  Reconstructed: {reconstructed_trajectory[0, :10, 0].cpu().numpy()}")

            # 許容誤差チェック
            if torch.allclose(test_trajectory, reconstructed_trajectory, atol=1e-5):
                print(f"\n✅ Roundtrip conversion is EXACT (within 1e-5 tolerance)")
            elif rmse < 0.01:
                print(f"\n⚠️ Roundtrip conversion has small errors (RMSE < 0.01)")
            else:
                print(f"\n❌ Roundtrip conversion has SIGNIFICANT errors (RMSE >= 0.01)")
        else:
            print(f"\n❌ Shape mismatch! Original: {test_trajectory.shape}, Reconstructed: {reconstructed_trajectory.shape}")

        print("="*60 + "\n")


class DiffusionVAEAdapter(ModelAdapter):
    """Adapter for diffusion VAE models"""

    def encode(self, trajectory: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        try:
            return self.model.encode(trajectory, attention_mask)
        except TypeError:
            return self.model.encode(trajectory)

    def decode(self, z_style: torch.Tensor, z_skill: torch.Tensor,
               metadata: Optional[DecodeMetadata] = None) -> Optional[torch.Tensor]:
        skip_decode = self.config.get('skip_diffusion_decode', True)
        if skip_decode:
            print("Warning: Diffusion decode skipped (time-consuming)")
            print("   Set config['skip_diffusion_decode'] = False to enable")
            return None
        print(f"Diffusion sampling: {len(z_style)} samples")
        sampled = self.model.sample(z_style, z_skill)
        print(f"Sampling completed: shape={sampled.shape}")
        return sampled

    def is_diffusion_model(self) -> bool:
        return True

    def get_model_type(self) -> str:
        return 'diffusion_vae'


class SkipConnectionVAEAdapter(StandardVAEAdapter):
    """Adapter for VAE models with skip connections"""

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        self._cached_skip_connections = {}
        print("SkipConnectionVAEAdapter initialized")

    def encode(self, trajectory: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        try:
            encoded = self.model.encode(trajectory, attention_mask)
        except TypeError:
            encoded = self.model.encode(trajectory)

        if 'skip_connections' in encoded and encoded['skip_connections'] is not None:
            batch_id = id(trajectory)
            self._cached_skip_connections[batch_id] = encoded['skip_connections']

        return encoded

    def decode(self, z_style: torch.Tensor, z_skill: torch.Tensor,
               metadata: Optional[DecodeMetadata] = None) -> torch.Tensor:
        skip_conn = None
        if metadata is not None:
            skip_conn = metadata.get('skip_connections')
            if skip_conn is None and 'batch_id' in metadata:
                skip_conn = self._cached_skip_connections.get(metadata['batch_id'])

        if skip_conn is not None:
            try:
                decoded = self.model.decode(z_style, z_skill, skip_conn)
            except TypeError:
                decoded = self.model.decode(z_style, z_skill)
        else:
            try:
                decoded = self.model.decode(z_style, z_skill, None)
            except TypeError:
                decoded = self.model.decode(z_style, z_skill)

        return decoded['trajectory']

    def get_model_type(self) -> str:
        return 'skip_connection_vae'