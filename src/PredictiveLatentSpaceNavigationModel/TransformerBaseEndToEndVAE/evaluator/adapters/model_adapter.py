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
        print(f"PatchedVAEAdapter: patch_size={self.patch_size}, patch_step={self.patch_step}")

    def encode(self, trajectory: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        return self.model.encode(trajectory, attention_mask)

    def decode(self, z_style: torch.Tensor, z_skill: torch.Tensor,
               metadata: Optional[DecodeMetadata] = None) -> torch.Tensor:
        num_patches = None
        attention_mask = None

        if metadata is not None:
            num_patches = metadata.get('num_patches')
            attention_mask = metadata.get('attention_mask')
            if num_patches is None and 'original_shape' in metadata:
                original_shape = metadata['original_shape']
                if len(original_shape) >= 2:
                    num_patches = original_shape[1]

        if num_patches is None:
            num_patches = 100

        decoded = self.model.decode(z_style, z_skill, num_patches)
        patches = decoded['trajectory']
        trajectory = self._patches_to_trajectory(patches, attention_mask)
        return trajectory

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