# -*- coding: utf-8 -*-
"""
CLAUDE_ADDED: Type definitions for adapter system
"""

from typing import TypedDict, List, Dict, Any, Optional
import torch

class StandardizedBatch(TypedDict, total=False):
    """Standardized batch data from DataAdapter"""
    trajectory: torch.Tensor
    subject_id: List[str]
    skill_metric: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    metadata: Dict[str, Any]

class DecodeMetadata(TypedDict, total=False):
    """Metadata for decode operation"""
    num_patches: Optional[int]
    attention_mask: Optional[torch.Tensor]
    skip_connections: Optional[Any]
    original_shape: Optional[tuple]
    batch_id: Optional[int]
