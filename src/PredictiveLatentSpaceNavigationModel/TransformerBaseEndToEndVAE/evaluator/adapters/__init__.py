# -*- coding: utf-8 -*-
# CLAUDE_ADDED: Adapter module initialization

from .types import StandardizedBatch, DecodeMetadata
from .data_adapter import DataAdapter, TupleDataAdapter, DictPatchDataAdapter
from .model_adapter import ModelAdapter, StandardVAEAdapter, PatchedVAEAdapter, DiffusionVAEAdapter, SkipConnectionVAEAdapter
from .adapter_factory import AdapterFactory

__all__ = [
    # Types
    'StandardizedBatch',
    'DecodeMetadata',

    # Data Adapters
    'DataAdapter',
    'TupleDataAdapter',
    'DictPatchDataAdapter',

    # Model Adapters
    'ModelAdapter',
    'StandardVAEAdapter',
    'PatchedVAEAdapter',
    'DiffusionVAEAdapter',
    'SkipConnectionVAEAdapter',

    # Factory
    'AdapterFactory',
]