# -*- coding: utf-8 -*-
"""
CLAUDE_ADDED: AdapterFactory
Automatically selects appropriate adapters based on data and model
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn

from .data_adapter import DataAdapter, TupleDataAdapter, DictPatchDataAdapter
from .model_adapter import (
    ModelAdapter,
    StandardVAEAdapter,
    PatchedVAEAdapter,
    DiffusionVAEAdapter,
    SkipConnectionVAEAdapter
)


class AdapterFactory:
    """Factory for automatic adapter selection"""

    @staticmethod
    def create_data_adapter(batch_sample: Any, config: Optional[Dict[str, Any]] = None) -> DataAdapter:
        """Auto-select DataAdapter from batch sample format"""
        print("=" * 60)
        print("DataAdapter auto-detection...")

        if isinstance(batch_sample, dict):
            print("  Detected: dict format")
            if 'attention_mask' in batch_sample:
                print("  -> DictPatchDataAdapter (with attention_mask)")
                factor_names = config.get('factor_names') if config else None
                return DictPatchDataAdapter(factor_names)
            else:
                print("  -> DictPatchDataAdapter (default)")
                return DictPatchDataAdapter()

        elif isinstance(batch_sample, (tuple, list)):
            print("  Detected: tuple/list format")
            print("  -> TupleDataAdapter")
            return TupleDataAdapter()

        else:
            raise ValueError(f"Unknown batch data format: {type(batch_sample)}")

    @staticmethod
    def create_model_adapter(model: nn.Module, config: Dict[str, Any]) -> ModelAdapter:
        """Auto-select ModelAdapter from model characteristics"""
        print("=" * 60)
        print("ModelAdapter auto-detection...")

        # 1. Diffusion model detection
        if hasattr(model, 'sample') and hasattr(model, 'num_timesteps'):
            print("  Detected: diffusion model")
            print("  -> DiffusionVAEAdapter")
            return DiffusionVAEAdapter(model, config)

        # 2. Patched model detection
        if hasattr(model, 'patch_size'):
            print("  Detected: patched model (model.patch_size)")
            print("  -> PatchedVAEAdapter")
            return PatchedVAEAdapter(model, config)

        data_config = config.get('data', {})
        if 'patch' in data_config or data_config.get('type', '').endswith('no_interpolate'):
            print("  Detected: patched model (config.data.patch)")
            print("  -> PatchedVAEAdapter")
            return PatchedVAEAdapter(model, config)

        # 3. Skip connection detection
        if AdapterFactory._has_skip_connections(model):
            print("  Detected: skip connection model")
            print("  -> SkipConnectionVAEAdapter")
            return SkipConnectionVAEAdapter(model, config)

        # 4. Standard VAE
        print("  Detected: standard VAE")
        print("  -> StandardVAEAdapter")
        return StandardVAEAdapter(model, config)

    @staticmethod
    def _has_skip_connections(model: nn.Module) -> bool:
        """Test if model has skip connections"""
        if not hasattr(model, 'encode'):
            return False

        try:
            test_input = torch.randn(1, 100, 6)
            model.eval()
            with torch.no_grad():
                test_output = model.encode(test_input)

            if isinstance(test_output, dict):
                if 'skip_connections' in test_output:
                    if test_output['skip_connections'] is not None:
                        print("    (Test result: skip_connections detected)")
                        return True
        except Exception as e:
            print(f"    (Skip connection test failed: {e})")
            return False

        return False

    @staticmethod
    def auto_detect_adapters(
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any]
    ) -> Tuple[DataAdapter, ModelAdapter]:
        """Auto-detect both DataAdapter and ModelAdapter"""
        print("\n" + "=" * 60)
        print("Adapter auto-detection started")
        print("=" * 60)

        try:
            batch_sample = next(iter(test_loader))
            print(f"Batch sample obtained: type={type(batch_sample)}")
        except Exception as e:
            raise RuntimeError(f"Failed to get batch from dataloader: {e}")

        data_adapter = AdapterFactory.create_data_adapter(batch_sample, config)
        model_adapter = AdapterFactory.create_model_adapter(model, config)

        print("\n" + "=" * 60)
        print("Adapter auto-detection completed")
        print(f"  DataAdapter: {data_adapter.__class__.__name__}")
        print(f"  ModelAdapter: {model_adapter.__class__.__name__}")
        print("=" * 60 + "\n")

        return data_adapter, model_adapter

    @staticmethod
    def create_adapters_from_types(
        data_adapter_type: str,
        model_adapter_type: str,
        model: nn.Module,
        config: Dict[str, Any]
    ) -> Tuple[DataAdapter, ModelAdapter]:
        """Create adapters by explicitly specifying types"""
        data_adapter_map = {
            'tuple': TupleDataAdapter,
            'dict_patch': DictPatchDataAdapter,
        }

        model_adapter_map = {
            'standard': StandardVAEAdapter,
            'patched': PatchedVAEAdapter,
            'diffusion': DiffusionVAEAdapter,
            'skip_connection': SkipConnectionVAEAdapter,
        }

        if data_adapter_type not in data_adapter_map:
            raise ValueError(
                f"Unknown DataAdapter type: {data_adapter_type}\n"
                f"Available: {list(data_adapter_map.keys())}"
            )

        if model_adapter_type not in model_adapter_map:
            raise ValueError(
                f"Unknown ModelAdapter type: {model_adapter_type}\n"
                f"Available: {list(model_adapter_map.keys())}"
            )

        data_adapter_class = data_adapter_map[data_adapter_type]
        model_adapter_class = model_adapter_map[model_adapter_type]

        if data_adapter_type == 'dict_patch':
            factor_names = config.get('factor_names')
            data_adapter = data_adapter_class(factor_names)
        else:
            data_adapter = data_adapter_class()

        model_adapter = model_adapter_class(model, config)

        print(f"Adapters created explicitly")
        print(f"  DataAdapter: {data_adapter.__class__.__name__}")
        print(f"  ModelAdapter: {model_adapter.__class__.__name__}")

        return data_adapter, model_adapter