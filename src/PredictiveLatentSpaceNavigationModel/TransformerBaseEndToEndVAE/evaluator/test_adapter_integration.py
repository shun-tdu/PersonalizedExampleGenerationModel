# -*- coding: utf-8 -*-
"""
CLAUDE_ADDED: Integration test for adapter system
Verifies that the adapter pattern implementation works correctly
"""

import sys
import torch
import torch.nn as nn
from typing import Dict, Any


def test_imports():
    """Test 1: Verify all adapter imports work"""
    print("=" * 60)
    print("TEST 1: Import Verification")
    print("=" * 60)

    try:
        from adapters import (
            StandardizedBatch, DecodeMetadata,
            DataAdapter, TupleDataAdapter, DictPatchDataAdapter,
            ModelAdapter, StandardVAEAdapter, PatchedVAEAdapter,
            DiffusionVAEAdapter, SkipConnectionVAEAdapter,
            AdapterFactory
        )
        print("[OK] All imports successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_data_adapter_tuple():
    """Test 2: TupleDataAdapter with legacy format"""
    print("\n" + "=" * 60)
    print("TEST 2: TupleDataAdapter")
    print("=" * 60)

    from adapters import TupleDataAdapter

    # Create mock batch data
    trajectory = torch.randn(4, 100, 6)  # [B, seq_len, features]
    subject_ids = ['S01', 'S02', 'S03', 'S04']
    skill_scores = torch.randn(4, 1)

    batch_data = (trajectory, subject_ids, skill_scores)

    adapter = TupleDataAdapter()
    standardized = adapter.extract_batch(batch_data)

    print(f"  Input: tuple with {len(batch_data)} elements")
    print(f"  Output trajectory shape: {standardized['trajectory'].shape}")
    print(f"  Output skill_metric shape: {standardized['skill_metric'].shape}")
    print(f"  Skill metric dim: {adapter.get_skill_metric_dim()}")
    print(f"  Skill metric names: {adapter.get_skill_metric_names()}")

    assert standardized['trajectory'].shape == (4, 100, 6)
    assert standardized['skill_metric'].shape == (4, 1)
    assert adapter.get_skill_metric_dim() == 1
    print("[OK] TupleDataAdapter test passed")
    return True


def test_data_adapter_dict():
    """Test 3: DictPatchDataAdapter with patch format"""
    print("\n" + "=" * 60)
    print("TEST 3: DictPatchDataAdapter")
    print("=" * 60)

    from adapters import DictPatchDataAdapter

    # Create mock patched batch data
    trajectory = torch.randn(4, 10, 20, 6)  # [B, num_patches, patch_size, features]
    subject_ids = ['S01', 'S02', 'S03', 'S04']
    skill_factor = torch.randn(4, 5)  # Multi-dimensional skill factors
    attention_mask = torch.zeros(4, 10, dtype=torch.bool)

    batch_data = {
        'trajectory': trajectory,
        'subject_id': subject_ids,
        'skill_factor': skill_factor,
        'attention_mask': attention_mask
    }

    factor_names = ['factor_1', 'factor_2', 'factor_3', 'factor_4', 'factor_5']
    adapter = DictPatchDataAdapter(factor_names)
    standardized = adapter.extract_batch(batch_data)

    print(f"  Input: dict with patches")
    print(f"  Output trajectory shape: {standardized['trajectory'].shape}")
    print(f"  Output skill_metric shape: {standardized['skill_metric'].shape}")
    print(f"  Skill metric dim: {adapter.get_skill_metric_dim()}")
    print(f"  Skill metric names: {adapter.get_skill_metric_names()}")
    print(f"  Has attention_mask: {standardized['attention_mask'] is not None}")
    print(f"  Metadata is_patched: {standardized['metadata'].get('is_patched')}")

    assert standardized['trajectory'].shape == (4, 10, 20, 6)
    assert standardized['skill_metric'].shape == (4, 5)
    assert adapter.get_skill_metric_dim() == 5
    assert standardized['metadata']['is_patched'] == True
    print("[OK] DictPatchDataAdapter test passed")
    return True


class MockStandardVAE(nn.Module):
    """Mock standard VAE for testing"""
    def __init__(self):
        super().__init__()

    def encode(self, trajectory):
        B = trajectory.shape[0]
        return {
            'z_style': torch.randn(B, 32),
            'z_skill': torch.randn(B, 16)
        }

    def decode(self, z_style, z_skill):
        B = z_style.shape[0]
        return {
            'trajectory': torch.randn(B, 100, 6)
        }


class MockPatchedVAE(nn.Module):
    """Mock patched VAE for testing"""
    def __init__(self):
        super().__init__()
        self.patch_size = 20

    def encode(self, trajectory, attention_mask=None):
        B = trajectory.shape[0]
        return {
            'z_style': torch.randn(B, 32),
            'z_skill': torch.randn(B, 16)
        }

    def decode(self, z_style, z_skill, num_patches):
        B = z_style.shape[0]
        return {
            'trajectory': torch.randn(B, num_patches, self.patch_size, 6)
        }


def test_model_adapter_standard():
    """Test 4: StandardVAEAdapter"""
    print("\n" + "=" * 60)
    print("TEST 4: StandardVAEAdapter")
    print("=" * 60)

    from adapters import StandardVAEAdapter

    model = MockStandardVAE()
    config = {}
    adapter = StandardVAEAdapter(model, config)

    trajectory = torch.randn(4, 100, 6)
    encoded = adapter.encode(trajectory)
    decoded = adapter.decode(encoded['z_style'], encoded['z_skill'])

    print(f"  Encoded z_style shape: {encoded['z_style'].shape}")
    print(f"  Encoded z_skill shape: {encoded['z_skill'].shape}")
    print(f"  Decoded trajectory shape: {decoded.shape}")
    print(f"  Is diffusion model: {adapter.is_diffusion_model()}")
    print(f"  Model type: {adapter.get_model_type()}")

    assert encoded['z_style'].shape == (4, 32)
    assert encoded['z_skill'].shape == (4, 16)
    assert decoded.shape == (4, 100, 6)
    assert adapter.is_diffusion_model() == False
    assert adapter.get_model_type() == 'standard_vae'
    print("[OK] StandardVAEAdapter test passed")
    return True


def test_model_adapter_patched():
    """Test 5: PatchedVAEAdapter"""
    print("\n" + "=" * 60)
    print("TEST 5: PatchedVAEAdapter")
    print("=" * 60)

    from adapters import PatchedVAEAdapter

    model = MockPatchedVAE()
    config = {
        'data': {
            'patch': {
                'size': 20,
                'step': 20
            }
        }
    }
    adapter = PatchedVAEAdapter(model, config)

    trajectory_patches = torch.randn(4, 10, 20, 6)
    encoded = adapter.encode(trajectory_patches)

    metadata = {
        'num_patches': 10,
        'attention_mask': None
    }
    decoded = adapter.decode(encoded['z_style'], encoded['z_skill'], metadata)

    print(f"  Input patches shape: {trajectory_patches.shape}")
    print(f"  Encoded z_style shape: {encoded['z_style'].shape}")
    print(f"  Encoded z_skill shape: {encoded['z_skill'].shape}")
    print(f"  Decoded trajectory shape: {decoded.shape}")
    print(f"  Expected seq_len: {(10-1)*20 + 20}")
    print(f"  Is diffusion model: {adapter.is_diffusion_model()}")
    print(f"  Model type: {adapter.get_model_type()}")

    assert encoded['z_style'].shape == (4, 32)
    assert encoded['z_skill'].shape == (4, 16)
    # seq_len = (num_patches - 1) * step + patch_size = 9*20 + 20 = 200
    assert decoded.shape == (4, 200, 6)
    assert adapter.is_diffusion_model() == False
    assert adapter.get_model_type() == 'patched_vae'
    print("[OK] PatchedVAEAdapter test passed")
    return True


def test_adapter_factory():
    """Test 6: AdapterFactory auto-detection"""
    print("\n" + "=" * 60)
    print("TEST 6: AdapterFactory Auto-Detection")
    print("=" * 60)

    from adapters import AdapterFactory

    # Test 6-1: Tuple data format
    print("\n  Test 6-1: Tuple format detection")
    batch_sample = (
        torch.randn(4, 100, 6),
        ['S01', 'S02', 'S03', 'S04'],
        torch.randn(4, 1)
    )
    data_adapter = AdapterFactory.create_data_adapter(batch_sample)
    print(f"    Detected adapter: {data_adapter.__class__.__name__}")
    assert data_adapter.__class__.__name__ == 'TupleDataAdapter'
    print("    [OK] Tuple format correctly detected")

    # Test 6-2: Dict patch format
    print("\n  Test 6-2: Dict patch format detection")
    batch_sample = {
        'trajectory': torch.randn(4, 10, 20, 6),
        'subject_id': ['S01', 'S02', 'S03', 'S04'],
        'skill_factor': torch.randn(4, 5),
        'attention_mask': torch.zeros(4, 10, dtype=torch.bool)
    }
    data_adapter = AdapterFactory.create_data_adapter(batch_sample)
    print(f"    Detected adapter: {data_adapter.__class__.__name__}")
    assert data_adapter.__class__.__name__ == 'DictPatchDataAdapter'
    print("    [OK] Dict patch format correctly detected")

    # Test 6-3: Standard VAE model
    print("\n  Test 6-3: Standard VAE model detection")
    model = MockStandardVAE()
    config = {}
    model_adapter = AdapterFactory.create_model_adapter(model, config)
    print(f"    Detected adapter: {model_adapter.__class__.__name__}")
    assert model_adapter.__class__.__name__ == 'StandardVAEAdapter'
    print("    [OK] Standard VAE correctly detected")

    # Test 6-4: Patched VAE model
    print("\n  Test 6-4: Patched VAE model detection")
    model = MockPatchedVAE()
    config = {}
    model_adapter = AdapterFactory.create_model_adapter(model, config)
    print(f"    Detected adapter: {model_adapter.__class__.__name__}")
    assert model_adapter.__class__.__name__ == 'PatchedVAEAdapter'
    print("    [OK] Patched VAE correctly detected")

    print("\n[OK] AdapterFactory test passed")
    return True


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("ADAPTER SYSTEM INTEGRATION TEST")
    print("=" * 60 + "\n")

    tests = [
        ("Import Verification", test_imports),
        ("TupleDataAdapter", test_data_adapter_tuple),
        ("DictPatchDataAdapter", test_data_adapter_dict),
        ("StandardVAEAdapter", test_model_adapter_standard),
        ("PatchedVAEAdapter", test_model_adapter_patched),
        ("AdapterFactory", test_adapter_factory),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"[FAIL] {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "[PASSED]" if success else "[FAILED]"
        print(f"  {status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All integration tests passed!")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)