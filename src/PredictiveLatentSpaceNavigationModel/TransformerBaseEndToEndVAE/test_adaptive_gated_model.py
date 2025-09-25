# CLAUDE_ADDED: Quick test script for AdaptiveGatedSkipConnectionNet

import torch
import sys
import os

# Add the models directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from models.adaptive_gated_skip_connection_model import AdaptiveGatedSkipConnectionNet

def test_model_instantiation():
    """Test if the AdaptiveGatedSkipConnectionNet can be instantiated and run a forward pass"""
    print("Testing AdaptiveGatedSkipConnectionNet instantiation...")

    # Create model with basic configuration
    model = AdaptiveGatedSkipConnectionNet(
        input_dim=6,
        seq_len=100,
        d_model=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        dropout=0.1,
        skip_dropout=0.3,
        style_latent_dim=16,
        skill_latent_dim=16,
        n_subjects=6
    )

    print("Model instantiated successfully")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 100, 6)  # [batch, seq_len, input_dim]
    subject_ids = ['h.nakamura', 'r.morishita', 'r.yanase', 's.miyama']
    skill_scores = torch.tensor([0.5, -0.2, 0.8, -0.1])

    print("Testing forward pass...")
    model.eval()
    with torch.no_grad():
        result = model(x, subject_ids, skill_scores)

    print("Forward pass completed successfully")
    print(f"  - Reconstructed shape: {result['reconstructed'].shape}")
    print(f"  - Style latent shape: {result['z_style'].shape}")
    print(f"  - Skill latent shape: {result['z_skill'].shape}")
    print(f"  - Total loss: {result['total_loss'].item():.4f}")

    # Test encode/decode methods
    print("Testing encode/decode methods...")
    encoded = model.encode(x)
    decoded = model.decode(encoded['z_style'], encoded['z_skill'], encoded['skip_connections'])

    print("Encode/decode methods work correctly")
    print(f"  - Decoded trajectory shape: {decoded['trajectory'].shape}")

    # Test gate statistics
    print("Testing gate statistics...")
    for i, gate in enumerate(model.decoder.adaptive_gates):
        stats = gate.get_gate_statistics()
        print(f"  - Gate {i}: mean={stats['mean']:.3f}, entropy={stats['entropy']:.3f}")

    print("All tests passed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_model_instantiation()
        print("\nAdaptiveGatedSkipConnectionNet is ready for training!")
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()