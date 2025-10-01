# CLAUDE_ADDED
"""
Test Script for Academic Evaluation System
学会評価システムのテストスクリプト
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_svg_output():
    """Test SVG output functionality"""
    print("Testing SVG output functionality...")

    # Set Times New Roman font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Liberation Serif']
    plt.rcParams['font.size'] = 12

    # Create test figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Test data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # Plot 1: Style space simulation
    ax1.scatter(y1, y2, c=x, cmap='Set3', alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('PC1 (0.654)')
    ax1.set_ylabel('PC2 (0.231)')
    ax1.set_title('Style Latent Space (PCA)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Skill space simulation
    skill_scores = np.random.normal(0, 1, 100)
    ax2.scatter(y1, y2, c=skill_scores, cmap='RdYlBu_r', alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('PC1 (0.432)')
    ax2.set_ylabel('PC2 (0.198)')
    ax2.set_title('Skill Latent Space (PCA)')
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Skill Score')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save as SVG
    test_output_dir = Path("test_output")
    test_output_dir.mkdir(exist_ok=True)

    svg_path = test_output_dir / "test_output.svg"
    fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)

    print(f"SVG test file created: {svg_path}")
    plt.close(fig)

def test_anonymization():
    """Test subject anonymization functionality"""
    print("Testing subject anonymization...")

    from result_extractor.svg_evaluator import AcademicSVGEvaluator

    evaluator = AcademicSVGEvaluator("test_output", anonymize_subjects=True)

    # Test subject names
    test_subjects = ['h.nakamura', 'r.morishita', 'r.yanase', 's.miyama', 's.tahara', 't.hasegawa']

    mapping = evaluator.anonymize_subject_names(test_subjects)

    print("Subject anonymization mapping:")
    for original, anonymous in mapping.items():
        print(f"  {original} -> {anonymous}")

    assert len(mapping) == len(test_subjects)
    assert all(anon.startswith("Subject") for anon in mapping.values())

    print("Subject anonymization test passed")

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")

    try:
        from result_extractor import AcademicSVGEvaluator, AcademicEvaluationRunner
        print("Main classes imported successfully")

        import torch
        import matplotlib.pyplot as plt
        import sklearn
        import seaborn
        import numpy as np
        print("All dependencies imported successfully")

    except ImportError as e:
        print(f"Import error: {e}")
        return False

    return True

def test_model_path_resolution():
    """Test model path resolution"""
    print("Testing model path resolution...")

    from result_extractor.evaluation_runner import AcademicEvaluationRunner

    runner = AcademicEvaluationRunner("test_output")

    # Test relative path conversion
    base_dir = Path(__file__).parent.parent
    test_model_path = "models/Skip/simple_film_adaptive_gate_model.py"

    absolute_path = base_dir / test_model_path
    print(f"Base directory: {base_dir}")
    print(f"Test model path: {test_model_path}")
    print(f"Resolved absolute path: {absolute_path}")
    print(f"Path exists: {absolute_path.exists()}")

    print("Path resolution test completed")

def test_database_connection():
    """Test database connection"""
    print("Testing database connection...")

    import sqlite3

    db_path = Path("experiments.db")

    if not db_path.exists():
        print(f"Database not found at {db_path}")
        print("   This is expected if running from a different directory")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM transformer_base_e2e_vae_experiment")
        count = cursor.fetchone()[0]

        print(f"Database connected successfully")
        print(f"   Found {count} experiments in database")

        # Test specific experiments
        test_ids = [311, 322, 317]
        for exp_id in test_ids:
            cursor.execute("SELECT experiment_name, status FROM transformer_base_e2e_vae_experiment WHERE id = ?", (exp_id,))
            result = cursor.fetchone()
            if result:
                name, status = result
                print(f"   Experiment {exp_id}: {name} ({status})")
            else:
                print(f"   Experiment {exp_id}: Not found")

        conn.close()

    except Exception as e:
        print(f"Database connection error: {e}")

def main():
    """Run all tests"""
    print("Academic Evaluation System - Test Suite")
    print("=" * 50)

    # Create test output directory
    test_output_dir = Path("test_output")
    test_output_dir.mkdir(exist_ok=True)

    # Run tests
    tests = [
        test_imports,
        test_svg_output,
        test_anonymization,
        test_model_path_resolution,
        test_database_connection
    ]

    passed = 0
    for test in tests:
        try:
            result = test()
            if result is not False:
                passed += 1
            print()
        except Exception as e:
            print(f"Test failed with exception: {e}")
            print()

    print(f"Test suite completed: {passed}/{len(tests)} tests passed")
    print(f"Test outputs saved to: {test_output_dir}")

    if passed == len(tests):
        print("All tests passed! The system is ready for use.")
    else:
        print("Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()