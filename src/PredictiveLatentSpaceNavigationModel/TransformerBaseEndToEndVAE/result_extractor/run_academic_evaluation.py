# CLAUDE_ADDED
"""
Academic Evaluation Script for Conference Papers
学会論文用評価実行スクリプト
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from result_extractor.evaluation_runner import AcademicEvaluationRunner


def evaluate_top_experiments():
    """Evaluate top performing experiments from previous analysis"""
    print("Academic Evaluation for Conference Paper")
    print("=" * 50)

    # Create runner
    runner = AcademicEvaluationRunner(
        output_dir="academic_results",
        anonymize_subjects=True
    )

    # Top 5 experiments from previous analysis
    top_experiments = [311, 322, 317, 318, 316]

    print(f"Evaluating top {len(top_experiments)} experiments...")
    print(f"Experiment IDs: {top_experiments}")
    print()

    # Evaluate each experiment
    for i, exp_id in enumerate(top_experiments, 1):
        print(f"[{i}/{len(top_experiments)}] Evaluating Experiment {exp_id}...")
        try:
            results = runner.evaluate_experiment(exp_id, experiments_db_path=None)  # Auto-detect database
            metrics = results['summary_metrics']

            print(f"Experiment {exp_id} completed:")
            print(f"   - Reconstruction MSE: {metrics['reconstruction_mse']:.6f}")
            print(f"   - Skill Regression R²: {metrics['skill_regression_r2']:.4f}")
            print(f"   - Best Regression Method: {metrics['best_regression_method']}")
            print(f"   - Number of Subjects: {metrics['n_subjects']}")
            print(f"   - Number of Samples: {metrics['n_samples']}")
            print()
        except Exception as e:
            print(f"Experiment {exp_id} failed: {e}")
            print()

    print("Academic evaluation completed!")
    print(f"Results saved to: academic_results/")
    print()
    print("Generated SVG files:")
    output_dir = Path("academic_results")
    svg_files = list(output_dir.glob("*.svg"))
    for svg_file in sorted(svg_files):
        print(f"   - {svg_file.name}")


def evaluate_single_experiment(experiment_id: int):
    """Evaluate a single experiment"""
    print(f"Academic Evaluation for Experiment {experiment_id}")
    print("=" * 50)

    runner = AcademicEvaluationRunner(
        output_dir=f"academic_results_exp{experiment_id}",
        anonymize_subjects=True
    )

    try:
        results = runner.evaluate_experiment(experiment_id, experiments_db_path=None)  # Auto-detect database
        metrics = results['summary_metrics']

        print(f"Experiment {experiment_id} evaluation completed:")
        print(f"   - Reconstruction MSE: {metrics['reconstruction_mse']:.6f}")
        print(f"   - Skill Regression R²: {metrics['skill_regression_r2']:.4f}")
        print(f"   - Best Regression Method: {metrics['best_regression_method']}")
        print(f"   - Number of Subjects: {metrics['n_subjects']}")
        print(f"   - Number of Samples: {metrics['n_samples']}")
        print()

        # Show regression details
        reg_results = results['skill_regression_results']
        print("Detailed Regression Results:")
        print(f"   - Linear Regression R²: {reg_results['linear_r2']:.4f}")
        print(f"   - SVM Regression R²: {reg_results['svm_r2']:.4f}")
        print(f"   - MLP Regression R²: {reg_results['mlp_r2']:.4f}")
        print()

        print(f"Results saved to: academic_results_exp{experiment_id}/")

    except Exception as e:
        print(f"Experiment {experiment_id} failed: {e}")


def compare_film_vs_cross_attention():
    """Compare FiLM vs Cross Attention models"""
    print("Academic Comparison: FiLM vs Cross Attention")
    print("=" * 50)

    runner = AcademicEvaluationRunner(
        output_dir="academic_comparison_film_vs_crossattn",
        anonymize_subjects=True
    )

    # Best FiLM models and Cross Attention models
    film_experiments = [311, 322, 317]  # Top FiLM models
    cross_attention_experiments = [336, 330, 335]  # Top Cross Attention models

    print("Comparing FiLM vs Cross Attention models...")
    print(f"FiLM models: {film_experiments}")
    print(f"Cross Attention models: {cross_attention_experiments}")
    print()

    all_experiments = film_experiments + cross_attention_experiments
    comparison_results = runner.compare_experiments(all_experiments, experiments_db_path=None)

    print("Comparison Summary:")
    for exp_name, metrics in comparison_results.items():
        if "error" not in metrics:
            exp_id = exp_name.split("_")[1]
            model_type = "FiLM" if int(exp_id) in film_experiments else "Cross Attention"
            print(f"   {exp_name} ({model_type}):")
            print(f"      R²: {metrics['skill_regression_r2']:.4f}")
            print(f"      Reconstruction MSE: {metrics['reconstruction_mse']:.6f}")
        else:
            print(f"   {exp_name}: Error - {metrics['error']}")

    print(f"\nResults saved to: academic_comparison_film_vs_crossattn/")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Academic Evaluation for Conference Papers")
    parser.add_argument("--mode", choices=["top", "single", "compare"], default="single",
                       help="Evaluation mode (default: single)")
    parser.add_argument("--experiment-id", type=int, default=311,
                       help="Experiment ID for single mode (default: 311)")

    args = parser.parse_args()

    if args.mode == "top":
        evaluate_top_experiments()
    elif args.mode == "single":
        evaluate_single_experiment(args.experiment_id)
    elif args.mode == "compare":
        compare_film_vs_cross_attention()


if __name__ == "__main__":
    main()