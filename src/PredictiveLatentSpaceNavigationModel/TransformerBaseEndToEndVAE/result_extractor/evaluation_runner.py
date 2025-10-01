# CLAUDE_ADDED
"""
Academic Evaluation Runner for Any Model
任意モデル用学会評価実行器
"""
import os
import sys
import torch
import importlib.util
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from datasets.dataloader_factory import DataLoaderFactory
from .svg_evaluator import AcademicSVGEvaluator


class AcademicEvaluationRunner:
    """Runner for academic evaluation of any model"""

    def __init__(self, output_dir: str = "academic_results", anonymize_subjects: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = AcademicSVGEvaluator(output_dir, anonymize_subjects)

    def load_model_from_path(self, model_path: str, model_class_name: str = None,
                           model_file_path: str = None, config: Dict[str, Any] = None) -> torch.nn.Module:
        """Load model from checkpoint file"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Determine model class
        if model_class_name is None or model_file_path is None or config is None:
            raise ValueError("model_class_name, model_file_path, and config must be provided")

        # Import model class dynamically
        model_file_path = Path(model_file_path)
        if not model_file_path.is_absolute():
            # Convert relative path to absolute
            base_dir = Path(__file__).parent.parent
            model_file_path = base_dir / model_file_path

        spec = importlib.util.spec_from_file_location("model_module", model_file_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        model_class = getattr(model_module, model_class_name)

        # Extract model parameters from config
        model_params = {k.replace('model_', ''): v for k, v in config.items() if k.startswith('model_')}

        # Create model instance
        model = model_class(**model_params)

        # Load state dict
        model.load_state_dict(state_dict)

        return model

    def load_model_from_experiment(self, experiment_id: int, experiments_db_path: str = "experiments.db") -> torch.nn.Module:
        """Load model from experiment database"""
        import sqlite3
        import json

        conn = sqlite3.connect(experiments_db_path)
        cursor = conn.cursor()

        # Get experiment details
        cursor.execute('''
            SELECT model_path, config_parameters
            FROM transformer_base_e2e_vae_experiment
            WHERE id = ?
        ''', (experiment_id,))

        result = cursor.fetchone()
        conn.close()

        if not result:
            raise ValueError(f"Experiment {experiment_id} not found in database")

        model_path, config_str = result
        config = json.loads(config_str)

        # Extract model information
        model_class_name = config.get('model_class_name')
        model_file_path = config.get('model_file_path')

        if not model_class_name or not model_file_path:
            raise ValueError(f"Model class or file path not found in experiment {experiment_id}")

        return self.load_model_from_path(model_path, model_class_name, model_file_path, config)

    def create_dataloader(self, data_config: Dict[str, Any] = None) -> DataLoader:
        """Create dataloader for evaluation"""
        if data_config is None:
            # Default configuration
            data_config = {
                'data_type': 'skill_metrics',
                'data_data_path': 'PredictiveLatentSpaceNavigationModel/DataPreprocess/AnalysisResults/Dataset_Generation_Test_20250908_193217/dataset',
                'data_val_split': 0.2,
                'data_random_seed': 42,
                'data_num_workers': 4,
                'data_pin_memory': True,
                'data_shuffle': False,  # Don't shuffle for consistent evaluation
                'training_batch_size': 32
            }

        try:
            # Create dataloader factory
            factory = DataLoaderFactory(data_config)

            # Get test dataloader
            _, _, test_loader = factory.create_dataloaders()

            return test_loader

        except Exception as e:
            print(f"Warning: Could not create dataloader with provided config: {e}")
            print("Please ensure the dataset path is correct and accessible.")
            raise

    def evaluate_experiment(self, experiment_id: int, experiments_db_path: str = "experiments.db",
                          data_config: Dict[str, Any] = None, device: str = "auto") -> Dict[str, Any]:
        """Evaluate a specific experiment by ID"""
        print(f"Evaluating experiment {experiment_id}...")

        # Setup device
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        # Load model
        model = self.load_model_from_experiment(experiment_id, experiments_db_path)
        model.to(device)
        model.eval()

        # Create dataloader
        test_loader = self.create_dataloader(data_config)

        # Run evaluation
        results = self.evaluator.evaluate_model(
            model, test_loader, device,
            experiment_name=f"experiment_{experiment_id}"
        )

        # Save summary
        summary_path = self.output_dir / f"experiment_{experiment_id}_summary.json"
        with open(summary_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            summary = {}
            for k, v in results['summary_metrics'].items():
                if hasattr(v, 'item'):  # numpy scalar
                    summary[k] = v.item()
                else:
                    summary[k] = v

            json.dump({
                'experiment_id': experiment_id,
                'summary_metrics': summary,
                'skill_regression_results': results['skill_regression_results']
            }, f, indent=2)

        print(f"Evaluation completed. Results saved to {self.output_dir}")
        return results

    def evaluate_model_file(self, model_path: str, model_class_name: str,
                          model_file_path: str, model_config: Dict[str, Any],
                          data_config: Dict[str, Any] = None,
                          experiment_name: str = "custom_model",
                          device: str = "auto") -> Dict[str, Any]:
        """Evaluate a model from file path"""
        print(f"Evaluating model from {model_path}...")

        # Setup device
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        # Load model
        model = self.load_model_from_path(model_path, model_class_name, model_file_path, model_config)
        model.to(device)
        model.eval()

        # Create dataloader
        test_loader = self.create_dataloader(data_config)

        # Run evaluation
        results = self.evaluator.evaluate_model(model, test_loader, device, experiment_name)

        # Save summary
        summary_path = self.output_dir / f"{experiment_name}_summary.json"
        with open(summary_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            summary = {}
            for k, v in results['summary_metrics'].items():
                if hasattr(v, 'item'):  # numpy scalar
                    summary[k] = v.item()
                else:
                    summary[k] = v

            json.dump({
                'experiment_name': experiment_name,
                'summary_metrics': summary,
                'skill_regression_results': results['skill_regression_results']
            }, f, indent=2)

        print(f"Evaluation completed. Results saved to {self.output_dir}")
        return results

    def compare_experiments(self, experiment_ids: List[int],
                          experiments_db_path: str = "experiments.db") -> Dict[str, Any]:
        """Compare multiple experiments"""
        print(f"Comparing experiments: {experiment_ids}")

        comparison_results = {}

        for exp_id in experiment_ids:
            try:
                results = self.evaluate_experiment(exp_id, experiments_db_path)
                comparison_results[f"experiment_{exp_id}"] = results['summary_metrics']
                print(f"✅ Experiment {exp_id} completed")
            except Exception as e:
                print(f"❌ Experiment {exp_id} failed: {e}")
                comparison_results[f"experiment_{exp_id}"] = {"error": str(e)}

        # Save comparison summary
        comparison_path = self.output_dir / "comparison_summary.json"
        with open(comparison_path, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = {}
            for exp_name, metrics in comparison_results.items():
                if "error" not in metrics:
                    serializable_metrics = {}
                    for k, v in metrics.items():
                        if hasattr(v, 'item'):
                            serializable_metrics[k] = v.item()
                        else:
                            serializable_metrics[k] = v
                    serializable_results[exp_name] = serializable_metrics
                else:
                    serializable_results[exp_name] = metrics

            json.dump(serializable_results, f, indent=2)

        print(f"Comparison completed. Results saved to {comparison_path}")
        return comparison_results


def main():
    """Command line interface for academic evaluation"""
    import argparse

    parser = argparse.ArgumentParser(description="Academic SVG Evaluation Tool")
    parser.add_argument("--mode", choices=["experiment", "model", "compare"], required=True,
                       help="Evaluation mode")
    parser.add_argument("--experiment-id", type=int, help="Experiment ID to evaluate")
    parser.add_argument("--experiment-ids", type=int, nargs="+", help="Multiple experiment IDs for comparison")
    parser.add_argument("--model-path", help="Path to model checkpoint")
    parser.add_argument("--model-class", help="Model class name")
    parser.add_argument("--model-file", help="Path to model definition file")
    parser.add_argument("--model-config", help="Path to model config JSON file")
    parser.add_argument("--output-dir", default="academic_results", help="Output directory")
    parser.add_argument("--db-path", default="experiments.db", help="Path to experiments database")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--anonymize", action="store_true", default=True, help="Anonymize subject names")

    args = parser.parse_args()

    # Create runner
    runner = AcademicEvaluationRunner(args.output_dir, args.anonymize)

    if args.mode == "experiment":
        if not args.experiment_id:
            print("Error: --experiment-id is required for experiment mode")
            return
        runner.evaluate_experiment(args.experiment_id, args.db_path, device=args.device)

    elif args.mode == "compare":
        if not args.experiment_ids:
            print("Error: --experiment-ids is required for compare mode")
            return
        runner.compare_experiments(args.experiment_ids, args.db_path)

    elif args.mode == "model":
        if not all([args.model_path, args.model_class, args.model_file, args.model_config]):
            print("Error: --model-path, --model-class, --model-file, and --model-config are required for model mode")
            return

        with open(args.model_config, 'r') as f:
            model_config = json.load(f)

        runner.evaluate_model_file(
            args.model_path, args.model_class, args.model_file, model_config,
            experiment_name="custom_model", device=args.device
        )


if __name__ == "__main__":
    main()