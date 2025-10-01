# Academic Evaluation System for Conference Papers

This package provides a comprehensive evaluation system for generating high-quality SVG figures suitable for academic conference papers. All outputs use Times New Roman font and English labels, with subject names anonymized (Subject1, Subject2, etc.).

## Features

- **High-quality SVG output** for publication-ready figures
- **Times New Roman font** for academic consistency
- **Subject anonymization** for privacy protection
- **Comprehensive evaluation** including:
  - Style latent space visualization (PCA & t-SNE)
  - Skill latent space visualization (PCA & t-SNE)
  - Skill regression performance analysis
  - Trajectory reconstruction analysis
  - Sample trajectory comparisons

## Quick Start

### 1. Evaluate Top Performing Experiments
```bash
cd result_extractor
python run_academic_evaluation.py --mode top
```

### 2. Evaluate a Single Experiment
```bash
python run_academic_evaluation.py --mode single --experiment-id 311
```

### 3. Compare FiLM vs Cross Attention Models
```bash
python run_academic_evaluation.py --mode compare
```

## Advanced Usage

### Evaluate Specific Experiment by ID
```python
from result_extractor import AcademicEvaluationRunner

runner = AcademicEvaluationRunner(output_dir="my_results")
results = runner.evaluate_experiment(experiment_id=311)
```

### Evaluate Custom Model
```python
runner = AcademicEvaluationRunner(output_dir="custom_results")

results = runner.evaluate_model_file(
    model_path="path/to/model.pth",
    model_class_name="SimpleFiLMAdaptiveGateNet",
    model_file_path="models/Skip/simple_film_adaptive_gate_model.py",
    model_config={
        "input_dim": 6,
        "d_model": 512,
        "n_heads": 8,
        # ... other model parameters
    }
)
```

### Compare Multiple Experiments
```python
runner = AcademicEvaluationRunner(output_dir="comparison_results")
comparison = runner.compare_experiments([311, 322, 317, 318, 316])
```

## Command Line Interface

The evaluation runner also provides a command line interface:

```bash
# Evaluate single experiment
python evaluation_runner.py --mode experiment --experiment-id 311

# Compare experiments
python evaluation_runner.py --mode compare --experiment-ids 311 322 317

# Evaluate custom model
python evaluation_runner.py --mode model \
    --model-path model.pth \
    --model-class SimpleFiLMAdaptiveGateNet \
    --model-file models/Skip/simple_film_adaptive_gate_model.py \
    --model-config config.json
```

## Output Files

For each evaluation, the following SVG files are generated:

- `{experiment_name}_style_space.svg` - Style latent space visualization
- `{experiment_name}_skill_space.svg` - Skill latent space visualization
- `{experiment_name}_skill_regression.svg` - Skill regression analysis
- `{experiment_name}_reconstruction.svg` - Reconstruction error analysis
- `{experiment_name}_trajectory_samples.svg` - Sample trajectory comparisons
- `{experiment_name}_summary.json` - Numerical results summary

## Configuration

### Subject Anonymization
By default, subject names are anonymized as Subject1, Subject2, etc. To disable:

```python
runner = AcademicEvaluationRunner(anonymize_subjects=False)
```

### Custom Data Configuration
```python
data_config = {
    'data_type': 'skill_metrics',
    'data_data_path': 'path/to/dataset',
    'data_val_split': 0.2,
    'training_batch_size': 32
}

runner.evaluate_experiment(311, data_config=data_config)
```

## Requirements

- torch
- matplotlib
- scikit-learn
- seaborn
- numpy
- pandas

## Academic Paper Integration

The generated SVG files are optimized for academic papers:

- **Vector format**: Scalable without quality loss
- **Times New Roman font**: Standard academic font
- **High DPI**: 300 DPI for publication quality
- **Consistent styling**: Professional color schemes
- **Anonymized data**: Subject privacy protection

Simply include the SVG files in your LaTeX document:

```latex
\begin{figure}[ht]
    \centering
    \includesvg[width=0.8\textwidth]{experiment_311_style_space.svg}
    \caption{Style latent space visualization showing subject clustering.}
    \label{fig:style_space}
\end{figure}
```

## Example Results

The evaluation system provides quantitative metrics including:

- **Reconstruction MSE**: Overall trajectory reconstruction quality
- **Skill Regression R²**: How well skill scores can be predicted from latent representations
- **Best Regression Method**: Optimal regression approach (Linear/SVM/MLP)
- **Subject Count**: Number of unique subjects in dataset
- **Sample Count**: Total number of trajectory samples

## Troubleshooting

### Missing Dependencies
```bash
pip install torch matplotlib scikit-learn seaborn numpy pandas
```

### Dataset Path Issues
Ensure the dataset path is correct in the configuration. The default path is:
```
PredictiveLatentSpaceNavigationModel/DataPreprocess/AnalysisResults/Dataset_Generation_Test_20250908_193217/dataset
```

### Model Loading Errors
Verify that:
- Model checkpoint file exists
- Model class name matches the actual class
- Model file path is correct
- All required model parameters are provided