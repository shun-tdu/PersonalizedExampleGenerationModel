import os
import yaml
import itertools
from copy import deepcopy


def load_base_config():
    """ベース設定を読み込み"""
    base_config = {
        'config_path': "PredictiveLatentSpaceNavigationModel/hierarchical_vae/configs",
        'data': {
            'data_path': "PredictiveLatentSpaceNavigationModel/DataPreprocess/my_data.parquet"
        },
        'model': {
            'input_dim': 2,
            'seq_len': 100,
            'hidden_dim': 128,
            'primitive_latent_dim': 32,
            'skill_latent_dim': 16,
            'style_latent_dim': 8,
            'beta_primitive': 1.0,
            'beta_skill': 2.0,
            'beta_style': 4.0,
            'precision_lr': 0.1
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 200,
            'lr': 1e-4,
            'weight_decay': 1e-5,
            'clip_grad_norm': 1.0,
            'warmup_epochs': 10,
            'scheduler_T_0': 15,
            'scheduler_T_mult': 2,
            'scheduler_eta_min': 1e-6,
            'patience': 20
        },
        'logging': {
            'output_dir': "PredictiveLatentSpaceNavigationModel/hierarchical_vae/outputs"
        },
        'hierarchical_settings': {
            'primitive_learning_start': 0.0,
            'skill_learning_start': 0.3,
            'style_learning_start': 0.6,
            'prediction_error_weights': {
                'level1': 10.0,
                'level2': 1.0,
                'level3': 0.1
            },
            'exemplar_generation': {
                'skill_enhancement_factor': 0.1,
                'style_preservation_weight': 1.0,
                'max_enhancement_steps': 5
            }
        },
        'evaluation': {
            'disentanglement_metrics': [
                "style_subject_separation",
                "skill_performance_correlation",
                "primitive_reconstruction_quality"
            ],
            'visualization': {
                'pca_components': 2,
                'save_individual_trajectories': True,
                'save_exemplar_comparisons': True
            }
        },
        'experiment': {
            'description': "Hierarchical VAE validation study",
            'tags': ["hierarchical_vae", "validation", "hyperparameter_search"]
        }
    }
    return base_config


def generate_phase1_configs(output_dir):
    """Phase 1: 重要パラメータの感度分析"""
    base_config = load_base_config()
    configs = []

    # 1. β値の階層効果検証
    beta_schedules = [
        ("uniform", [1.0, 1.0, 1.0]),
        ("baseline", [1.0, 2.0, 4.0]),
        ("extreme", [0.5, 2.0, 8.0]),
        ("reverse", [4.0, 2.0, 1.0])
    ]

    for name, (bp, bs, bst) in beta_schedules:
        config = deepcopy(base_config)
        config['model']['beta_primitive'] = bp
        config['model']['beta_skill'] = bs
        config['model']['beta_style'] = bst
        config['experiment']['description'] = f"Beta schedule validation: {name}"
        config['experiment']['tags'].append(f"beta_{name}")

        filename = f"phase1_beta_{name}.yaml"
        configs.append((filename, config))

    # 2. 潜在次元の表現力検証
    latent_dims = [
        ("small", [16, 8, 4]),
        ("baseline", [32, 16, 8]),
        ("large", [64, 32, 16])
    ]

    for name, (prim, skill, style) in latent_dims:
        config = deepcopy(base_config)
        config['model']['primitive_latent_dim'] = prim
        config['model']['skill_latent_dim'] = skill
        config['model']['style_latent_dim'] = style
        config['experiment']['description'] = f"Latent dimension validation: {name}"
        config['experiment']['tags'].append(f"latent_{name}")

        filename = f"phase1_latent_{name}.yaml"
        configs.append((filename, config))

    # 3. 隠れ層次元の影響
    hidden_dims = [64, 128, 256]
    for hidden_dim in hidden_dims:
        config = deepcopy(base_config)
        config['model']['hidden_dim'] = hidden_dim
        config['experiment']['description'] = f"Hidden dimension validation: {hidden_dim}"
        config['experiment']['tags'].append(f"hidden_{hidden_dim}")

        filename = f"phase1_hidden_{hidden_dim}.yaml"
        configs.append((filename, config))

    return save_configs(configs, output_dir)


def generate_phase2_configs(output_dir):
    """Phase 2: 階層学習スケジュールの効果"""
    base_config = load_base_config()
    configs = []

    # 学習スケジュールの比較
    schedules = [
        ("simultaneous", [0.0, 0.0, 0.0]),
        ("gradual", [0.0, 0.3, 0.6]),  # baseline
        ("strict", [0.0, 0.5, 0.8]),
        ("delayed", [0.0, 0.2, 0.9])
    ]

    for name, (prim_start, skill_start, style_start) in schedules:
        config = deepcopy(base_config)
        config['hierarchical_settings']['primitive_learning_start'] = prim_start
        config['hierarchical_settings']['skill_learning_start'] = skill_start
        config['hierarchical_settings']['style_learning_start'] = style_start
        config['experiment']['description'] = f"Learning schedule validation: {name}"
        config['experiment']['tags'].append(f"schedule_{name}")

        filename = f"phase2_schedule_{name}.yaml"
        configs.append((filename, config))

    return save_configs(configs, output_dir)


def generate_phase3_configs(output_dir):
    """Phase 3: 予測誤差重みの最適化"""
    base_config = load_base_config()
    configs = []

    # 予測誤差の重み付け
    error_weights = [
        ("high_precision", [100.0, 10.0, 1.0]),
        ("balanced", [10.0, 1.0, 0.1]),  # baseline
        ("low_precision", [1.0, 0.1, 0.01]),
        ("flat", [1.0, 1.0, 1.0])
    ]

    for name, (level1, level2, level3) in error_weights:
        config = deepcopy(base_config)
        config['hierarchical_settings']['prediction_error_weights']['level1'] = level1
        config['hierarchical_settings']['prediction_error_weights']['level2'] = level2
        config['hierarchical_settings']['prediction_error_weights']['level3'] = level3
        config['experiment']['description'] = f"Prediction error weights: {name}"
        config['experiment']['tags'].append(f"error_weights_{name}")

        filename = f"phase3_error_weights_{name}.yaml"
        configs.append((filename, config))

    # 精度学習率の影響
    precision_lrs = [0.01, 0.1, 0.5]
    for lr in precision_lrs:
        config = deepcopy(base_config)
        config['model']['precision_lr'] = lr
        config['experiment']['description'] = f"Precision learning rate: {lr}"
        config['experiment']['tags'].append(f"precision_lr_{lr}")

        filename = f"phase3_precision_lr_{lr:.2f}.yaml"
        configs.append((filename, config))

    return save_configs(configs, output_dir)


def generate_ablation_configs(output_dir):
    """アブレーション研究"""
    base_config = load_base_config()
    configs = []

    # 1. 階層なしVAE（比較用）
    config = deepcopy(base_config)
    config['model']['beta_primitive'] = 1.0
    config['model']['beta_skill'] = 0.0  # スキル階層を無効化
    config['model']['beta_style'] = 0.0  # スタイル階層を無効化
    config['experiment']['description'] = "Ablation: No hierarchy (standard VAE)"
    config['experiment']['tags'].append("ablation_no_hierarchy")

    filename = "ablation_no_hierarchy.yaml"
    configs.append((filename, config))

    # 2. スタイル保持なしお手本生成
    config = deepcopy(base_config)
    config['hierarchical_settings']['exemplar_generation']['style_preservation_weight'] = 0.0
    config['experiment']['description'] = "Ablation: No style preservation in exemplars"
    config['experiment']['tags'].append("ablation_no_style_preservation")

    filename = "ablation_no_style_preservation.yaml"
    configs.append((filename, config))

    # 3. 予測誤差なし
    config = deepcopy(base_config)
    config['hierarchical_settings']['prediction_error_weights'] = {
        'level1': 0.0, 'level2': 0.0, 'level3': 0.0
    }
    config['experiment']['description'] = "Ablation: No prediction error"
    config['experiment']['tags'].append("ablation_no_prediction_error")

    filename = "ablation_no_prediction_error.yaml"
    configs.append((filename, config))

    return save_configs(configs, output_dir)


def save_configs(configs, output_dir):
    """設定ファイルを保存"""
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    for filename, config in configs:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        saved_files.append(filepath)
        print(f"Saved: {filepath}")

    return saved_files


def generate_all_validation_configs(output_dir="configs_validation"):
    """全ての検証用設定ファイルを生成"""
    print("=== 階層型VAE検証用設定ファイル生成 ===")

    all_configs = []

    print("\n--- Phase 1: 重要パラメータの感度分析 ---")
    phase1_configs = generate_phase1_configs(output_dir)
    all_configs.extend(phase1_configs)

    print("\n--- Phase 2: 階層学習スケジュールの効果 ---")
    phase2_configs = generate_phase2_configs(output_dir)
    all_configs.extend(phase2_configs)

    print("\n--- Phase 3: 予測誤差重みの最適化 ---")
    phase3_configs = generate_phase3_configs(output_dir)
    all_configs.extend(phase3_configs)

    print("\n--- アブレーション研究 ---")
    ablation_configs = generate_ablation_configs(output_dir)
    all_configs.extend(ablation_configs)

    print(f"\n=== 生成完了: {len(all_configs)}個の設定ファイル ===")
    print(f"保存先: {output_dir}")
    print("\n実行方法:")
    print(f"python hierarchical_experiment_manager.py --config_dir {output_dir}")

    return all_configs


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname((os.path.abspath(__file__)))
    CONFIG_DIR = os.path.join(SCRIPT_DIR, 'configs')
    generate_all_validation_configs(CONFIG_DIR)