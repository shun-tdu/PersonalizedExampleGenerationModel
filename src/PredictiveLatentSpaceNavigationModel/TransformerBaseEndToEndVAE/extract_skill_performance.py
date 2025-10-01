# CLAUDE_ADDED
"""
FiLM/Cross Attention実験のスキル回帰性能を抽出・ランキング
"""
import os
import re
import sqlite3
import json
from pathlib import Path

def extract_skill_performance_from_html(html_path):
    """HTMLレポートからスキル回帰性能を抽出"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # skill_best_regression_r2 を探す（最も重要な指標）
        skill_r2_pattern = r'<div class="metric-name">skill_best_regression_r2</div>\s*<div class="metric-value">([0-9.-]+)</div>'
        match = re.search(skill_r2_pattern, content)

        if match:
            return float(match.group(1))

        # 代替として skill_mlp_r2 を探す
        skill_mlp_pattern = r'<div class="metric-name">skill_mlp_r2</div>\s*<div class="metric-value">([0-9.-]+)</div>'
        match = re.search(skill_mlp_pattern, content)

        if match:
            return float(match.group(1))

        return None

    except Exception as e:
        print(f"Error reading {html_path}: {e}")
        return None

def get_experiment_info_from_db(exp_id):
    """データベースから実験情報を取得"""
    conn = sqlite3.connect('experiments.db')
    cursor = conn.cursor()

    query = '''
    SELECT experiment_name, config_parameters
    FROM transformer_base_e2e_vae_experiment
    WHERE id = ?
    '''

    cursor.execute(query, (exp_id,))
    result = cursor.fetchone()

    if result:
        exp_name, config_str = result
        config = {}
        if config_str:
            try:
                config = json.loads(config_str)
            except:
                pass

        return {
            'experiment_name': exp_name,
            'model_class': config.get('model_class_name', 'Unknown'),
            'skip_layers': config.get('model_skip_layers', 'Unknown'),
            'd_model': config.get('model_d_model', 'Unknown'),
            'n_heads': config.get('model_n_heads', 'Unknown')
        }

    conn.close()
    return None

def main():
    print("=== FiLM/Cross Attention Skill Regression Performance Ranking ===\n")

    # film関連の評価レポートを探す
    film_reports = []

    # outputs/film_gated_deep_network_fixed/reports/
    film_fixed_dir = Path("outputs/film_gated_deep_network_fixed/reports/")
    if film_fixed_dir.exists():
        for html_file in film_fixed_dir.glob("evaluation_report_exp*.html"):
            exp_id = int(re.search(r'exp(\d+)', html_file.name).group(1))
            film_reports.append((exp_id, str(html_file)))

    # outputs/film_gated_deep_network_test/reports/
    film_test_dir = Path("outputs/film_gated_deep_network_test/reports/")
    if film_test_dir.exists():
        for html_file in film_test_dir.glob("evaluation_report_exp*.html"):
            exp_id = int(re.search(r'exp(\d+)', html_file.name).group(1))
            film_reports.append((exp_id, str(html_file)))

    # スキル回帰性能を抽出
    experiment_results = []

    for exp_id, report_path in film_reports:
        # スキル回帰性能を抽出
        skill_r2 = extract_skill_performance_from_html(report_path)

        if skill_r2 is not None:
            # データベースから実験情報を取得
            exp_info = get_experiment_info_from_db(exp_id)

            if exp_info:
                experiment_results.append({
                    'experiment_id': exp_id,
                    'skill_r2': skill_r2,
                    'experiment_name': exp_info['experiment_name'],
                    'model_class': exp_info['model_class'],
                    'skip_layers': exp_info['skip_layers'],
                    'd_model': exp_info['d_model'],
                    'n_heads': exp_info['n_heads']
                })

    # スキル回帰性能でソート（降順）
    experiment_results.sort(key=lambda x: x['skill_r2'], reverse=True)

    # Top 5を表示
    print("=== Top 5 Skill Regression Performance ===\n")

    for rank, result in enumerate(experiment_results[:5], 1):
        print(f"Rank {rank}: Experiment ID {result['experiment_id']}")
        print(f"   Name: {result['experiment_name']}")
        print(f"   Model: {result['model_class']}")
        print(f"   Skip layers: {result['skip_layers']}")
        print(f"   d_model: {result['d_model']}, n_heads: {result['n_heads']}")
        print(f"   Skill regression R2: {result['skill_r2']:.4f}")
        print()

    # All results summary
    print(f"=== All Results ({len(experiment_results)} experiments) ===\n")

    for result in experiment_results:
        print(f"ID {result['experiment_id']:3d}: R2={result['skill_r2']:.4f} "
              f"(skip={result['skip_layers']}) {result['experiment_name']}")

    # Statistics
    if experiment_results:
        skill_r2_values = [r['skill_r2'] for r in experiment_results]
        print(f"\n=== Statistics ===")
        print(f"Best performance: {max(skill_r2_values):.4f}")
        print(f"Worst performance: {min(skill_r2_values):.4f}")
        print(f"Average performance: {sum(skill_r2_values)/len(skill_r2_values):.4f}")
        print(f"Number of experiments: {len(experiment_results)}")

if __name__ == "__main__":
    main()