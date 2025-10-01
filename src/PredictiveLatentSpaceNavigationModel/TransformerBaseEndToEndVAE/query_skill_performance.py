# CLAUDE_ADDED
"""
スキル回帰性能の高い実験を検索するスクリプト
"""
import sqlite3
import json
import pandas as pd

def query_skill_regression_performance():
    # データベースに接続
    conn = sqlite3.connect('experiments.db')
    cursor = conn.cursor()

    # FiLMまたはCross Attention関連の実験を取得
    query = '''
    SELECT id, experiment_name, evaluation_results, config_parameters
    FROM transformer_base_e2e_vae_experiment
    WHERE evaluation_results IS NOT NULL
    AND evaluation_results != ''
    AND (experiment_name LIKE '%film%' OR experiment_name LIKE '%cross_attention%')
    AND status = 'completed'
    ORDER BY id
    '''

    cursor.execute(query)
    results = cursor.fetchall()

    print("=== FiLM/Cross Attention実験のスキル回帰性能分析 ===\n")

    experiments_data = []

    for exp_id, exp_name, eval_results_str, config_str in results:
        try:
            # 評価結果をパース
            eval_results = json.loads(eval_results_str) if eval_results_str else {}
            config = json.loads(config_str) if config_str else {}

            # 最初の実験の構造を詳細表示
            if exp_id == results[0][0]:
                print(f"実験ID {exp_id} の評価結果構造:")
                for key, value in eval_results.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for subkey, subvalue in value.items():
                            print(f"    - {subkey}: {type(subvalue).__name__}")
                    else:
                        print(f"  {key}: {type(value).__name__}")
                print()

            # スキル回帰性能を探す（様々なキーパターンを確認）
            skill_r2 = None
            skill_mse = None
            skill_mae = None

            # 評価結果から回帰性能を探索
            for key, value in eval_results.items():
                if 'skill' in key.lower():
                    if isinstance(value, dict):
                        # R2スコアを探す
                        for subkey, subvalue in value.items():
                            if 'r2' in subkey.lower() or 'r_squared' in subkey.lower():
                                skill_r2 = subvalue
                            elif 'mse' in subkey.lower():
                                skill_mse = subvalue
                            elif 'mae' in subkey.lower():
                                skill_mae = subvalue
                    elif 'r2' in key.lower():
                        skill_r2 = value
                    elif 'mse' in key.lower():
                        skill_mse = value
                    elif 'mae' in key.lower():
                        skill_mae = value

            # 直接的なキーも確認
            possible_r2_keys = [
                'skill_regression_r2', 'skill_r2_score', 'skill_score_regression_r2',
                'skill_latent_dimension_vs_score_r2', 'skill_score_r2'
            ]

            for key in possible_r2_keys:
                if key in eval_results and skill_r2 is None:
                    skill_r2 = eval_results[key]
                    break

            if skill_r2 is not None:
                experiments_data.append({
                    'experiment_id': exp_id,
                    'experiment_name': exp_name,
                    'skill_r2': float(skill_r2),
                    'skill_mse': skill_mse,
                    'skill_mae': skill_mae,
                    'model_class': config.get('model_class_name', 'Unknown')
                })

        except Exception as e:
            print(f"Error processing experiment {exp_id}: {e}")
            continue

    # 結果の表示
    if experiments_data:
        df = pd.DataFrame(experiments_data)
        # R2スコアでソート（降順）
        df_sorted = df.sort_values('skill_r2', ascending=False)

        print("=== スキル回帰性能ランキング（Top 5） ===\n")

        top_5 = df_sorted.head(5)
        for rank, (idx, row) in enumerate(top_5.iterrows(), 1):
            print(f"{rank}位: 実験ID {row['experiment_id']}")
            print(f"   実験名: {row['experiment_name']}")
            print(f"   モデル: {row['model_class']}")
            print(f"   スキル回帰 R²: {row['skill_r2']:.4f}")
            if row['skill_mse'] is not None:
                print(f"   スキル回帰 MSE: {row['skill_mse']:.4f}")
            if row['skill_mae'] is not None:
                print(f"   スキル回帰 MAE: {row['skill_mae']:.4f}")
            print()

        print(f"総実験数: {len(df_sorted)}")

        # 全実験のリスト
        print("\n=== 全実験のスキル回帰性能 ===")
        for idx, row in df_sorted.iterrows():
            print(f"ID {row['experiment_id']}: R²={row['skill_r2']:.4f} ({row['experiment_name']})")

    else:
        print("スキル回帰性能データが見つかりませんでした")
        print("\n利用可能な実験:")
        for exp_id, exp_name, _, _ in results:
            print(f"  ID {exp_id}: {exp_name}")

    conn.close()

if __name__ == "__main__":
    query_skill_regression_performance()