# 簡易版ベストモデル選択
import os
import sqlite3
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'hierarchical_experiments.db')


def filter_viable_models(db_path):
    """実用的なモデルの候補を絞り込み"""

    with sqlite3.connect(db_path) as conn:
        query = """
                SELECT id, \
                       experiment_name, \
                       reconstruction_mse,
                       style_separation_score, \
                       best_val_loss, \
                       final_epoch
                FROM hierarchical_experiments
                WHERE status = 'completed'
                  AND reconstruction_mse IS NOT NULL
                  AND reconstruction_mse < 0.5 -- 基本的な再構成品質
                  AND final_epoch >= 30        -- 最低限の学習
                ORDER BY reconstruction_mse ASC \
                """

        df = pd.read_sql_query(query, conn)

        print("実用可能なモデル候補:")
        print(df.to_string(index=False))

        return df

def calculate_composite_score(candidates_df):
    """複合スコアでモデルを評価"""

    df = candidates_df.copy()

    # 正規化（0-1スケール）
    df['recon_score'] = 1.0 / (1.0 + df['reconstruction_mse'])  # 低いほど良い → 高いスコア
    df['style_score'] = df['style_separation_score'].fillna(0)  # 高いほど良い
    df['stability_score'] = 1.0 / (1.0 + df['best_val_loss'])  # 低いほど良い → 高いスコア

    # 重み付け合計
    df['composite_score'] = (
            df['recon_score'] * 0.5 +  # 再構成品質 50%
            df['style_score'] * 0.3 +  # スタイル分離 30%
            df['stability_score'] * 0.2  # 学習安定性 20%
    )

    # ランキング
    df_ranked = df.sort_values('composite_score', ascending=False)

    print("\nモデルランキング (複合スコア):")
    print(df_ranked[['id', 'experiment_name', 'composite_score',
                     'recon_score', 'style_score', 'stability_score']].to_string(index=False))

    return df_ranked

def validate_best_model(model_id, db_path):
    """ベストモデルの最終検証"""

    with sqlite3.connect(db_path) as conn:
        query = """
                SELECT model_path, \
                       config_path, \
                       reconstruction_mse,
                       style_separation_score, \
                       notes
                FROM hierarchical_experiments
                WHERE id = ? \
                """

        result = conn.execute(query, (model_id,)).fetchone()

        if result:
            model_path, config_path, mse, style_score, notes = result

            print(f"\n選択されたベストモデル (ID: {model_id}):")
            print(f"  モデルパス: {model_path}")
            print(f"  設定ファイル: {config_path}")
            print(f"  再構成MSE: {mse:.6f}")
            print(f"  スタイル分離: {style_score if style_score else 'N/A'}")
            print(f"  備考: {notes if notes else 'なし'}")

            # ファイル存在確認
            if model_path and os.path.exists(model_path):
                print("  ✅ モデルファイル確認済み")
            else:
                print("  ❌ モデルファイルが見つかりません")
                return None

            return {
                'id': model_id,
                'model_path': model_path,
                'config_path': config_path
            }
        else:
            print(f"モデルID {model_id} が見つかりません")
            return None

def select_best_model_for_diagnosis(db_path):
    """スタイル診断に最適なモデルを選択"""

    print("=" * 50)
    print("ベストモデル選択（スタイル診断用）")
    print("=" * 50)

    # Step 1: 候補絞り込み
    candidates = filter_viable_models(db_path)

    if len(candidates) == 0:
        print("❌ 実用可能なモデルが見つかりません")
        return None

    # Step 2: 複合評価
    ranked = calculate_composite_score(candidates)

    # Step 3: Top 3 を表示
    top3 = ranked.head(3)
    print(f"\nTOP 3 候補:")
    for i, row in top3.iterrows():
        print(f"  {i + 1}. ID {int(row['id'])}: {row['experiment_name']} "
              f"(スコア: {row['composite_score']:.3f})")

    # Step 4: 最終選択
    best_id = int(top3.iloc[0]['id'])
    best_info = validate_best_model(best_id, db_path)

    if best_info:
        print(f"\n✅ ベストモデル決定: ID {best_id}")
        return best_info
    else:
        print(f"\n❌ ベストモデルの検証に失敗")
        return None


# 実行例
best_model = select_best_model_for_diagnosis(DB_PATH)

if best_model:
    print(f"\nスタイル診断実行:")
    print(f"python style_diagnosis.py \\")
    print(f"  --model_path {best_model['model_path']} \\")
    print(f"  --data_path your_data.parquet \\")
    print(f"  --config_path {best_model['config_path']}")
