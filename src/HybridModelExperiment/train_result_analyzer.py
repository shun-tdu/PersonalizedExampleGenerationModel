import marimo as mo
import pandas as pd
import sqlite3
from pathlib import Path
import plotly.express as px

# 1. セットアップとデータ読み込み
mo.md("# 実験結果解析ダッシュボード")

# データベースへのパス
DB_PATH = Path("./experiments.db")

@mo.chache
def load_data():
    """データベースから実験データを読み込み，DataFrameとして返す"""

    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        # statusがcompletedのデータのみを対象とする
        df = pd.read_sql_query("SELECT * FROM experiments", conn)

    # 不要なカラムや表示を整える
    if not df.empty:
        df['start_time'] = pd.to_datetime(df['start_time'])
        df = df.round(5)
    return df

# データをロード
all_experiments_df = load_data()

mo.md(f"データベース '{DB_PATH}' から，合計 **{len(all_experiments_df)}** 件の実験データを読み込みました．")

# 2. インタラクティブなフィルターの定義
mo.md("## フィルター")

# all_experiments_dfが空でない場合のみフィルターを作成
if not all_experiments_df.empty:
    # ステータスフィルター
    status_filter = mo.ui.multiselect(

    )