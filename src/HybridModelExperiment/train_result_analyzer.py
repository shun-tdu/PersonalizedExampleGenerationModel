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
        options=list(all_experiments_df['status'].unique()),
        value=['completed'],
        label = "実験ステータス:"
    )

    # ベスト検証ロスフィルター
    min_loss = all_experiments_df['best_val_loss'].min()
    max_loss = all_experiments_df['best_val_loss'].max()
    loss_filter = mo.ui.slider(
        start=min_loss,
        stop=max_loss,
        value=max_loss,
        label="ベスト検証ロス(これ以下の値を表示):"
    )

    # モデルアーキテクチャフィルター
    model_types = list(all_experiments_df['low_freq_model_type'].unique())
    model_filter = mo.ui.multiselect(
        options=model_types,
        value=model_types,  # デフォルトで全て選択
        label="モデルアーキテクチャ:"
    )
else:
    # データがない場合はダミーのUIを表示
    status_filter, loss_filter, model_filter = mo.md("データがありません"), mo.md(""), mo.md("")

# フィルターを横に並べて表示
mo.hstack([status_filter, loss_filter, model_filter], justify='space-around')

# --- 3. フィルターされたデータの表示 ---
mo.md("## 実験結果一覧")


@mo.capture
def filter_dataframe(df, status, loss, models):
    """UI要素の値に基づいてDataFrameをフィルタリングする"""
    if df.empty:
        return df
    filtered = df[df['status'].isin(status)]
    filtered = filtered[filtered['best_val_loss'] <= loss]
    filtered = filtered[filtered['low_freq_model_type'].isin(models)]
    return filtered


# フィルターの値を使ってデータを絞り込む (ここがmarimoのリアクティブな部分)
filtered_df = filter_dataframe(all_experiments_df, status_filter.value, loss_filter.value, model_filter.value)

# 表示するカラムを定義
display_columns = [
    'id', 'status', 'best_val_loss', 'final_total_loss',
    'low_freq_model_type', 'lr_low_freq', 'batch_size', 'config_file'
]

# データエディタとして表を表示 (ソートや選択が可能)
experiment_table = mo.ui.data_editor(
    filtered_df[display_columns] if not filtered_df.empty else pd.DataFrame(columns=display_columns),
    selection='single'  # 1行だけ選択可能にする
)

experiment_table

# --- 4. パラメータと性能の可視化 ---
mo.md("## ハイパーパラメータと性能の関係")


@mo.capture
def create_scatter_plot(df):
    """インタラクティブな散布図を作成する"""
    if df.empty or len(df) < 2:
        return mo.md("表示するのに十分なデータがありません。")

    fig = px.scatter(
        df,
        x="lr_low_freq",
        y="best_val_loss",
        color="low_freq_model_type",
        log_x=True,  # 学習率は対数スケールで見ることが多い
        log_y=True,  # 損失も対数スケールで見やすい
        hover_data=['id', 'batch_size', 'transformer_layers'],
        title="学習率 vs ベスト検証ロス"
    )
    # PlotlyのグラフをHTMLとして埋め込む
    return mo.as_html(fig.to_html(include_plotlyjs='cdn'))


create_scatter_plot(filtered_df)

# --- 5. 選択した実験の詳細と成果物の表示 ---
mo.md("## 選択した実験の詳細")

# experiment_tableで選択された行を取得
selected_rows = experiment_table.value

if not selected_rows:
    mo.md("上の表から実験を1つ選択すると、詳細と学習曲線プロットが表示されます。")
else:
    # 選択された行のインデックスから、元のDataFrameの完全な情報を取得
    selected_index = selected_rows[0]
    selected_experiment_data = filtered_df.loc[selected_index]

    # 選択された実験の画像パスを取得
    image_path = Path(selected_experiment_data['image_path'])

    # 詳細情報を表示
    mo.md(f"""
    - **実験ID**: {selected_experiment_data['id']}
    - **設定ファイル**: `{selected_experiment_data['config_file']}`
    - **モデル**: `{selected_experiment_data['low_freq_model_type']}`
    - **ベスト検証ロス**: `{selected_experiment_data['best_val_loss']}`
    - **Gitコミット**: `{selected_experiment_data['git_commit_hash']}`
    """)

    # 対応するプロット画像を表示
    if image_path.exists():
        mo.image(src=image_path)
    else:
        mo.md(f"⚠️ **プロット画像が見つかりません:** `{image_path}`")
