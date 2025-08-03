import marimo as mo
import pandas as pd

mo.md(
    """
    # 🐳 Marimo Docker 接続テスト
    このページが表示され、下のスライダーを動かして表が変化すれば、
    コンテナへのアクセスは成功しています！
    """
)

number_slider = mo.ui.slider(
    start=1, stop=10, step=1, value=5, label="表示する行数"
)

mo.md(f"スライダーの現在の値: **{number_slider}**")

@mo.memo
def create_dataframe():
    """
    サンプルデータを生成する関数。
    @mo.memoデコレータにより、このセルの再実行は効率的に行われます。
    """
    data = {
        '商品カテゴリ': [f'カテゴリ{i}' for i in range(1, 11)],
        '売上（万円）': [150, 230, 80, 410, 300, 120, 500, 280, 190, 340],
        '在庫数': [30, 45, 15, 20, 50, 25, 60, 33, 28, 40],
    }
    return pd.DataFrame(data)

df = create_dataframe()

mo.md(df.head(number_slider.value))