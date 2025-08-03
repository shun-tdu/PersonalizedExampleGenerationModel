import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    return mo, pd


@app.cell
def _(mo):
    mo.md(
        """
    # 🐳 Marimo Docker 接続テスト
    このページが表示され、下のスライダーを動かして表が変化すれば、
    コンテナへのアクセスは成功しています！
    """
    )
    return


@app.cell
def _(mo):
    number_slider = mo.ui.slider(
        start=1, stop=10, step=1, value=5, label="表示する行数"
    )


    return (number_slider,)


@app.cell
def _(mo, number_slider):
    mo.md(f"""スライダーの現在の値: **{number_slider}**""")
    return


@app.cell
def _(pd):
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


    return (df,)


@app.cell
def _(df, number_slider):
    df.head(number_slider.value)
    return


if __name__ == "__main__":
    app.run()
