import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sqlite3 
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import json
    from pathlib import Path
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import warnings
    warnings.filterwarnings('ignore')

    # スタイル設定
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    mo.md("# 🧪 実験結果分析ダッシュボード")
    return (mo,)


@app.cell
def _(db_path_input, mo):
    # データベースパスを設定
    dp_path_input = mo.ui.text(
        value="PredictiveLatentSpaceNavigationModel/experiments.db",
        label="データベースへのパス: ",
        placeholder="experiments.dbのパスを入力"
    )

    mo.md(f"""
    ## 📊 データベース接続設定
    {db_path_input}
    """)
    return


app._unparsable_cell(
    r"""
    # データベースからデータを読み込み
    try:
        conn = sqlite3.concat(db_path_input.value)
        # 全実験データを取得
            query = \"\"\"
            SELECT * FROM experiments 
            WHERE status IN ('completed', 'failed')
            ORDER BY created_at DESC
            \"\"\"
        
            df_experiments = pd.read_sql_query(query, conn)
        
            # JSON形式の相関データを展開
            def parse_correlations(corr_str):
                if pd.isna(corr_str) or corr_str == '':
                    return {}
                try:
                    return json.loads(corr_str)
                except:
                    return {}
        
            df_experiments['correlations_parsed'] = df_experiments['skill_correlations'].apply(parse_correlations)
        
            conn.close()
        
            # 成功した実験のみフィルタ
            df_completed = df_experiments[df_experiments['status'] == 'completed'].copy()
        
            mo.md(f\"\"\"
            ✅ **データベース接続成功!**
            - 総実験数: {len(df_experiments)}
            - 完了した実験: {len(df_completed)}
            - 失敗した実験: {len(df_experiments[df_experiments['status'] == 'failed'])}
            \"\"\")
        
        except Exception as e:
            df_experiments = pd.DataFrame()
            df_completed = pd.DataFrame()
            mo.md(f\"❌ **データベース接続エラー:** {e}\")
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    if not df_completed.empty:
            # 実験選択用のUI
            experiment_selector = mo.ui.dropdown(
                options={f\"実験{row['id']}: {row['config_file']} (MSE: {row['reconstruction_mse']:.6f})\": row['id'] 
                        for _, row in df_completed.iterrows()},
                value=df_completed.iloc[0]['id'] if not df_completed.empty else None,
                label=\"分析する実験を選択:\"
            )
        
            mo.md(f\"\"\"
            ## 🔍 実験選択
            {experiment_selector}
            \"\"\")
        else:
            experiment_selector = mo.ui.dropdown(options={}, label=\"実験が見つかりません\")
            mo.md(\"⚠️ 完了した実験が見つかりません\")
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # 選択された実験の詳細情報
        if experiment_selector.value and not df_completed.empty:
            selected_exp = df_completed[df_completed['id'] == experiment_selector.value].iloc[0]
        
            mo.md(f\"\"\"
            ## 📋 実験詳細: {selected_exp['config_file']}
        
            ### ハイパーパラメータ
            - **学習率**: {selected_exp['lr']}
            - **バッチサイズ**: {selected_exp['batch_size']}
            - **エポック数**: {selected_exp['num_epochs']}
            - **隠れ層次元**: {selected_exp['hidden_dim']}
            - **スタイル潜在次元**: {selected_exp['style_latent_dim']}
            - **スキル潜在次元**: {selected_exp['skill_latent_dim']}
            - **β値**: {selected_exp['beta']}
        
            ### 学習結果
            - **最終学習損失**: {selected_exp['final_total_loss']:.6f}
            - **最良検証損失**: {selected_exp['best_val_loss']:.6f}
            - **再構成MSE**: {selected_exp['reconstruction_mse']:.6f}
        
            ### 実行情報
            - **開始時刻**: {selected_exp['start_time']}
            - **終了時刻**: {selected_exp['end_time']}
            - **Git コミット**: {selected_exp['git_commit_hash'][:8] if selected_exp['git_commit_hash'] else 'N/A'}
            \"\"\")
        else:
            selected_exp = None
            mo.md(\"実験を選択してください\")
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # 実験比較チャート
        if not df_completed.empty and len(df_completed) > 1:
        
            # 再構成MSEの比較
            fig_mse = px.bar(
                df_completed.sort_values('reconstruction_mse'), 
                x='config_file', 
                y='reconstruction_mse',
                title=\"📈 実験別再構成MSE比較\",
                labels={'reconstruction_mse': 'Reconstruction MSE', 'config_file': 'Configuration'},
                color='reconstruction_mse',
                color_continuous_scale='viridis'
            )
            fig_mse.update_layout(xaxis_tickangle=-45)
        
            # パラメータとパフォーマンスの関係
            fig_params = make_subplots(
                rows=2, cols=2,
                subplot_titles=('学習率 vs MSE', 'β値 vs MSE', 'バッチサイズ vs MSE', '潜在次元 vs MSE')
            )
        
            # 学習率 vs MSE
            fig_params.add_trace(
                go.Scatter(x=df_completed['lr'], y=df_completed['reconstruction_mse'], 
                          mode='markers', name='LR vs MSE', marker=dict(size=8)),
                row=1, col=1
            )
        
            # β値 vs MSE
            fig_params.add_trace(
                go.Scatter(x=df_completed['beta'], y=df_completed['reconstruction_mse'], 
                          mode='markers', name='Beta vs MSE', marker=dict(size=8)),
                row=1, col=2
            )
        
            # バッチサイズ vs MSE
            fig_params.add_trace(
                go.Scatter(x=df_completed['batch_size'], y=df_completed['reconstruction_mse'], 
                          mode='markers', name='Batch vs MSE', marker=dict(size=8)),
                row=2, col=1
            )
        
            # 潜在次元 vs MSE
            total_latent_dim = df_completed['style_latent_dim'] + df_completed['skill_latent_dim']
            fig_params.add_trace(
                go.Scatter(x=total_latent_dim, y=df_completed['reconstruction_mse'], 
                          mode='markers', name='Latent Dim vs MSE', marker=dict(size=8)),
                row=2, col=2
            )
        
            fig_params.update_layout(height=600, title_text=\"🔄 ハイパーパラメータとパフォーマンスの関係\")
        
            mo.md(\"## 📊 実験比較\")
            mo.Html(fig_mse.to_html())
            mo.Html(fig_params.to_html())
        else:
            mo.md(\"### ⚠️ 比較には2つ以上の完了した実験が必要です\")
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # 相関分析の可視化
        if selected_exp is not None and selected_exp['correlations_parsed']:
            correlations = selected_exp['correlations_parsed']
        
            # 相関データを整理
            corr_data = []
            for metric, corr_list in correlations.items():
                for dim, (corr, p_val) in enumerate(corr_list):
                    corr_data.append({
                        'metric': metric,
                        'z_skill_dim': f'z_skill_{dim}',
                        'correlation': corr,
                        'p_value': p_val,
                        'significant': p_val < 0.05
                    })
        
            corr_df = pd.DataFrame(corr_data)
        
            # 相関ヒートマップ
            pivot_corr = corr_df.pivot(index='metric', columns='z_skill_dim', values='correlation')
            pivot_pval = corr_df.pivot(index='metric', columns='z_skill_dim', values='p_value')
        
            fig_heatmap = px.imshow(
                pivot_corr,
                title=\"🔥 z_skillとパフォーマンス指標の相関ヒートマップ\",
                labels=dict(x=\"z_skill次元\", y=\"パフォーマンス指標\", color=\"相関係数\"),
                color_continuous_scale='RdBu_r',
                aspect=\"auto\"
            )
        
            # 有意な相関のみ表示
            significant_corr = corr_df[corr_df['significant']].sort_values('correlation', key=abs, ascending=False)
        
            if not significant_corr.empty:
                fig_significant = px.bar(
                    significant_corr.head(10),
                    x='correlation',
                    y=[f\"{row['metric']} - {row['z_skill_dim']}\" for _, row in significant_corr.head(10).iterrows()],
                    title=\"📊 有意な相関 (p < 0.05) TOP 10\",
                    orientation='h',
                    color='correlation',
                    color_continuous_scale='RdBu_r'
                )
            
                mo.md(\"## 🧠 潜在変数とパフォーマンスの相関分析\")
                mo.Html(fig_heatmap.to_html())
                mo.Html(fig_significant.to_html())
            
                # 相関サマリーテーブル
                mo.md(\"### 📋 有意な相関サマリー\")
                mo.ui.table(significant_corr[['metric', 'z_skill_dim', 'correlation', 'p_value']].round(4))
            else:
                mo.md(\"### ⚠️ 有意な相関が見つかりませんでした (p < 0.05)\")
        else:
            mo.md(\"### ℹ️ 相関データが利用できません\")
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # 生成された画像の表示
        if selected_exp is not None:
        
            # 学習曲線
            if selected_exp['image_path'] and Path(selected_exp['image_path']).exists():
                mo.md(\"## 📈 学習曲線\")
                mo.image(src=selected_exp['image_path'])
            else:
                mo.md(\"### ⚠️ 学習曲線画像が見つかりません\")
        
            # 潜在空間可視化
            if selected_exp['latent_visualization_path'] and Path(selected_exp['latent_visualization_path']).exists():
                mo.md(\"## 🎯 潜在空間可視化\")
                mo.image(src=selected_exp['latent_visualization_path'])
            else:
                mo.md(\"### ⚠️ 潜在空間可視化画像が見つかりません\")
        
            # 生成軌道
            if selected_exp['generated_trajectories_path'] and Path(selected_exp['generated_trajectories_path']).exists():
                mo.md(\"## 🚀 生成軌道\")
                mo.image(src=selected_exp['generated_trajectories_path'])
            else:
                mo.md(\"### ⚠️ 生成軌道画像が見つかりません\")
        else:
            mo.md(\"実験を選択してください\")
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # 実験の時系列分析
        if not df_completed.empty:
            df_time = df_completed.copy()
            df_time['start_time'] = pd.to_datetime(df_time['start_time'])
            df_time = df_time.sort_values('start_time')
        
            # MSEの時系列変化
            fig_timeline = px.line(
                df_time,
                x='start_time',
                y='reconstruction_mse',
                title=\"📅 実験の時系列変化 (再構成MSE)\",
                markers=True,
                hover_data=['config_file', 'lr', 'beta']
            )
        
            # 実験間隔の分析
            df_time['duration_hours'] = (
                pd.to_datetime(df_time['end_time']) - pd.to_datetime(df_time['start_time'])
            ).dt.total_seconds() / 3600
        
            fig_duration = px.bar(
                df_time,
                x='config_file',
                y='duration_hours',
                title=\"⏱️ 実験別実行時間\",
                labels={'duration_hours': '実行時間 (時間)'}
            )
            fig_duration.update_layout(xaxis_tickangle=-45)
        
            mo.md(\"## ⏰ 時系列分析\")
            mo.Html(fig_timeline.to_html())
            mo.Html(fig_duration.to_html())
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # 実験統計サマリー
        if not df_completed.empty:
            summary_stats = {
                \"📊 基本統計\": {
                    \"総実験数\": len(df_completed),
                    \"平均再構成MSE\": f\"{df_completed['reconstruction_mse'].mean():.6f}\",
                    \"最良MSE\": f\"{df_completed['reconstruction_mse'].min():.6f}\",
                    \"最悪MSE\": f\"{df_completed['reconstruction_mse'].max():.6f}\",
                    \"MSE標準偏差\": f\"{df_completed['reconstruction_mse'].std():.6f}\"
                },
                \"⚙️ パラメータ範囲\": {
                    \"学習率範囲\": f\"{df_completed['lr'].min():.2e} - {df_completed['lr'].max():.2e}\",
                    \"β値範囲\": f\"{df_completed['beta'].min():.3f} - {df_completed['beta'].max():.3f}\",
                    \"バッチサイズ範囲\": f\"{df_completed['batch_size'].min()} - {df_completed['batch_size'].max()}\",
                    \"潜在次元範囲\": f\"{(df_completed['style_latent_dim'] + df_completed['skill_latent_dim']).min()} - {(df_completed['style_latent_dim'] + df_completed['skill_latent_dim']).max()}\"
                }
            }
        
            mo.md(\"## 📋 実験統計サマリー\")
            for category, stats in summary_stats.items():
                mo.md(f\"### {category}\")
                for key, value in stats.items():
                    mo.md(f\"- **{key}**: {value}\")
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # データエクスポート機能
        if not df_completed.empty:
        
            # CSV形式でのエクスポート
            export_button = mo.ui.button(
                label=\"📤 実験結果をCSVでエクスポート\",
                on_click=lambda: df_completed.to_csv('experiment_results.csv', index=False)
            )
        
            # ベストパフォーマンス実験の詳細
            best_exp = df_completed.loc[df_completed['reconstruction_mse'].idxmin()]
        
            mo.md(f\"\"\"
            ## 🏆 最良実験結果
        
            **設定ファイル**: {best_exp['config_file']}
        
            **ハイパーパラメータ**:
            - 学習率: {best_exp['lr']}
            - β値: {best_exp['beta']}
            - バッチサイズ: {best_exp['batch_size']}
            - 潜在次元: {best_exp['style_latent_dim']} + {best_exp['skill_latent_dim']} = {best_exp['style_latent_dim'] + best_exp['skill_latent_dim']}
        
            **性能**:
            - 再構成MSE: {best_exp['reconstruction_mse']:.6f}
            - 最終学習損失: {best_exp['final_total_loss']:.6f}
            - 最良検証損失: {best_exp['best_val_loss']:.6f}
        
            ---
        
            {export_button}
            \"\"\")
    """,
    name="_"
)


app._unparsable_cell(
    r"""
     mo.md(\"\"\"
        ## 💡 分析のヒント
    
        ### 🔍 見るべきポイント
        1. **再構成MSE**: モデルの基本性能を示す最重要指標
        2. **相関分析**: z_skillがパフォーマンス指標とどの程度関連しているか
        3. **潜在空間可視化**: スタイルとスキルが適切に分離されているか
        4. **生成軌道**: αの変化により軌道の特徴が変化しているか
    
        ### 📊 良い結果の特徴
        - 低い再構成MSE
        - z_skillとパフォーマンス指標に有意な相関
        - 潜在空間でのクラスター分離
        - 生成軌道でのスムーズな変化
    
        ### ⚠️ 問題のある結果
        - 高い再構成MSE (過学習または学習不足)
        - 相関がほとんどない (分離学習の失敗)
        - 潜在空間での混在
        - 生成軌道での不自然な変化
        \"\"\")
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
