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
    from scipy import stats
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings('ignore')

    # スタイル設定
    sns.set_palette("husl")
    # plt.style.use('seaborn-v0_8')

    mo.md("# 🧬 β-VAE実験結果 包括的分析ダッシュボード")
    return go, json, make_subplots, mo, np, pd, plt, px, sqlite3, stats


@app.cell
def _(mo):
    # データベースパス設定
    db_path_input = mo.ui.text(
        value="PredictiveLatentSpaceNavigationModel/experiments.db",
        label="データベースパス:",
        placeholder="experiments.dbのパスを入力"
    )

    analysis_mode = mo.ui.dropdown(
        options={
            "🔍 基本分析": "basic",
            "📊 詳細統計": "detailed", 
            "🧠 ハイパーパラメータ最適化": "hyperopt",
            "🎯 潜在空間分析": "latent",
            "📈 学習プロセス分析": "training"
        },
        value="🔍 基本分析",
        label="分析モード"
    )

    mo.vstack([
        mo.md("## ⚙️ 分析設定"),
        db_path_input,
        analysis_mode
    ])
    return analysis_mode, db_path_input


@app.cell
def _(db_path_input, json, mo, pd, sqlite3):
    # データ読み込みと前処理
    def load_experiment_data(db_path):
        """データベースから実験データを読み込み、概要を返す関数"""
        try:
            # 関数内でconnを定義。これはこの関数だけのものになる。
            conn = sqlite3.connect(db_path)

            # 全実験データ取得
            df_experiments = pd.read_sql_query("""
                SELECT * FROM experiments 
                WHERE status = 'completed' AND reconstruction_mse IS NOT NULL
                ORDER BY created_at DESC
            """, conn)

            # JSON相関データの展開
            def parse_correlations(corr_str):
                if pd.isna(corr_str) or not corr_str:
                    return {}
                try:
                    return json.loads(corr_str)
                except:
                    return {}

            df_experiments['correlations_parsed'] = df_experiments['skill_correlations'].apply(parse_correlations)

            # 実験統計
            n_experiments = len(df_experiments)
            date_range = pd.to_datetime(df_experiments['start_time']).dt.date
            experiment_period = f"{date_range.min()} ~ {date_range.max()}"

            # パフォーマンス統計
            best_mse = df_experiments['reconstruction_mse'].min()
            worst_mse = df_experiments['reconstruction_mse'].max()
            avg_mse = df_experiments['reconstruction_mse'].mean()

            conn.close()

            # 結果をMarkdownオブジェクトとして作成
            status_content = mo.md(f"""
            ## 📊 実験概要
            - **総実験数**: {n_experiments}
            - **実験期間**: {experiment_period}
            - **最良MSE**: {best_mse:.2e}
            - **最悪MSE**: {worst_mse:.2e} 
            - **平均MSE**: {avg_mse:.2e}
            - **性能改善**: {((worst_mse - best_mse) / worst_mse * 100):.1f}%
            """)

            # データフレームと表示内容を両方返す
            return df_experiments, status_content

        except Exception as e:
            df_experiments = pd.DataFrame()
            status_content = mo.md(f"❌ データ読み込みエラー: {e}")
            return df_experiments, status_content

    df_experiments, status_content = load_experiment_data(db_path_input.value)
    status_content
    return (df_experiments,)


@app.cell
def _(analysis_mode, df_experiments, mo, pd, plt):
    # 基本分析 - パフォーマンス概要
    if analysis_mode.value == "basic" and not df_experiments.empty:

        # Matplotlibを使用してより軽量な図を作成
        import io
        import base64

        # 1. MSE分布（matplotlib版）
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # ヒストグラム
        ax1.hist(df_experiments['reconstruction_mse'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Reconstruction MSE')
        ax1.set_ylabel('Num Experiments')
        ax1.set_title('Reconstruction MSE Distribution')
        ax1.grid(True, alpha=0.3)

        # ボックスプロット
        ax1.boxplot(df_experiments['reconstruction_mse'], vert=False, patch_artist=True)

        # 2. 実験の時系列推移（matplotlib版）
        df_time = df_experiments.copy()
        df_time['start_time'] = pd.to_datetime(df_time['start_time'])
        df_time = df_time.sort_values('start_time')
        df_time['experiment_order'] = range(1, len(df_time) + 1)

        ax2.plot(df_time['experiment_order'], df_time['reconstruction_mse'], 
                'bo-', linewidth=2, markersize=6, label='MSE Trend')

        # 移動平均を追加
        if len(df_time) >= 3:
            df_time['mse_ma'] = df_time['reconstruction_mse'].rolling(window=3).mean()
            ax2.plot(df_time['experiment_order'], df_time['mse_ma'], 
                    'r--', linewidth=2, label='Moving Average')

        ax2.set_xlabel('Experiment Index')
        ax2.set_ylabel('Reconstruction MSE')
        ax2.set_title('MSE Progression Over Experiments')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # 画像をBase64エンコードして軽量化
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        plt.close()

        # 3. 設定別パフォーマンス（効率化）
        param_performance = []
        for _, row in df_experiments.iterrows():
            param_performance.append({
                'experiment_id': row['id'],
                'config_file': row['config_file'],
                'lr': row['lr'],
                'beta': row['beta'],
                'batch_size': row['batch_size'],
                'latent_total': row['style_latent_dim'] + row['skill_latent_dim'],
                'reconstruction_mse': row['reconstruction_mse'],
                'training_time': (pd.to_datetime(row['end_time']) - pd.to_datetime(row['start_time'])).total_seconds() / 3600
            })

        df_perf = pd.DataFrame(param_performance)

        # トップ3とワースト3
        top3 = df_perf.nsmallest(3, 'reconstruction_mse')[['config_file', 'reconstruction_mse', 'lr', 'beta']]
        worst3 = df_perf.nlargest(3, 'reconstruction_mse')[['config_file', 'reconstruction_mse', 'lr', 'beta']]

        # 軽量化された出力
        basic_content = mo.vstack([
            mo.md("## 🔍 基本分析結果"),
            mo.image(src=f"data:image/png;base64,{img_base64}"),
            mo.md("### 🏆 トップ3実験"),
            mo.ui.table(top3.round(6)),
            mo.md("### 📉 改善が必要な実験"),
            mo.ui.table(worst3.round(6)),
            mo.md(f"""
            ### 📊 サマリー統計
            - **実験数**: {len(df_experiments)}
            - **最良MSE**: {df_experiments['reconstruction_mse'].min():.2e}
            - **最悪MSE**: {df_experiments['reconstruction_mse'].max():.2e}
            - **平均MSE**: {df_experiments['reconstruction_mse'].mean():.2e}
            - **標準偏差**: {df_experiments['reconstruction_mse'].std():.2e}
            """)
        ])
    else:
        basic_content = mo.md("基本分析モードを選択してください")

    basic_content
    return


@app.cell
def _(analysis_mode, df_experiments, mo, np, pd, plt, stats):
    def _():
        # 詳細統計分析
        if analysis_mode.value == "detailed" and not df_experiments.empty:

            # 1. ハイパーパラメータの統計的影響分析
            params = ['lr', 'beta', 'batch_size', 'hidden_dim', 'style_latent_dim', 'skill_latent_dim']
            param_correlations = {}

            for param in params:
                if param in df_experiments.columns:
                    corr, p_val = stats.pearsonr(df_experiments[param], df_experiments['reconstruction_mse'])
                    param_correlations[param] = {'correlation': corr, 'p_value': p_val, 'significant': p_val < 0.05}

            # matplotlibで軽量な相関図を作成
            import io
            import base64

            if param_correlations:
                corr_df = pd.DataFrame(param_correlations).T
                corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)

                fig, ax = plt.subplots(figsize=(10, 4))
                colors = ['red' if sig else 'lightblue' for sig in corr_df['significant']]
                bars = ax.bar(corr_df.index, corr_df['correlation'], color=colors, alpha=0.7)
                ax.set_title('🔗 ハイパーパラメータとMSEの相関')
                ax.set_ylabel('相関係数')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

                # 凡例
                import matplotlib.patches as mpatches
                sig_patch = mpatches.Patch(color='red', label='有意 (p<0.05)')
                non_sig_patch = mpatches.Patch(color='lightblue', label='非有意')
                ax.legend(handles=[sig_patch, non_sig_patch])

                plt.xticks(rotation=45)
                plt.tight_layout()

                # 画像をBase64エンコード
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.read()).decode()
                plt.close()

                correlation_image = mo.image(src=f"data:image/png;base64,{img_base64}")
            else:
                correlation_image = mo.md("相関データがありません")

            # 統計サマリー
            numeric_cols = ['reconstruction_mse', 'final_total_loss', 'best_val_loss'] + params
            existing_cols = [col for col in numeric_cols if col in df_experiments.columns]
            stats_summary = df_experiments[existing_cols].describe()

            # 異常値検出
            from scipy import stats as scipy_stats
            z_scores = np.abs(scipy_stats.zscore(df_experiments['reconstruction_mse']))
            outliers = df_experiments[z_scores > 2]['config_file'].tolist()

            detailed_content = mo.vstack([
                mo.md("## 📊 詳細統計分析"),
                correlation_image,
                mo.md("### 📈 相関分析結果"),
                mo.ui.table(corr_df.round(4)) if param_correlations else mo.md("データなし"),
                mo.md("### 📋 統計サマリー"),
                mo.ui.table(stats_summary.round(6)),
                mo.md(f"### ⚠️ 異常値検出 (Z-score > 2): {len(outliers)}件"),
                mo.md(f"異常値: {', '.join(outliers) if outliers else 'なし'}")
            ])
        else:
            detailed_content = mo.md("詳細統計モードを選択してください")
        return detailed_content


    _()
    return


@app.cell
def _(analysis_mode, df_experiments, go, make_subplots, mo, np, pd):
    # ハイパーパラメータ最適化分析
    if analysis_mode.value == "hyperopt" and not df_experiments.empty:

        # 1. パレート最適解の特定
        # 複数目的での最適化: MSE最小化 & 学習時間最小化
        df_opt = df_experiments.copy()
        df_opt['training_time'] = (pd.to_datetime(df_opt['end_time']) - pd.to_datetime(df_opt['start_time'])).dt.total_seconds() / 3600

        # パレートフロント計算
        def is_pareto_optimal(costs):
            is_efficient = np.ones(costs.shape[0], dtype=bool)
            for i, c in enumerate(costs):
                if is_efficient[i]:
                    is_efficient[is_efficient] = np.any(costs[is_efficient] <= c, axis=1)
                    is_efficient[i] = True
            return is_efficient

        objectives = df_opt[['reconstruction_mse', 'training_time']].values
        pareto_mask = is_pareto_optimal(objectives)

        # パレートフロントの可視化
        fig_pareto = go.Figure()

        # 非効率解
        fig_pareto.add_trace(go.Scatter(
            x=df_opt[~pareto_mask]['reconstruction_mse'],
            y=df_opt[~pareto_mask]['training_time'],
            mode='markers',
            name='一般解',
            marker=dict(color='lightblue', size=8)
        ))

        # パレート最適解
        fig_pareto.add_trace(go.Scatter(
            x=df_opt[pareto_mask]['reconstruction_mse'],
            y=df_opt[pareto_mask]['training_time'],
            mode='markers',
            name='パレート最適解',
            marker=dict(color='red', size=12, symbol='star')
        ))

        fig_pareto.update_layout(
            title="🎯 パレート最適解分析 (MSE vs 学習時間)",
            xaxis_title="Reconstruction MSE",
            yaxis_title="学習時間 (時間)"
        )

        # 2. 推奨ハイパーパラメータ
        pareto_experiments = df_opt[pareto_mask]
        best_balance = pareto_experiments.loc[pareto_experiments['reconstruction_mse'].idxmin()]

        # 3. パラメータ感度分析
        param_ranges = {}
        for param in ['lr', 'beta', 'batch_size']:
            if param in df_experiments.columns:
                param_values = df_experiments[param].values
                mse_values = df_experiments['reconstruction_mse'].values

                # パラメータ値を区間に分割
                n_bins = min(5, len(df_experiments))
                bins = pd.qcut(param_values, n_bins, duplicates='drop')
                bin_stats = df_experiments.groupby(bins)['reconstruction_mse'].agg(['mean', 'std', 'count'])

                param_ranges[param] = bin_stats

        # 感度分析の可視化
        fig_sensitivity = make_subplots(
            rows=1, cols=len(param_ranges),
            subplot_titles=list(param_ranges.keys())
        )

        for i, (param, stats_df) in enumerate(param_ranges.items()):
            fig_sensitivity.add_trace(
                go.Bar(
                    x=[str(interval) for interval in stats_df.index],
                    y=stats_df['mean'],
                    error_y=dict(type='data', array=stats_df['std']),
                    name=param,
                    showlegend=False
                ),
                row=1, col=i+1
            )

        fig_sensitivity.update_layout(height=400, title_text="📈 パラメータ感度分析")

        hyperopt_content = mo.vstack([
            mo.md("## 🧠 ハイパーパラメータ最適化分析"),
            mo.Html(fig_pareto.to_html()),
            mo.md("### 🏆 推奨設定 (パレート最適解から)"),
            mo.md(f"""
            - **設定ファイル**: {best_balance['config_file']}
            - **学習率**: {best_balance['lr']:.2e}
            - **β値**: {best_balance['beta']:.3f}
            - **バッチサイズ**: {int(best_balance['batch_size'])}
            - **MSE**: {best_balance['reconstruction_mse']:.2e}
            - **学習時間**: {best_balance['training_time']:.2f}時間
            """),
            mo.Html(fig_sensitivity.to_html()),
            mo.md("### 📋 パレート最適解一覧"),
            mo.ui.table(pareto_experiments[['config_file', 'lr', 'beta', 'reconstruction_mse', 'training_time']].round(6))
        ])
    else:
        hyperopt_content = mo.md("ハイパーパラメータ最適化モードを選択してください")

    hyperopt_content
    return


@app.cell
def _(analysis_mode, df_experiments, mo, pd, px):
    def _():
        # 潜在空間分析
        if analysis_mode.value == "latent" and not df_experiments.empty:

            # 相関データの詳細分析
            all_correlations = []

            for _, row in df_experiments.iterrows():
                if row['correlations_parsed']:
                    exp_id = row['id']
                    config = row['config_file']
                    mse = row['reconstruction_mse']

                    for metric, corr_list in row['correlations_parsed'].items():
                        for dim, (corr, p_val) in enumerate(corr_list):
                            all_correlations.append({
                                'experiment_id': exp_id,
                                'config_file': config,
                                'reconstruction_mse': mse,
                                'metric': metric,
                                'z_skill_dim': dim,
                                'correlation': corr,
                                'p_value': p_val,
                                'significant': p_val < 0.05,
                                'abs_correlation': abs(corr)
                            })

            if all_correlations:
                corr_df = pd.DataFrame(all_correlations)

                # 1. 最強相関の特定
                significant_corr = corr_df[corr_df['significant']]
                if not significant_corr.empty:
                    strongest_corr = significant_corr.loc[significant_corr['abs_correlation'].idxmax()]

                    # 2. 実験別相関強度
                    exp_corr_strength = corr_df.groupby('experiment_id').agg({
                        'abs_correlation': ['mean', 'max'],
                        'significant': 'sum'
                    }).round(4)
                    exp_corr_strength.columns = ['平均相関強度', '最大相関強度', '有意相関数']

                    # 3. メトリック別相関パターン
                    metric_analysis = corr_df.groupby('metric').agg({
                        'abs_correlation': ['mean', 'std', 'max'],
                        'significant': ['sum', 'count']
                    }).round(4)

                    # 4. 相関ヒートマップ（実験×メトリック）
                    pivot_corr = corr_df.groupby(['experiment_id', 'metric'])['abs_correlation'].max().unstack(fill_value=0)

                    fig_heatmap = px.imshow(
                        pivot_corr,
                        title="🔥 実験別メトリック相関強度ヒートマップ",
                        labels=dict(x="パフォーマンス指標", y="実験ID", color="相関強度"),
                        color_continuous_scale='Viridis'
                    )

                    # 5. 次元別相関分析
                    dim_analysis = corr_df.groupby('z_skill_dim').agg({
                        'abs_correlation': ['mean', 'std', 'count'],
                        'significant': 'sum'
                    }).round(4)

                    fig_dim_analysis = px.bar(
                        x=dim_analysis.index,
                        y=dim_analysis[('abs_correlation', 'mean')],
                        error_y=dim_analysis[('abs_correlation', 'std')],
                        title="📊 z_skill次元別平均相関強度"
                    )

                    latent_content = mo.vstack([
                        mo.md("## 🎯 潜在空間分析"),
                        mo.md(f"""
                        ### 🏆 最強相関
                        - **実験**: {strongest_corr['config_file']}
                        - **指標**: {strongest_corr['metric']}
                        - **次元**: z_skill_{strongest_corr['z_skill_dim']}
                        - **相関**: {strongest_corr['correlation']:.4f}
                        - **p値**: {strongest_corr['p_value']:.4f}
                        """),
                        mo.Html(fig_heatmap.to_html()),
                        mo.Html(fig_dim_analysis.to_html()),
                        mo.md("### 📈 実験別相関強度"),
                        mo.ui.table(exp_corr_strength),
                        mo.md("### 📊 メトリック別分析"),
                        mo.ui.table(metric_analysis)
                    ])
                else:
                    latent_content = mo.md("### ⚠️ 有意な相関が見つかりませんでした")
            else:
                latent_content = mo.md("### ℹ️ 相関データがありません")
        else:
            latent_content = mo.md("潜在空間分析モードを選択してください")
        return latent_content


    _()
    return


@app.cell
def _(analysis_mode, df_experiments, go, mo, pd, px):
    # 学習プロセス分析
    if analysis_mode.value == "training" and not df_experiments.empty:

        # 1. 学習効率分析
        df_training = df_experiments.copy()
        df_training['training_time'] = (pd.to_datetime(df_training['end_time']) - pd.to_datetime(df_training['start_time'])).dt.total_seconds() / 3600
        df_training['efficiency'] = 1 / (df_training['reconstruction_mse'] * df_training['training_time'])
        df_training['convergence_ratio'] = df_training['best_val_loss'] / df_training['final_total_loss']

        # 2. 学習時間 vs パフォーマンス
        fig_efficiency = px.scatter(
            df_training,
            x='training_time',
            y='reconstruction_mse',
            size='num_epochs',
            color='lr',
            hover_data=['config_file', 'batch_size'],
            title="⚡ 学習効率分析 (時間 vs 性能)",
            labels={'training_time': '学習時間 (時間)', 'reconstruction_mse': 'MSE'}
        )

        # 3. 収束性分析
        fig_convergence = px.scatter(
            df_training,
            x='final_total_loss',
            y='best_val_loss',
            color='convergence_ratio',
            size='num_epochs',
            title="🎯 学習収束性分析",
            labels={'final_total_loss': '最終学習損失', 'best_val_loss': '最良検証損失'}
        )

        # 理想的な収束線を追加
        min_loss = min(df_training['final_total_loss'].min(), df_training['best_val_loss'].min())
        max_loss = max(df_training['final_total_loss'].max(), df_training['best_val_loss'].max())
        fig_convergence.add_trace(go.Scatter(
            x=[min_loss, max_loss],
            y=[min_loss, max_loss],
            mode='lines',
            name='理想収束線',
            line=dict(dash='dash', color='red')
        ))

        # 4. 学習効率ランキング
        efficiency_ranking = df_training.nlargest(5, 'efficiency')[['config_file', 'reconstruction_mse', 'training_time', 'efficiency']]

        # 5. 早期停止効果分析
        if 'num_epochs' in df_training.columns:
            epoch_analysis = df_training.groupby('num_epochs').agg({
                'reconstruction_mse': ['mean', 'std', 'count'],
                'training_time': 'mean'
            }).round(4)

            fig_epochs = px.scatter(
                df_training,
                x='num_epochs',
                y='reconstruction_mse',
                size='training_time',
                title="📚 エポック数と性能の関係"
            )

        training_content = mo.vstack([
            mo.md("## 📈 学習プロセス分析"),
            mo.Html(fig_efficiency.to_html()),
            mo.Html(fig_convergence.to_html()),
            mo.md("### ⚡ 学習効率ランキング (効率 = 1/(MSE × 時間))"),
            mo.ui.table(efficiency_ranking.round(6)),
            mo.Html(fig_epochs.to_html()) if 'num_epochs' in df_training.columns else mo.md(""),
            mo.md("### 📊 収束性サマリー"),
            mo.md(f"""
            - **平均学習時間**: {df_training['training_time'].mean():.2f}時間
            - **最高効率実験**: {efficiency_ranking.iloc[0]['config_file']}
            - **平均収束比**: {df_training['convergence_ratio'].mean():.4f}
            - **過学習率**: {(df_training['convergence_ratio'] > 1.1).sum() / len(df_training) * 100:.1f}%
            """)
        ])
    else:
        training_content = mo.md("学習プロセス分析モードを選択してください")

    training_content
    return


if __name__ == "__main__":
    app.run()
