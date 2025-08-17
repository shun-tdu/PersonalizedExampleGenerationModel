import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os


class HierarchicalVAEValidationAnalyzer:
    def __init__(self, db_path):
        self.db_path = db_path
        self.results_df = None
        self.load_results()

    def load_results(self):
        """データベースから実験結果を読み込み"""
        query = """
                SELECT * \
                FROM hierarchical_experiments
                WHERE status = 'completed' \
                  AND tags LIKE '%validation%'
                ORDER BY created_at \
                """

        with sqlite3.connect(self.db_path) as conn:
            self.results_df = pd.read_sql_query(query, conn)

        # タグを解析してカテゴリ分け
        self.results_df['phase'] = self.results_df['experiment_name'].str.extract(r'(phase\d+)')
        self.results_df['parameter_type'] = self.results_df['experiment_name'].str.extract(r'phase\d+_(\w+)')
        self.results_df['parameter_value'] = self.results_df['experiment_name'].str.extract(r'phase\d+_\w+_(\w+)')

        print(f"分析対象実験数: {len(self.results_df)}")

    def analyze_phase1_sensitivity(self):
        """Phase 1: パラメータ感度分析"""
        print("\n=== Phase 1: パラメータ感度分析 ===")

        phase1_data = self.results_df[self.results_df['phase'] == 'phase1'].copy()

        if len(phase1_data) == 0:
            print("Phase 1のデータが見つかりません")
            return

        # β値の影響分析
        beta_results = phase1_data[phase1_data['parameter_type'] == 'beta']
        if len(beta_results) > 0:
            self._analyze_beta_effects(beta_results)

        # 潜在次元の影響分析
        latent_results = phase1_data[phase1_data['parameter_type'] == 'latent']
        if len(latent_results) > 0:
            self._analyze_latent_effects(latent_results)

        # 隠れ層次元の影響分析
        hidden_results = phase1_data[phase1_data['parameter_type'] == 'hidden']
        if len(hidden_results) > 0:
            self._analyze_hidden_effects(hidden_results)

    def _analyze_beta_effects(self, beta_results):
        """β値の効果分析"""
        print("\n--- β値スケジュールの効果 ---")

        metrics = ['reconstruction_mse', 'style_separation_score', 'skill_performance_correlation']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, metric in enumerate(metrics):
            if metric in beta_results.columns:
                beta_results_clean = beta_results.dropna(subset=[metric])

                # boxplot
                sns.boxplot(data=beta_results_clean, x='parameter_value', y=metric, ax=axes[i])
                axes[i].set_title(f'{metric} by β schedule')
                axes[i].tick_params(axis='x', rotation=45)

                # 統計検定
                groups = [group[metric].values for name, group in beta_results_clean.groupby('parameter_value')]
                if len(groups) > 1:
                    f_stat, p_value = stats.f_oneway(*groups)
                    axes[i].text(0.05, 0.95, f'ANOVA p={p_value:.4f}',
                                 transform=axes[i].transAxes, verticalalignment='top',
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig('beta_schedule_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 最適β値の特定
        best_beta = beta_results.loc[beta_results['reconstruction_mse'].idxmin(), 'parameter_value']
        print(f"最適β値スケジュール: {best_beta}")

        # 結果サマリー
        summary = beta_results.groupby('parameter_value')[metrics].agg(['mean', 'std']).round(4)
        print("\nβ値スケジュール別性能サマリー:")
        print(summary)

    def _analyze_latent_effects(self, latent_results):
        """潜在次元の効果分析"""
        print("\n--- 潜在次元の効果 ---")

        # 次元数と性能の関係
        dimension_mapping = {'small': 1, 'baseline': 2, 'large': 3}
        latent_results['dim_order'] = latent_results['parameter_value'].map(dimension_mapping)

        metrics = ['reconstruction_mse', 'style_separation_score', 'final_total_loss']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, metric in enumerate(metrics):
            if metric in latent_results.columns:
                clean_data = latent_results.dropna(subset=[metric, 'dim_order'])

                # 散布図とトレンドライン
                axes[i].scatter(clean_data['dim_order'], clean_data[metric], alpha=0.7)

                # 線形回帰
                if len(clean_data) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        clean_data['dim_order'], clean_data[metric])
                    line_x = np.array([1, 3])
                    line_y = slope * line_x + intercept
                    axes[i].plot(line_x, line_y, 'r--', alpha=0.8)
                    axes[i].text(0.05, 0.95, f'R²={r_value ** 2:.3f}, p={p_value:.4f}',
                                 transform=axes[i].transAxes, verticalalignment='top',
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                axes[i].set_xlabel('Latent Dimension Size')
                axes[i].set_ylabel(metric)
                axes[i].set_title(f'{metric} vs Latent Dimensions')
                axes[i].set_xticks([1, 2, 3])
                axes[i].set_xticklabels(['Small', 'Baseline', 'Large'])

        plt.tight_layout()
        plt.savefig('latent_dimension_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _analyze_hidden_effects(self, hidden_results):
        """隠れ層次元の効果分析"""
        print("\n--- 隠れ層次元の効果 ---")

        hidden_results['hidden_dim_num'] = hidden_results['parameter_value'].astype(int)

        # 計算効率 vs 性能のトレードオフ
        if 'final_epoch' in hidden_results.columns and 'reconstruction_mse' in hidden_results.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # 性能 vs 隠れ層次元
            ax1.scatter(hidden_results['hidden_dim_num'], hidden_results['reconstruction_mse'])
            ax1.set_xlabel('Hidden Dimension')
            ax1.set_ylabel('Reconstruction MSE')
            ax1.set_title('Performance vs Hidden Dimension')

            # 収束速度 vs 隠れ層次元
            ax2.scatter(hidden_results['hidden_dim_num'], hidden_results['final_epoch'])
            ax2.set_xlabel('Hidden Dimension')
            ax2.set_ylabel('Training Epochs')
            ax2.set_title('Convergence Speed vs Hidden Dimension')

            plt.tight_layout()
            plt.savefig('hidden_dimension_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

    def analyze_phase2_hierarchy(self):
        """Phase 2: 階層学習効果分析"""
        print("\n=== Phase 2: 階層学習効果分析 ===")

        phase2_data = self.results_df[self.results_df['phase'] == 'phase2'].copy()

        if len(phase2_data) == 0:
            print("Phase 2のデータが見つかりません")
            return

        # 学習スケジュール別の収束パターン分析
        schedule_results = phase2_data[phase2_data['parameter_type'] == 'schedule']

        if len(schedule_results) > 0:
            self._analyze_learning_schedules(schedule_results)

    def _analyze_learning_schedules(self, schedule_results):
        """学習スケジュールの効果分析"""
        print("\n--- 学習スケジュール効果 ---")

        metrics = ['final_total_loss', 'best_val_loss', 'final_epoch', 'early_stopped']

        # 学習スケジュール別性能比較
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            if metric in schedule_results.columns:
                if metric == 'early_stopped':
                    # 早期停止率の可視化
                    early_stop_rate = schedule_results.groupby('parameter_value')[metric].mean()
                    early_stop_rate.plot(kind='bar', ax=axes[i])
                    axes[i].set_title('Early Stopping Rate by Schedule')
                    axes[i].set_ylabel('Early Stop Rate')
                else:
                    sns.boxplot(data=schedule_results, x='parameter_value', y=metric, ax=axes[i])
                    axes[i].set_title(f'{metric} by Learning Schedule')
                    axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('learning_schedule_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 最適スケジュール
        if 'best_val_loss' in schedule_results.columns:
            best_schedule = schedule_results.loc[schedule_results['best_val_loss'].idxmin(), 'parameter_value']
            print(f"最適学習スケジュール: {best_schedule}")

    def analyze_phase3_prediction_errors(self):
        """Phase 3: 予測誤差重み分析"""
        print("\n=== Phase 3: 予測誤差重み分析 ===")

        phase3_data = self.results_df[self.results_df['phase'] == 'phase3'].copy()

        if len(phase3_data) == 0:
            print("Phase 3のデータが見つかりません")
            return

        # 予測誤差重み別分析
        error_weight_results = phase3_data[phase3_data['parameter_type'] == 'error']
        if len(error_weight_results) > 0:
            self._analyze_prediction_error_weights(error_weight_results)

        # 精度学習率別分析
        precision_lr_results = phase3_data[phase3_data['parameter_type'] == 'precision']
        if len(precision_lr_results) > 0:
            self._analyze_precision_learning_rates(precision_lr_results)

    def _analyze_prediction_error_weights(self, error_results):
        """予測誤差重みの効果分析"""
        print("\n--- 予測誤差重み効果 ---")

        # 階層別KL損失の分析
        kl_metrics = ['final_kl_primitive', 'final_kl_skill', 'final_kl_style']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, metric in enumerate(kl_metrics):
            if metric in error_results.columns:
                sns.boxplot(data=error_results, x='parameter_value', y=metric, ax=axes[i])
                axes[i].set_title(f'{metric} by Error Weights')
                axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('prediction_error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _analyze_precision_learning_rates(self, precision_results):
        """精度学習率の効果分析"""
        print("\n--- 精度学習率効果 ---")

        precision_results['precision_lr_num'] = precision_results['parameter_value'].astype(float)

        if 'reconstruction_mse' in precision_results.columns:
            plt.figure(figsize=(8, 6))
            plt.scatter(precision_results['precision_lr_num'], precision_results['reconstruction_mse'])
            plt.xlabel('Precision Learning Rate')
            plt.ylabel('Reconstruction MSE')
            plt.title('Reconstruction Quality vs Precision Learning Rate')
            plt.xscale('log')
            plt.savefig('precision_lr_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

    def generate_comprehensive_report(self, output_file='validation_report.txt'):
        """包括的な検証レポートを生成"""
        print("\n=== 包括的検証レポート生成 ===")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("階層型VAE検証実験レポート\n")
            f.write("=" * 50 + "\n\n")

            # 実験概要
            f.write(f"実験数: {len(self.results_df)}\n")
            f.write(
                f"成功率: {len(self.results_df[self.results_df['status'] == 'completed']) / len(self.results_df) * 100:.1f}%\n\n")

            # 最優秀実験の特定
            if 'reconstruction_mse' in self.results_df.columns:
                best_exp = self.results_df.loc[self.results_df['reconstruction_mse'].idxmin()]
                f.write("最優秀実験:\n")
                f.write(f"  実験名: {best_exp['experiment_name']}\n")
                f.write(f"  再構成MSE: {best_exp['reconstruction_mse']:.6f}\n")
                if 'style_separation_score' in best_exp:
                    f.write(f"  スタイル分離ARI: {best_exp['style_separation_score']:.4f}\n")
                if 'skill_performance_correlation' in best_exp:
                    f.write(f"  スキル相関: {best_exp['skill_performance_correlation']:.4f}\n")
                f.write("\n")

            # 推奨設定
            f.write("推奨ハイパーパラメータ設定:\n")

            # 各フェーズの最適解
            for phase in ['phase1', 'phase2', 'phase3']:
                phase_data = self.results_df[self.results_df['phase'] == phase]
                if len(phase_data) > 0 and 'reconstruction_mse' in phase_data.columns:
                    best_in_phase = phase_data.loc[phase_data['reconstruction_mse'].idxmin()]
                    f.write(f"  {phase}: {best_in_phase['experiment_name']}\n")

            f.write("\n詳細な分析結果は生成された画像ファイルを参照してください。\n")

        print(f"レポートを保存しました: {output_file}")

    def run_complete_analysis(self):
        """全ての分析を実行"""
        print("=== 階層型VAE検証結果分析開始 ===")

        self.analyze_phase1_sensitivity()
        self.analyze_phase2_hierarchy()
        self.analyze_phase3_prediction_errors()
        self.generate_comprehensive_report()

        print("\n=== 分析完了 ===")
        print("生成されたファイル:")
        for file in ['beta_schedule_analysis.png', 'latent_dimension_analysis.png',
                     'hidden_dimension_analysis.png', 'learning_schedule_analysis.png',
                     'prediction_error_analysis.png', 'precision_lr_analysis.png',
                     'validation_report.txt']:
            if os.path.exists(file):
                print(f"  - {file}")


def main():
    analyzer = HierarchicalVAEValidationAnalyzer("hierarchical_experiments.db")
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()