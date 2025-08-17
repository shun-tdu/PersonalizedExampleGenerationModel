-- 階層型VAE実験管理用データベーススキーマ
-- 自由エネルギー原理に基づく予測符号化アーキテクチャ対応

-- =====================================================
-- メイン実験テーブル
-- =====================================================
CREATE TABLE IF NOT EXISTS hierarchical_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- 基本実験情報
    experiment_name TEXT NOT NULL,
    description TEXT,
    config_path TEXT,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    start_time TEXT,
    end_time TEXT,
    duration_minutes REAL,

    -- データ設定
    data_path TEXT,
    input_dim INTEGER,
    seq_len INTEGER,

    -- モデル基本設定
    hidden_dim INTEGER,
    primitive_latent_dim INTEGER,
    skill_latent_dim INTEGER,
    style_latent_dim INTEGER,

    -- 階層別β値（複雑性制約）
    beta_primitive REAL,
    beta_skill REAL,
    beta_style REAL,
    precision_lr REAL,

    -- 学習設定
    batch_size INTEGER,
    num_epochs INTEGER,
    lr REAL,
    weight_decay REAL,
    clip_grad_norm REAL,

    -- スケジューラ設定
    warmup_epochs INTEGER,
    scheduler_T_0 INTEGER,
    scheduler_T_mult INTEGER,
    scheduler_eta_min REAL,
    patience INTEGER,

    -- 階層別学習スケジュール
    primitive_learning_start REAL,  -- 0.0
    skill_learning_start REAL,      -- 0.3
    style_learning_start REAL,      -- 0.6

    -- 予測誤差重み（階層別）
    prediction_error_weight_level1 REAL,  -- 10.0 データレベル
    prediction_error_weight_level2 REAL,  -- 1.0 スキルレベル
    prediction_error_weight_level3 REAL,  -- 0.1 スタイルレベル

    -- お手本生成設定
    skill_enhancement_factor REAL,      -- 0.1
    style_preservation_weight REAL,     -- 1.0
    max_enhancement_steps INTEGER,      -- 5

    -- 学習結果
    final_total_loss REAL,
    best_val_loss REAL,
    final_epoch INTEGER,
    early_stopped BOOLEAN DEFAULT FALSE,

    -- 階層別損失（最終値）
    final_recon_loss REAL,
    final_prediction_error REAL,
    final_kl_primitive REAL,
    final_kl_skill REAL,
    final_kl_style REAL,

    -- 評価指標
    reconstruction_mse REAL,
    style_separation_score REAL,        -- ARI score for subject separation
    skill_performance_correlation REAL, -- 最大相関値
    primitive_reconstruction_quality REAL,

    -- スキル軸分析結果
    skill_axis_analysis_completed BOOLEAN DEFAULT FALSE,
    overall_skill_axis_r_squared REAL,
    best_skill_correlation_metric TEXT,
    best_skill_correlation_value REAL,
    skill_improvement_directions_count INTEGER,

    -- 軸ベース改善評価
    axis_based_improvement_enabled BOOLEAN DEFAULT FALSE,
    exemplar_method_comparison_completed BOOLEAN DEFAULT FALSE,
    axis_vs_random_performance_ratio REAL,  -- 軸ベース/ランダムの性能比

    -- ファイルパス
    model_path TEXT,
    best_model_path TEXT,
    config_backup_path TEXT,

    -- 可視化・結果ファイル
    training_curves_path TEXT,
    hierarchical_latent_visualization_path TEXT,
    skill_axes_visualization_path TEXT,
    personalized_exemplars_path TEXT,
    axis_based_exemplars_path TEXT,
    exemplar_comparison_path TEXT,
    evaluation_results_path TEXT,

    -- アブレーション・比較実験
    is_ablation_study BOOLEAN DEFAULT FALSE,
    ablation_type TEXT,  -- 'no_hierarchy', 'different_beta_schedules', 'fixed_vs_adaptive_precision'
    baseline_experiment_id INTEGER,  -- 比較対象の実験ID

    -- メタ情報
    git_commit_hash TEXT,
    python_version TEXT,
    pytorch_version TEXT,
    cuda_version TEXT,
    gpu_info TEXT,
    system_info TEXT,
    random_seed INTEGER,

    -- タグ・注釈
    tags TEXT,  -- JSON array format: ["hierarchical_vae", "free_energy_principle"]
    notes TEXT,
    researcher_comments TEXT,

    -- タイムスタンプ
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 外部キー制約
    FOREIGN KEY (baseline_experiment_id) REFERENCES hierarchical_experiments(id)
);

-- =====================================================
-- 階層別詳細分析テーブル
-- =====================================================
CREATE TABLE IF NOT EXISTS hierarchy_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    hierarchy_level TEXT NOT NULL CHECK (hierarchy_level IN ('primitive', 'skill', 'style')),

    -- 基本統計
    latent_dim INTEGER,
    mean_activation REAL,
    std_activation REAL,
    sparsity_ratio REAL,  -- ゼロに近い値の割合

    -- 分散表現評価
    disentanglement_score REAL,
    mutual_information_score REAL,
    beta_vae_metric REAL,
    sap_score REAL,  -- Separated Attribute Predictability
    mig_score REAL,  -- Mutual Information Gap

    -- 相関分析
    max_correlation_with_performance REAL,
    avg_correlation_with_performance REAL,
    most_correlated_performance_metric TEXT,

    -- 生成品質
    reconstruction_quality REAL,
    interpolation_smoothness REAL,
    extrapolation_stability REAL,

    -- 階層特異性
    cross_hierarchy_correlation REAL,  -- 他階層との相関（低いほど良い）
    within_hierarchy_consistency REAL,  -- 階層内の一貫性

    -- 予測符号化評価
    prediction_accuracy REAL,
    prediction_error_magnitude REAL,
    error_reduction_ratio REAL,  -- 上位からの予測による誤差削減率

    -- 計算日時
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (experiment_id) REFERENCES hierarchical_experiments(id)
);

-- =====================================================
-- スキル軸分析テーブル
-- =====================================================
CREATE TABLE IF NOT EXISTS skill_axis_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,

    -- 分析対象
    target_performance_metric TEXT NOT NULL,
    skill_dimension INTEGER,

    -- 相関結果
    correlation_coefficient REAL,
    p_value REAL,
    confidence_interval_lower REAL,
    confidence_interval_upper REAL,

    -- 回帰分析結果
    r_squared REAL,
    regression_coefficient REAL,
    regression_intercept REAL,
    regression_p_value REAL,

    -- 改善方向
    improvement_direction_vector TEXT,  -- JSON配列として保存
    improvement_direction_magnitude REAL,
    improvement_direction_confidence REAL,

    -- 検証結果
    cross_validation_score REAL,
    holdout_validation_score REAL,
    improvement_consistency_score REAL,

    -- 分析メタ情報
    sample_size INTEGER,
    outliers_removed INTEGER,
    preprocessing_method TEXT,
    analysis_method TEXT,

    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (experiment_id) REFERENCES hierarchical_experiments(id)
);

-- =====================================================
-- 個人最適化お手本評価テーブル
-- =====================================================
CREATE TABLE IF NOT EXISTS exemplar_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    subject_id TEXT NOT NULL,

    -- お手本生成方法
    generation_method TEXT NOT NULL CHECK (generation_method IN ('random_noise', 'axis_based', 'hybrid')),
    target_metric TEXT,  -- 'overall', 'trial_time', 'trial_error', etc.
    enhancement_factor REAL,

    -- スタイル保持評価
    style_preservation_score REAL,     -- [0, 1] 1に近いほど良い
    style_deviation_magnitude REAL,    -- スタイルからの逸脱量
    individual_characteristic_retained BOOLEAN,

    -- スキル向上評価
    skill_enhancement_score REAL,      -- 向上の適切さ
    predicted_performance_improvement REAL,
    actual_performance_improvement REAL,  -- 実際の検証結果（あれば）
    improvement_direction_accuracy REAL,

    -- 軌道品質評価
    trajectory_smoothness REAL,
    trajectory_realism REAL,
    biomechanical_plausibility REAL,
    motion_consistency REAL,

    -- 予測誤差の最適性
    prediction_error_magnitude REAL,
    optimal_error_range_compliance BOOLEAN,
    learning_challenge_appropriateness REAL,  -- 自由エネルギー原理適合性

    -- 学習効果予測
    estimated_learning_acceleration REAL,
    motivation_sustainability_score REAL,
    cognitive_load_appropriateness REAL,
    transfer_learning_potential REAL,

    -- 比較評価（他手法との）
    vs_random_improvement_ratio REAL,
    vs_uniform_exemplar_advantage REAL,
    vs_expert_demonstration_similarity REAL,

    -- 信頼性・統計
    confidence_score REAL,
    statistical_significance REAL,
    sample_variance REAL,

    -- 評価メタ情報
    evaluation_method TEXT,
    evaluator_type TEXT,  -- 'automated', 'expert_human', 'user_study'
    evaluation_criteria TEXT,  -- JSON format

    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (experiment_id) REFERENCES hierarchical_experiments(id)
);

-- =====================================================
-- 学習過程詳細テーブル
-- =====================================================
CREATE TABLE IF NOT EXISTS training_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    epoch INTEGER NOT NULL,

    -- 基本損失
    train_total_loss REAL,
    val_total_loss REAL,

    -- 階層別損失
    train_recon_loss REAL,
    train_prediction_error REAL,
    train_kl_primitive REAL,
    train_kl_skill REAL,
    train_kl_style REAL,

    val_recon_loss REAL,
    val_prediction_error REAL,
    val_kl_primitive REAL,
    val_kl_skill REAL,
    val_kl_style REAL,

    -- 学習状態
    learning_rate REAL,
    effective_beta_primitive REAL,  -- エポックに応じて変化
    effective_beta_skill REAL,
    effective_beta_style REAL,

    -- 性能指標
    reconstruction_mse REAL,
    gradient_norm REAL,
    weight_norm REAL,

    -- 分散表現品質（サンプリング）
    style_separation_sample REAL,
    skill_correlation_sample REAL,

    -- 時間情報
    epoch_duration_seconds REAL,
    cumulative_time_seconds REAL,

    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (experiment_id) REFERENCES hierarchical_experiments(id)
);

-- =====================================================
-- パフォーマンス比較テーブル
-- =====================================================
CREATE TABLE IF NOT EXISTS performance_comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_experiment_id INTEGER NOT NULL,
    comparison_experiment_id INTEGER NOT NULL,
    comparison_type TEXT NOT NULL,  -- 'ablation', 'hyperparameter', 'architecture'

    -- 比較指標
    reconstruction_mse_improvement REAL,
    style_separation_improvement REAL,
    skill_correlation_improvement REAL,
    exemplar_quality_improvement REAL,

    -- 統計的有意性
    statistical_test_type TEXT,
    p_value REAL,
    effect_size REAL,
    confidence_interval TEXT,  -- JSON format

    -- 計算効率比較
    training_time_ratio REAL,
    memory_usage_ratio REAL,
    inference_time_ratio REAL,

    -- 定性的評価
    improvement_summary TEXT,
    advantages TEXT,
    disadvantages TEXT,
    recommendations TEXT,

    compared_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (baseline_experiment_id) REFERENCES hierarchical_experiments(id),
    FOREIGN KEY (comparison_experiment_id) REFERENCES hierarchical_experiments(id)
);

-- =====================================================
-- インデックス作成
-- =====================================================
CREATE INDEX IF NOT EXISTS idx_hierarchical_experiments_status ON hierarchical_experiments(status);
CREATE INDEX IF NOT EXISTS idx_hierarchical_experiments_created_at ON hierarchical_experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_hierarchical_experiments_tags ON hierarchical_experiments(tags);
CREATE INDEX IF NOT EXISTS idx_hierarchical_experiments_ablation ON hierarchical_experiments(is_ablation_study, ablation_type);

CREATE INDEX IF NOT EXISTS idx_hierarchy_analysis_experiment ON hierarchy_analysis(experiment_id, hierarchy_level);
CREATE INDEX IF NOT EXISTS idx_skill_axis_experiment_metric ON skill_axis_analysis(experiment_id, target_performance_metric);
CREATE INDEX IF NOT EXISTS idx_exemplar_evaluations_experiment_method ON exemplar_evaluations(experiment_id, generation_method);
CREATE INDEX IF NOT EXISTS idx_training_progress_experiment_epoch ON training_progress(experiment_id, epoch);
CREATE INDEX IF NOT EXISTS idx_performance_comparisons_baseline ON performance_comparisons(baseline_experiment_id);

-- =====================================================
-- 便利なビュー
-- =====================================================

-- 実験概要ビュー
CREATE VIEW IF NOT EXISTS experiment_summary AS
SELECT
    id,
    experiment_name,
    status,
    CASE
        WHEN status = 'completed' THEN 'Success'
        WHEN status = 'failed' THEN 'Failed'
        WHEN status = 'running' THEN 'In Progress'
        ELSE 'Pending'
    END as status_display,
    reconstruction_mse,
    style_separation_score,
    skill_performance_correlation,
    skill_axis_analysis_completed,
    axis_based_improvement_enabled,
    final_total_loss,
    best_val_loss,
    duration_minutes,
    tags,
    start_time,
    end_time,
    created_at
FROM hierarchical_experiments
ORDER BY created_at DESC;

-- 最高性能実験ビュー
CREATE VIEW IF NOT EXISTS best_experiments AS
SELECT
    *,
    RANK() OVER (ORDER BY reconstruction_mse ASC) as mse_rank,
    RANK() OVER (ORDER BY style_separation_score DESC) as separation_rank,
    RANK() OVER (ORDER BY skill_performance_correlation DESC) as correlation_rank
FROM hierarchical_experiments
WHERE status = 'completed'
ORDER BY (mse_rank + separation_rank + correlation_rank) ASC;

-- スキル軸分析サマリービュー
CREATE VIEW IF NOT EXISTS skill_axis_summary AS
SELECT
    e.experiment_name,
    saa.target_performance_metric,
    saa.r_squared,
    saa.correlation_coefficient,
    saa.improvement_direction_confidence,
    saa.analyzed_at
FROM skill_axis_analysis saa
JOIN hierarchical_experiments e ON saa.experiment_id = e.id
WHERE saa.r_squared > 0.1  -- 意味のある相関のみ
ORDER BY saa.r_squared DESC;

-- お手本評価比較ビュー
CREATE VIEW IF NOT EXISTS exemplar_method_comparison AS
SELECT
    e.experiment_name,
    ee.subject_id,
    ee.generation_method,
    AVG(ee.style_preservation_score) as avg_style_preservation,
    AVG(ee.skill_enhancement_score) as avg_skill_enhancement,
    AVG(ee.trajectory_smoothness) as avg_smoothness,
    AVG(ee.learning_challenge_appropriateness) as avg_challenge_appropriateness,
    COUNT(*) as evaluation_count
FROM exemplar_evaluations ee
JOIN hierarchical_experiments e ON ee.experiment_id = e.id
GROUP BY e.experiment_name, ee.subject_id, ee.generation_method
ORDER BY avg_skill_enhancement DESC;

-- =====================================================
-- トリガー（自動更新）
-- =====================================================

-- 更新時刻の自動更新
CREATE TRIGGER IF NOT EXISTS update_hierarchical_experiments_timestamp
    AFTER UPDATE ON hierarchical_experiments
BEGIN
    UPDATE hierarchical_experiments
    SET updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.id;
END;

-- 実験完了時の継続時間計算
CREATE TRIGGER IF NOT EXISTS calculate_duration_on_completion
    AFTER UPDATE OF status ON hierarchical_experiments
    WHEN NEW.status = 'completed' AND OLD.status != 'completed'
BEGIN
    UPDATE hierarchical_experiments
    SET duration_minutes = (
        CAST((julianday(NEW.end_time) - julianday(NEW.start_time)) * 24 * 60 AS REAL)
    )
    WHERE id = NEW.id;
END;

-- =====================================================
-- 初期データ・設定値
-- =====================================================

-- 実験タグの定義
INSERT OR IGNORE INTO sqlite_master (type, name, tbl_name, sql) VALUES
('table', 'experiment_tags', 'experiment_tags', '-- Virtual table for tag definitions');

-- 評価基準の定義
INSERT OR IGNORE INTO sqlite_master (type, name, tbl_name, sql) VALUES
('table', 'evaluation_criteria', 'evaluation_criteria', '-- Virtual table for evaluation criteria');

-- =====================================================
-- データベース設定
-- =====================================================

-- 外部キー制約を有効化
PRAGMA foreign_keys = ON;

-- WALモードでパフォーマンス向上
PRAGMA journal_mode = WAL;

-- 自動バキューム
PRAGMA auto_vacuum = INCREMENTAL;

-- =====================================================
-- 初期化完了メッセージ用ビュー
-- =====================================================
CREATE VIEW IF NOT EXISTS database_info AS
SELECT
    'Hierarchical VAE Experiment Database' as name,
    '1.0.0' as version,
    'Free Energy Principle based Predictive Coding Architecture' as description,
    CURRENT_TIMESTAMP as initialized_at,
    (SELECT COUNT(*) FROM hierarchical_experiments) as total_experiments,
    (SELECT COUNT(*) FROM hierarchical_experiments WHERE status = 'completed') as completed_experiments;