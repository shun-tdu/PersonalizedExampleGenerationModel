-- hierarchical_vae_schema.sql
-- 階層型VAE実験管理用データベーススキーマ

-- データベース情報テーブル
CREATE TABLE IF NOT EXISTS database_info (
    name TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 初期データ挿入
INSERT OR REPLACE INTO database_info (name, version, description)
VALUES ('hierarchical_vae_experiments', '1.0', 'Hierarchical VAE experiment tracking database');

-- 階層型VAE実験テーブル
CREATE TABLE IF NOT EXISTS hierarchical_experiments (
    -- 基本情報
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT NOT NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'dry_run')),
    config_path TEXT,
    config_backup_path TEXT,
    description TEXT,
    tags TEXT, -- JSON形式

    -- タイムスタンプ
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- データ設定
    data_path TEXT,

    -- モデル設定
    input_dim INTEGER,
    seq_len INTEGER,
    hidden_dim INTEGER,
    primitive_latent_dim INTEGER,
    skill_latent_dim INTEGER,
    style_latent_dim INTEGER,

    -- 階層別β値
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
    warmup_epochs INTEGER,
    scheduler_T_0 INTEGER,
    scheduler_T_mult INTEGER,
    scheduler_eta_min REAL,
    patience INTEGER,

    -- 階層型VAE特有設定
    primitive_learning_start REAL,
    skill_learning_start REAL,
    style_learning_start REAL,

    -- 予測誤差重み
    prediction_error_weight_level1 REAL,
    prediction_error_weight_level2 REAL,
    prediction_error_weight_level3 REAL,

    -- お手本生成設定
    skill_enhancement_factor REAL,
    style_preservation_weight REAL,
    max_enhancement_steps INTEGER,

    -- 実験結果
    final_total_loss REAL,
    best_val_loss REAL,
    final_epoch INTEGER,
    early_stopped BOOLEAN,

    -- 階層別最終損失
    final_recon_loss REAL,
    final_kl_primitive REAL,
    final_kl_skill REAL,
    final_kl_style REAL,

    -- 評価指標
    reconstruction_mse REAL,
    style_separation_score REAL,
    skill_performance_correlation REAL,
    best_skill_correlation_metric TEXT,

    -- スキル軸分析結果
    skill_axis_analysis_completed BOOLEAN DEFAULT FALSE,
    skill_improvement_directions_count INTEGER DEFAULT 0,
    axis_based_improvement_enabled BOOLEAN DEFAULT FALSE,

    -- ファイルパス
    model_path TEXT,
    best_model_path TEXT,
    training_curves_path TEXT,
    axis_based_exemplars_path TEXT,
    evaluation_results_path TEXT,

    -- ログ設定
    output_dir TEXT,

    -- システム情報
    python_version TEXT,
    pytorch_version TEXT,
    cuda_version TEXT,
    gpu_info TEXT,
    system_info TEXT,

    -- アブレーション研究
    is_ablation_study BOOLEAN DEFAULT FALSE,

    -- 備考
    notes TEXT
);

-- インデックス作成
CREATE INDEX IF NOT EXISTS idx_experiments_status ON hierarchical_experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON hierarchical_experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_experiments_tags ON hierarchical_experiments(tags);
CREATE INDEX IF NOT EXISTS idx_experiments_performance ON hierarchical_experiments(reconstruction_mse);

-- トリガー: updated_at自動更新
CREATE TRIGGER IF NOT EXISTS update_experiments_timestamp
    AFTER UPDATE ON hierarchical_experiments
    FOR EACH ROW
BEGIN
    UPDATE hierarchical_experiments
    SET updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.id;
END;