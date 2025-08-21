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
VALUES ('beta_vae_experiments', '1.0', 'Beta VAE experiment tracking database');

-- 階層型VAE実験テーブル
CREATE TABLE IF NOT EXISTS beta_vae_generalized_coordinate_experiments (
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
    style_latent_dim INTEGER,
    skill_latent_dim INTEGER,
    beta REAL,
    n_layers INTEGER,
    contrastive_weight REAL,
    use_triplet BOOLEAN,

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
    final_kl_style REAL,
    final_kl_skill REAL,
    final_contrastive_loss REAL,

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
CREATE INDEX IF NOT EXISTS idx_experiments_status ON beta_vae_generalized_coordinate_experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON beta_vae_generalized_coordinate_experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_experiments_tags ON beta_vae_generalized_coordinate_experiments(tags);
CREATE INDEX IF NOT EXISTS idx_experiments_performance ON beta_vae_generalized_coordinate_experiments(reconstruction_mse);

-- トリガー: updated_at自動更新
CREATE TRIGGER IF NOT EXISTS update_experiments_timestamp
    AFTER UPDATE ON beta_vae_generalized_coordinate_experiments
    FOR EACH ROW
BEGIN
    UPDATE beta_vae_generalized_coordinate_experiments
    SET updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.id;
END;