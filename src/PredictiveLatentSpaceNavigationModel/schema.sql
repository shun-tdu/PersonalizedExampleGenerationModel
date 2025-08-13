-- 実験管理テーブル
CREATE TABLE IF NOT EXISTS experiments(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    config_file TEXT,
    notes TEXT,
    git_commit_hash TEXT,

    -- ハイパーパラメータ
    lr REAL,
    batch_size INTEGER,
    num_epochs INTEGER,
    hidden_dim INTEGER,
    style_latent_dim INTEGER,
    skill_latent_dim INTEGER,
    n_layers INTEGER,

    -- 学習結果
    final_total_loss REAL,
    best_val_loss REAL,
    model_path TEXT,
    image_path TEXT,

    -- 評価結果
    reconstruction_mse REAL,
    latent_visualization_path TEXT,
    generated_trajectories_path TEXT,
    evaluation_results_path TEXT,

    -- 相関分析結果(JSON形式で保存)
    skill_correlations TEXT,

    -- 作成・更新時刻
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);