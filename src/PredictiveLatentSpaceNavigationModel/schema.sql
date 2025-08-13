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

    -- 結果
    final_total_loss REAL,
    best_val_loss REAL,
    model_path TEXT,
    image_path TEXT,

    -- 作成・更新時刻
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);