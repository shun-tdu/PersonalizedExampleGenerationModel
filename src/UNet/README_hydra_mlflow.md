# CLAUDE_ADDED
# Hydra + MLFlow統合 UNet拡散モデル

このドキュメントでは、HydraとMLFlowを統合したUNet拡散モデルの使用方法を説明します。

## 概要

- **Hydra**: 設定管理とハイパーパラメータ調整
- **MLFlow**: 実験トラッキングとモデル管理
- **UNet拡散モデル**: 個人特性に基づく軌道生成

## セットアップ

### 1. 依存関係のインストール
```bash
# Docker環境の場合
docker-compose exec app pip install -r requirements.txt

# ローカル環境の場合
pip install -r requirements.txt
```

### 2. MLFlow UIの起動

**重要**: MLFlow UIにアクセスするためには、docker-compose.ymlで以下のポートマッピングが設定されている必要があります：

```yaml
services:
  app:
    ports:
      - "8888:8888"  # Jupyter
      - "5000:5000"  # MLFlow UI
```

```bash
# Dockerコンテナ内で実行
cd /app/src/UNet
mlflow ui --host 0.0.0.0 --port 5000
```

ブラウザで `http://localhost:5000` にアクセスしてMLFlow UIを確認できます。

## 使用方法

### 1. 訓練実行

#### Hydra設定ファイルを使用した訓練
```bash
cd /app/src/UNet
python train.py
```

#### 設定をオーバーライドして訓練
```bash
# エポック数を変更
python train.py training.epochs=50

# バッチサイズとラーニングレートを変更
python train.py training.batch_size=16 training.learning_rate=5e-5

# ダミーデータを使用
python train.py training.use_dummy=true

# 実データを使用
python train.py data.train_data=data/Datasets/overfitting_dataset_cond_3.npz
```

#### 複数実験の並列実行
```bash
# 異なるハイパーパラメータで複数実験を実行
python train.py -m training.learning_rate=1e-4,5e-5,1e-5 training.batch_size=16,32,64
```

### 2. 軌道生成

#### Hydra設定ファイルを使用した生成
```bash
python -c "from generate import hydra_generate; hydra_generate()"
```

#### 従来のコマンドライン引数を使用した生成
```bash
python generate.py --checkpoint checkpoints/checkpoint_epoch_100.pth --data_path data/Datasets/overfitting_dataset_cond_3.npz --batch_size 8 --method ddim --steps 50
```

### 3. モデル評価

```bash
python evaluate.py
```

評価結果は `evaluation_results/` ディレクトリに保存され、MLFlowにも記録されます。

## 設定ファイル (config.yaml)

### 主要な設定項目

```yaml
# モデル設定
model:
  input_dim: 2
  condition_dim: 5  # 自動検出されます
  time_embed_dim: 128
  base_channels: 64

# 訓練設定
training:
  batch_size: 32
  epochs: 100
  learning_rate: 1e-4
  save_interval: 10
  use_dummy: false

# データ設定
data:
  train_data: "data/Datasets/overfitting_dataset_cond_3.npz"
  val_data: null
  val_split_ratio: 0.2

# 生成設定
generation:
  method: "ddpm"  # ddpm or ddim
  num_inference_steps: null
  seq_len: 101
  num_samples: 8

# MLFlow設定
mlflow:
  tracking_uri: "mlruns"
  experiment_name: "unet_diffusion_experiments"
  run_name: null
  log_model: true
  log_artifacts: true
```

### 設定のオーバーライド例

```bash
# 異なる生成手法で実験
python train.py generation.method=ddim generation.num_inference_steps=50

# 異なるモデル構造で実験
python train.py model.base_channels=128 model.time_embed_dim=256

# 異なるデータセットで実験
python train.py data.train_data=path/to/your/data.npz
```

## MLFlowでの実験管理

### 実験の確認
1. MLFlow UIで実験履歴を確認
2. パラメータ、メトリクス、アーティファクトを比較
3. 最適なモデルを特定

### ログされる情報
- **パラメータ**: Hydra設定、モデルパラメータ数
- **メトリクス**: 訓練損失、検証損失、評価メトリクス
- **アーティファクト**: チェックポイント、訓練曲線、生成軌道

### モデルの取得
```python
import mlflow.pytorch

# MLFlowからモデルをロード
model_uri = "runs:/<RUN_ID>/model"
model = mlflow.pytorch.load_model(model_uri)
```

## 評価メトリクス

評価スクリプトは以下のメトリクスを計算します：

### 1. 軌道多様性
- 平均ペアワイズ距離
- 開始点・終点の多様性

### 2. 軌道滑らかさ
- 速度、加速度、ジャークの統計
- 滑らかさスコア

### 3. 条件一貫性
- 条件と軌道特徴の相関
- 平均絶対相関

### 4. 終点精度
- 平均終点誤差
- 終点誤差の分布

## トラブルシューティング

### よくある問題

1. **CUDA メモリ不足**
   ```bash
   python train.py training.batch_size=16  # バッチサイズを削減
   ```

2. **条件次元の不一致**
   - モデルが自動的にチェックポイントから条件次元を検出します
   - 実データを使用する場合は、データの条件次元が一致することを確認してください

3. **MLFlow UI にアクセスできない**
   - docker-compose.ymlでポートマッピング `"5000:5000"` が設定されていることを確認
   - コンテナを再起動： `docker-compose down && docker-compose up -d`
   ```bash
   # Dockerの場合、ポートマッピングを確認
   docker-compose exec app mlflow ui --host 0.0.0.0 --port 5000
   ```

### デバッグ情報の確認
```bash
# Hydra設定の確認
python train.py --cfg job

# 詳細ログの出力
python train.py hydra.verbose=true
```

## ファイル構成

```
src/UNet/
├── config.yaml              # Hydra設定ファイル
├── train.py                 # Hydra+MLFlow統合訓練スクリプト
├── generate.py              # Hydra統合生成スクリプト
├── evaluate.py              # 評価スクリプト
├── Model.py                 # UNetモデル定義
├── TrajectoryDataset.py     # データセットクラス
├── visualize.py             # 可視化スクリプト
├── visualization.ipynb     # Jupyter可視化ノートブック
├── mlruns/                  # MLFlow実験データ
├── checkpoints/             # モデルチェックポイント
├── evaluation_results/      # 評価結果
└── generated_trajectories/  # 生成結果
```

## Docker環境での使用

```bash
# コンテナに入る
docker-compose exec app bash

# UNetディレクトリに移動
cd /app/src/UNet

# 訓練実行
python train.py

# MLFlow UI起動
mlflow ui --host 0.0.0.0 --port 5000
```

## 高度な使用方法

### カスタム設定ファイルの作成
```yaml
# custom_config.yaml
defaults:
  - config

# カスタム設定
training:
  epochs: 200
  batch_size: 64

model:
  base_channels: 128
```

```bash
python train.py --config-name=custom_config
```

### Hydra Sweeps (ハイパーパラメータ調整)
```bash
# config.yamlに以下を追加
# hydra:
#   sweep:
#     dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
#     subdir: ${hydra:job.override_dirname}

python train.py -m training.learning_rate=1e-4,5e-5,1e-5 training.batch_size=16,32
```

これで、HydraとMLFlowを活用した効率的な実験管理が可能になります。

● HydraとMLFlowを統合したUNet拡散モデルの開発フローについて、段階的に解説します。

  開発フロー全体図

  1. 環境準備・設定
      ↓
  2. データ準備・確認
      ↓
  3. モデル訓練・実験管理
      ↓
  4. 結果分析・比較
      ↓
  5. モデル評価
      ↓
  6. 軌道生成・可視化
      ↓
  7. 最適化・改善

  1. 環境準備・設定 🔧

  1.1 Docker環境セットアップ

  # プロジェクトディレクトリに移動
  cd /path/to/PersonalizedExampleGeneration

  # Dockerコンテナを起動
  docker-compose up -d

  # コンテナに入る
  docker-compose exec app bash

  1.2 依存関係インストール

  # コンテナ内で実行
  pip install -r requirements.txt

  1.3 MLFlow UI起動

  cd /app/src/UNet
  mlflow ui --host 0.0.0.0 --port 5000 &
  → ブラウザで http://localhost:5000 を開いてMLFlow UIを確認

  2. データ準備・確認 📊

  2.1 データセットの確認

  # データの存在確認
  ls /data/Datasets/

  # データセットの内容確認（Jupyter使用推奨）
  python -c "
  import numpy as np
  data = np.load('/data/Datasets/overfitting_dataset_cond_3.npz')
  print('Available keys:', list(data.keys()))
  print('Trajectories shape:', data['trajectories'].shape)
  print('Conditions shape:', data['conditions'].shape)
  "

  2.2 設定ファイルの調整

  # config.yamlを編集してデータパスを設定
  vim config.yaml

  # データ設定例
  data:
    train_data: "data/Datasets/overfitting_dataset_cond_3.npz"
    val_data: null
    val_split_ratio: 0.2

  3. モデル訓練・実験管理 🚀

  3.1 ベースライン実験

  # デフォルト設定で初回訓練
  python train.py

  # 実行後、MLFlow UIで結果を確認
  # - Experiment: unet_diffusion_experiments
  # - Run名、パラメータ、メトリクスを確認

  3.2 ハイパーパラメータ実験

  # 学習率の実験
  python train.py training.learning_rate=5e-5 mlflow.run_name="lr_5e-5"
  python train.py training.learning_rate=1e-5 mlflow.run_name="lr_1e-5"

  # バッチサイズの実験
  python train.py training.batch_size=16 mlflow.run_name="batch_16"
  python train.py training.batch_size=64 mlflow.run_name="batch_64"

  # モデル構造の実験
  python train.py model.base_channels=128 mlflow.run_name="channels_128"

  3.3 並列実験（Hydra Multirun）

  # 複数の学習率を同時実験
  python train.py -m training.learning_rate=1e-4,5e-5,1e-5 \
    mlflow.run_name="lr_sweep"

  # 学習率×バッチサイズのグリッドサーチ
  python train.py -m \
    training.learning_rate=1e-4,5e-5 \
    training.batch_size=16,32 \
    mlflow.run_name="grid_search"

  4. 結果分析・比較 📈

  4.1 MLFlow UIでの比較

  1. 実験一覧の確認
    - Experiments → unet_diffusion_experiments
    - 各Runの訓練損失、検証損失を比較
  2. パラメータ vs メトリクスの分析
    - Compare runs機能を使用
    - 最適なハイパーパラメータ組み合わせを特定
  3. 訓練曲線の確認
    - Artifacts → plots → training_curves.png
    - 過学習の有無、収束状況を確認

  4.2 プログラマティックな分析

  import mlflow
  import pandas as pd

  # 実験結果の取得
  client = mlflow.tracking.MlflowClient()
  experiment = client.get_experiment_by_name("unet_diffusion_experiments")
  runs = client.search_runs(experiment.experiment_id)

  # DataFrame化して分析
  runs_df = pd.DataFrame([{
      'run_id': run.info.run_id,
      'learning_rate': run.data.params.get('training.learning_rate'),
      'batch_size': run.data.params.get('training.batch_size'),
      'final_train_loss': run.data.metrics.get('train_loss'),
      'final_val_loss': run.data.metrics.get('val_loss')
  } for run in runs])

  print(runs_df.sort_values('final_val_loss'))

  5. モデル評価 🎯

  5.1 最適モデルの特定

  # MLFlow UIで最適なRunを特定後、そのRun IDを使用
  export BEST_RUN_ID="your_best_run_id"

  # または最新のチェックポイントを使用
  ls checkpoints/ | tail -1

  5.2 包括的評価の実行

  # 評価スクリプト実行
  python evaluate.py

  # 結果確認
  ls evaluation_results/
  cat evaluation_results/evaluation_metrics.json

  5.3 評価結果の分析

  # 主要メトリクスの確認
  python -c "
  import json
  with open('evaluation_results/evaluation_metrics.json') as f:
      metrics = json.load(f)

  print('=== 主要評価メトリクス ===')
  print(f'軌道多様性: {metrics.get(\"diversity_mean_pairwise_distance\", 0):.4f}')
  print(f'滑らかさ: {metrics.get(\"smoothness_smoothness_score\", 0):.4f}')
  print(f'終点精度: {metrics.get(\"endpoint_mean_endpoint_error\", 0):.4f}')
  print(f'条件一貫性: {metrics.get(\"consistency_mean_abs_correlation\", 0):.4f}')
  "

  6. 軌道生成・可視化 🎨

  6.1 軌道生成

  # DDPM生成
  python generate.py \
    --checkpoint checkpoints/checkpoint_epoch_100.pth \
    --data_path data/Datasets/overfitting_dataset_cond_3.npz \
    --method ddpm \
    --batch_size 16

  # DDIM生成（高速）
  python generate.py \
    --checkpoint checkpoints/checkpoint_epoch_100.pth \
    --data_path data/Datasets/overfitting_dataset_cond_3.npz \
    --method ddim \
    --steps 50 \
    --batch_size 16

  6.2 Hydra統合生成

  # 設定ファイル経由での生成
  python -c "from generate import hydra_generate; hydra_generate()"

  # 設定オーバーライド
  python -c "from generate import hydra_generate; hydra_generate()" \
    generation.method=ddim \
    generation.num_samples=32

  6.3 可視化

  # 基本可視化
  python visualize.py --checkpoint checkpoints/checkpoint_epoch_100.pth

  # Jupyter Notebookでの詳細分析
  jupyter notebook visualization.ipynb

  7. 最適化・改善 🔄

  7.1 結果に基づく改善策の検討

  性能が低い場合の対処法:

  # 1. より長い訓練
  python train.py training.epochs=200

  # 2. モデル構造の拡張
  python train.py model.base_channels=128 model.time_embed_dim=256

  # 3. 学習率スケジューリング（カスタム実装が必要）
  python train.py training.learning_rate=1e-4

  # 4. データ拡張や前処理の改善
  # → TrajectoryDataset.pyの修正が必要

  7.2 継続的な実験管理

  # 定期的な実験の実行
  for lr in 1e-4 5e-5 1e-5; do
    for batch in 16 32 64; do
      python train.py \
        training.learning_rate=$lr \
        training.batch_size=$batch \
        mlflow.run_name="lr_${lr}_batch_${batch}"
    done
  done

  7.3 A/Bテスト的な比較

  # モデルAとモデルBの比較
  python train.py model.base_channels=64 mlflow.run_name="model_A"
  python train.py model.base_channels=128 mlflow.run_name="model_B"

  # 両方を評価して比較
  python evaluate.py  # model_A
  python evaluate.py  # model_B

  実際の開発セッション例 💼

  セッション1: 初期開発

  # 1. 環境確認
  docker-compose exec app bash
  cd /app/src/UNet

  # 2. ベースライン構築
  python train.py training.epochs=50 mlflow.run_name="baseline"

  # 3. 結果確認
  mlflow ui --host 0.0.0.0 --port 5000 &
  # ブラウザでMLFlow UI確認

  # 4. 初期生成テスト
  python generate.py --checkpoint checkpoints/checkpoint_epoch_50.pth \
    --use_dummy --batch_size 8

  セッション2: ハイパーパラメータ調整

  # 1. 学習率実験
  python train.py -m training.learning_rate=1e-4,5e-5,1e-5 \
    training.epochs=100 \
    mlflow.run_name="lr_sweep"

  # 2. 結果分析（MLFlow UI）
  # 3. 最適パラメータで本格訓練
  python train.py training.learning_rate=5e-5 \
    training.epochs=200 \
    mlflow.run_name="final_model"

  セッション3: 評価・デプロイ

  # 1. 最終評価
  python evaluate.py

  # 2. 複数手法での生成比較
  python generate.py --checkpoint checkpoints/checkpoint_epoch_200.pth \
    --method ddpm --batch_size 16 --output_dir results_ddpm

  python generate.py --checkpoint checkpoints/checkpoint_epoch_200.pth \
    --method ddim --steps 50 --batch_size 16 --output_dir results_ddim

  # 3. 結果の比較分析
  jupyter notebook visualization.ipynb

  重要なポイント ⚠️

  1. 実験の記録: MLFlowで全実験を記録し、再現可能性を確保
  2. 段階的改善: 一度に多くの変更をせず、一つずつ検証
  3. 可視化重視: 数値だけでなく、生成結果を視覚的に確認
  4. 設定管理: Hydraで設定を体系的に管理
  5. 継続的評価: 定期的に包括的評価を実行

  このフローに従うことで、効率的かつ体系的なモデル開発が可能になります。
