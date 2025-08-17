# 階層型VAE潜在空間分析レポート

生成日時: 2025-08-17 20:12:31

## 実験情報
- 実験ID: 5
- 実験名: phase1_beta_extreme
- ステータス: completed
- 説明: Beta schedule validation: extreme
- 作成日時: 2025-08-17T07:41:19.471036
- 完了日時: 2025-08-17T07:42:16.769651

## 実験結果
- 再構成MSE: 0.000046
- スタイル分離スコア: 1.0000
- スキル性能相関: -0.4796

## データ概要
- 総サンプル数: 210
- 被験者数: 6
- エキスパート比率: 50.0%

## 潜在空間次元
- z_style: 8次元
- z_skill: 16次元
- z_primitive: 32次元

## 生成ファイル
### 可視化画像
- z_style_pca.png
- z_style_tsne.png
- z_style_umap.png
- z_skill_pca.png
- z_skill_tsne.png
- z_skill_umap.png
- z_primitive_pca.png
- z_primitive_tsne.png
- z_primitive_umap.png
- hierarchical_comparison.png

### データファイル
- latent_representations.csv
- latent_representations.parquet
- cluster_analysis.json
- latent_summary.txt

## 使用方法
```python
import pandas as pd
df = pd.read_parquet('latent_representations.parquet')
# 被験者別分析などに活用
```
