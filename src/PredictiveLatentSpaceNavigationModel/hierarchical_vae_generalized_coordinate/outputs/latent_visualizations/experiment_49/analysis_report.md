# 一般化座標VAE潜在空間分析レポート

生成日時: 2025-08-20 01:15:09

## 実験情報
- 実験ID: 49
- 実験名: beta_search_hierarchical_2
- ステータス: completed
- 説明: Ablation: No hierarchy (standard VAE)
- 作成日時: 2025-08-20T00:20:30.666819
- 完了日時: 2025-08-20T00:22:40.498698

## 実験結果
- 再構成MSE: 0.802126
- スタイル分離スコア: 0.0856
- スキル性能相関: -0.3240

## モデル特徴
- モデル: 一般化座標VAE (Hierarchical VAE Generalized Coordinate)
- 階層構造: 3レベル（スタイル→スキル→プリミティブ）
- 特徴: 自由エネルギー原理に基づく予測符号化

## データ概要
- 総サンプル数: 210
- 被験者数: 6
- エキスパート比率: 50.0%

## 潜在空間次元
- z_style: 16次元
- z_skill: 32次元
- z_primitive: 64次元

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
- skill_axes_analysis.png (一般化座標VAE特有)

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
