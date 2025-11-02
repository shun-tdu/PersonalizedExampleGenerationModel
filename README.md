<p align="center">
  <img src="path/to/your/logo.png" width="150" alt="Project Logo">
</p>

<h1 align="center">Skill-Style Disentanglement:運動技能支援のためのスキル・スタイル分離モデルの構築</h1>

<p align="center">
  個人最適な運動学習のための、スタイルとスキルを分離するモデルを提案します。TransformerをベースとしたVAEであり、運動データをスキル空間とスタイル空間に射影し、独立に操作した運動データを生成するモデルです。
</p>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://pytorch.org/">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://www.sqlite.org/index.html">
    <img src="https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white" alt="SQLite">
  </a>
  <a href="https://plotly.com/">
    <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly">
  </a>
</p>

---

## 📚 目次 (Table of Contents)

- [📝 概要](#-概要)
- [🎯 研究背景と目的](#️-研究背景と目的)
- [⚙️ システム構成](#️-システム構成)
- [🛠️ 使用技術](#-使用技術)
- [📈 成果](#-成果)
- [🔭 今後の展望](#-今後の展望)

---

## 📝 概要 (Overview)

本リポジトリは、画一的な指導から脱却し、個々人に最適化されたお手本生成を通して、個人最適な熟達支援を実現するための研究プロジェクトです。

核となるのは、人間の複雑な運動データを**「スキル（技能）」**と**「スタイル（個分の癖）」**という独立した要因に分離する、独自の**Transformer-VAE**モデルです。

このAIモデルを、力覚（フォース）フィードバックが可能な**自作の実験デバイス**と統合。
これにより、「個人の癖を活かしたまま、技能の核心部分だけを向上させる」という、従来にない高次元な物理的フィードバックを生成・提示します。
---

## 🎯 研究背景と目的 (Background and Objectives)

- 従来の課題: 従来の運動支援は、画一的な理想フォームを目指すものが多く、個人の身体特性や「癖（スタイル）」を活かした指導が困難でした。

- 本研究の目的: 動作データから「スキル」と「スタイル」をVAEで分離し、スタイルを維持したままスキル情報のみを操作・支援することで、個人に最適化された運動技能支援を実現します。

---
## ⚙️ システム構成 (System Architecture)
### ハードウェア（実験装置）
- デバイス: 直交二軸リニアアクチュエータ 
  - device image
- 特徴
  - ハンドル操作による2次元の運動データ（位置、速度、加速度）を測定
  - 生成モデルが生成した支援情報を力覚フィードバックとして操作者にリアルタイムで物理提示

### ソフトウェア（提案モデル）
- モデル：TransformerベースのVAE
- アーキテクチャ
  - エンコーダが動作データをスキル潜在空間とスタイル潜在空間に射影
  - 各空間に個別の損失関数を適用することで意味的な分離を実現
- 損失関数
  - 再構築誤差
   $$\\ L_{\text{recon}} = || \mathbf{x} - \hat{\mathbf{x}} ||^2$$
  - KLDiv
   $$\\ L_{\text{KL}} = D_{\text{KL}}(q_(\mathbf{z} | \mathbf{x}) || p(\mathbf{z}))$$
  - 独立性損失
  - スキル因子損失
    $$L_{\text{skill}} = ( s - f(\mathbf{z}_{\text{skill}}) )^2$$
    

---
## 🛠️ 使用技術 (Methods)
| Category          | Technology Stack      |
|-------------------|-----------------------|
| AI Model          | PyTorch               |
| Data Analysis     | Pandas, schikit-learn |
| Device Control    | C++, Ether CAT        |
| Environment Setup | Docker                |
| Database          | SQLite                |
| Visualization     | Plotly, Matplot       |


---
## 📈 成果 (Results)
あああ

---
## 🔭 今後の展望 (Future Work)
- 全身のボーンデータへのモデル適用
- 運動学を考慮した、より生理学的に妥当な動作生成
- 強化学習を用いたインタラクティブにスキル空間を探索して、支援するエージェントの開発