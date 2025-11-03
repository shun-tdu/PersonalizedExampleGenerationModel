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

    <p align="center">
        <img src="assets/Device_LinearActuator.png" alt="Result_StyleSeparation" width="50%">
    </p>
    

- 特徴
  - ハンドル操作による2次元の運動データ（位置、速度、加速度）を測定
  - Admittance制御により任意の仮想ダイナミクスを提示可能
  - 生成モデルが生成した支援情報を力覚フィードバックとして操作者にリアルタイムで物理提示

### ソフトウェア（提案モデル）

<p align="center">
  <img src="assets/BertLikeStyleSkillSeparationNet.png" alt="Proposed Architecture" width="50%">
</p>

- アーキテクチャ
  - エンコーダが動作データをスキル潜在空間とスタイル潜在空間に射影
  - 各空間に個別の損失関数を適用することで意味的な分離を実現
- 損失関数
  - 再構築誤差
    ```math
    \mathcal{L}_{\text{rec}} = \frac{1}{\sum M} \sum_{i} M_i (\hat{x}_i - x_i)^2
    ```
  - KL損失
    ```math
    \mathcal{L}_{\text{KL,style}} = D_{KL}(q(z_{\text{style}}|x) \parallel p(z_{\text{style}}))
    ```
    ```math
    \mathcal{L}_{\text{KL,skill}} = D_{KL}(q(z_{\text{skill}}|x) \parallel p(z_{\text{skill}}))
    ```
  - 直行性損失
    ```math
    \mathcal{L}_{\text{orth}} = \frac{1}{Sty \cdot Ski} \sum_{j=1}^{Sty} \sum_{k=1}^{Ski} (C_{j,k})^2
    \quad \text{where} \quad
    C = \frac{1}{B} \bar{Z}_{\text{style}}^T \bar{Z}_{\text{skill}}
    ```
  - スキル因子損失
    ```math
    \mathcal{L}_{\text{factor}} = \mathbb{E} [ \| \hat{y} - y \|^2_2 ]
    \quad \text{where} \quad
    \hat{y} = f_{\text{reg}}(z_{\text{skill}})
    ```
  - 統合損失
    ```math
    \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \beta_{\text{style}} \mathcal{L}_{\text{KL,style}} + \beta_{\text{skill}} \mathcal{L}_{\text{KL,skill}} + \gamma_{\text{orth}} \mathcal{L}_{\text{orth}} + \gamma_{\text{factor}} \mathcal{L}_{\text{factor}}
    ```

---
## 📈 成果 (Results)
提案モデルの有効性を入力データに対する再構築誤差、スタイル潜在変数による被験者分類、スタイル潜在空間の定性的評価、スキル潜在変数によるスキル因子回帰、スキル潜在空間の定性的評価で評価した。
以下にそれぞれの評価結果を示す。
- 入力データに対する再構築誤差
  - 元データと再構築軌道のプロット(上から位置２次元、位置、速度、加速度)
    
    モデルの生成性能を定性的に評価する。
  
    入力データに対して、高周波のノイズが重畳したようなデータが確認できるが、軌道の概形は捉えられていることがわかる。
  
    <p align="center">
        <img src="assets/Result_TrajectoryOverlay.png" alt="Result Trajectory Overlay" width="30%">
    </p>
    
  - 位置、速度、加速度のRMSE
    
    モデルの生成性能を定量的に評価する。
    
    位置、速度、加速度において同一スケールの誤差が確認できるが、全体的なRMSEは0.1程度に留まっており、モデルが軌道の主要な運動学的特徴を再構築できていることが定量的に示された。

    <div align="center">
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th>X-Component</th>
              <th>Y-Component</th>
              <th>Overall</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Position [m]</td>
              <td>0.098</td>
              <td>0.094</td>
              <td>0.096</td>
            </tr>
            <tr>
              <td>Velocity [m/s]</td>
              <td>0.113 </td>
              <td>0.106</td>
              <td>0.110</td>
            </tr>
            <tr>
              <td>Acceleration[m/s^2]</td>
              <td>0.130</td>
              <td>0.132</td>
              <td>0.131</td>
            </tr>
          </tbody>
        </table>
    </div>

- スタイル空間評価
  - スタイル変数から被験者の分類精度の評価

    被験者固有の筋骨格構造や認知特性に由来するスタイルをモデルが分離できているかを評価する。
  
    動作データからエンコードされたスタイル潜在変数を単純な分類器であるMLP、SVM、RandomForestに学習データとして与え、被験者分類が実現されるかを確認した。
    
    結果から90%以上の精度でどの分類器でも被験者分類ができていることがわかる。また、スタイル潜在空間を主成分分析：PCAで可視化したプロットを見ると潜在空間上で異なる領域にそれぞれの被験者の動作データが射影されていることが確認できる。
  
    以上から、本モデルはスタイル空間の構造化を促す損失関数を用いていないのにもかかわらず、モデルが自然にスタイル情報をエンコードしたことが示唆された。
    
    <p align="center">
        <img src="assets/Result_StyleSeparation.png" alt="Result_StyleSeparation" width="50%">
    </p>


- スキル空間評価
  - スキル潜在変数によるスキル因子回帰精度
  
    被験者の巧拙を表すスキル情報をモデルが分離できているかを評価する。
    
    動作データからエンコードされたスキル潜在変数を単純な回帰モデルであるMLP、SVM、線形回帰モデルに学習データとして与え、被験者のスキルスコアが回帰できるかを確認した。
  
    結果からどのモデルでも0.75以上の決定係数であることがわかった。
    
    以上から、本モデルは被験者の運動データから巧拙を表すスキルの表現を獲得していることが示唆された。

    <p align="center">
        <img src="assets/Result_SkillRegressionScore.png" alt="Result_StyleSeparation" width="50%">
    </p>
    
  - スキル潜在空間の定性的評価
    
    スキル潜在空間をPCAで可視化し、スキル空間に熟達者のデータが集まるような領域が形成されているかを確認した。
  
    これは個人最適なお手本生成を行うための重要な前提条件であり、熟達者が集まる領域が確認されれば、学習者の運動データを測定した後、
    潜在空間に射影し、スタイル潜在変数を固定しながらスキル潜在変数を熟達者が集まる領域に段階的に移動させながら動作データのサンプリングを行い、
    学習者にフィードバックを行えば、学習者には常に学習者自身のスタイルが反映された少しだけ熟達した動作データが提示されることになり、効率的な運動学習が実現されることが期待される。
    
    結果から主成分2が増加するに応じて熟達度を表すスキルスコアが増加していることがわかる。このことからモデルはスキル因子を回帰する過程でスキル空間をスキルスコアが連続的に変化するように構造化したことがわかる。
  
    以上から、本モデルは個人最適なお手本生成が可能な潜在空間の構造を有していることが示された。一方で、動作データの生成精度には課題が残っている。今後は、エンコーダの重みを固定し、デコーダをより表現力を持つ拡散モデルに切り替えることで再構成誤差の低減に努めていく。

    <p align="center">
        <img src="assets/Result_SkillSpacePCA.png" alt="Result_StyleSeparation" width="50%">
    </p>    

---
## 🔭 今後の展望 (Future Work)
- スキル変換の実現
- 全身のボーンデータへのモデル適用
- 強化学習を用いたインタラクティブにスキル空間を探索して、支援するエージェントの開発

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

