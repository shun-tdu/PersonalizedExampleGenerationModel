from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.base_model import BaseExperimentModel
from models.components.loss_weight_scheduler import LossWeightScheduler


class ImprovedLossWeightScheduler(LossWeightScheduler):
    """改良された損失重みスケジューラ"""

    def __init__(self, loss_schedule_config, structure_priority_epochs=60):
        super().__init__(loss_schedule_config)
        self.structure_priority_epochs = structure_priority_epochs
        self.current_epoch = 0

    def step(self, epoch):
        """エポックを更新"""
        super().step(epoch)
        self.current_epoch = epoch

    def get_adaptive_weights(self, epoch, reconstruction_loss, structure_losses):
        """適応的重み計算"""
        base_weights = self.get_weights()

        # 構造損失の平均（NaN/inf チェック付き）
        valid_losses = [loss for loss in structure_losses.values()
                        if torch.isfinite(loss)]
        avg_structure_loss = sum(valid_losses) / max(len(valid_losses), 1)

        if epoch < self.structure_priority_epochs:
            # 構造優先期間：再構成損失を適度に制限
            recon_loss_val = reconstruction_loss.item() if torch.isfinite(reconstruction_loss) else 1.0
            recon_weight = min(1.5, max(0.5, recon_loss_val / 0.05))

            # 構造損失が十分下がっていない場合は構造学習を強化
            if avg_structure_loss > 0.3:
                structure_multiplier = 1.8
            elif avg_structure_loss > 0.1:
                structure_multiplier = 1.3
            else:
                structure_multiplier = 1.0

        else:
            # 再構成最適化期間
            recon_weight = 1.0
            structure_multiplier = 0.7  # 構造損失の重みを減らす

        # 重みを調整
        adapted_weights = base_weights.copy()

        # 再構成重みの調整
        adapted_weights['reconstruction'] = recon_weight

        # 構造関連損失の重み調整
        for key in ['orthogonal_loss', 'contrastive_loss', 'style_classification_loss',
                    'skill_regression_loss', 'manifold_loss']:
            if key in adapted_weights:
                adapted_weights[key] *= structure_multiplier

        return adapted_weights

class ContrastiveLoss(nn.Module):
    """対比学習損失"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)

        # バッチサイズが小さい場合の対処
        if batch_size <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # 正例・負例マスク
        mask = torch.eq(labels, labels.T).float()

        # 特徴量正規化 数値安定性向上
        features = F.normalize(features, p=2, dim=1, eps=1e-8)

        # 類似度計算 温度パラメータをクリップ
        temperature_clamped = torch.clamp(torch.tensor(self.temperature), min=0.01)
        similarity_matrix = torch.matmul(features, features.T) / temperature_clamped

        # 類似度をクリップして数値安定性を向上
        similarity_matrix = torch.clamp(similarity_matrix, min=-10, max=10)

        # 対角成分除去
        mask = mask - torch.eye(batch_size, device=mask.device)

        # 損失計算 数値安定性の改善
        exp_sim = torch.exp(similarity_matrix)
        pos_sum = torch.sum(exp_sim * mask, dim=1)
        neg_sum = torch.sum(exp_sim * (1 - torch.eye(batch_size, device=mask.device)), dim=1)

        # ゼロ除算とlog(0)を防ぐ
        pos_sum = torch.clamp(pos_sum, min=1e-8)
        neg_sum = torch.clamp(neg_sum, min=1e-8)

        loss = -torch.log(pos_sum / neg_sum)

        # NaNチェック
        if torch.any(torch.isnan(loss)):
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        return loss.mean()


class PositionalEncoding(nn.Module):
    """時系列用位置エンコーディング"""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class ConstrainedSkipConnection(nn.Module):
    def __init__(self, d_model, bottleneck_ratio=0.25, noise_std=0.05):
        super().__init__()
        self.d_model = d_model
        self.bottleneck_dim = max(int(d_model*bottleneck_ratio), 16)
        self.noise_std = noise_std

        # 情報ボトルネック層
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model, self.bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.bottleneck_dim, d_model)
        )

        # 重み付きゲート
        self.gate = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, decoder_feature, encoder_feature, training=True):
        """
        Args:
            decoder_feature:デコーダの現在の特徴量
            encoder_feature: エンコーダからのスキップ特徴量
            training: 学習中かどうか
        """
        # 1. 情報ボトルネックを通してスキップ情報を集約
        constrained_skip = self.bottleneck(encoder_feature)

        # 2. 訓練中はついうかのノイズを加える(正則化のため)
        if training:
            noise = torch.randn_like(constrained_skip) * self.noise_std
            constrained_skip = constrained_skip + noise

        # 3. ゲート機構で適応的に重み付け
        combined_input = torch.cat([decoder_feature, encoder_feature], dim=2)
        gate_weight = self.gate(combined_input)

        # 4. ゲート重み付きでスキップ接続を適用
        output = decoder_feature + gate_weight * constrained_skip

        return output


class StyleSkillSeparationNetEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 seq_len,
                 d_model,
                 n_heads,
                 n_layers,
                 dropout,
                 style_latent_dim,
                 skill_latent_dim
                 ):
        super().__init__()

        # 入力射影
        self.input_proj = nn.Linear(input_dim, d_model)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model)

        # 特徴量抽出Transformer層
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dropout=dropout,
            dim_feedforward=d_model*2
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)

        # スキップ接続
        self.skip_features = []

        # スタイル潜在空間用ヘッド（効率化）
        self.style_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, style_latent_dim * 2)
        )

        # スキル潜在空間用ヘッド（効率化）
        self.skill_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, skill_latent_dim * 2)
        )

        # 重み初期化
        self._initialize_weights()

    def _initialize_weights(self):
        """CLAUDE_ADDED: Xavier初期化で重みを安定化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform初期化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        batch_size, _, _ = x.shape

        # スキップ特徴量を初期化
        self.skip_features = []

        # 入力射影
        encoded = self.input_proj(x)
        self.skip_features.append(encoded)

        # 位置エンコーディング
        pos_encoded = self.pos_encoding(encoded)

        # Transformer層ごとに中間特徴量を保存
        current = pos_encoded
        for i, layer in enumerate(self.transformer_encoder.layers):
            current = layer(current)
            if (i + 1) % 2 == 0:
                self.skip_features.append(current)

        # シーケンス次元でAverage Pooling
        pooled_feature = torch.mean(current, dim=1) #[batch_size, d_model]

        # スタイル・スキル射影
        style_params = self.style_head(pooled_feature)
        skill_params = self.skill_head(pooled_feature)

        return {
            'style_mu': style_params[:, :style_params.size(1) // 2],
            'style_logvar': style_params[:, style_params.size(1) // 2:],
            'skill_mu': skill_params[:, :skill_params.size(1) // 2],
            'skill_logvar': skill_params[:, skill_params.size(1) // 2:],
            'skip_feature': self.skip_features
        }


class StyleSkillSeparationNetDecoder(nn.Module):
    def __init__(self,
                 output_dim,
                 seq_len,
                 d_model,
                 n_heads,
                 n_layers,
                 dropout,
                 style_latent_dim,
                 skill_latent_dim):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model

        # 中間次元を制限
        intermediate_dim = min(d_model * 4, 2048)  # 中間次元を制限

        # 潜在変数サンプリング層
        self.from_style_latent = nn.Sequential(
            nn.Linear(style_latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim, d_model * seq_len)
        )
        self.from_skill_latent = nn.Sequential(
            nn.Linear(skill_latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim, d_model * seq_len)
        )

        # スタイル・スキル合成層
        self.fusion_proj = nn.Linear(2 * d_model, d_model)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformerデコーダ層
        decoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dropout=dropout,
            dim_feedforward=d_model*2
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layers, num_layers=n_layers)

        # 制限付きスキップ接続層
        self.constrained_skip_layers = nn.ModuleList([
            ConstrainedSkipConnection(d_model, bottleneck_ratio=0.2)
            for _ in range(min(3, n_layers//2))
        ])

        # 出力射影
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, z_style, z_skill, skip_features=None, skip_weight=0.0):
        batch_size, _ = z_style.shape

        # スタイル・スキル潜在変数を系列データに変換
        style_seq = self.from_style_latent(z_style).view(batch_size, self.seq_len, self.d_model)
        skill_seq = self.from_skill_latent(z_skill).view(batch_size, self.seq_len, self.d_model)

        # スタイル・スキル系列データを合成
        concatenated_seq = torch.concat([style_seq, skill_seq], dim=2)
        fusion_seq = self.fusion_proj(concatenated_seq)

        # 位置エンコーディング
        pos_encoded = self.pos_encoding(fusion_seq)

        # Transformer Decode
        current = pos_encoded
        skip_idx = 0

        for i, layer in enumerate(self.transformer_decoder.layers):
            current = layer(current)

            # 制限付きスキップ接続の適用
            if (skip_features is not None and
                    skip_idx < len(skip_features) and
                    skip_idx < len(self.constrained_skip_layers) and
                    i % 2 == 1 and skip_weight > 0):

                    skip_feature = skip_features[skip_idx]

                    if skip_feature.size(1) == current.size(1):
                        skip_output = self.constrained_skip_layers[skip_idx](current, skip_feature, self.training)

                        # 重み付きでスキップ接続を活用
                        current = (1 - skip_weight) * current + skip_weight * skip_output

                    skip_idx += 1

        # 出力射影
        reconstructed = self.output_proj(current)

        return reconstructed


class StyleSkillSeparationNet(BaseExperimentModel):
    def __init__(self,
                 input_dim=6,
                 seq_len=100,
                 d_model=128,
                 n_heads=4,
                 n_encoder_layers=2,
                 n_decoder_layers=2,
                 dropout=0.1,
                 style_latent_dim=8,
                 skill_latent_dim=8,
                 n_subjects=6,
                 loss_schedule_config: Dict[str, Any] = None,
                 progressive_training_config: Dict[str, Any] = None,
                 **kwargs
                 ):
        super().__init__(input_dim=input_dim,
                         seq_len=seq_len,
                         d_model=d_model,
                         n_heads=n_heads,
                         n_encoder_layers=n_encoder_layers,
                         n_decoder_layers=n_decoder_layers,
                         dropout=dropout,
                         style_latent_dim=style_latent_dim,
                         skill_latent_dim=skill_latent_dim,
                         n_subjects=n_subjects,
                         loss_schedule_config=loss_schedule_config,
                         progressive_training_config=progressive_training_config,
                         **kwargs
                         )
        self.seq_len = seq_len
        self.d_model = d_model
        self.style_latent_dim = style_latent_dim
        self.skill_latent_dim = skill_latent_dim
        self.current_epoch = 0

        # 段階的学習の設定
        if progressive_training_config is None:
            progressive_training_config = {
                'enabled': True,
                'structure_learning_epochs': 60,
                'transition_epochs': 30,
                'reconstruction_priority_epochs': 110
            }

        self.progressive_config = progressive_training_config
        self.progressive_training = progressive_training_config.get('enabled', True)
        self.structure_learning_epochs = progressive_training_config.get('structure_learning_epochs', 60)
        self.transition_epochs = progressive_training_config.get('transition_epochs', 30)

        # 動的パラメータ
        self.skip_connection_weight = 0.0
        self.current_dropout = dropout

        # 損失関数の重みスケジューラの初期化
        if loss_schedule_config is None:
            loss_schedule_config = {
                # Phase 1: 基礎構造学習
                'beta_style': {'schedule': 'linear', 'start_epoch': 5, 'end_epoch': 25, 'start_val': 0.01,
                               'end_val': 0.1},
                'beta_skill': {'schedule': 'linear', 'start_epoch': 5, 'end_epoch': 25, 'start_val': 0.01,
                               'end_val': 0.05},
                'orthogonal_loss': {'schedule': 'linear', 'start_epoch': 10, 'end_epoch': 30, 'start_val': 0.0,
                                    'end_val': 0.3},

                # Phase 2: スタイル構造化
                'contrastive_loss': {'schedule': 'linear', 'start_epoch': 30, 'end_epoch': 50, 'start_val': 0.0,
                                     'end_val': 0.4},
                'style_classification_loss': {'schedule': 'linear', 'start_epoch': 35, 'end_epoch': 60,
                                              'start_val': 0.0, 'end_val': 0.3},

                # Phase 3: スキル構造化
                'skill_regression_loss': {'schedule': 'linear', 'start_epoch': 60, 'end_epoch': 100, 'start_val': 0.0,
                                          'end_val': 0.4},
                'manifold_loss': {'schedule': 'linear', 'start_epoch': 80, 'end_epoch': 140, 'start_val': 0.0,
                                  'end_val': 0.6},

                # Phase 4: 後期KL強化
                'beta_style_late': {'schedule': 'linear', 'start_epoch': 120, 'end_epoch': 150, 'start_val': 0.1,
                                    'end_val': 0.2},
                'beta_skill_late': {'schedule': 'linear', 'start_epoch': 120, 'end_epoch': 150, 'start_val': 0.05,
                                    'end_val': 0.1},
            }

        self.loss_scheduler = ImprovedLossWeightScheduler(loss_schedule_config, self.structure_learning_epochs)

        # スケジューラ設定からlossの計算フラグを受け取り
        self.calc_orthogonal_loss = 'orthogonal_loss' in loss_schedule_config
        self.calc_contrastive_loss = 'contrastive_loss' in loss_schedule_config
        self.calc_manifold_loss = 'manifold_loss' in loss_schedule_config
        self.calc_style_subtask = 'style_classification_loss' in loss_schedule_config
        self.calc_skill_subtask = 'skill_regression_loss' in loss_schedule_config

        print(f"StyleSkillSeparationNet (Improved) instantiated with:")
        print(f"  d_model: {d_model}, n_heads: {n_heads}")
        print(f"  n_encoder_layers: {n_encoder_layers}, n_decoder_layers: {n_decoder_layers}")
        print(f"  style_latent_dim: {style_latent_dim}, skill_latent_dim: {skill_latent_dim}")
        print(f"  progressive_training: {self.progressive_training}")

        # エンコーダ定義
        self.encoder = StyleSkillSeparationNetEncoder(input_dim,
                                                      seq_len,
                                                      d_model,
                                                      n_heads,
                                                      n_encoder_layers,
                                                      dropout,
                                                      style_latent_dim,
                                                      skill_latent_dim
                                                      )
        # デコーダ定義
        self.decoder = StyleSkillSeparationNetDecoder(input_dim,
                                                      seq_len,
                                                      d_model,
                                                      n_heads,
                                                      n_decoder_layers,  # CLAUDE_ADDED: typo修正
                                                      dropout,
                                                      style_latent_dim,
                                                      skill_latent_dim)

        # CLAUDE_ADDED: プロトタイプベースのスタイル識別 (被験者数に依存しない)
        if self.calc_style_subtask:
            self.style_prototype_network = nn.Sequential(
                nn.Linear(style_latent_dim, style_latent_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(style_latent_dim, style_latent_dim)  # プロトタイプ空間への写像
            )
            # プロトタイプを保存するバッファ（学習中に更新）
            self.register_buffer('style_prototypes', torch.zeros(n_subjects, style_latent_dim))
            self.register_buffer('prototype_counts', torch.zeros(n_subjects))

        # 補助タスク用ネット (スキルスコア回帰)
        if self.calc_skill_subtask:
            self.skill_score_regressor = nn.Sequential(
                nn.Linear(skill_latent_dim, skill_latent_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(skill_latent_dim // 2, 1)
            )

        # 対照学習の損失
        self.contrastive_loss = ContrastiveLoss()

    def set_training_phase(self, epoch: int):
        """訓練フェーズに応じてパラメータを調整"""
        self.current_epoch = epoch

        if not self.progressive_training:
            return

        if epoch < self.structure_learning_epochs:
            # Phase 1: 構造学習期間（スキップ接続無効）
            self.skip_connection_weight = 0.0
            self.current_dropout = 0.3

        elif epoch < self.structure_learning_epochs + self.transition_epochs:
            # Phase 2: 移行期間（徐々にスキップ接続を有効化）
            progress = (epoch - self.structure_learning_epochs) / self.transition_epochs
            self.skip_connection_weight = progress
            self.current_dropout = 0.3 - 0.1 * progress  # 0.3 -> 0.2

        else:
            # Phase 3: 再構成最適化期間（スキップ接続有効）
            self.skip_connection_weight = 1.0
            self.current_dropout = 0.1

        # ドロップアウトの動的更新
        self._update_dropout_rates()

    def _update_dropout_rates(self):
        """モデル全体のドロップアウト率を更新"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.current_dropout

    def on_epoch_start(self, epoch: int):
        """学習ループからエポックの開始時呼び出されるメソッド"""
        self.loss_scheduler.step(epoch)
        if self.progressive_training:
            self.set_training_phase(epoch)
            if epoch % 10 == 0:  # 10エポックごとにログ出力
                print(f"Epoch {epoch}: Skip weight = {self.skip_connection_weight:.3f}, "
                      f"Dropout = {self.current_dropout:.3f}")

    def reparameterize(self, mu, logvar):
        """改良された再パラメータ化トリック"""
        # 数値安定性のためのクリッピング
        logvar = torch.clamp(logvar, min=-8, max=8)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, subject_ids: str = None, skill_scores: float = None) -> Dict[str, torch.Tensor]:
        batch_size, _, _ = x.shape

        # エンコード
        encoded = self.encoder(x)
        skip_features = encoded.get('skip_features', None)

        # 潜在変数サンプリング
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])

        # デコード
        reconstructed = self.decoder(
            z_style, z_skill,
            skip_features = skip_features,
            skip_weight= self.skip_connection_weight
        )

        # 補助タスク
        subject_pred = None
        skill_score_pred = None
        if self.calc_style_subtask:
            subject_pred = self._prototype_based_classification(z_style, subject_ids)
        if self.calc_skill_subtask:
            skill_score_pred = self.skill_score_regressor(z_skill)

        # 損失計算
        losses = self.compute_losses(
            x, reconstructed, encoded, z_style, z_skill,
            subject_pred, skill_score_pred,
            subject_ids, skill_scores
        )

        result = {
            'reconstructed': reconstructed,
            'z_style': z_style,
            'z_skill': z_skill,
        }

        return result | losses

    def compute_losses(self, x, reconstructed, encoded, z_style, z_skill,
                       subject_pred, skill_score_pred, subject_ids, skill_scores):
        """修正された損失計算（KL重複問題を解決）"""
        device = x.device

        def safe_loss(loss_tensor, default_val=0.0):
            """NaN/infを安全に処理"""
            if torch.isnan(loss_tensor) or torch.isinf(loss_tensor):
                return torch.tensor(default_val, device=device, requires_grad=True)
            return loss_tensor

        # 再構成損失 - 段階的損失関数
        if self.current_epoch < 50:
            # 構造学習期間：Huber損失
            loss_pos = F.smooth_l1_loss(reconstructed[:, :, 0:2], x[:, :, 0:2])
            loss_vel = F.smooth_l1_loss(reconstructed[:, :, 2:4], x[:, :, 2:4])
            loss_acc = F.smooth_l1_loss(reconstructed[:, :, 4:6], x[:, :, 4:6])
        else:
            # 通常期間：MSE損失
            loss_pos = F.mse_loss(reconstructed[:, :, 0:2], x[:, :, 0:2])
            loss_vel = F.mse_loss(reconstructed[:, :, 2:4], x[:, :, 2:4])
            loss_acc = F.mse_loss(reconstructed[:, :, 4:6], x[:, :, 4:6])

        total_recon_loss = safe_loss(1.0 * loss_pos + 1.0 * loss_vel + 0.1 * loss_acc)
        losses = {'reconstruction_loss': total_recon_loss}

        # KLダイバージェンス - 数値安定性向上
        style_mu_clamped = torch.clamp(encoded['style_mu'], min=-5, max=5)
        skill_mu_clamped = torch.clamp(encoded['skill_mu'], min=-5, max=5)
        style_logvar_clamped = torch.clamp(encoded['style_logvar'], min=-8, max=8)
        skill_logvar_clamped = torch.clamp(encoded['skill_logvar'], min=-8, max=8)

        style_kl_terms = 1 + style_logvar_clamped - style_mu_clamped.pow(2) - style_logvar_clamped.exp()
        skill_kl_terms = 1 + skill_logvar_clamped - skill_mu_clamped.pow(2) - skill_logvar_clamped.exp()

        losses['kl_style_loss'] = safe_loss(-0.5 * torch.mean(torch.sum(style_kl_terms, dim=1)))
        losses['kl_skill_loss'] = safe_loss(-0.5 * torch.mean(torch.sum(skill_kl_terms, dim=1)))

        # 構造化損失の計算
        structure_losses = {}

        # 直交性損失
        if self.calc_orthogonal_loss:
            z_style_norm = F.normalize(z_style - z_style.mean(dim=0), p=2, dim=1)
            z_skill_norm = F.normalize(z_skill - z_skill.mean(dim=0), p=2, dim=1)

            correlation = torch.mm(z_style_norm.T, z_skill_norm) / z_style.size(0)
            orthogonal_loss = safe_loss(torch.norm(correlation, p='fro') ** 2)
            losses['orthogonal_loss'] = orthogonal_loss
            structure_losses['orthogonal'] = orthogonal_loss

        # スタイル空間関連の損失
        if subject_ids is not None:
            all_subjects = ['h.nakamura', 'r.morishita', 'r.yanase', 's.miyama', 's.tahara', 't.hasegawa']
            subject_to_idx = {subj: i for i, subj in enumerate(all_subjects)}
            subject_indices = torch.tensor([subject_to_idx[subj] for subj in subject_ids], device=z_style.device)

            if self.calc_contrastive_loss:
                contrastive_loss = safe_loss(self.contrastive_loss(z_style, subject_indices))
                losses['contrastive_loss'] = contrastive_loss
                structure_losses['contrastive'] = contrastive_loss

            if self.calc_style_subtask and subject_pred is not None:
                style_cls_loss = safe_loss(F.cross_entropy(subject_pred, subject_indices))
                losses['style_classification_loss'] = style_cls_loss
                structure_losses['style_classification'] = style_cls_loss

        # スキル空間関連の損失
        if skill_scores is not None:
            if self.calc_skill_subtask and skill_score_pred is not None:
                skill_score_pred_flat = skill_score_pred.squeeze(-1)
                skill_reg_loss = safe_loss(F.mse_loss(skill_score_pred_flat, skill_scores))
                losses['skill_regression_loss'] = skill_reg_loss
                structure_losses['skill_regression'] = skill_reg_loss

            if self.calc_manifold_loss:
                manifold_loss = safe_loss(self.compute_manifold_loss(z_skill, skill_scores))
                losses['manifold_loss'] = manifold_loss
                structure_losses['manifold'] = manifold_loss

        # 適応的重み取得
        weights = self.loss_scheduler.get_adaptive_weights(
            self.current_epoch, total_recon_loss, structure_losses
        )

        # 修正された総合損失計算（KL重複なし）
        # beta_styleとbeta_skillは既に区分線形で統合済み、重複加算しない
        total_loss = (weights.get('reconstruction', 1.0) * total_recon_loss
                      + weights.get('beta_style', 0.0) * losses['kl_style_loss']  # 統合済みの重み
                      + weights.get('beta_skill', 0.0) * losses['kl_skill_loss']  # 統合済みの重み
                      + weights.get('orthogonal_loss', 0.0) * losses.get('orthogonal_loss',
                                                                         torch.tensor(0.0, device=device))
                      + weights.get('contrastive_loss', 0.0) * losses.get('contrastive_loss',
                                                                          torch.tensor(0.0, device=device))
                      + weights.get('manifold_loss', 0.0) * losses.get('manifold_loss',
                                                                       torch.tensor(0.0, device=device))
                      + weights.get('style_classification_loss', 0.0) * losses.get('style_classification_loss',
                                                                                   torch.tensor(0.0, device=device))
                      + weights.get('skill_regression_loss', 0.0) * losses.get('skill_regression_loss',
                                                                               torch.tensor(0.0, device=device)))

        losses['total_loss'] = safe_loss(total_loss)

        # 学習進捗の監視（10エポックごと）
        if self.current_epoch % 10 == 0:
            avg_structure_loss = sum(structure_losses.values()) / max(len(structure_losses),
                                                                      1) if structure_losses else 0.0
            print(f"Epoch {self.current_epoch}: Recon={total_recon_loss:.4f}, Struct_avg={avg_structure_loss:.4f}")
            print(
                f"  Current weights - beta_style: {weights.get('beta_style', 0.0):.4f}, beta_skill: {weights.get('beta_skill', 0.0):.4f}")

        return losses

    def compute_manifold_loss(self, z_skill: torch.Tensor, skill_score: torch.Tensor, min_separation: float = 1.0):
        """
        相対的分離設計による熟達多様体形成損失
        CLAUDE_ADDED: Option 2 - 熟達者と非熟達者の重心分離 + 各グループ内凝集性
        Args:
            z_skill (torch.Tensor): スキル潜在変数[batch_size, skill_latent_dim]
            skill_score (torch.Tensor): 標準化されたスキルスコア [batch_size, ]
            min_separation (float): 熟達者と非熟達者の最小分離距離
        """
        batch_size = z_skill.size(0)
        device = z_skill.device

        # 熟達者と非熟達者のマスク
        expert_mask = skill_score > 0.0
        novice_mask = skill_score <= 0.0

        n_experts = expert_mask.sum()
        n_novices = novice_mask.sum()

        # 各項の損失を初期化
        loss_separation = torch.tensor(0.0, device=device, requires_grad=True)
        loss_expert_cohesion = torch.tensor(0.0, device=device, requires_grad=True)
        loss_novice_cohesion = torch.tensor(0.0, device=device, requires_grad=True)

        # CLAUDE_ADDED: 1. 重心分離損失（熟達者と非熟達者の分離を促進）
        if n_experts > 0 and n_novices > 0:
            # 各グループの重心を計算
            expert_centroid = z_skill[expert_mask].mean(dim=0)
            novice_centroid = z_skill[novice_mask].mean(dim=0)

            # 重心間の距離
            centroid_distance = torch.norm(expert_centroid - novice_centroid, p=2)

            # 分離が不十分な場合にペナルティ
            loss_separation = torch.clamp(min_separation - centroid_distance, min=0.0) ** 2

        # CLAUDE_ADDED: 2. 熟達者グループ内凝集性（スキルスコアに応じた重み付き）
        if n_experts > 1:
            z_skill_experts = z_skill[expert_mask]
            expert_scores = skill_score[expert_mask]
            expert_centroid = z_skill_experts.mean(dim=0)

            # 各熟達者の重心からの距離（スキルスコアで重み付け）
            expert_distances_to_centroid = torch.norm(z_skill_experts - expert_centroid.unsqueeze(0), p=2, dim=1)
            expert_weights = torch.sigmoid(expert_scores * 2.0)  # 高スキルほど強い凝集
            loss_expert_cohesion = torch.mean(expert_weights * expert_distances_to_centroid)

        # CLAUDE_ADDED: 3. 非熟達者グループ内凝集性（適度な凝集）
        if n_novices > 1:
            z_skill_novices = z_skill[novice_mask]
            novice_scores = skill_score[novice_mask]
            novice_centroid = z_skill_novices.mean(dim=0)

            # 各非熟達者の重心からの距離（低スキルほど緩い凝集）
            novice_distances_to_centroid = torch.norm(z_skill_novices - novice_centroid.unsqueeze(0), p=2, dim=1)
            novice_weights = torch.sigmoid(-novice_scores * 1.0)  # 低スキルほど緩い凝集
            loss_novice_cohesion = torch.mean(novice_weights * novice_distances_to_centroid)

        # CLAUDE_ADDED: 4. 適応的スキルベース分離（連続的なスキルスコアを活用）
        loss_skill_based_separation = torch.tensor(0.0, device=device, requires_grad=True)
        if batch_size > 1:
            # スキルスコア差と潜在空間距離の相関を促進
            skill_diff_matrix = torch.abs(skill_score.unsqueeze(1) - skill_score.unsqueeze(0))
            z_skill_distance_matrix = torch.cdist(z_skill, z_skill, p=2)

            # 対角成分を除外
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            skill_diffs = skill_diff_matrix[mask]
            z_distances = z_skill_distance_matrix[mask]

            # スキル差と空間距離の正の相関を促進（高スキル差ペアは遠く配置）
            if skill_diffs.numel() > 0:
                # スキル差に比例した最小距離を設定
                target_distances = skill_diffs * 0.5  # スケーリング係数
                distance_violations = torch.clamp(target_distances - z_distances, min=0.0)
                loss_skill_based_separation = torch.mean(distance_violations ** 2)

        # CLAUDE_ADDED: 5. 重み付き総合損失
        alpha_separation = 2.0  # 重心分離（最重要）
        alpha_expert_cohesion = 1.0  # 熟達者凝集性
        alpha_novice_cohesion = 0.3  # 非熟達者凝集性（緩め）
        alpha_skill_separation = 1.5  # スキルベース分離

        total_manifold_loss = (
                alpha_separation * loss_separation +
                alpha_expert_cohesion * loss_expert_cohesion +
                alpha_novice_cohesion * loss_novice_cohesion +
                alpha_skill_separation * loss_skill_based_separation
        )

        return total_manifold_loss

    def encode(self, x):
        """エンコードのみ"""
        encoded = self.encoder(x)
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])
        return {'z_style': z_style, 'z_skill': z_skill}

    def decode(self, z_style, z_skill):
        """デコードのみ"""
        trajectory = self.decoder(z_style, z_skill, skip_features=None, skip_weight=0.0)
        return {'trajectory': trajectory}

    def _prototype_based_classification(self, z_style: torch.Tensor, subject_ids: list = None):
        """プロトタイプベースのスタイル識別"""
        # CLAUDE_ADDED: プロトタイプ空間への写像
        style_features = self.style_prototype_network(z_style)
        style_features = F.normalize(style_features, p=2, dim=1)  # L2正規化

        batch_size = z_style.size(0)

        if self.training and subject_ids is not None:
            # 学習時：プロトタイプを更新しながら分類
            return self._update_prototypes_and_classify(style_features, subject_ids)
        else:
            # テスト時：既存プロトタイプとの距離ベース分類
            return self._distance_based_classification(style_features)

    def _update_prototypes_and_classify(self, style_features: torch.Tensor, subject_ids: list):
        """学習時：プロトタイプ更新と分類を同時実行"""
        batch_size = style_features.size(0)
        device = style_features.device

        # CLAUDE_ADDED: 全被験者を固定的なインデックスにマッピング（一貫性を保つ）
        all_subjects = ['h.nakamura', 'r.morishita', 'r.yanase', 's.miyama', 's.tahara', 't.hasegawa']
        subject_to_idx = {subj: i for i, subj in enumerate(all_subjects)}
        subject_indices = [subject_to_idx[subj] for subj in subject_ids]

        # 予測用の類似度行列を計算
        similarities = torch.zeros(batch_size, self.style_prototypes.size(0), device=device)

        # バッチ内の各サンプルについてプロトタイプを更新
        for i, (feature, subj_idx, subj_id) in enumerate(zip(style_features, subject_indices, subject_ids)):
            if subj_idx < self.style_prototypes.size(0):
                # プロトタイプの移動平均更新 (momentum=0.9)
                momentum = 0.9
                current_count = self.prototype_counts[subj_idx].item()

                if current_count == 0:
                    # 初回: プロトタイプを直接設定
                    self.style_prototypes[subj_idx] = feature.detach()
                    self.prototype_counts[subj_idx] = 1.0
                else:
                    # 移動平均で更新
                    self.style_prototypes[subj_idx] = (
                            momentum * self.style_prototypes[subj_idx] +
                            (1 - momentum) * feature.detach()
                    )
                    self.prototype_counts[subj_idx] += 1.0

        # 正規化されたプロトタイプとの類似度計算
        normalized_prototypes = F.normalize(self.style_prototypes, p=2, dim=1)
        similarities = torch.mm(style_features, normalized_prototypes.T)

        # 温度スケーリングで確率分布に変換
        temperature = 0.5
        logits = similarities / temperature

        return logits

    def _distance_based_classification(self, style_features: torch.Tensor):
        """テスト時：距離ベース分類"""
        device = style_features.device

        # 正規化されたプロトタイプとの類似度計算
        normalized_prototypes = F.normalize(self.style_prototypes, p=2, dim=1)
        similarities = torch.mm(style_features, normalized_prototypes.T)

        # 温度スケーリング
        temperature = 0.5
        logits = similarities / temperature

        return logits

    def get_prototype_info(self):
        """プロトタイプの情報を取得（デバッグ用）"""
        return {
            'prototypes': self.style_prototypes.cpu().numpy(),
            'counts': self.prototype_counts.cpu().numpy(),
            'active_prototypes': (self.prototype_counts > 0).sum().item()
        }






