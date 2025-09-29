# CLAUDE_ADDED: Robust Style Encoder with FiLM
# スタイル表現学習の根本的問題を解決する改良版

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.base_model import BaseExperimentModel
from models.components.loss_weight_scheduler import LossWeightScheduler


class SupervisedContrastiveLoss(nn.Module):
    """改良された対照学習損失"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        # L2正規化
        features = F.normalize(features, p=2, dim=1)

        # 類似度行列
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 対角要素除外
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=features.device)
        mask = mask * logits_mask

        if mask.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # 数値安定性を考慮したlogsumexp
        logits = similarity_matrix - (1 - logits_mask) * 1e9
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        # 正例ペアの平均log確率
        mean_log_prob_pos = (mask * log_prob).sum(1) / torch.clamp(mask.sum(1), min=1.0)

        return -mean_log_prob_pos.mean()


class PositionalEncoding(nn.Module):
    """強化された位置エンコーディング"""
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


class FilmLayer(nn.Module):
    """FiLM変調層"""
    def __init__(self, condition_dim, feature_channels):
        super().__init__()
        self.film_generator = nn.Linear(condition_dim, feature_channels*2)

    def forward(self, features, condition):
        gamma_beta = self.film_generator(condition)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        return gamma * features + beta


class RobustStyleEncoder(nn.Module):
    """CLAUDE_ADDED: 堅牢なスタイルエンコーダ（被験者一貫性重視）"""

    def __init__(self, input_dim, seq_len, d_model, n_heads, n_layers, dropout,
                 style_latent_dim, skill_latent_dim):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len

        # 入力処理（被験者特徴抽出に最適化）
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        # 被験者特徴抽出用Multi-Scale Attention
        self.multi_scale_attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads//2, batch_first=True, dropout=dropout)
            for _ in range(2)  # Short-term & Long-term patterns
        ])

        # 軽量化されたTransformer（過学習防止）
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=2*d_model,  # 軽量化
                batch_first=True,
                dropout=dropout,
                activation='gelu'
            ) for _ in range(min(n_layers, 3))  # 最大3層に制限
        ])

        # 被験者特徴統合（複数の統計的特徴を組み合わせ）
        self.subject_feature_extractor = nn.ModuleList([
            # Mean pooling
            nn.AdaptiveAvgPool1d(1),
            # Max pooling
            nn.AdaptiveMaxPool1d(1),
            # Attention pooling with learnable query
        ])

        # Learnable attention query for subject characterization
        self.subject_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.subject_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)

        # 被験者特徴融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model),  # 3つのpooling結果を融合
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # 潜在空間ヘッド（正則化を段階的に適用）
        self.style_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, style_latent_dim * 2)
        )

        self.skill_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, skill_latent_dim * 2)
        )

        # 被験者一貫性用のメモリバンク（推論時の安定性向上）
        self.register_buffer('subject_memory', torch.zeros(6, style_latent_dim))  # 6 subjects
        self.register_buffer('memory_counts', torch.zeros(6))

    def forward(self, x, subject_ids=None, training_phase='mid'):
        batch_size = x.size(0)
        skip_connections = []

        # 入力処理
        encoded = self.input_proj(x)
        pos_encoded = self.pos_encoding(encoded)

        # Multi-scale attention for subject feature extraction
        short_term_features, _ = self.multi_scale_attention[0](
            pos_encoded[:, :50, :], pos_encoded[:, :50, :], pos_encoded[:, :50, :]
        )
        long_term_features, _ = self.multi_scale_attention[1](
            pos_encoded, pos_encoded, pos_encoded
        )

        # Combine multi-scale features
        combined_features = pos_encoded + 0.3 * F.pad(short_term_features, (0, 0, 0, 50), "constant", 0) + 0.3 * long_term_features

        # Lightweight Transformer processing
        current_input = combined_features
        for layer in self.encoder_layers:
            current_input = layer(current_input)
            skip_connections.append(current_input)

        # 複数の被験者特徴抽出
        # 1. Mean pooling
        mean_features = current_input.mean(dim=1)  # [batch, d_model]

        # 2. Max pooling
        max_features = current_input.max(dim=1)[0]  # [batch, d_model]

        # 3. Learnable attention pooling
        query = self.subject_query.unsqueeze(0).expand(batch_size, -1, -1)
        attended_features, _ = self.subject_attention(query, current_input, current_input)
        attended_features = attended_features.squeeze(1)  # [batch, d_model]

        # 特徴融合
        fused_features = torch.cat([mean_features, max_features, attended_features], dim=-1)
        subject_features = self.feature_fusion(fused_features)

        # 被験者メモリバンクとの一貫性チェック（学習時のみ）
        if self.training and subject_ids is not None:
            subject_features = self._apply_memory_consistency(subject_features, subject_ids)

        # 潜在空間射影
        style_params = self.style_head(subject_features)
        skill_params = self.skill_head(subject_features)

        return {
            'style_mu': style_params[:, :style_params.size(1) // 2],
            'style_logvar': style_params[:, style_params.size(1) // 2:],
            'skill_mu': skill_params[:, :skill_params.size(1) // 2],
            'skill_logvar': skill_params[:, skill_params.size(1) // 2:],
            'skip_connections': skip_connections,
            'subject_features': subject_features  # For consistency loss
        }

    def _apply_memory_consistency(self, subject_features, subject_ids):
        """被験者メモリバンクとの一貫性を適用"""
        all_subjects = ['h.nakamura', 'r.morishita', 'r.yanase', 's.miyama', 's.tahara', 't.hasegawa']
        subject_to_idx = {subj: i for i, subj in enumerate(all_subjects)}

        consistent_features = subject_features.clone()

        for i, subj_id in enumerate(subject_ids):
            if subj_id in subject_to_idx:
                subj_idx = subject_to_idx[subj_id]
                current_count = self.memory_counts[subj_idx].item()

                if current_count > 0:
                    # 既存メモリとの一貫性を促進
                    memory_feature = self.subject_memory[subj_idx]
                    consistency_weight = min(current_count / 100.0, 0.3)  # 最大30%の影響
                    consistent_features[i] = (1 - consistency_weight) * subject_features[i] + consistency_weight * memory_feature

                # メモリ更新
                momentum = 0.95
                self.subject_memory[subj_idx] = momentum * self.subject_memory[subj_idx] + (1 - momentum) * subject_features[i].detach()
                self.memory_counts[subj_idx] += 1

        return consistent_features


class RobustStyleSkipDecoder(nn.Module):
    """FiLM強化デコーダ（シンプル化）"""

    def __init__(self, output_dim, seq_len, d_model, n_heads, n_layers, dropout,
                 style_latent_dim, skill_latent_dim, skip_dropout=0.3):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layers = min(n_layers, 3)  # 軽量化

        # 潜在変数プロジェクタ
        self.style_projector = nn.Sequential(
            nn.Linear(style_latent_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        self.skill_projector = nn.Sequential(
            nn.Linear(skill_latent_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        # 融合層
        self.fusion_layers = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        # FiLM層（各デコーダ層に対応）
        self.film_layers = nn.ModuleList([
            FilmLayer(style_latent_dim, d_model) for _ in range(self.n_layers)
        ])

        # 軽量化されたデコーダ層
        self.decoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model*2,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(self.n_layers)
        ])

        # Skip接続制御（シンプル化）
        self.skip_weights = nn.Parameter(torch.ones(self.n_layers) * 0.5)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        # 出力層
        self.output_layers = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, z_style, z_skill, skip_connections):
        batch_size = z_style.size(0)

        # 潜在変数処理
        style_features = self.style_projector(z_style)
        skill_features = self.skill_projector(z_skill)
        fused_features = self.fusion_layers(torch.cat([style_features, skill_features], dim=-1))

        # 時系列展開
        memory_from_z = fused_features.unsqueeze(1).expand(-1, self.seq_len, -1)
        decoded_features = self.pos_encoding(memory_from_z)

        # デコーダ処理（FiLM + Skip）
        for i, (layer, film) in enumerate(zip(self.decoder_layers, self.film_layers)):
            # Transformer処理
            transformed_features = layer(decoded_features)

            # FiLM変調
            filmed_features = film(transformed_features, z_style)

            # Skip接続（適応的重み）
            if i < len(skip_connections):
                skip = skip_connections[min(i, len(skip_connections)-1)]
                skip_weight = torch.sigmoid(self.skip_weights[i])
                decoded_features = (1 - skip_weight) * filmed_features + skip_weight * skip
            else:
                decoded_features = filmed_features

        # 出力
        output = self.output_layers(decoded_features)
        return output


class RobustStyleFilmNet(BaseExperimentModel):
    """CLAUDE_ADDED: 堅牢なスタイル学習を重視したFiLMネットワーク"""

    def __init__(self,
                 input_dim=6,
                 seq_len=100,
                 d_model=256,
                 n_heads=8,
                 n_encoder_layers=3,
                 n_decoder_layers=3,
                 dropout=0.1,
                 skip_dropout=0.3,
                 style_latent_dim=16,
                 skill_latent_dim=16,
                 n_subjects=6,
                 loss_schedule_config: Dict[str, Any] = None,
                 **kwargs):

        super().__init__(input_dim=input_dim, seq_len=seq_len, d_model=d_model,
                         n_heads=n_heads, n_encoder_layers=n_encoder_layers,
                         n_decoder_layers=n_decoder_layers, dropout=dropout,
                         skip_dropout=skip_dropout, style_latent_dim=style_latent_dim,
                         skill_latent_dim=skill_latent_dim, n_subjects=n_subjects,
                         loss_schedule_config=loss_schedule_config, **kwargs)

        print(f"RobustStyleFilmNet instantiated with focus on subject consistency")

        # 段階的学習用の損失スケジューラ
        if loss_schedule_config is None:
            loss_schedule_config = {
                # Stage 1: 基本VAE（軽い正則化）
                'beta_style': {'schedule': 'linear', 'start_epoch': 10, 'end_epoch': 30, 'start_val': 0.0, 'end_val': 0.005},
                'beta_skill': {'schedule': 'linear', 'start_epoch': 10, 'end_epoch': 30, 'start_val': 0.0, 'end_val': 0.002},

                # Stage 2: スタイル学習強化
                'style_classification_loss': {'schedule': 'linear', 'start_epoch': 5, 'end_epoch': 25, 'start_val': 0.0, 'end_val': 2.0},
                'contrastive_loss': {'schedule': 'linear', 'start_epoch': 8, 'end_epoch': 30, 'start_val': 0.0, 'end_val': 1.0},
                'consistency_loss': {'schedule': 'linear', 'start_epoch': 15, 'end_epoch': 40, 'start_val': 0.0, 'end_val': 0.5},

                # Stage 3: 分離最適化
                'orthogonal_loss': {'schedule': 'linear', 'start_epoch': 25, 'end_epoch': 50, 'start_val': 0.0, 'end_val': 0.3},
                'manifold_loss': {'schedule': 'linear', 'start_epoch': 30, 'end_epoch': 60, 'start_val': 0.0, 'end_val': 0.2},
                'skill_regression_loss': {'schedule': 'linear', 'start_epoch': 20, 'end_epoch': 50, 'start_val': 0.0, 'end_val': 0.1}
            }
        self.loss_scheduler = LossWeightScheduler(loss_schedule_config)

        # フラグ
        self.calc_orthogonal_loss = 'orthogonal_loss' in loss_schedule_config
        self.calc_contrastive_loss = 'contrastive_loss' in loss_schedule_config
        self.calc_manifold_loss = 'manifold_loss' in loss_schedule_config
        self.calc_style_subtask = 'style_classification_loss' in loss_schedule_config
        self.calc_skill_subtask = 'skill_regression_loss' in loss_schedule_config
        self.calc_consistency_loss = 'consistency_loss' in loss_schedule_config

        # エンコーダ・デコーダ
        self.encoder = RobustStyleEncoder(
            input_dim, seq_len, d_model, n_heads, n_encoder_layers,
            dropout, style_latent_dim, skill_latent_dim
        )

        self.decoder = RobustStyleSkipDecoder(
            input_dim, seq_len, d_model, n_heads, n_decoder_layers,
            dropout, style_latent_dim, skill_latent_dim, skip_dropout
        )

        # サブタスクネットワーク
        if self.calc_style_subtask:
            self.style_classifier = nn.Sequential(
                nn.Linear(style_latent_dim, style_latent_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(style_latent_dim, n_subjects)
            )

        if self.calc_skill_subtask:
            self.skill_regressor = nn.Sequential(
                nn.Linear(skill_latent_dim, skill_latent_dim//2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(skill_latent_dim//2, 1)
            )

        # 対照学習
        self.contrastive_loss = SupervisedContrastiveLoss()

        # エポック管理
        self.current_epoch = 0

    def on_epoch_start(self, epoch: int):
        self.loss_scheduler.step(epoch)
        self.current_epoch = epoch

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, subject_ids: str = None, skill_scores: float = None) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]

        # エンコード
        encoded = self.encoder(x, subject_ids, training_phase='mid')

        # 潜在変数サンプリング
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])

        # デコード
        reconstructed = self.decoder(z_style, z_skill, encoded['skip_connections'])

        # サブタスク
        subject_pred = None
        skill_score_pred = None
        if self.calc_style_subtask:
            subject_pred = self.style_classifier(z_style)
        if self.calc_skill_subtask:
            skill_score_pred = self.skill_regressor(z_skill)

        # 損失計算
        losses = self.compute_losses(
            x, reconstructed, encoded, z_style, z_skill,
            subject_pred, skill_score_pred, subject_ids, skill_scores
        )

        result = {
            'reconstructed': reconstructed,
            'z_style': z_style,
            'z_skill': z_skill,
        }

        return result | losses

    def compute_losses(self, x, reconstructed, encoded, z_style, z_skill,
                      subject_pred, skill_score_pred, subject_ids, skill_scores):
        weights = self.loss_scheduler.get_weights()

        # 基本損失
        losses = {'reconstruction_loss': F.mse_loss(reconstructed, x)}

        # 適応的KL損失（段階的正則化）
        kl_weight_style = weights.get('beta_style', 0.0)
        kl_weight_skill = weights.get('beta_skill', 0.0)

        style_mu = torch.clamp(encoded['style_mu'], min=-3, max=3)
        skill_mu = torch.clamp(encoded['skill_mu'], min=-3, max=3)
        style_logvar = torch.clamp(encoded['style_logvar'], min=-5, max=5)
        skill_logvar = torch.clamp(encoded['skill_logvar'], min=-5, max=5)

        style_kl_terms = 1 + style_logvar - style_mu.pow(2) - style_logvar.exp()
        skill_kl_terms = 1 + skill_logvar - skill_mu.pow(2) - skill_logvar.exp()

        losses['kl_style_loss'] = -0.5 * torch.mean(torch.sum(style_kl_terms, dim=1))
        losses['kl_skill_loss'] = -0.5 * torch.mean(torch.sum(skill_kl_terms, dim=1))

        # 被験者一貫性損失
        if self.calc_consistency_loss and subject_ids is not None:
            losses['consistency_loss'] = self._compute_consistency_loss(z_style, subject_ids)

        # 直交性損失
        if self.calc_orthogonal_loss:
            z_style_norm = F.normalize(z_style - z_style.mean(dim=0), p=2, dim=1)
            z_skill_norm = F.normalize(z_skill - z_skill.mean(dim=0), p=2, dim=1)
            cross_correlation = torch.matmul(z_style_norm.T, z_skill_norm) / z_style.shape[0]
            losses['orthogonal_loss'] = torch.mean(cross_correlation ** 2)

        # サブタスク損失
        if subject_ids is not None:
            all_subjects = ['h.nakamura', 'r.morishita', 'r.yanase', 's.miyama', 's.tahara', 't.hasegawa']
            subject_to_idx = {subj: i for i, subj in enumerate(all_subjects)}
            subject_indices = torch.tensor([subject_to_idx[subj] for subj in subject_ids], device=z_style.device)

            if self.calc_contrastive_loss:
                losses['contrastive_loss'] = self.contrastive_loss(z_style, subject_indices)

            if self.calc_style_subtask and subject_pred is not None:
                losses['style_classification_loss'] = F.cross_entropy(subject_pred, subject_indices)

        if skill_scores is not None:
            if self.calc_skill_subtask and skill_score_pred is not None:
                skill_score_pred_flat = skill_score_pred.squeeze(-1)
                losses['skill_regression_loss'] = F.mse_loss(skill_score_pred_flat, skill_scores)

            if self.calc_manifold_loss:
                losses['manifold_loss'] = self.compute_manifold_loss(z_skill, skill_scores)

        # 総合損失計算
        total_loss = (
            losses['reconstruction_loss'] +
            kl_weight_style * losses['kl_style_loss'] +
            kl_weight_skill * losses['kl_skill_loss'] +
            weights.get('orthogonal_loss', 0.0) * losses.get('orthogonal_loss', torch.tensor(0.0, device=x.device)) +
            weights.get('contrastive_loss', 0.0) * losses.get('contrastive_loss', torch.tensor(0.0, device=x.device)) +
            weights.get('manifold_loss', 0.0) * losses.get('manifold_loss', torch.tensor(0.0, device=x.device)) +
            weights.get('style_classification_loss', 0.0) * losses.get('style_classification_loss', torch.tensor(0.0, device=x.device)) +
            weights.get('skill_regression_loss', 0.0) * losses.get('skill_regression_loss', torch.tensor(0.0, device=x.device)) +
            weights.get('consistency_loss', 0.0) * losses.get('consistency_loss', torch.tensor(0.0, device=x.device))
        )

        losses['total_loss'] = total_loss
        return losses

    def _compute_consistency_loss(self, z_style, subject_ids):
        """被験者一貫性損失（同一被験者のスタイル表現を近づける）"""
        unique_subjects = list(set(subject_ids))
        if len(unique_subjects) < 2:
            return torch.tensor(0.0, device=z_style.device)

        consistency_loss = 0.0
        count = 0

        for subject in unique_subjects:
            subject_indices = [i for i, subj in enumerate(subject_ids) if subj == subject]
            if len(subject_indices) > 1:
                subject_styles = z_style[subject_indices]
                # 同一被験者内の分散を最小化
                subject_mean = subject_styles.mean(dim=0)
                subject_variance = torch.mean((subject_styles - subject_mean.unsqueeze(0))**2)
                consistency_loss += subject_variance
                count += 1

        if count > 0:
            return consistency_loss / count
        else:
            return torch.tensor(0.0, device=z_style.device)

    def compute_manifold_loss(self, z_skill: torch.Tensor, skill_score: torch.Tensor):
        """シンプルなmanifold loss"""
        device = z_skill.device
        expert_mask = skill_score > 0.0
        novice_mask = skill_score <= 0.0

        n_experts = expert_mask.sum()
        n_novices = novice_mask.sum()

        if n_experts > 1 and n_novices > 1:
            expert_centroid = z_skill[expert_mask].mean(dim=0)
            novice_centroid = z_skill[novice_mask].mean(dim=0)
            separation = torch.norm(expert_centroid - novice_centroid, p=2)
            return torch.clamp(2.0 - separation, min=0.0) ** 2
        else:
            return torch.tensor(0.0, device=device)

    def encode(self, x, subject_ids=None):
        encoded = self.encoder(x, subject_ids)
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])
        return {'z_style': z_style, 'z_skill': z_skill, 'skip_connections': encoded['skip_connections']}

    def decode(self, z_style, z_skill, skip_connections=None):
        if skip_connections is None:
            batch_size = z_style.size(0)
            dummy_skip = torch.zeros(batch_size, self.seq_len, self.d_model, device=z_style.device)
            skip_connections = [dummy_skip.clone() for _ in range(self.decoder.n_layers)]

        trajectory = self.decoder(z_style, z_skill, skip_connections)
        return {'trajectory': trajectory}