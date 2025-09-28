# CLAUDE_ADDED: Improved Adaptive Gated Skip Connection Model
# 潜在空間学習を優先する段階的学習戦略を実装

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


class ProgressiveSkipGate(nn.Module):
    """CLAUDE_ADDED: Progressive Skip Gate - 段階的にskip重みを調整"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 軽量化されたゲートネットワーク（1次元入力対応）
        self.gate_network = nn.Sequential(
            nn.Linear(1, d_model // 8),  # 1次元入力（特徴差分）
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 8, 1),
            nn.Sigmoid()
        )

        # 統計追跡
        self.register_buffer('gate_stats', torch.zeros(2))  # [mean, count]

    def forward(self, latent_features, skip_features, progressive_weight=0.1):
        """
        Args:
            latent_features: 潜在空間からの特徴量
            skip_features: skip接続からの特徴量
            progressive_weight: 段階的重み（学習進行に応じて増加）
        """
        batch_size, seq_len = latent_features.shape[:2]

        # 特徴量差分に基づくゲート決定
        feature_diff = torch.abs(latent_features - skip_features).mean(dim=(1, 2))  # [batch]
        gate_input = feature_diff.unsqueeze(1)  # [batch, 1]

        # ゲート重み計算
        gate_weights = self.gate_network(gate_input)  # [batch, 1]

        # 段階的重み適用
        final_weights = gate_weights * progressive_weight

        # 統計更新
        if self.training:
            with torch.no_grad():
                self.gate_stats[0] = 0.9 * self.gate_stats[0] + 0.1 * final_weights.mean()
                self.gate_stats[1] += 1

        return final_weights.unsqueeze(1)  # [batch, 1, 1]


class ImprovedInformationBottleneck(nn.Module):
    """CLAUDE_ADDED: より厳しい情報ボトルネック"""

    def __init__(self, d_model, compression_ratio=4, dropout=0.1):
        super().__init__()
        bottleneck_dim = max(d_model // compression_ratio, 16)  # 最小16次元

        self.compressor = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),  # 更なる圧縮
            nn.GELU()
        )

        self.expander = nn.Sequential(
            nn.Linear(bottleneck_dim // 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, d_model),
            nn.LayerNorm(d_model)
        )

        self.bottleneck_dim = bottleneck_dim // 2

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            compressed: 情報圧縮された特徴量
            info_loss: 情報損失
        """
        # 情報圧縮
        compressed = self.compressor(x)  # [batch, seq_len, bottleneck_dim//2]

        # 情報復元
        restored = self.expander(compressed)  # [batch, seq_len, d_model]

        # 復元誤差
        info_loss = F.mse_loss(restored, x, reduction='mean')

        return restored, info_loss


class LatentFirstEncoder(nn.Module):
    """CLAUDE_ADDED: 潜在空間学習を優先するエンコーダ"""

    def __init__(self, input_dim, seq_len, d_model, n_heads, n_layers, dropout,
                 style_latent_dim, skill_latent_dim):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len

        # 基本コンポーネント
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        # Transformer層（軽量化）
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=2*d_model,  # 軽量化
                batch_first=True,
                dropout=dropout,
                activation='gelu'
            ) for _ in range(n_layers)
        ])

        # 注意深いプーリング
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout
        )
        self.pooling_query = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # 潜在空間ヘッド（強化）
        self.style_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, style_latent_dim * 2)
        )

        self.skill_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, skill_latent_dim * 2)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """重み初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        batch_size = x.size(0)
        skip_connections = []

        # 入力処理
        encoded = self.input_proj(x)
        pos_encoded = self.pos_encoding(encoded)

        # Transformer層
        current_input = pos_encoded
        for layer in self.encoder_layers:
            current_input = layer(current_input)
            skip_connections.append(current_input)

        # アテンションプーリング
        query = self.pooling_query.unsqueeze(0).expand(batch_size, -1, -1)
        pooled_features, _ = self.attention_pooling(query, current_input, current_input)
        pooled_features = pooled_features.squeeze(1)

        # 潜在空間ヘッド
        style_params = self.style_head(pooled_features)
        skill_params = self.skill_head(pooled_features)

        return {
            'style_mu': style_params[:, :style_params.size(1) // 2],
            'style_logvar': style_params[:, style_params.size(1) // 2:],
            'skill_mu': skill_params[:, :skill_params.size(1) // 2],
            'skill_logvar': skill_params[:, skill_params.size(1) // 2:],
            'skip_connections': skip_connections
        }


class ProgressiveSkipDecoder(nn.Module):
    """CLAUDE_ADDED: 段階的Skip接続デコーダ"""

    def __init__(self, output_dim, seq_len, d_model, n_heads, n_layers, dropout,
                 style_latent_dim, skill_latent_dim):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layers = n_layers

        # 潜在変数プロジェクタ（強化）
        self.style_projector = nn.Sequential(
            nn.Linear(style_latent_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        self.skill_projector = nn.Sequential(
            nn.Linear(skill_latent_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # 融合層
        self.fusion_layers = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        # Progressive Skip Gates
        self.progressive_gates = nn.ModuleList([
            ProgressiveSkipGate(d_model, dropout) for _ in range(n_layers)
        ])

        # Improved Information Bottlenecks
        self.information_bottlenecks = nn.ModuleList([
            ImprovedInformationBottleneck(d_model, compression_ratio=6, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Skip融合層
        self.skip_fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model)
            ) for _ in range(n_layers)
        ])

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        # デコーダ層
        self.decoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 2,  # 軽量化
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(n_layers)
        ])

        # 出力層
        self.output_layers = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim)
        )

        # 損失記録
        self.register_buffer('bottleneck_losses', torch.zeros(n_layers))

    def forward(self, z_style, z_skill, skip_connections, progressive_weight=0.1):
        batch_size = z_style.size(0)

        # 潜在変数処理
        style_features = self.style_projector(z_style)
        skill_features = self.skill_projector(z_skill)
        fused_features = self.fusion_layers(
            torch.cat([style_features, skill_features], dim=-1)
        )

        # 時系列展開
        memory_from_z = fused_features.unsqueeze(1).expand(-1, self.seq_len, -1)
        decoded_features = self.pos_encoding(memory_from_z)

        # Progressive Skip Connection処理
        bottleneck_losses = []
        gate_weights_list = []

        for i, (layer, gate, bottleneck) in enumerate(
            zip(self.decoder_layers, self.progressive_gates, self.information_bottlenecks)
        ):
            # Skip接続取得
            skip_idx = len(skip_connections) - i - 1
            skip_idx = max(0, min(skip_idx, len(skip_connections) - 1))
            skip = skip_connections[skip_idx]

            # 厳格な情報ボトルネック適用
            processed_skip, info_loss = bottleneck(skip)
            bottleneck_losses.append(info_loss)

            # Progressive Gate計算
            gate_weights = gate(decoded_features, processed_skip, progressive_weight)
            gate_weights_list.append(gate_weights.mean().item())

            # ゲート適用（段階的重み制御）
            gated_skip = gate_weights * processed_skip

            # 特徴量融合
            combined_memory = torch.cat([decoded_features, gated_skip], dim=2)
            fused_input = self.skip_fusion_layers[i](combined_memory)

            # Transformer処理
            decoded_features = layer(fused_input)

        # 損失記録
        self.bottleneck_losses = torch.stack(bottleneck_losses)

        # 出力
        output = self.output_layers(decoded_features)

        return output, {
            'bottleneck_losses': self.bottleneck_losses,
            'gate_weights': gate_weights_list
        }


class ImprovedAdaptiveGatedSkipNet(BaseExperimentModel):
    """CLAUDE_ADDED: 潜在空間学習を優先する改良版Adaptive Gated Skip Connection Network"""

    def __init__(self,
                 input_dim=6,
                 seq_len=100,
                 d_model=128,
                 n_heads=4,
                 n_encoder_layers=2,
                 n_decoder_layers=2,
                 dropout=0.1,
                 style_latent_dim=16,
                 skill_latent_dim=16,
                 n_subjects=6,
                 loss_schedule_config: Dict[str, Any] = None,
                 **kwargs):

        super().__init__(input_dim=input_dim, seq_len=seq_len, d_model=d_model,
                         n_heads=n_heads, n_encoder_layers=n_encoder_layers,
                         n_decoder_layers=n_decoder_layers, dropout=dropout,
                         style_latent_dim=style_latent_dim, skill_latent_dim=skill_latent_dim,
                         n_subjects=n_subjects, loss_schedule_config=loss_schedule_config,
                         **kwargs)

        self.seq_len = seq_len
        self.d_model = d_model
        self.style_latent_dim = style_latent_dim
        self.skill_latent_dim = skill_latent_dim

        print(f"ImprovedAdaptiveGatedSkipNet instantiated:")
        print(f"  Focus: Latent space learning priority")
        print(f"  Progressive skip strategy: Enabled")
        print(f"  Enhanced information bottleneck: Enabled")

        # CLAUDE_ADDED: 潜在空間学習を優先する段階的スケジューラ
        if loss_schedule_config is None:
            loss_schedule_config = {
                # 強化されたKL損失（潜在空間構造化促進）
                'beta_style': {'schedule': 'linear', 'start_epoch': 5, 'end_epoch': 30, 'start_val': 0.0, 'end_val': 0.01},
                'beta_skill': {'schedule': 'linear', 'start_epoch': 5, 'end_epoch': 30, 'start_val': 0.0, 'end_val': 0.005},
                'orthogonal_loss': {'schedule': 'constant', 'val': 0.1},
                'contrastive_loss': {'schedule': 'constant', 'val': 0.2},  # 強化
                'manifold_loss': {'schedule': 'constant', 'val': 0.15},
                'style_classification_loss': {'schedule': 'constant', 'val': 0.3},  # 大幅強化
                'skill_regression_loss': {'schedule': 'constant', 'val': 0.1},

                # Skip接続の段階的制御（潜在空間優先→skip活用）
                'progressive_skip_weight': {
                    'schedule': 'sigmoid_delayed',
                    'start_epoch': 60, 'end_epoch': 120,
                    'start_val': 0.05, 'end_val': 0.8  # 初期は潜在空間優先
                },

                # 改良されたゲート制御
                'gate_entropy_loss': {'schedule': 'linear', 'start_epoch': 40, 'end_epoch': 80, 'start_val': 0.2, 'end_val': 0.05},
                'gate_balance_loss': {'schedule': 'constant', 'val': 0.02},  # 軽減
                'bottleneck_loss': {'schedule': 'linear', 'start_epoch': 20, 'end_epoch': 60, 'start_val': 0.05, 'end_val': 0.02}
            }
        self.loss_scheduler = LossWeightScheduler(loss_schedule_config)

        # 損失計算フラグ
        self.calc_orthogonal_loss = 'orthogonal_loss' in loss_schedule_config
        self.calc_contrastive_loss = 'contrastive_loss' in loss_schedule_config
        self.calc_manifold_loss = 'manifold_loss' in loss_schedule_config
        self.calc_style_subtask = 'style_classification_loss' in loss_schedule_config
        self.calc_skill_subtask = 'skill_regression_loss' in loss_schedule_config

        # エンコーダ・デコーダ
        self.encoder = LatentFirstEncoder(
            input_dim, seq_len, d_model, n_heads, n_encoder_layers,
            dropout, style_latent_dim, skill_latent_dim
        )

        self.decoder = ProgressiveSkipDecoder(
            input_dim, seq_len, d_model, n_heads, n_decoder_layers,
            dropout, style_latent_dim, skill_latent_dim
        )

        # サブタスクネットワーク
        if self.calc_style_subtask:
            self.style_classifier = nn.Sequential(
                nn.Linear(style_latent_dim, style_latent_dim),
                nn.GELU(),
                nn.LayerNorm(style_latent_dim),
                nn.Dropout(0.1),
                nn.Linear(style_latent_dim, n_subjects)
            )

        if self.calc_skill_subtask:
            self.skill_regressor = nn.Sequential(
                nn.Linear(skill_latent_dim, skill_latent_dim // 2),
                nn.GELU(),
                nn.LayerNorm(skill_latent_dim // 2),
                nn.Dropout(0.1),
                nn.Linear(skill_latent_dim // 2, 1)
            )

        # 対照学習
        self.contrastive_loss = SupervisedContrastiveLoss()

        # エポック追跡
        self.current_epoch = 0
        self.max_epochs = 200

    def on_epoch_start(self, epoch: int):
        """エポック開始時のコールバック"""
        self.loss_scheduler.step(epoch)
        self.current_epoch = epoch

        # 段階的学習の進捗ログ
        if epoch % 20 == 0:
            weights = self.loss_scheduler.get_weights()
            skip_weight = weights.get('progressive_skip_weight', 0.05)
            print(f"Epoch {epoch}: Progressive skip weight = {skip_weight:.3f}")

    def reparameterize(self, mu, logvar):
        """再パラメータ化"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, subject_ids: str = None, skill_scores: float = None) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]

        # エンコード
        encoded = self.encoder(x)

        # 潜在変数サンプリング
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])

        # Progressive Skip Weight取得
        weights = self.loss_scheduler.get_weights()
        progressive_weight = weights.get('progressive_skip_weight', 0.05)

        # デコード（Progressive Skip制御付き）
        reconstructed, decoder_info = self.decoder(
            z_style, z_skill, encoded['skip_connections'], progressive_weight
        )

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
            subject_pred, skill_score_pred, subject_ids, skill_scores,
            decoder_info, progressive_weight
        )

        result = {
            'reconstructed': reconstructed,
            'z_style': z_style,
            'z_skill': z_skill,
        }

        return result | losses

    def compute_losses(self, x, reconstructed, encoded, z_style, z_skill,
                      subject_pred, skill_score_pred, subject_ids, skill_scores,
                      decoder_info, progressive_weight):
        weights = self.loss_scheduler.get_weights()

        # 基本損失
        losses = {'reconstruction_loss': F.mse_loss(reconstructed, x)}

        # 強化されたKL損失（潜在空間構造化促進）
        style_mu = torch.clamp(encoded['style_mu'], min=-3, max=3)  # より厳しいクランプ
        skill_mu = torch.clamp(encoded['skill_mu'], min=-3, max=3)
        style_logvar = torch.clamp(encoded['style_logvar'], min=-5, max=5)
        skill_logvar = torch.clamp(encoded['skill_logvar'], min=-5, max=5)

        style_kl_terms = 1 + style_logvar - style_mu.pow(2) - style_logvar.exp()
        skill_kl_terms = 1 + skill_logvar - skill_mu.pow(2) - skill_logvar.exp()

        losses['kl_style_loss'] = -0.5 * torch.mean(torch.sum(style_kl_terms, dim=1))
        losses['kl_skill_loss'] = -0.5 * torch.mean(torch.sum(skill_kl_terms, dim=1))

        # Progressive Skip関連損失
        losses['bottleneck_loss'] = decoder_info['bottleneck_losses'].mean()

        # ゲート統計
        gate_weights = torch.tensor(decoder_info['gate_weights'], device=x.device)
        if len(gate_weights) > 0:
            gate_entropy = -(gate_weights * torch.log(gate_weights + 1e-8) +
                           (1-gate_weights) * torch.log(1-gate_weights + 1e-8)).mean()
            losses['gate_entropy_loss'] = -gate_entropy  # エントロピー最大化
            losses['gate_balance_loss'] = torch.abs(gate_weights.mean() - 0.5)
        else:
            losses['gate_entropy_loss'] = torch.tensor(0.0, device=x.device)
            losses['gate_balance_loss'] = torch.tensor(0.0, device=x.device)

        # 直交性損失（強化）
        if self.calc_orthogonal_loss:
            z_style_norm = F.normalize(z_style - z_style.mean(dim=0), p=2, dim=1)
            z_skill_norm = F.normalize(z_skill - z_skill.mean(dim=0), p=2, dim=1)
            cross_correlation = torch.matmul(z_style_norm.T, z_skill_norm) / z_style.shape[0]
            losses['orthogonal_loss'] = torch.mean(cross_correlation ** 2)

        # サブタスク損失（強化）
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

        # 総合損失計算（段階的重み適用）
        beta_style_weight = weights.get('beta_style', 0.0)
        beta_skill_weight = weights.get('beta_skill', 0.0)

        total_loss = (
            losses['reconstruction_loss'] +
            beta_style_weight * losses['kl_style_loss'] +
            beta_skill_weight * losses['kl_skill_loss'] +
            weights.get('orthogonal_loss', 0.0) * losses.get('orthogonal_loss', torch.tensor(0.0, device=x.device)) +
            weights.get('contrastive_loss', 0.0) * losses.get('contrastive_loss', torch.tensor(0.0, device=x.device)) +
            weights.get('manifold_loss', 0.0) * losses.get('manifold_loss', torch.tensor(0.0, device=x.device)) +
            weights.get('style_classification_loss', 0.0) * losses.get('style_classification_loss', torch.tensor(0.0, device=x.device)) +
            weights.get('skill_regression_loss', 0.0) * losses.get('skill_regression_loss', torch.tensor(0.0, device=x.device)) +
            weights.get('gate_entropy_loss', 0.0) * losses['gate_entropy_loss'] +
            weights.get('gate_balance_loss', 0.0) * losses['gate_balance_loss'] +
            weights.get('bottleneck_loss', 0.0) * losses['bottleneck_loss']
        )

        losses['total_loss'] = total_loss

        return losses

    def compute_manifold_loss(self, z_skill: torch.Tensor, skill_score: torch.Tensor):
        """改良されたmanifold loss"""
        device = z_skill.device
        batch_size = z_skill.size(0)

        expert_mask = skill_score > 0.0
        novice_mask = skill_score <= 0.0

        n_experts = expert_mask.sum()
        n_novices = novice_mask.sum()

        loss_expert_cohesion = torch.tensor(0.0, device=device, requires_grad=True)
        loss_separation = torch.tensor(0.0, device=device, requires_grad=True)

        # 熟達者凝集性（動的中心）
        if n_experts > 1:
            z_skill_experts = z_skill[expert_mask]
            expert_scores = skill_score[expert_mask]

            # スキルスコアで重み付けした重心
            expert_weights = torch.softmax(expert_scores * 2.0, dim=0)
            expert_centroid = torch.sum(expert_weights.unsqueeze(1) * z_skill_experts, dim=0)

            # 重心からの距離
            distances_to_centroid = torch.norm(z_skill_experts - expert_centroid.unsqueeze(0), p=2, dim=1)
            cohesion_weights = torch.sigmoid(expert_scores * 3.0)
            loss_expert_cohesion = torch.mean(cohesion_weights * distances_to_centroid)

        elif n_experts == 1:
            expert_centroid = z_skill[expert_mask].squeeze(0)
        else:
            expert_centroid = torch.zeros(z_skill.size(1), device=device)

        # 分離損失
        if n_novices > 0 and n_experts > 0:
            z_skill_novices = z_skill[novice_mask]
            novice_scores = skill_score[novice_mask]

            distances_from_expert_center = torch.norm(z_skill_novices - expert_centroid.unsqueeze(0), p=2, dim=1)
            separation_weights = torch.sigmoid(-novice_scores * 2.0)

            margin = 2.0
            margin_violations = torch.clamp(margin - distances_from_expert_center, min=0.0)
            loss_separation = torch.mean(separation_weights * margin_violations ** 2)

        return 2.0 * loss_expert_cohesion + 1.5 * loss_separation

    def encode(self, x):
        """エンコードのみ"""
        encoded = self.encoder(x)
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])
        return {'z_style': z_style, 'z_skill': z_skill, 'skip_connections': encoded['skip_connections']}

    def decode(self, z_style, z_skill, skip_connections=None):
        """デコードのみ"""
        if skip_connections is None:
            batch_size = z_style.size(0)
            dummy_skip = torch.zeros(batch_size, self.seq_len, self.d_model, device=z_style.device)
            skip_connections = [dummy_skip.clone() for _ in range(self.decoder.n_layers)]

        weights = self.loss_scheduler.get_weights()
        progressive_weight = weights.get('progressive_skip_weight', 0.05)

        trajectory, _ = self.decoder(z_style, z_skill, skip_connections, progressive_weight)
        return {'trajectory': trajectory}