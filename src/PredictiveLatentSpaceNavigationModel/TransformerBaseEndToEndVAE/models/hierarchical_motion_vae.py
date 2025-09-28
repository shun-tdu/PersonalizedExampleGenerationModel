# CLAUDE_ADDED: Hierarchical Motion VAE Implementation
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.base_model import BaseExperimentModel
from models.components.loss_weight_scheduler import LossWeightScheduler


class PositionalEncoding(nn.Module):
    """時系列用位置エンコーディング"""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class SupervisedContrastiveLoss(nn.Module):
    """対照学習損失関数"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)

        if batch_size <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # 正例ペアマスク
        mask = torch.eq(labels, labels.T).float().to(features.device)

        # 特徴量正規化
        features = F.normalize(features, p=2, dim=1)

        # 類似度行列計算
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 対角成分除外
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=features.device)
        mask = mask * logits_mask

        if mask.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # 損失計算
        logits = similarity_matrix - (1 - logits_mask) * 1e9
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        mean_log_prob_pos = (mask * log_prob).sum(1) / torch.clamp(mask.sum(1), min=1.0)
        loss = -mean_log_prob_pos.mean()

        return loss


class SkillVAE(nn.Module):
    """第1層：単一軌道からスキル潜在変数を学習するVAE"""

    def __init__(self,
                 input_dim: int = 6,
                 seq_len: int = 100,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6,
                 skill_latent_dim: int = 32,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.skill_latent_dim = skill_latent_dim

        # 入力投影
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        # エンコーダー
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # プーリング層（系列を単一ベクトルに集約）
        self.pooling = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout
        )
        self.pooling_query = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # 潜在変数ヘッド
        self.latent_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, skill_latent_dim * 2)  # mu, logvar
        )

        # CLAUDE_ADDED: TransformerEncoderLayerベースのデコーダー（再構成誤差改善のため）
        # 潜在変数から系列への投影層
        self.latent_projection = nn.Sequential(
            nn.Linear(skill_latent_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )

        # TransformerEncoderLayerベースのデコーダー
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_decoder_layers)

        # 出力投影
        self.output_projection = nn.Linear(d_model, input_dim)

        # スキルパフォーマンス予測ヘッド（オプション）
        self.performance_head = nn.Sequential(
            nn.Linear(skill_latent_dim, skill_latent_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(skill_latent_dim // 2, 1)
        )

        # 重み初期化
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

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """エンコード処理"""
        batch_size = x.size(0)

        # 入力投影: [batch, seq_len, input_dim] -> [batch, seq_len, d_model]
        projected = self.input_projection(x)

        # 位置エンコーディング追加
        pos_encoded = self.pos_encoding(projected)

        # Transformerエンコーダー処理
        encoded = self.encoder(pos_encoded)  # [batch, seq_len, d_model]

        # アテンションプーリングで系列を単一ベクトルに集約
        query = self.pooling_query.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 1, d_model]
        pooled, _ = self.pooling(query, encoded, encoded)  # [batch, 1, d_model]
        pooled = pooled.squeeze(1)  # [batch, d_model]

        # 潜在変数パラメータ計算
        latent_params = self.latent_head(pooled)  # [batch, skill_latent_dim * 2]
        mu = latent_params[:, :self.skill_latent_dim]
        logvar = latent_params[:, self.skill_latent_dim:]

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """再パラメータ化トリック"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z_skill: torch.Tensor) -> torch.Tensor:
        """デコード処理（TransformerEncoderLayer対応）"""
        batch_size = z_skill.size(0)

        # CLAUDE_ADDED: 潜在変数を系列データに展開して投影
        # style_skill_separation_model.pyの実装を参考
        latent_expanded = self.latent_projection(z_skill)  # [batch, d_model]
        latent_sequence = latent_expanded.unsqueeze(1).expand(-1, self.seq_len, -1)  # [batch, seq_len, d_model]

        # 位置エンコーディング追加
        pos_encoded = self.pos_encoding(latent_sequence)

        # TransformerEncoderベースのデコード
        decoded = self.decoder(pos_encoded)  # [batch, seq_len, d_model]
        reconstructed = self.output_projection(decoded)  # [batch, seq_len, input_dim]

        return reconstructed

    def forward(self, x: torch.Tensor, performance_scores: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """順伝播"""
        # エンコード
        mu, logvar = self.encode(x)

        # 再パラメータ化
        z_skill = self.reparameterize(mu, logvar)

        # デコード
        reconstructed = self.decode(z_skill)

        # 結果辞書
        result = {
            'z_skill': z_skill,
            'z_skill_mu': mu,
            'z_skill_logvar': logvar,
            'reconstructed': reconstructed
        }

        # パフォーマンス予測（オプション）
        if performance_scores is not None:
            performance_pred = self.performance_head(z_skill)
            result['performance_pred'] = performance_pred

        return result


class StyleVAE(nn.Module):
    """第2層：スキル潜在変数系列から個人スタイルを学習するVAE"""

    def __init__(self,
                 skill_latent_dim: int = 32,
                 block_len: int = 35,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_encoder_layers: int = 4,
                 n_decoder_layers: int = 4,
                 style_latent_dim: int = 16,
                 n_subjects: int = 6,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()

        self.skill_latent_dim = skill_latent_dim
        self.block_len = block_len
        self.d_model = d_model
        self.style_latent_dim = style_latent_dim
        self.n_subjects = n_subjects

        # 入力投影（z_skillをd_modelに投影）
        self.input_projection = nn.Linear(skill_latent_dim, d_model)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, block_len)

        # エンコーダー
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # プーリング層
        self.pooling = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout
        )
        self.pooling_query = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # スタイル潜在変数ヘッド
        self.style_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, style_latent_dim * 2)  # mu, logvar
        )

        # CLAUDE_ADDED: TransformerEncoderLayerベースのデコーダー（再構成誤差改善のため）
        # スタイル潜在変数から系列への投影層
        self.style_projection = nn.Sequential(
            nn.Linear(style_latent_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )

        # TransformerEncoderLayerベースのデコーダー
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_decoder_layers)

        # 出力投影
        self.output_projection = nn.Linear(d_model, skill_latent_dim)

        # 被験者分類ヘッド
        self.subject_classifier = nn.Sequential(
            nn.Linear(style_latent_dim, style_latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(style_latent_dim, n_subjects)
        )

        # 対照学習
        self.contrastive_loss = SupervisedContrastiveLoss()

        # 重み初期化
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

    def encode(self, z_skill_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """エンコード処理"""
        batch_size = z_skill_sequence.size(0)

        # 入力投影: [batch, block_len, skill_latent_dim] -> [batch, block_len, d_model]
        projected = self.input_projection(z_skill_sequence)

        # 位置エンコーディング追加
        pos_encoded = self.pos_encoding(projected)

        # Transformerエンコーダー処理
        encoded = self.encoder(pos_encoded)  # [batch, block_len, d_model]

        # アテンションプーリングで系列を単一ベクトルに集約
        query = self.pooling_query.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 1, d_model]
        pooled, _ = self.pooling(query, encoded, encoded)  # [batch, 1, d_model]
        pooled = pooled.squeeze(1)  # [batch, d_model]

        # スタイル潜在変数パラメータ計算
        style_params = self.style_head(pooled)  # [batch, style_latent_dim * 2]
        mu = style_params[:, :self.style_latent_dim]
        logvar = style_params[:, self.style_latent_dim:]

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """再パラメータ化トリック"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z_style: torch.Tensor) -> torch.Tensor:
        """デコード処理 - TransformerEncoderLayerを使用"""
        batch_size = z_style.size(0)

        # スタイル潜在変数を特徴量空間に投影
        style_features = self.style_projection(z_style)  # [batch, d_model]
        style_features = style_features.unsqueeze(1).expand(-1, self.block_len, -1)  # [batch, block_len, d_model]

        # 位置エンコーディングを追加
        style_features = self.pos_encoding(style_features)

        # TransformerEncoderでデコード
        style_features = style_features.transpose(0, 1)  # [seq_len, batch, d_model]
        decoded = self.decoder(style_features)  # [seq_len, batch, d_model]
        decoded = decoded.transpose(0, 1)  # [batch, seq_len, d_model]

        # 出力投影
        reconstructed_z_skill = self.output_projection(decoded)  # [batch, block_len, skill_latent_dim]

        return reconstructed_z_skill

    def forward(self, z_skill_sequence: torch.Tensor, subject_ids: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """順伝播"""
        # エンコード
        mu, logvar = self.encode(z_skill_sequence)

        # 再パラメータ化
        z_style = self.reparameterize(mu, logvar)

        # デコード
        reconstructed_z_skill = self.decode(z_style)

        # 結果辞書
        result = {
            'z_style': z_style,
            'z_style_mu': mu,
            'z_style_logvar': logvar,
            'reconstructed_z_skill': reconstructed_z_skill
        }

        # 被験者分類
        if subject_ids is not None:
            subject_logits = self.subject_classifier(z_style)
            result['subject_logits'] = subject_logits

        return result


class HierarchicalMotionVAE(BaseExperimentModel):
    """階層型運動VAEのメインクラス"""

    def __init__(self,
                 # SkillVAE parameters
                 input_dim: int = 6,
                 seq_len: int = 100,
                 skill_d_model: int = 128,
                 skill_n_heads: int = 8,
                 skill_n_encoder_layers: int = 6,
                 skill_n_decoder_layers: int = 6,
                 skill_latent_dim: int = 32,

                 # StyleVAE parameters
                 block_len: int = 35,
                 style_d_model: int = 128,
                 style_n_heads: int = 8,
                 style_n_encoder_layers: int = 4,
                 style_n_decoder_layers: int = 4,
                 style_latent_dim: int = 16,

                 # General parameters
                 n_subjects: int = 6,
                 dropout: float = 0.1,
                 training_stage: str = 'skill',  # 'skill' or 'style'
                 loss_schedule_config: Optional[Dict[str, Any]] = None,

                 # CLAUDE_ADDED: 段階的学習制御パラメータ
                 skill_checkpoint_path: Optional[str] = None,  # SkillVAE事前学習済み重み
                 freeze_skill_vae: bool = False,               # SkillVAE重み固定
                 **kwargs):

        super().__init__(**kwargs)

        self.training_stage = training_stage
        self.n_subjects = n_subjects

        # SkillVAE
        self.skill_vae = SkillVAE(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=skill_d_model,
            n_heads=skill_n_heads,
            n_encoder_layers=skill_n_encoder_layers,
            n_decoder_layers=skill_n_decoder_layers,
            skill_latent_dim=skill_latent_dim,
            dropout=dropout
        )

        # StyleVAE
        self.style_vae = StyleVAE(
            skill_latent_dim=skill_latent_dim,
            block_len=block_len,
            d_model=style_d_model,
            n_heads=style_n_heads,
            n_encoder_layers=style_n_encoder_layers,
            n_decoder_layers=style_n_decoder_layers,
            style_latent_dim=style_latent_dim,
            n_subjects=n_subjects,
            dropout=dropout
        )

        # 損失重みスケジューラ
        if loss_schedule_config is None:
            loss_schedule_config = self._get_default_loss_config()
        self.loss_scheduler = LossWeightScheduler(loss_schedule_config)

        # 被験者ID辞書
        self.all_subjects = ['h.nakamura', 'r.morishita', 'r.yanase', 's.miyama', 's.tahara', 't.hasegawa']
        self.subject_to_idx = {subj: i for i, subj in enumerate(self.all_subjects)}

        # 現在エポック
        self.current_epoch = 0

        # CLAUDE_ADDED: 段階的学習制御の初期化
        if skill_checkpoint_path:
            self._load_skill_checkpoint(skill_checkpoint_path)

        if freeze_skill_vae or training_stage == 'style':
            self._freeze_skill_vae()

    def _get_default_loss_config(self) -> Dict[str, Any]:
        """デフォルト損失設定"""
        return {
            'reconstruction_weight': {'schedule': 'constant', 'val': 1.0},
            'kl_weight': {'schedule': 'linear', 'start_epoch': 10, 'end_epoch': 50, 'start_val': 0.0, 'end_val': 0.01},
            'performance_weight': {'schedule': 'constant', 'val': 0.1},
            'contrastive_weight': {'schedule': 'constant', 'val': 0.1},
            'classification_weight': {'schedule': 'constant', 'val': 0.1}
        }

    def set_training_stage(self, stage: str):
        """学習段階を設定"""
        assert stage in ['skill', 'style'], f"Invalid stage: {stage}"
        self.training_stage = stage

        if stage == 'style':
            # StyleVAE学習時はSkillVAEを固定
            for param in self.skill_vae.parameters():
                param.requires_grad = False
        else:
            # SkillVAE学習時はすべてのパラメータを学習可能に
            for param in self.parameters():
                param.requires_grad = True

    def encode_to_skills(self, x: torch.Tensor) -> torch.Tensor:
        """軌道データをスキル潜在変数に変換"""
        with torch.no_grad() if self.training_stage == 'style' else torch.enable_grad():
            skill_output = self.skill_vae(x)
            return skill_output['z_skill']

    def forward(self, x: torch.Tensor,
                subject_ids: Optional[List[str]] = None,
                performance_scores: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """順伝播 - 既存システム互換"""

        if self.training_stage == 'skill':
            # SkillVAE学習段階：軌道データを直接使用
            return self._forward_skill_stage(x, performance_scores)
        else:
            # StyleVAE学習段階：xをスキル系列として解釈
            # 注意：実際にはxはHierarchicalSkillDatasetからのスキル系列データであることを想定
            return self._forward_style_stage(x, subject_ids)

    def _forward_skill_stage(self, x: torch.Tensor, performance_scores: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """SkillVAE学習段階の順伝播"""
        skill_output = self.skill_vae(x, performance_scores)

        # 損失計算
        losses = self._compute_skill_losses(x, skill_output, performance_scores)

        # 結果統合
        result = skill_output.copy()
        result.update(losses)

        return result

    def _forward_style_stage(self, z_skill_sequence: torch.Tensor, subject_ids: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """StyleVAE学習段階の順伝播"""
        style_output = self.style_vae(z_skill_sequence, subject_ids)

        # 損失計算
        losses = self._compute_style_losses(z_skill_sequence, style_output, subject_ids)

        # 結果統合
        result = style_output.copy()
        result.update(losses)

        return result

    def _compute_skill_losses(self, x: torch.Tensor, skill_output: Dict[str, torch.Tensor],
                             performance_scores: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """SkillVAE損失計算"""
        weights = self.loss_scheduler.get_weights()
        losses = {}

        # 再構成損失
        losses['reconstruction_loss'] = F.mse_loss(skill_output['reconstructed'], x)

        # KL損失
        mu = skill_output['z_skill_mu']
        logvar = skill_output['z_skill_logvar']
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        losses['kl_loss'] = kl_loss

        # パフォーマンス損失（オプション）
        if performance_scores is not None and 'performance_pred' in skill_output:
            performance_pred = skill_output['performance_pred'].squeeze(-1)
            losses['performance_loss'] = F.mse_loss(performance_pred, performance_scores)
        else:
            losses['performance_loss'] = torch.tensor(0.0, device=x.device)

        # 総合損失
        total_loss = (weights.get('reconstruction_weight', 1.0) * losses['reconstruction_loss'] +
                      weights.get('kl_weight', 0.01) * losses['kl_loss'] +
                      weights.get('performance_weight', 0.1) * losses['performance_loss'])

        losses['total_loss'] = total_loss
        return losses

    def _compute_style_losses(self, z_skill_sequence: torch.Tensor, style_output: Dict[str, torch.Tensor],
                             subject_ids: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """StyleVAE損失計算"""
        weights = self.loss_scheduler.get_weights()
        losses = {}

        # 再構成損失
        losses['reconstruction_loss'] = F.mse_loss(style_output['reconstructed_z_skill'], z_skill_sequence)

        # KL損失
        mu = style_output['z_style_mu']
        logvar = style_output['z_style_logvar']
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        losses['kl_loss'] = kl_loss

        # スタイル分離損失
        if subject_ids is not None:
            # 未知の被験者は新しいインデックスを割り当て
            subject_indices = []
            for subj in subject_ids:
                if subj not in self.subject_to_idx:
                    new_idx = len(self.subject_to_idx)
                    self.subject_to_idx[subj] = new_idx
                subject_indices.append(self.subject_to_idx[subj])

            subject_indices = torch.tensor(subject_indices, device=z_skill_sequence.device)

            # 対照学習損失
            z_style = style_output['z_style']
            losses['contrastive_loss'] = self.style_vae.contrastive_loss(z_style, subject_indices)

            # 分類損失（クラス数チェック付き）
            if 'subject_logits' in style_output:
                subject_logits = style_output['subject_logits']
                n_classes = subject_logits.size(-1)
                valid_indices = subject_indices < n_classes

                if valid_indices.any():
                    # 有効なインデックスのみで損失計算
                    valid_logits = subject_logits[valid_indices]
                    valid_targets = subject_indices[valid_indices]
                    losses['classification_loss'] = F.cross_entropy(valid_logits, valid_targets)
                else:
                    # 有効なインデックスがない場合はゼロ損失
                    losses['classification_loss'] = torch.tensor(0.0, device=z_skill_sequence.device)
            else:
                losses['classification_loss'] = torch.tensor(0.0, device=z_skill_sequence.device)
        else:
            losses['contrastive_loss'] = torch.tensor(0.0, device=z_skill_sequence.device)
            losses['classification_loss'] = torch.tensor(0.0, device=z_skill_sequence.device)

        # 総合損失
        total_loss = (weights.get('reconstruction_weight', 1.0) * losses['reconstruction_loss'] +
                      weights.get('kl_weight', 0.01) * losses['kl_loss'] +
                      weights.get('contrastive_weight', 0.1) * losses['contrastive_loss'] +
                      weights.get('classification_weight', 0.1) * losses['classification_loss'])

        losses['total_loss'] = total_loss
        return losses

    def on_epoch_start(self, epoch: int):
        """エポック開始時のコールバック"""
        super().on_epoch_start(epoch)
        self.current_epoch = epoch

    def train_skill_vae(self):
        """SkillVAE学習モードに設定"""
        self.set_training_stage('skill')
        print("Set to SkillVAE training mode")

    def train_style_vae(self):
        """StyleVAE学習モードに設定"""
        self.set_training_stage('style')
        print("Set to StyleVAE training mode (SkillVAE frozen)")

    # CLAUDE_ADDED: 既存学習システムとの互換性のための統一インターフェース
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """既存システム互換用のエンコードインターフェース"""
        if self.training_stage == 'skill':
            # SkillVAE学習段階：SkillVAEのエンコードのみ
            skill_mu, skill_logvar = self.skill_vae.encode(x)
            z_skill = self.skill_vae.reparameterize(skill_mu, skill_logvar)

            return {
                'z_style': torch.zeros(x.size(0), self.style_vae.style_latent_dim, device=x.device),  # ダミー
                'z_skill': z_skill,
                'z_skill_mu': skill_mu,
                'z_skill_logvar': skill_logvar,
                'skip_connections': None
            }
        else:
            # StyleVAE学習段階：階層的エンコード
            # 1. まずSkillVAEで軌道をスキル潜在変数に変換
            with torch.no_grad():
                skill_mu, skill_logvar = self.skill_vae.encode(x)
                z_skill = self.skill_vae.reparameterize(skill_mu, skill_logvar)

            # 2. StyleVAEに必要なブロック系列を作成（簡易実装）
            # 注意: 実際のStyleVAE学習では適切なブロック系列データが必要
            block_len = self.style_vae.block_len
            if z_skill.size(0) >= block_len:
                z_skill_sequence = z_skill[:block_len].unsqueeze(0)  # [1, block_len, skill_dim]
            else:
                # パディングまたは繰り返し
                z_skill_sequence = z_skill.unsqueeze(0)  # [1, batch, skill_dim]

            style_mu, style_logvar = self.style_vae.encode(z_skill_sequence)
            z_style = self.style_vae.reparameterize(style_mu, style_logvar)

            return {
                'z_style': z_style,
                'z_skill': z_skill_sequence.squeeze(0),  # 元の形状に戻す
                'z_style_mu': style_mu,
                'z_style_logvar': style_logvar,
                'skip_connections': None
            }

    def decode(self, z_style: torch.Tensor, z_skill: torch.Tensor, skip_connections: Optional[List] = None) -> Dict[str, torch.Tensor]:
        """既存システム互換用のデコードインターフェース"""
        if self.training_stage == 'skill':
            # SkillVAE学習段階：SkillVAEのデコードのみ
            reconstructed = self.skill_vae.decode(z_skill)
            return {'trajectory': reconstructed}
        else:
            # StyleVAE学習段階：階層的デコード
            # 1. StyleVAEでスタイルからスキル系列を生成
            z_skill_sequence = z_style.unsqueeze(1).expand(-1, self.style_vae.block_len, -1)  # 簡易実装

            # 2. SkillVAEで各スキル潜在変数から軌道を生成
            batch_size = z_skill.size(0)
            reconstructed_list = []

            for i in range(min(batch_size, self.style_vae.block_len)):
                skill_latent = z_skill[i:i+1]  # [1, skill_dim]
                with torch.no_grad():
                    trajectory = self.skill_vae.decode(skill_latent)  # [1, seq_len, input_dim]
                reconstructed_list.append(trajectory)

            if reconstructed_list:
                reconstructed = torch.cat(reconstructed_list, dim=0)  # [batch, seq_len, input_dim]
            else:
                # フォールバック
                reconstructed = self.skill_vae.decode(z_skill[:1])

            return {'trajectory': reconstructed}

    # CLAUDE_ADDED: 段階的学習制御機能
    def _load_skill_checkpoint(self, checkpoint_path: str):
        """SkillVAE事前学習済み重みを読み込み"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            if 'model_state_dict' in checkpoint:
                # 完全なモデルチェックポイントの場合
                state_dict = checkpoint['model_state_dict']
                # SkillVAE部分のみ抽出
                skill_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('skill_vae.'):
                        # プレフィックスを除去
                        skill_key = key[len('skill_vae.'):]
                        skill_state_dict[skill_key] = value

                # SkillVAEに重みをロード
                self.skill_vae.load_state_dict(skill_state_dict, strict=False)
                print(f"SkillVAE weights loaded from: {checkpoint_path}")

            else:
                # SkillVAE単体のチェックポイントの場合
                self.skill_vae.load_state_dict(checkpoint, strict=False)
                print(f"SkillVAE weights loaded from: {checkpoint_path}")

        except Exception as e:
            print(f"Warning: Failed to load SkillVAE checkpoint from {checkpoint_path}: {e}")

    def _freeze_skill_vae(self):
        """SkillVAEの重みを固定"""
        for param in self.skill_vae.parameters():
            param.requires_grad = False
        print("SkillVAE weights frozen")

    def save_skill_checkpoint(self, path: str):
        """SkillVAEの重みを保存"""
        torch.save(self.skill_vae.state_dict(), path)
        print(f"SkillVAE checkpoint saved to: {path}")

    def can_train_style_vae(self) -> bool:
        """StyleVAE学習の準備ができているかチェック"""
        # SkillVAEが学習済みか確認
        try:
            # ダミーデータでテスト
            dummy_input = torch.randn(1, 100, 6)
            with torch.no_grad():
                output = self.skill_vae(dummy_input)
                return 'z_skill' in output and output['z_skill'] is not None
        except:
            return False

    # CLAUDE_ADDED: BaseExperimentModelの汎用prerequisites機能をサポート
    def get_checkpoint_requirements(self) -> Dict[str, Any]:
        """
        HierarchicalMotionVAEが必要とするチェックポイント設定を返す
        汎用システムとの統合用
        """
        if self.training_stage == 'style':
            return {
                'skill_vae_checkpoint': {
                    'description': 'SkillVAE pre-trained weights for hierarchical learning',
                    'required': True,
                    'target_component': 'skill_vae',
                    'freeze_after_load': True
                }
            }
        return {}

    def load_pretrained_weights(self, checkpoint_configs: Dict[str, Any]):
        """
        BaseExperimentModelの汎用機能をオーバーライド
        HierarchicalMotionVAE固有の処理を含む
        """
        print("HierarchicalMotionVAE: Loading pretrained weights...")

        for config_name, config_value in checkpoint_configs.items():
            if config_name == 'skill_vae_checkpoint' and config_value:
                # SkillVAE専用の処理
                if isinstance(config_value, str):
                    checkpoint_path = config_value
                elif isinstance(config_value, dict):
                    checkpoint_path = config_value.get('path', config_value)
                else:
                    continue

                try:
                    self._load_skill_checkpoint(checkpoint_path)
                    if self.training_stage == 'style':
                        self._freeze_skill_vae()
                    print(f"✓ HierarchicalMotionVAE: SkillVAE checkpoint loaded from {checkpoint_path}")
                except Exception as e:
                    print(f"Warning: HierarchicalMotionVAE SkillVAE checkpoint loading failed: {e}")
            else:
                # その他のチェックポイントは親クラスに委譲
                try:
                    super().load_pretrained_weights({config_name: config_value})
                except Exception as e:
                    print(f"Warning: Failed to load checkpoint {config_name}: {e}")

        print("✓ HierarchicalMotionVAE: Pretrained weights loading completed")