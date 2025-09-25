from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.base_model import BaseExperimentModel
from models.components.loss_weight_scheduler import LossWeightScheduler


class ContrastiveLoss(nn.Module):
    """対比学習損失"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)

        # CLAUDE_ADDED: バッチサイズが小さい場合の対処
        if batch_size <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # 正例・負例マスク
        mask = torch.eq(labels, labels.T).float()

        # 特徴量正規化 - CLAUDE_ADDED: 数値安定性向上
        features = F.normalize(features, p=2, dim=1, eps=1e-8)

        # 類似度計算 - CLAUDE_ADDED: 温度パラメータをクリップ
        temperature_clamped = torch.clamp(torch.tensor(self.temperature), min=0.01)
        similarity_matrix = torch.matmul(features, features.T) / temperature_clamped

        # CLAUDE_ADDED: 類似度をクリップして数値安定性を向上
        similarity_matrix = torch.clamp(similarity_matrix, min=-10, max=10)

        # 対角成分除去
        mask = mask - torch.eye(batch_size, device=mask.device)

        # 損失計算 - CLAUDE_ADDED: 数値安定性の改善
        exp_sim = torch.exp(similarity_matrix)
        pos_sum = torch.sum(exp_sim * mask, dim=1)
        neg_sum = torch.sum(exp_sim * (1 - torch.eye(batch_size, device=mask.device)), dim=1)

        # CLAUDE_ADDED: ゼロ除算とlog(0)を防ぐ
        pos_sum = torch.clamp(pos_sum, min=1e-8)
        neg_sum = torch.clamp(neg_sum, min=1e-8)

        loss = -torch.log(pos_sum / neg_sum)

        # CLAUDE_ADDED: NaNチェック
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


class ResidualConnection(nn.Module):
    """残差接続付きの線形層"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 2)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return residual + x


class ImprovedStyleSkillSeparationNetEncoder(nn.Module):
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

        # 各段階での正規化
        self.layer_norm_after_proj = nn.LayerNorm(d_model)
        self.layer_norm_after_transformer = nn.LayerNorm(d_model)
        self.layer_norm_before_heads = nn.LayerNorm(d_model)

        # 入力射影
        self.input_proj = nn.Linear(input_dim, d_model)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        # 残差接続
        self.self_residual_proj = ResidualConnection(d_model, dropout)

        # 特徴量抽出Transformer層
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)

        # プーリング
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads = n_heads,
            batch_first=True,
            dropout=dropout
        )
        self.pooling_query = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # 改善されたヘッド
        self.improved_style_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model//2, style_latent_dim * 2)
        )

        self.improved_skill_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
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
        batch_size, seq_len, input_dim = x.shape

        # 1. 入力射影 + 正規化
        encoded = self.input_proj(x)
        encoded = self.layer_norm_after_proj(encoded)

        # 2. 位置エンコーディング
        pos_encoded = self.pos_encoding(encoded)

        # 3 . Transformer処理
        attended = self.transformer_encoder(pos_encoded)
        attended = self.layer_norm_after_transformer(attended)

        # 4 . 残差接続
        attended = self.self_residual_proj(attended)

        # 5. アテンション付きプーリング層
        query = self.pooling_query.unsqueeze(0).expand(batch_size, -1, -1)
        pooled_features, _ = self.attention_pooling(query, attended, attended)
        pooled_features = pooled_features.squeeze(1) # [batch_size, d_model]

        # 6. 最終正規化
        pooled_features = self.layer_norm_before_heads(pooled_features)

        # 7. 改善されたヘッド
        style_params = self.improved_style_head(pooled_features)
        skill_params = self.improved_skill_head(pooled_features)

        return {
            'style_mu': style_params[:, :style_params.size(1) // 2],
            'style_logvar': style_params[:, style_params.size(1) // 2:],
            'skill_mu': skill_params[:, :skill_params.size(1) // 2],
            'skill_logvar': skill_params[:, skill_params.size(1) // 2:],
        }


class TransformerAutoRegressiveDecoder(nn.Module):
    """CLAUDE_ADDED: TransformerDecoderを使用した自己回帰型デコーダ"""

    def __init__(self,
                 output_dim,
                 seq_len,
                 d_model,
                 n_heads,
                 n_layers,
                 dropout,
                 style_latent_dim,
                 skill_latent_dim
                 ):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.output_dim = output_dim

        # 潜在変数からメモリ（キー・バリュー）の生成
        self.memory_encoder = nn.Sequential(
            nn.Linear(style_latent_dim + skill_latent_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # 入力埋め込み層（出力次元からd_modelへ）
        self.input_embedding = nn.Linear(output_dim, d_model)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        # TransformerDecoder層
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_layers
        )

        # 出力射影層
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        # CLAUDE_FIXED: 学習可能な開始トークン（より小さな初期化）
        self.start_token = nn.Parameter(torch.zeros(1, 1, output_dim))  # ゼロ初期化で安定化

        # CLAUDE_ADDED: 重み初期化
        self._initialize_weights()

    def _initialize_weights(self):
        """CLAUDE_ADDED: より控えめな重み初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初期化でgainを小さく
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def _generate_causal_mask(self, seq_len, device):
        """CLAUDE_ADDED: 因果マスク生成（未来の情報を見えなくする）"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, z_style, z_skill, ground_truth=None, teacher_forcing_ratio=0.5):
        """CLAUDE_ADDED: TransformerDecoderによる自己回帰生成"""
        batch_size = z_style.size(0)
        device = z_style.device

        # 潜在変数をメモリに変換（クロスアテンションで使用）
        condition = torch.cat([z_style, z_skill], dim=-1)  # [batch, style_dim + skill_dim]
        memory = self.memory_encoder(condition).unsqueeze(1)  # [batch, 1, d_model]
        memory = memory.expand(-1, self.seq_len, -1)  # [batch, seq_len, d_model]

        # 学習時（Teacher Forcing使用） - CLAUDE_FIXED: teacher_forcing_ratioを正しく実装
        if self.training and ground_truth is not None:
            # 時刻毎にTeacher Forcingを適用
            outputs = []
            current_input = self.start_token.expand(batch_size, -1, -1)  # [batch, 1, output_dim]

            for t in range(self.seq_len):
                # 現在までの入力を埋め込み
                embedded_input = self.input_embedding(current_input)  # [batch, t+1, d_model]
                positioned_input = self.pos_encoding(embedded_input)

                # 因果マスク生成（現在の長さに対応）
                current_len = positioned_input.size(1)
                causal_mask = self._generate_causal_mask(current_len, device)

                # CLAUDE_FIXED: メモリは全時刻分を保持（TransformerDecoderの正しい使い方）
                current_memory = memory  # [batch, seq_len, d_model] 全時刻分を使用

                # TransformerDecoder適用
                decoded = self.transformer_decoder(
                    tgt=positioned_input,
                    memory=current_memory,
                    tgt_mask=causal_mask
                )  # [batch, t+1, d_model]

                # 最後のトークンのみ射影
                next_output = self.output_projection(decoded[:, -1:, :])  # [batch, 1, output_dim]
                outputs.append(next_output)

                # CLAUDE_FIXED: Teacher Forcing確率に基づいて次の入力を決定
                if t < self.seq_len - 1:  # 最後のステップでは不要
                    if torch.rand(1).item() < teacher_forcing_ratio:
                        # Teacher Forcing: 正解データを使用
                        next_input = ground_truth[:, t:t+1, :]
                    else:
                        # 自己回帰: 自分の予測を使用
                        next_input = next_output

                    current_input = torch.cat([current_input, next_input], dim=1)

            output = torch.cat(outputs, dim=1)  # [batch, seq_len, output_dim]

        else:
            # 推論時（自己回帰生成）
            outputs = []
            current_input = self.start_token.expand(batch_size, -1, -1)  # [batch, 1, output_dim]

            for t in range(self.seq_len):
                # 現在までの入力を埋め込み
                embedded_input = self.input_embedding(current_input)  # [batch, t+1, d_model]
                positioned_input = self.pos_encoding(embedded_input)

                # 因果マスク生成（現在の長さに対応）
                current_len = positioned_input.size(1)
                causal_mask = self._generate_causal_mask(current_len, device)

                # CLAUDE_FIXED: メモリは全時刻分を保持（TransformerDecoderの正しい使い方）
                current_memory = memory  # [batch, seq_len, d_model] 全時刻分を使用

                # TransformerDecoder適用
                decoded = self.transformer_decoder(
                    tgt=positioned_input,
                    memory=current_memory,
                    tgt_mask=causal_mask
                )  # [batch, t+1, d_model]

                # 最後のトークンのみ射影
                next_output = self.output_projection(decoded[:, -1:, :])  # [batch, 1, output_dim]
                outputs.append(next_output)

                # 次の入力を準備
                if t < self.seq_len - 1:
                    current_input = torch.cat([current_input, next_output], dim=1)

            output = torch.cat(outputs, dim=1)  # [batch, seq_len, output_dim]

        return output


class SelfRegressiveStyleSkillSeparationNet(BaseExperimentModel):
    """CLAUDE_ADDED: AutoRegressiveDecoderを使用するスタイル・スキル分離モデル"""

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
                 teacher_forcing_config: Dict[str, float] = None,
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
                         **kwargs
                         )

        self.seq_len = seq_len
        self.d_model = d_model
        self.style_latent_dim = style_latent_dim
        self.skill_latent_dim = skill_latent_dim

        # CLAUDE_ADDED: モデルインスタンス化時のパラメータをログ出力
        print(f"SelfRegressiveStyleSkillSeparationNet instantiated with:")
        print(f"  n_decoder_layers: {n_decoder_layers}")
        print(f"  d_model: {d_model}")
        print(f"  n_heads: {n_heads}")
        print(f"  n_encoder_layers: {n_encoder_layers}")
        print(f"  style_latent_dim: {style_latent_dim}")
        print(f"  skill_latent_dim: {skill_latent_dim}")
        print(f"  n_subjects: {n_subjects}")

        # 損失関数の重みスケジューラの初期化
        if loss_schedule_config is None:
            loss_schedule_config = {
                # CLAUDE_ADDED: 分離されたKL損失スケジュール
                'beta_style': {'schedule': 'linear', 'start_epoch': 40, 'end_epoch': 70, 'start_val': 0.0, 'end_val': 0.001},
                'beta_skill': {'schedule': 'linear', 'start_epoch': 40, 'end_epoch': 70, 'start_val': 0.0, 'end_val': 0.0001},
                # 後方互換性のため旧betaも保持
                'beta': {'schedule': 'linear', 'start_epoch': 40, 'end_epoch': 70, 'start_val': 0.0, 'end_val': 0.001},
                'orthogonal_loss': {'schedule': 'constant', 'val': 0.1},
                'contrastive_loss': {'schedule': 'constant', 'val': 0.1},
                'manifold_loss': {'schedule': 'constant', 'val': 0.1},
                'style_classification_loss': {'schedule': 'constant', 'val': 0.1},
                'skill_regression_loss': {'schedule': 'constant', 'val': 0.1},
            }
        self.loss_scheduler = LossWeightScheduler(loss_schedule_config)

        # CLAUDE_ADDED: Teacher Forcing設定の初期化
        if teacher_forcing_config is None:
            teacher_forcing_config = {
                'training': 0.5,    # 学習時のTeacher Forcing確率
                'inference': 0.0    # 推論時のTeacher Forcing確率（通常は0.0）
            }
        self.teacher_forcing_config = teacher_forcing_config

        # スケジューラ設定からlossの計算フラグを受け取り
        self.calc_orthogonal_loss = 'orthogonal_loss' in loss_schedule_config
        self.calc_contrastive_loss = 'contrastive_loss' in loss_schedule_config
        self.calc_manifold_loss = 'manifold_loss' in loss_schedule_config
        self.calc_style_subtask = 'style_classification_loss' in loss_schedule_config
        self.calc_skill_subtask = 'skill_regression_loss' in loss_schedule_config

        # エンコーダ定義（既存の改善されたエンコーダを使用）
        self.encoder = ImprovedStyleSkillSeparationNetEncoder(input_dim,
                                                              seq_len,
                                                              d_model,
                                                              n_heads,
                                                              n_encoder_layers,
                                                              dropout,
                                                              style_latent_dim,
                                                              skill_latent_dim
                                                              )

        # CLAUDE_ADDED: TransformerAutoRegressiveDecoderを使用
        self.decoder = TransformerAutoRegressiveDecoder(input_dim,
                                                        seq_len,
                                                        d_model,
                                                        n_heads,
                                                        n_decoder_layers,
                                                        dropout,
                                                        style_latent_dim,
                                                        skill_latent_dim
                                                        )

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
            self.skill_regression_network = nn.Sequential(
                nn.Linear(skill_latent_dim, skill_latent_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(skill_latent_dim, skill_latent_dim // 2),
                nn.ReLU(),
                nn.Linear(skill_latent_dim // 2, 1)  # スキルスコア回帰
            )

        # 損失関数初期化
        if self.calc_contrastive_loss:
            self.contrastive_loss = ContrastiveLoss(temperature=0.07)

        # 初期化
        self._initialize_weights()

    def on_epoch_start(self, epoch: int):
        """CLAUDE_ADDED: 学習ループからエポックの開始時呼び出されるメソッド"""
        self.loss_scheduler.step(epoch)

    def _initialize_weights(self):
        """CLAUDE_ADDED: Xavier初期化で重みを安定化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x, subject_indices=None, skill_scores=None, teacher_forcing_ratio=None):
        """CLAUDE_ADDED: AutoRegressiveDecoderを使用するforward処理"""
        batch_size, seq_len, input_dim = x.shape

        # CLAUDE_ADDED: teacher_forcing_ratioの自動設定（既存システムとの互換性保持）
        if teacher_forcing_ratio is None:
            if hasattr(self, 'teacher_forcing_config'):
                # 設定から読み取り
                if self.training:
                    teacher_forcing_ratio = self.teacher_forcing_config.get('training', 0.5)
                else:
                    teacher_forcing_ratio = self.teacher_forcing_config.get('inference', 0.0)
            else:
                # デフォルト値（学習時0.5、推論時0.0）
                teacher_forcing_ratio = 0.5 if self.training else 0.0

        # エンコーダで潜在変数を計算
        encoded = self.encoder(x)

        # スタイルとスキルの潜在変数をサンプリング
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])

        # CLAUDE_ADDED: TransformerAutoRegressiveDecoderで再構築（Teacher Forcing対応）
        # 訓練時は元データをground_truthとして渡す
        ground_truth = x if self.training else None
        reconstructed = self.decoder(z_style, z_skill, ground_truth, teacher_forcing_ratio)

        # 補助タスクの予測
        subject_pred = None
        skill_score_pred = None

        if self.calc_style_subtask and subject_indices is not None:
            # プロトタイプベースのスタイル分類
            style_features = self.style_prototype_network(z_style)
            # プロトタイプとの距離を計算
            distances = torch.cdist(style_features, self.style_prototypes)
            subject_pred = -distances  # 距離の負値をスコアとする

        if self.calc_skill_subtask and skill_scores is not None:
            skill_score_pred = self.skill_regression_network(z_skill).squeeze(-1)

        # CLAUDE_ADDED: 既存システムに合わせて損失計算を含める
        losses = self.compute_losses(
            x, reconstructed, encoded, z_style, z_skill,
            subject_pred, skill_score_pred,
            subject_indices, skill_scores
        )

        result = {
            'reconstructed': reconstructed,
            'encoded': encoded,
            'z_style': z_style,
            'z_skill': z_skill,
            'subject_pred': subject_pred,
            'skill_score_pred': skill_score_pred
        }

        return result | losses

    def reparameterize(self, mu, logvar):
        """CLAUDE_ADDED: VAEの再パラメータ化トリック"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_losses(self, x, reconstructed, encoded, z_style, z_skill, subject_pred, skill_score_pred, subject_indices, skill_scores):
        """CLAUDE_ADDED: AutoRegressiveDecoder対応の損失計算"""
        # スケジューラから現在の重みを取得
        weights = self.loss_scheduler.get_weights()

        # CLAUDE_ADDED: NaN/inf検出とクリッピング
        def safe_tensor(tensor, name="unknown"):
            if self.training and hasattr(self, '_debug_nan_check') and self._debug_nan_check:
                if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
                    print(f"Warning: {name} contains NaN or inf, replacing with zeros")
                    return torch.zeros_like(tensor)
            return tensor

        # 再構築損失 - 安全な計算
        reconstructed = safe_tensor(reconstructed, "reconstructed")
        x = safe_tensor(x, "input_x")
        losses = {'reconstruction_loss': F.mse_loss(reconstructed, x)}

        # KLダイバージェンス - より厳格なクリッピング
        style_mu_clamped = torch.clamp(encoded['style_mu'], min=-5, max=5)
        skill_mu_clamped = torch.clamp(encoded['skill_mu'], min=-5, max=5)
        style_logvar_clamped = torch.clamp(encoded['style_logvar'], min=-8, max=8)
        skill_logvar_clamped = torch.clamp(encoded['skill_logvar'], min=-8, max=8)

        # KL損失の安全な計算
        style_kl_terms = 1 + style_logvar_clamped - style_mu_clamped.pow(2) - style_logvar_clamped.exp()
        skill_kl_terms = 1 + skill_logvar_clamped - skill_mu_clamped.pow(2) - skill_logvar_clamped.exp()

        style_kl_terms = safe_tensor(style_kl_terms, "style_kl_terms")
        skill_kl_terms = safe_tensor(skill_kl_terms, "skill_kl_terms")

        losses['kl_style_loss'] = -0.5 * torch.mean(torch.sum(style_kl_terms, dim=1))
        losses['kl_skill_loss'] = -0.5 * torch.mean(torch.sum(skill_kl_terms, dim=1))

        # CLAUDE_ADDED: スタイル・スキル直交性損失
        if self.calc_orthogonal_loss:
            # 各サンプルのスタイルとスキル表現の内積を計算
            style_skill_products = torch.sum(z_style * z_skill, dim=1)
            losses['orthogonal_loss'] = torch.mean(style_skill_products.pow(2))

        # スタイル空間関連の損失
        if subject_indices is not None:
            # CLAUDE_ADDED: 全被験者を固定的なインデックスにマッピング（一貫性を保つ）
            all_subjects = ['h.nakamura', 'r.morishita', 'r.yanase', 's.miyama', 's.tahara', 't.hasegawa']
            subject_to_idx = {subj: i for i, subj in enumerate(all_subjects)}
            subject_indices_mapped = torch.tensor([subject_to_idx[subj] for subj in subject_indices],
                                                device=z_style.device)

            # 対比学習
            if self.calc_contrastive_loss:
                losses['contrastive_loss'] = self.contrastive_loss(z_style, subject_indices_mapped)

            # CLAUDE_ADDED: スタイル分類サブタスクの安全な実行
            if self.calc_style_subtask and subject_pred is not None:
                losses['style_classification_loss'] = F.cross_entropy(subject_pred, subject_indices_mapped)

        # スキル空間関連の損失
        if skill_scores is not None:
            # CLAUDE_ADDED: スキルスコア回帰損失の安全な実行
            if self.calc_skill_subtask and skill_score_pred is not None:
                # skill_score_pred: [batch, 1] -> [batch] に変換
                skill_score_pred_flat = skill_score_pred.squeeze(-1)
                losses['skill_regression_loss'] = F.mse_loss(skill_score_pred_flat, skill_scores)

        # CLAUDE_ADDED: Manifold損失（スタイル・スキル空間の分離促進）
        if self.calc_manifold_loss:
            # スタイル空間内の類似性と多様性のバランス
            style_pairwise_dist = torch.pdist(z_style)
            skill_pairwise_dist = torch.pdist(z_skill)

            # 適度な分散を促すための損失
            style_variance_loss = torch.mean((style_pairwise_dist - 1.0).pow(2))
            skill_variance_loss = torch.mean((skill_pairwise_dist - 1.0).pow(2))

            losses['manifold_loss'] = style_variance_loss + skill_variance_loss

        # 重み付き総損失の計算
        total_loss = losses['reconstruction_loss']

        # KL損失の追加（分離されたスケジューリング）
        if 'beta_style' in weights:
            total_loss += weights['beta_style'] * losses['kl_style_loss']
        if 'beta_skill' in weights:
            total_loss += weights['beta_skill'] * losses['kl_skill_loss']
        # 後方互換性のため
        elif 'beta' in weights:
            total_loss += weights['beta'] * (losses['kl_style_loss'] + losses['kl_skill_loss'])

        # その他の損失の追加
        for loss_name in ['orthogonal_loss', 'style_classification_loss', 'skill_regression_loss',
                         'contrastive_loss', 'manifold_loss']:
            if loss_name in losses and loss_name in weights:
                total_loss += weights[loss_name] * losses[loss_name]

        losses['total_loss'] = total_loss
        return losses

    def encode(self, x):
        """CLAUDE_ADDED: エンコードのみ"""
        encoded = self.encoder(x)
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])
        return {'z_style': z_style, 'z_skill': z_skill}

    def decode(self, z_style, z_skill, ground_truth=None, teacher_forcing_ratio=0.0):
        """CLAUDE_ADDED: デコードのみ

        Args:
            z_style: スタイル潜在変数 [batch_size, style_latent_dim]
            z_skill: スキル潜在変数 [batch_size, skill_latent_dim]
            ground_truth: Teacher Forcing用の正解データ（オプション）
            teacher_forcing_ratio: Teacher Forcingの確率（デフォルト: 0.0）
        """
        trajectory = self.decoder(z_style, z_skill, ground_truth, teacher_forcing_ratio)
        return {'trajectory': trajectory}