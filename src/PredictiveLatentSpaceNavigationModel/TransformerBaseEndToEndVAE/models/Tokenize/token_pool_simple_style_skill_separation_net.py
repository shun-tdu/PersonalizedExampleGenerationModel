"""
- BERTライクに潜在変数の平均と分散を表すトークンを系列データの先頭に付加して学習するモデル
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.base_model import BaseExperimentModel
from models.components.loss_weight_scheduler import LossWeightScheduler


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


# CLAUDE_ADDED: Discriminatorを削除し、相関行列ベースの直交性損失を使用


class TokenPoolSeparationEncoder(nn.Module):
    """分布トークンで潜在空間を学習するエンコーダ"""

    def __init__(self,
                 input_dim,
                 seq_len,
                 d_model,
                 n_heads,
                 n_layers,
                 dropout,
                 style_latent_dim,
                 skill_latent_dim):
        super().__init__()

        assert d_model == style_latent_dim, "d_model and style_latent_dim must be the same for ACTOR-style encoder"
        assert d_model == skill_latent_dim, "d_model and skill_latent_dim must be the same for ACTOR-style encoder"

        self.d_model = d_model
        self.seq_len = seq_len

        # 入力射影
        self.input_proj = nn.Linear(input_dim, d_model)

        # 分布トークン
        self.style_mu_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.style_logvar_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.skill_mu_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.skill_logvar_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, seq_len + 4)

        # 特徴量抽出Transformer層
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)

    def forward(self, x):
        # CLAUDE_ADDED: 修正 - .sha -> .shape
        batch_size, _, _ = x.shape

        # 入力射影
        encoded = self.input_proj(x)

        # 分布トークンをバッチサイズ分に拡張してシーケンスに連結
        # CLAUDE_ADDED: 修正 - .expand()メソッドのみを使用（トークンはパラメータであり呼び出し可能ではない）
        style_mu_token = self.style_mu_token.expand(batch_size, -1, -1)
        style_logvar_token = self.style_logvar_token.expand(batch_size, -1, -1)
        skill_mu_token = self.skill_mu_token.expand(batch_size, -1, -1)
        skill_logvar_token = self.skill_logvar_token.expand(batch_size, -1, -1)

        full_sequence = torch.cat([
            style_mu_token,
            style_logvar_token,
            skill_mu_token,
            skill_logvar_token,
            encoded
        ], dim=1)  # [batch_size, seq_len+4, d_model]

        # 位置エンコーディング
        pos_encoded = self.pos_encoding(full_sequence)

        # Transformer処理
        attended = self.transformer_encoder(pos_encoded)

        return {
            'style_mu': attended[:, 0, :],
            'style_logvar': attended[:, 1, :],
            'skill_mu': attended[:, 2, :],
            'skill_logvar': attended[:, 3, :],
        }


class TokenPoolSeparationDecoder(nn.Module):
    """分布トークンで潜在空間を学習するデコーダ"""

    def __init__(self,
                 output_dim,
                 seq_len,
                 d_model,
                 n_heads,
                 n_layers,
                 dropout,
                 style_latent_dim,
                 skill_latent_dim,
                 ):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model

        # スタイル変数とスキル変数を統合して単一のコンテキストに変換する層
        self.latent_fusion = nn.Linear(style_latent_dim + skill_latent_dim, d_model)

        # 出力系列を生成するためのクエリ
        self.queries = nn.Parameter(torch.randn(1, seq_len, d_model))

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        # TransformerDecoderを定義
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=n_layers)

        # 出力層
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, z_style, z_skill, skip_connections):
        batch_size, _ = z_style.shape

        # スタイルとスキルを結合してコンテキストを生成
        z_combined = torch.cat([z_style, z_skill], dim=1)
        memory = self.latent_fusion(z_combined)
        memory = memory.unsqueeze(1)

        # クエリを準備
        queries = self.queries.expand(batch_size, -1, -1)
        pos_encoded_queries = self.pos_encoding(queries)

        transformed = self.transformer_decoder(tgt=pos_encoded_queries, memory=memory)

        # 共通の出力層
        reconstructed = self.output_proj(transformed)

        return reconstructed


class TokenPoolSeparationNet(BaseExperimentModel):
    """CLAUDE_ADDED: Simple FiLM Adaptive Gate Skip Connection Network"""

    def __init__(self,
                 input_dim=6,
                 seq_len=100,
                 d_model=256,
                 n_heads=4,
                 n_encoder_layers=4,
                 n_decoder_layers=4,
                 dropout=0.1,
                 style_latent_dim=256,
                 skill_latent_dim=256,
                 factor_num=3,
                 n_subjects=6,
                 loss_schedule_config: Dict[str, Any] = None,
                 **kwargs):

        super().__init__(input_dim=input_dim,
                         seq_len=seq_len,
                         d_model=d_model,
                         n_heads=n_heads,
                         n_encoder_layers=n_encoder_layers,
                         n_decoder_layers=n_decoder_layers,
                         dropout=dropout,
                         style_latent_dim=style_latent_dim,
                         skill_latent_dim=skill_latent_dim,
                         factor_num=factor_num,
                         n_subjects=n_subjects,
                         loss_schedule_config=loss_schedule_config,
                         **kwargs)

        self.seq_len = seq_len
        self.d_model = d_model
        self.style_latent_dim = style_latent_dim
        self.skill_latent_dim = skill_latent_dim

        print(f"SimpleFiLMAdaptiveGateNet instantiated:")
        print(f"  d_model: {d_model}, n_heads: {n_heads}")
        print(f"  encoder_layers: {n_encoder_layers}, decoder_layers: {n_decoder_layers}")
        print(f"  latent_dims: style={style_latent_dim}, skill={skill_latent_dim}")

        # CLAUDE_ADDED: 損失スケジューラ初期化 - 相関行列ベースの直交性損失を使用
        if loss_schedule_config is None:
            loss_schedule_config = {
                'beta_style': {'schedule': 'linear', 'start_epoch': 21, 'end_epoch': 50, 'start_val': 0.0,
                               'end_val': 0.0001},
                'beta_skill': {'schedule': 'linear', 'start_epoch': 21, 'end_epoch': 50, 'start_val': 0.0,
                               'end_val': 0.0001},
                'factor_regression_loss': {'schedule': 'linear', 'start_epoch': 20, 'end_epoch': 60, 'start_val': 0.0,
                                           'end_val': 0.2},
                'orthogonal_loss': {'schedule': 'constant', 'val': 0.1}  # CLAUDE_ADDED: adversarial -> orthogonal
            }
        self.loss_scheduler = LossWeightScheduler(loss_schedule_config)

        # スケジューラ設定からlossの計算フラグを受け取り
        self.calc_factor_subtask = 'factor_regression_loss' in loss_schedule_config
        self.calc_orthogonal_loss = 'orthogonal_loss' in loss_schedule_config  # CLAUDE_ADDED: orthogonal_loss

        # エンコーダ・デコーダ
        self.encoder = TokenPoolSeparationEncoder(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            dropout=dropout,
            style_latent_dim=style_latent_dim,
            skill_latent_dim=skill_latent_dim,
        )

        self.decoder = TokenPoolSeparationDecoder(
            output_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            dropout=dropout,
            style_latent_dim=style_latent_dim,
            skill_latent_dim=skill_latent_dim,
        )

        # CLAUDE_ADDED: Discriminatorは削除

        # サブタスクネットワーク
        if self.calc_factor_subtask:
            self.factor_regressor = nn.Sequential(
                nn.Linear(skill_latent_dim, skill_latent_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(skill_latent_dim // 2, factor_num)
            )

        # エポック追跡
        self.current_epoch = 0
        self.max_epochs = 200

    def on_epoch_start(self, epoch: int):
        """学習ループからエポックの開始時呼び出されるメソッド"""
        self.loss_scheduler.step(epoch)
        self.current_epoch = epoch

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor = None, skill_factor: torch.Tensor = None) -> Dict[
        str, torch.Tensor]:
        """CLAUDE_ADDED: Discriminatorを削除し、相関行列ベースの直交性損失を使用"""
        batch_size = x.shape[0]

        # エンコード
        encoded = self.encoder(x)

        # 潜在変数サンプリング
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])

        # デコード
        reconstructed = self.decoder(z_style, z_skill, None)

        # CLAUDE_ADDED: Discriminatorは削除

        # サブタスク: スキル空間から因子スコアを回帰
        skill_factor_pred = None
        if self.calc_factor_subtask:
            skill_factor_pred = self.factor_regressor(z_skill)

        # 損失計算
        losses = self.compute_losses(
            x,
            reconstructed,
            encoded,
            z_style,
            z_skill,
            skill_factor_pred,
            skill_factor
        )

        result = {
            'reconstructed': reconstructed,
            'z_style': z_style,
            'z_skill': z_skill,
        }

        return result | losses

    def compute_losses(self, x, reconstructed, encoded, z_style, z_skill,
                       skill_factor_pred, skill_factor):
        """CLAUDE_ADDED: 相関行列ベースの直交性損失を使用"""
        weights = self.loss_scheduler.get_weights()

        # 基本損失
        losses = {'reconstruction_loss': F.mse_loss(reconstructed, x)}

        # KL損失
        style_kl_terms = 1 + encoded['style_logvar'] - encoded['style_mu'].pow(2) - encoded['style_logvar'].exp()
        skill_kl_terms = 1 + encoded['skill_logvar'] - encoded['skill_mu'].pow(2) - encoded['skill_logvar'].exp()

        losses['kl_style_loss'] = -0.5 * torch.mean(torch.sum(style_kl_terms, dim=1))
        losses['kl_skill_loss'] = -0.5 * torch.mean(torch.sum(skill_kl_terms, dim=1))

        # CLAUDE_ADDED: 相関行列ベースの直交性損失 (z_styleとz_skillの独立性を促進)
        if self.calc_orthogonal_loss:
            # バッチ方向で標準化
            z_style_norm = (z_style - z_style.mean(dim=0)) / (z_style.std(dim=0) + 1e-8)
            z_skill_norm = (z_skill - z_skill.mean(dim=0)) / (z_skill.std(dim=0) + 1e-8)

            # 相関行列を計算
            cross_correlation_matrix = torch.matmul(z_style_norm.T, z_skill_norm) / z_style.shape[0]

            # 相関行列の要素を0に近づける
            losses['orthogonal_loss'] = torch.mean(cross_correlation_matrix ** 2)

        # サブタスク損失: スキル空間から因子スコアを回帰
        if skill_factor is not None:
            if self.calc_factor_subtask and skill_factor_pred is not None:
                losses['factor_regression_loss'] = F.mse_loss(skill_factor_pred, skill_factor)

        # 総合損失計算
        total_loss = (losses['reconstruction_loss']
                      + weights.get('beta_style', weights.get('beta', 0.0)) * losses['kl_style_loss']
                      + weights.get('beta_skill', weights.get('beta', 0.0)) * losses['kl_skill_loss']
                      + weights.get('orthogonal_loss', 0.0) * losses.get('orthogonal_loss',
                                                                         torch.tensor(0.0, device=x.device))
                      + weights.get('factor_regression_loss', 0.0) * losses.get('factor_regression_loss',
                                                                                torch.tensor(0.0, device=x.device))
                      )

        losses['total_loss'] = total_loss
        return losses

    def configure_optimizers(self, training_config: Dict[str, Any]):
        """単一optimizerとschedulerの設定"""
        super().configure_optimizers(training_config)

        optimizer = self._create_optimizer(self.parameters())
        scheduler = self._create_scheduler(optimizer)

        return optimizer, scheduler

    def training_step(self, batch, optimizers, device: torch.device, max_norm=None) -> Dict[str, torch.Tensor]:
        """1バッチ分の学習処理"""
        # バッチデータを展開
        trajectory, subject_ids, skill_factors = batch
        trajectory = trajectory.to(device)
        skill_factors = skill_factors.to(device)

        # 単一オプティマイザの場合
        if isinstance(optimizers, tuple):
            optimizer = optimizers[0]
        else:
            optimizer = optimizers

        # 順伝播
        optimizer.zero_grad()
        outputs = self.forward(trajectory, subject_ids, skill_factors)

        # 逆伝播
        loss = outputs['total_loss']
        loss.backward()

        # 勾配クリッピング
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm)

        optimizer.step()

        # CLAUDE_ADDED: 損失辞書を数値に変換して返す
        loss_dict = {}
        for key, value in outputs.items():
            if 'loss' in key.lower():
                if torch.is_tensor(value):
                    # スカラーテンソルの場合のみ.item()、それ以外は.mean().item()
                    loss_dict[key] = value.item() if value.numel() == 1 else value.mean().item()
                else:
                    loss_dict[key] = float(value)

        return loss_dict

    def validation_step(self, batch, device: torch.device) -> Dict[str, torch.Tensor]:
        """1バッチ分の検証処理"""
        # バッチデータを展開
        trajectory, subject_ids, skill_factors = batch
        trajectory = trajectory.to(device)
        skill_factors = skill_factors.to(device)

        # 順伝播のみ
        with torch.no_grad():
            outputs = self.forward(trajectory, subject_ids, skill_factors)

        # CLAUDE_ADDED: 損失辞書を数値に変換して返す
        loss_dict = {}
        for key, value in outputs.items():
            if 'loss' in key.lower():
                if torch.is_tensor(value):
                    # スカラーテンソルの場合のみ.item()、それ以外は.mean().item()
                    loss_dict[key] = value.item() if value.numel() == 1 else value.mean().item()
                else:
                    loss_dict[key] = float(value)

        return loss_dict

    def encode(self, x):
        """エンコードのみ"""
        encoded = self.encoder(x)
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])
        return {'z_style': z_style, 'z_skill': z_skill}

    def decode(self, z_style, z_skill):
        """デコードのみ"""
        trajectory = self.decoder(z_style, z_skill, None)
        return {'trajectory': trajectory}
