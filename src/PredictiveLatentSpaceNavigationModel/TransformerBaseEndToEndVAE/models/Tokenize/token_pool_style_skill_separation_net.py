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


class SkillPredictorDiscriminator(nn.Module):
    """z_styleから因子スコアを予測する識別器"""
    def __init__(self, style_latent_dim, factor_num, d_model, dropout):
        super().__init__()
        self.discriminate_net = nn.Sequential(
            nn.Linear(style_latent_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, factor_num)
        )

    def forward(self, z_style):
        return self.discriminate_net(z_style)


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
        ], dim=1) # [batch_size, seq_len+4, d_model]
        
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

        # 損失スケジューラ初期化
        if loss_schedule_config is None:
            loss_schedule_config = {
                'beta_style': {'schedule': 'linear', 'start_epoch': 21, 'end_epoch': 50, 'start_val': 0.0, 'end_val': 0.0001},
                'beta_skill': {'schedule': 'linear', 'start_epoch': 21, 'end_epoch': 50, 'start_val': 0.0, 'end_val': 0.0001},
                'factor_regression_loss': {'schedule': 'linear', 'start_epoch': 20, 'end_epoch': 60, 'start_val': 0.0, 'end_val': 0.2},
                'adversarial_loss' : {'schedule':'constant', 'val':0.1}
            }
        self.loss_scheduler = LossWeightScheduler(loss_schedule_config)

        # スケジューラ設定からlossの計算フラグを受け取り
        self.calc_factor_subtask = 'factor_regression_loss' in loss_schedule_config
        self.adversarial_loss = 'adversarial_loss' in loss_schedule_config

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

        # Discriminator
        self.discriminator = SkillPredictorDiscriminator(style_latent_dim, factor_num, d_model, dropout)

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

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor = None, skill_factor: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """CLAUDE_ADDED: subject_idsを追加してデータローダーとの互換性を確保"""
        batch_size = x.shape[0]

        # エンコード
        encoded = self.encoder(x)

        # 潜在変数サンプリング
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])

        # デコード
        # CLAUDE_ADDED: skip_connections引数を削除
        reconstructed = self.decoder(z_style, z_skill, None)

        # Discriminatorでスタイルからスキル因子を予測
        predicted_skill_factor_from_style = self.discriminator(z_style)

        # サブタスク
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
            skill_factor,
            predicted_skill_factor_from_style
        )

        result = {
            'reconstructed': reconstructed,
            'z_style': z_style,
            'z_skill': z_skill,
        }

        return result | losses

    def compute_losses(self, x, reconstructed, encoded, z_style, z_skill,
                      skill_factor_pred, skill_factor, predicted_skill_factor_from_style):
        weights = self.loss_scheduler.get_weights()

        # 基本損失
        losses = {'reconstruction_loss': F.mse_loss(reconstructed, x)}

        # KL損失
        style_kl_terms = 1 + encoded['style_logvar'] - encoded['style_mu'].pow(2) - encoded['style_logvar'].exp()
        skill_kl_terms = 1 + encoded['skill_logvar'] - encoded['skill_mu'].pow(2) - encoded['skill_logvar'].exp()

        losses['kl_style_loss'] = -0.5 * torch.mean(torch.sum(style_kl_terms, dim=1))
        losses['kl_skill_loss'] = -0.5 * torch.mean(torch.sum(skill_kl_terms, dim=1))

        # Discriminator損失
        if skill_factor is not None and self.calc_factor_subtask:
            # 1. Discriminator自身の損失
            # エンコーダに勾配が流れないようにブロック
            # CLAUDE_ADDED: .squeeze()を削除 - 既に[batch_size, factor_num]の形状
            loss_discriminator = F.mse_loss(self.discriminator(z_style.detach()), skill_factor)
            losses['discriminator_loss'] = loss_discriminator

            # 2. エンコーダのための敵体性損失
            # CLAUDE_ADDED: .squeeze()を削除 - 既に[batch_size, factor_num]の形状
            loss_adversarial = -F.mse_loss(predicted_skill_factor_from_style, skill_factor)
            losses['adversarial_loss'] = loss_adversarial

        # サブタスク損失
        if skill_factor is not None:
            if self.calc_factor_subtask and skill_factor_pred is not None:
                # CLAUDE_ADDED: .squeeze(-1)を削除 - 既に[batch_size, factor_num]の形状
                losses['factor_regression_loss'] = F.mse_loss(skill_factor_pred, skill_factor)

        # 総合損失計算
        total_loss = (losses['reconstruction_loss']
                      + weights.get('beta_style', weights.get('beta', 0.0)) * losses['kl_style_loss']
                      + weights.get('beta_skill', weights.get('beta', 0.0)) * losses['kl_skill_loss']
                      + weights.get('factor_regression_loss', 0.0) * losses.get('factor_regression_loss', torch.tensor(0.0, device=x.device))
                      + weights.get('adversarial_loss', 0.0) * losses.get('adversarial_loss', torch.tensor(0.0, device=x.device))
                      )

        losses['total_loss'] = total_loss
        return losses

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

        trajectory, _ = self.decoder(z_style, z_skill, skip_connections)
        return {'trajectory': trajectory}

    def configure_optimizers(self, training_config: Dict[str, Any]) -> Any:
        super().configure_optimizers(training_config)

        # setup for optimizers
        params_g = list(self.encoder.parameters()) + list(self.decoder.parameters())
        if self.calc_factor_subtask:
            params_g += list(self.factor_regressor.parameters())
        params_d = self.discriminator.parameters()

        opt_g = self._create_optimizer(params_g)
        opt_d = self._create_optimizer(params_d)

        sched_g = self._create_scheduler(opt_g)
        sched_d = self._create_scheduler(opt_d)

        return (opt_g, opt_d), (sched_g, sched_d)

    # CLAUDE_ADDED: 交互最適化の実装
    def training_step(self, batch:Any, optimizers:Tuple[torch.optim.Optimizer, torch.optim.Optimizer], device:torch.device, max_norm=None) -> Dict[str, torch.Tensor]:
        """
        Discriminatorを使った交互最適化を行う計算ステップ
        1. Discriminatorの学習: エンコーダを固定してDiscriminatorを更新
        2. Generator(Encoder/Decoder)の学習: Discriminatorを固定してGeneratorを更新
        """
        opt_g, opt_d = optimizers
        batch_data = [data.to(device) if torch.is_tensor(data) else data for data in batch]

        # データの取得
        x, subject_ids, skill_factor = batch_data[0], batch_data[1], batch_data[2]


        # 1. Discriminatorの学習
        if skill_factor is not None and self.adversarial_loss:
            opt_d.zero_grad()

            # エンコーダで潜在変数を取得（勾配は流さない）
            with torch.no_grad():
                encoded = self.encoder(x)
                z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])

            # Discriminatorの損失計算
            predicted_skill_factor_from_style = self.discriminator(z_style)
            # CLAUDE_ADDED: .squeeze()を削除 - 既に[batch_size, factor_num]の形状
            loss_d = F.mse_loss(predicted_skill_factor_from_style, skill_factor)

            loss_d.backward()
            opt_d.step()
        else:
            loss_d = torch.tensor(0.0, device=device)

        # 2. Generator (Encoder/Decoder)の学習
        opt_g.zero_grad()

        # 順伝播
        outputs = self.forward(x, subject_ids, skill_factor)
        total_loss = outputs['total_loss']

        total_loss.backward()
        opt_g.step()

        # 損失辞書を返す（数値に変換）
        loss_dict = {}
        for key, value in outputs.items():
            if 'loss' in key.lower():
                loss_dict[key] = value.item() if torch.is_tensor(value) else value

        # Discriminator損失も追加
        loss_dict['discriminator_step_loss'] = loss_d.item() if torch.is_tensor(loss_d) else float(loss_d)

        return loss_dict

    # CLAUDE_ADDED: 検証ステップの実装
    def validation_step(self, batch:Any, device:torch.device) -> Dict[str, torch.Tensor]:
        """
        1バッチ分の検証処理を行い、ログ用の損失辞書を返す
        """
        batch_data = [data.to(device) if torch.is_tensor(data) else data for data in batch]

        # データの取得
        if len(batch_data) >= 3:
            x, subject_ids, skill_factor = batch_data[0], batch_data[1], batch_data[2]
        elif len(batch_data) == 2:
            x, subject_ids = batch_data[0], batch_data[1]
            skill_factor = None
        else:
            x = batch_data[0]
            subject_ids = None
            skill_factor = None

        # 順伝播
        with torch.no_grad():
            outputs = self.forward(x, subject_ids, skill_factor)

        # 損失辞書を返す（数値に変換）
        loss_dict = {}
        for key, value in outputs.items():
            if 'loss' in key.lower():
                loss_dict[key] = value.item() if torch.is_tensor(value) else value

        return loss_dict

    # CLAUDE_ADDED: encodeメソッドの修正 - skip_connectionsは存在しない
    def encode(self, x):
        """エンコードのみ"""
        encoded = self.encoder(x)
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])
        return {'z_style': z_style, 'z_skill': z_skill}

    # CLAUDE_ADDED: decodeメソッドの修正 - skip_connectionsを削除
    def decode(self, z_style, z_skill):
        """デコードのみ"""
        trajectory = self.decoder(z_style, z_skill, None)
        return {'trajectory': trajectory}
