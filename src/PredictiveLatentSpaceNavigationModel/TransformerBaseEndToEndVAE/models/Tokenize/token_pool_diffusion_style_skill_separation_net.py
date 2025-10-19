"""
- BERTライクに潜在変数の平均と分散を表すトークンを系列データの先頭に付加
- デコーダに拡散モデルを用いたモデル
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.base_model import BaseExperimentModel
from models.components.loss_weight_scheduler import LossWeightScheduler
# CLAUDE_ADDED: scipy.stats.alphaのimportを削除（未使用）


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

class SinusoidalPosEmb(nn.Module):
    """拡散ステップの埋め込み表現"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim//2
        emb = math.log(10000)/(half_dim-1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None]*emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


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


class DiffusionDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        # 時間埋め込みをAdaptive Layer Normで埋め込み
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model * 6)
        )

        # Pre LayerNorm + self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)

        # Pre LayerNorm + Cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        # Pre LayerNorm + FeedForward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)

        # 時間埋め込みを射影する層
        # self.time_proj = nn.Linear(d_model, d_model)  # CLAUDE_ADDED: スペース修正

    def forward(self, x, time_emb, context):
        # 時間埋め込みを加算
        # x = x + self.time_proj(time_emb).unsqueeze(1)
        s1, b1, s2, b2, s3, b3 = self.adaLN_modulation(time_emb).chunk(6, dim=1)

        # Self Attention
        normed_x1 = self.norm1(x) * (1 + s1.unsqueeze(1)) + b1.unsqueeze(1)
        x = x + self.self_attn(normed_x1, normed_x1, normed_x1)[0]

        # Cross-Attention (z_style, z_skillがコンテキストとして機能)
        normed_x2 = self.norm2(x) * (1 + s2.unsqueeze(1)) + b2.unsqueeze(1)
        x = x + self.cross_attn(query=normed_x2, key=context, value=context)[0]

        # FeedForward
        normed_x3 = self.norm3(x) * (1 + s3.unsqueeze(1)) + b3.unsqueeze(1)
        x = x + self.ffn(normed_x3)
        return x

class ConditionalDiffusionDecoder(nn.Module):
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
        self.input_proj = nn.Linear(output_dim, d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

        # 時間埋め込み
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

        # z_style, z_skillを結合してコンテキストに合成する層
        self.context_proj = nn.Linear(style_latent_dim + skill_latent_dim, d_model)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            DiffusionDecoderBlock(d_model, n_heads) for _ in range(n_layers)
        ])

    def forward(self,x_t, diffusion_step, z_style, z_skill):
        # x_t: [batch, seq_len, features]
        # time: [batch]
        # z_style, z_skill: [batch, latent_dim]

        x = self.input_proj(x_t)
        time_emb = self.time_embedding(diffusion_step)

        # スタイルとスキルを結合して単一のコンテキストベクトルに
        context = torch.cat([z_style, z_skill], dim=-1)
        context = self.context_proj(context).unsqueeze(1)  # [batch, 1, d_model]

        for block in self.blocks:
            x = block(x, time_emb, context)

        return self.output_proj(x)


class TokenPoolSeparationDiffusionNet(BaseExperimentModel):
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
                 ddim_sampling_eta: float = 0.0,
                 ddim_steps: int = 50,
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
                         ddim_sampling_eta=ddim_sampling_eta,
                         ddim_steps=ddim_steps,
                         loss_schedule_config=loss_schedule_config,
                         **kwargs)

        self.seq_len = seq_len
        self.d_model = d_model
        self.style_latent_dim = style_latent_dim
        self.skill_latent_dim = skill_latent_dim

        self.current_epoch = 0
        self.max_epoch = 200
        # CLAUDE_ADDED: input_dimを保存（sample()メソッドで使用）
        self.input_dim = input_dim

        # CLAUDE_ADDED: DDIM parameters
        self.ddim_sampling_eta = ddim_sampling_eta
        self.ddim_steps = ddim_steps

        # CLAUDE_ADDED: 正しいクラス名に修正
        print(f"TokenPoolSeparationDiffusionNet instantiated:")
        print(f"  d_model: {d_model}, n_heads: {n_heads}")
        print(f"  encoder_layers: {n_encoder_layers}, decoder_layers: {n_decoder_layers}")
        print(f"  latent_dims: style={style_latent_dim}, skill={skill_latent_dim}")
        print(f"  diffusion_timesteps: {1000}, DDIM_steps: {ddim_steps}, DDIM_eta: {ddim_sampling_eta}")

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

        self.decoder = ConditionalDiffusionDecoder(
            output_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            dropout=dropout,  # CLAUDE_ADDED: dropoutパラメータを追加
            style_latent_dim=style_latent_dim,
            skill_latent_dim=skill_latent_dim
            )

        # 補助タスク用ネットワーク
        if self.calc_factor_subtask:
            self.factor_regressor = nn.Sequential(
                nn.Linear(skill_latent_dim, skill_latent_dim//2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(skill_latent_dim//2, factor_num)
            )

        # 拡散プロセスのハイパーパラメータ
        self.num_timesteps = 1000
        betas = torch.linspace(0.0001, 0.02, self.num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # 5. サンプリングに必要な係数をすべて事前計算
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 逆方向プロセス q(x_{t-1} | x_t, x_0) の計算に使用
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # 逆方向プロセス p(x_{t-1} | x_t) の分散を計算
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

    def _extract(self, a, t, x_shape):
        """バッファから拡散ステップtに対応する値を取得し、入力xの形状に合わせる"""
        # CLAUDE_ADDED: デバイス統一 - tをaと同じデバイスに移動してからgather
        batch_size = t.shape[0]
        out = a.gather(-1, t.to(a.device))
        return out.reshape(batch_size, *((1, ) * (len(x_shape) - 1)))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_process(self, x0, t):
        """x0にtステップ分のノイズを加える"""
        noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return xt, noise

    def forward(self, x, skill_factors):
        # 1. エンコーダで潜在変数を計算
        encoded = self.encoder(x)
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])

        # 2. 拡散損失を計算
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        xt, noise = self.forward_process(x, t)
        predicted_noise = self.decoder(xt, t, z_style, z_skill)

        # 3. 損失を計算
        losses = self.compute_losses(x, encoded, z_style, z_skill, noise, predicted_noise, skill_factors)

        return {
            'predicted_noise': predicted_noise,
            'z_style': z_style,
            'z_skill': z_skill,
            **losses
        }

    def compute_losses(self, x, encoded, z_style, z_skill, noise, predicted_noise, skill_factors):
        """全ての損失を計算"""
        weights = self.loss_scheduler.get_weights()
        losses = {}

        # 拡散損失
        losses['diffusion_loss'] = F.mse_loss(predicted_noise, noise)

        # KL損失
        style_kl = -0.5 * torch.mean(
            torch.sum(1 + encoded['style_logvar'] - encoded['style_mu'].pow(2) - encoded['style_logvar'].exp(), dim=1))
        skill_kl = -0.5 * torch.mean(
            torch.sum(1 + encoded['skill_logvar'] - encoded['skill_mu'].pow(2) - encoded['skill_logvar'].exp(), dim=1))
        losses['kl_style_loss'] = style_kl
        losses['kl_skill_loss'] = skill_kl

        # 直交性損失
        z_style_norm = (z_style - z_style.mean(dim=0)) / (z_style.std(dim=0) + 1e-8)
        z_skill_norm = (z_skill - z_skill.mean(dim=0)) / (z_skill.std(dim=0) + 1e-8)
        cross_corr = torch.matmul(z_style_norm.T, z_skill_norm) / z_style.shape[0]
        losses['orthogonal_loss'] = torch.mean(cross_corr ** 2)

        # サブタスク損失 (スキル因子回帰)
        skill_factor_pred = self.factor_regressor(z_skill)
        losses['factor_regression_loss'] = F.mse_loss(skill_factor_pred, skill_factors)

        # 総合損失
        total_loss = (losses['diffusion_loss']
                      + weights.get('beta_style', 0.0) * losses['kl_style_loss']
                      + weights.get('beta_skill', 0.0) * losses['kl_skill_loss']
                      + weights.get('orthogonal_loss', 0.0) * losses['orthogonal_loss']
                      + weights.get('factor_regression_loss', 0.0) * losses['factor_regression_loss'])
        losses['total_loss'] = total_loss

        return losses

    @torch.no_grad()
    def ddim_sample(self, z_style, z_skill, ddim_steps=None, eta=None):
        """CLAUDE_ADDED: DDIM sampling - faster and deterministic sampling

        Args:
            z_style: Style latent variable [batch, style_dim]
            z_skill: Skill latent variable [batch, skill_dim]
            ddim_steps: Number of sampling steps (default: self.ddim_steps)
            eta: Stochasticity parameter (0.0=deterministic, 1.0=stochastic like DDPM)

        Returns:
            trajectory: Generated trajectory [batch, seq_len, input_dim]
        """
        device = z_style.device
        batch_size = z_style.shape[0]
        seq_len = self.seq_len
        input_dim = self.input_dim

        # Use default values if not provided
        if ddim_steps is None:
            ddim_steps = self.ddim_steps
        if eta is None:
            eta = self.ddim_sampling_eta

        # Create subsequence of timesteps for DDIM
        # Uniformly sample ddim_steps indices from [0, num_timesteps-1]
        c = self.num_timesteps // ddim_steps
        ddim_timesteps = torch.arange(0, self.num_timesteps, c, device=device).long()

        # Start from pure noise
        trajectory = torch.randn(batch_size, seq_len, input_dim, device=device)

        # Reverse diffusion process
        timesteps_iter = reversed(ddim_timesteps)
        for i, t in enumerate(timesteps_iter):
            # Current and previous timestep
            time = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Get previous timestep
            t_prev = ddim_timesteps[-(i+2)] if i < len(ddim_timesteps) - 1 else torch.tensor(-1, device=device)

            # Predict noise
            predicted_noise = self.decoder(trajectory, time, z_style, z_skill)

            # Get alpha values
            alpha_t = self._extract(self.alphas_cumprod, time, trajectory.shape)

            if t_prev >= 0:
                time_prev = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
                alpha_t_prev = self._extract(self.alphas_cumprod, time_prev, trajectory.shape)
            else:
                alpha_t_prev = torch.ones_like(alpha_t)

            # Predict x0 from xt and predicted noise
            # x0 = (xt - sqrt(1 - alpha_t) * noise) / sqrt(alpha_t)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1. - alpha_t)
            pred_x0 = (trajectory - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
            pred_x0 = torch.clamp(pred_x0, -1., 1.)

            # DDIM direction pointing to xt
            # Direction from x0 to xt at timestep t_prev
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)

            # Compute variance (controlled by eta)
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))

            # Compute direction component
            dir_xt = torch.sqrt(1. - alpha_t_prev - sigma_t**2) * predicted_noise

            # Random noise
            noise = torch.randn_like(trajectory) if eta > 0 else torch.zeros_like(trajectory)

            # DDIM update step
            trajectory = sqrt_alpha_t_prev * pred_x0 + dir_xt + sigma_t * noise

        return trajectory

    @torch.no_grad()
    def sample(self, z_style, z_skill):
        """CLAUDE_ADDED: Default sampling method - uses DDIM for faster generation"""
        return self.ddim_sample(z_style, z_skill)

    def training_step(self, batch, optimizers, device, max_norm=None) -> Dict[str, torch.Tensor]:
        trajectory, _, skill_factors = batch
        trajectory = trajectory.to(device)
        skill_factors = skill_factors.to(device)

        # CLAUDE_ADDED: オプティマイザ確認（単一optimizerモデル用）
        optimizer = optimizers[0] if isinstance(optimizers, (list, tuple)) else optimizers

        # 順伝搬
        optimizer.zero_grad()
        outputs = self.forward(trajectory, skill_factors)

        # 逆伝搬
        loss = outputs['total_loss']
        loss.backward()

        # 勾配クリッピング
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm)

        optimizer.step()

        # 損失を辞書に変換
        loss_dict = {}
        for key, value in outputs.items():
            if 'loss' in key.lower():
                if torch.is_tensor(value):
                    loss_dict[key] = value.item() if value.numel() == 1 else value.mean().item()
                else:
                    loss_dict[key] = float(value)

        return loss_dict

    def validation_step(self, batch, device) -> Dict[str, torch.Tensor]:
        trajectory, _, skill_factors = batch
        trajectory = trajectory.to(device)
        skill_factors = skill_factors.to(device)

        # 準伝搬
        with torch.no_grad():
            outputs = self.forward(trajectory, skill_factors)

        loss = outputs['total_loss']

        # 損失を辞書に変換
        loss_dict = {}
        for key, value in outputs.items():
            if 'loss' in key.lower():
                if torch.is_tensor(value):
                    loss_dict[key] = value.item() if value.numel() == 1 else value.mean().item()
                else:
                    loss_dict[key] = float(value)

        return loss_dict


    def on_train_epoch_start(self, epoch: int):
        self.loss_scheduler.step(epoch)
        self.current_epoch = epoch

    def configure_optimizers(self, training_config: Dict[str, Any]):
        """CLAUDE_ADDED: 単一optimizerとschedulerを返す（experiment_managerとの互換性のため）"""
        super().configure_optimizers(training_config)

        optimizer = self._create_optimizer(self.parameters())
        scheduler = self._create_scheduler(optimizer)

        # CLAUDE_ADDED: schedulerがNoneの場合はoptimizerのみ返す、それ以外はタプルで返す
        if scheduler is None:
            return optimizer, None
        else:
            return optimizer, scheduler

    def encode(self, x):
        encoded = self.encoder(x)
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])
        return {'z_style': z_style, 'z_skill': z_skill}

    def decode(self, z_style, z_skill):
        """CLAUDE_ADDED: 評価器との互換性のため、辞書形式で返す"""
        trajectory = self.sample(z_style, z_skill)
        return {'trajectory': trajectory}





