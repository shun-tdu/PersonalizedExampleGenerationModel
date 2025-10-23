"""
CLAUDE_ADDED: 事前学習済みエンコーダ + 拡散デコーダ
- エンコーダは patched_token_pool_compressed_physical_style_skill_separation_net.py と同じ構造
- デコーダは条件付き拡散モデル
- パッチ処理されたデータに対応
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
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
    """
    CLAUDE_ADDED: 事前学習済みエンコーダと同じ構造
    patched_token_pool_compressed_physical_style_skill_separation_net.py から移植
    パッチ処理 + attention mask 対応
    """

    def __init__(self,
                 input_dim,
                 patch_size,
                 d_model,
                 n_heads,
                 n_layers,
                 dropout,
                 style_latent_dim,
                 skill_latent_dim,
                 max_patches=100
                 ):
        super().__init__()

        self.d_model = d_model
        self.patch_size = patch_size

        # 入力射影
        self.input_proj = nn.Linear(input_dim*patch_size, d_model)

        # 分布トークン
        self.style_mu_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.style_logvar_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.skill_mu_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.skill_logvar_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, max_patches + 4)

        # 特徴量抽出Transformer層
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)

        # 潜在変数圧縮層
        self.style_mu_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model//2, style_latent_dim)
        )
        self.style_logvar_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, style_latent_dim)
        )
        self.skill_mu_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, skill_latent_dim)
        )
        self.skill_logvar_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, skill_latent_dim)
        )

    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: [B, Patches, PatchSize, Features]
            src_key_padding_mask: [B, Patches] (True = パディング)
        """
        batch_size, num_patches, _, _ = x.shape

        # パッチをフラット化 [B, Patches, PatchSize * Features]
        x_flat = x.flatten(start_dim=2)

        # 入力射影
        encoded = self.input_proj(x_flat)

        # 分布トークンをバッチサイズ分に拡張してシーケンスに連結
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
        ], dim=1)  # [batch_size, num_patches+4, d_model]

        # 位置エンコーディング
        pos_encoded = self.pos_encoding(full_sequence)

        # データトークン部分をマスク
        if src_key_padding_mask is not None:
            token_padding = torch.zeros(batch_size, 4, dtype=torch.bool, device=x.device)
            full_mask = torch.cat([token_padding, src_key_padding_mask], dim=1)
        else:
            full_mask = None

        # Transformer処理
        attended = self.transformer_encoder(pos_encoded, src_key_padding_mask=full_mask)

        # 任意の潜在空間次元に圧縮
        style_mu = self.style_mu_head(attended[:, 0, :])
        style_logvar = self.style_logvar_head(attended[:, 1, :])
        skill_mu = self.skill_mu_head(attended[:, 2, :])
        skill_logvar = self.skill_logvar_head(attended[:, 3, :])

        return {
            'style_mu': style_mu,
            'style_logvar': style_logvar,
            'skill_mu': skill_mu,
            'skill_logvar': skill_logvar,
        }


class DiffusionDecoderBlock(nn.Module):
    """CLAUDE_ADDED: 拡散デコーダのブロック（AdaLN + Cross-Attention）"""
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

    def forward(self, x, time_emb, context):
        """
        Args:
            x: [B, Patches*PatchSize, d_model]
            time_emb: [B, d_model]
            context: [B, 1, d_model] - スタイル+スキルの条件
        """
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
    """
    CLAUDE_ADDED: パッチ処理対応の条件付き拡散デコーダ
    入力: [B, Patches, PatchSize, Features] 形状のノイズ付き軌道
    条件: z_style, z_skill
    出力: [B, Patches, PatchSize, Features] 形状のノイズ予測
    """
    def __init__(self,
                 output_dim,
                 patch_size,
                 d_model,
                 n_heads,
                 n_layers,
                 dropout,
                 style_latent_dim,
                 skill_latent_dim,
                 max_patches=100
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim

        # パッチをフラット化して処理
        self.input_proj = nn.Linear(output_dim * patch_size, d_model)
        self.output_proj = nn.Linear(d_model, output_dim * patch_size)

        # 時間埋め込み
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU()
        )

        # z_style, z_skillを結合してコンテキストに合成する層
        self.context_proj = nn.Linear(style_latent_dim + skill_latent_dim, d_model)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            DiffusionDecoderBlock(d_model, n_heads) for _ in range(n_layers)
        ])

    def forward(self, x_t, diffusion_step, z_style, z_skill):
        """
        Args:
            x_t: [B, Patches, PatchSize, Features] ノイズ付き軌道
            diffusion_step: [B] タイムステップ
            z_style: [B, style_dim]
            z_skill: [B, skill_dim]
        Returns:
            noise_pred: [B, Patches, PatchSize, Features] ノイズ予測
        """
        batch_size, num_patches, patch_size, features = x_t.shape

        # パッチをフラット化: [B, Patches, PatchSize*Features]
        x_flat = x_t.flatten(start_dim=2)

        # 射影: [B, Patches, d_model]
        x = self.input_proj(x_flat)

        # 時間埋め込み
        time_emb = self.time_embedding(diffusion_step)

        # スタイルとスキルを結合して単一のコンテキストベクトルに
        context = torch.cat([z_style, z_skill], dim=-1)
        context = self.context_proj(context).unsqueeze(1)  # [B, 1, d_model]

        # Transformer blocks
        for block in self.blocks:
            x = block(x, time_emb, context)

        # 出力射影: [B, Patches, PatchSize*Features]
        output_flat = self.output_proj(x)

        # 元の形状に戻す: [B, Patches, PatchSize, Features]
        output = output_flat.view(batch_size, num_patches, patch_size, features)

        return output


class PreTrainedTokenPoolDiffusionNet(BaseExperimentModel):
    """
    CLAUDE_ADDED: 事前学習済みエンコーダ + 拡散デコーダ
    - エンコーダは凍結可能
    - パッチ処理対応
    - 2段階学習対応
    """
    def __init__(self,
                 input_dim=6,
                 patch_size=10,
                 max_patches=100,
                 d_model=512,
                 n_heads=8,
                 n_encoder_layers=8,
                 n_decoder_layers=8,
                 dropout=0.1,
                 style_latent_dim=16,
                 skill_latent_dim=16,
                 factor_num=2,
                 n_subjects=8,
                 ddim_sampling_eta: float = 0.0,
                 ddim_steps: int = 50,
                 loss_schedule_config: Dict[str, Any] = None,
                 pretrained_encoder_path: Optional[str] = None,
                 freeze_encoder: bool = False,
                 **kwargs):
        super().__init__(input_dim=input_dim,
                         patch_size=patch_size,
                         max_patches=max_patches,
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

        self.patch_size = patch_size
        self.d_model = d_model
        self.style_latent_dim = style_latent_dim
        self.skill_latent_dim = skill_latent_dim
        self.input_dim = input_dim
        self.max_patches = max_patches
        self.freeze_encoder = freeze_encoder

        self.current_epoch = 0
        self.max_epoch = 500

        # DDIM parameters
        self.ddim_sampling_eta = ddim_sampling_eta
        self.ddim_steps = ddim_steps

        print(f"PreTrainedTokenPoolDiffusionNet instantiated:")
        print(f"  d_model: {d_model}, n_heads: {n_heads}")
        print(f"  encoder_layers: {n_encoder_layers}, decoder_layers: {n_decoder_layers}")
        print(f"  latent_dims: style={style_latent_dim}, skill={skill_latent_dim}")
        print(f"  patch_size: {patch_size}, max_patches: {max_patches}")
        print(f"  diffusion_timesteps: 1000, DDIM_steps: {ddim_steps}, DDIM_eta: {ddim_sampling_eta}")
        print(f"  freeze_encoder: {freeze_encoder}")

        # 損失スケジューラ
        if loss_schedule_config is None:
            loss_schedule_config = {
                'beta_style': {'schedule': 'linear', 'start_epoch': 21, 'end_epoch': 50, 'start_val': 0.0,
                               'end_val': 0.0001},
                'beta_skill': {'schedule': 'linear', 'start_epoch': 21, 'end_epoch': 50, 'start_val': 0.0,
                               'end_val': 0.0001},
                'factor_regression_loss': {'schedule': 'linear', 'start_epoch': 20, 'end_epoch': 60,
                                           'start_val': 0.0, 'end_val': 0.2},
                'orthogonal_loss': {'schedule': 'constant', 'val': 0.1}
            }
        self.loss_scheduler = LossWeightScheduler(loss_schedule_config)

        # 損失計算フラグ
        self.calc_factor_subtask = 'factor_regression_loss' in loss_schedule_config
        self.calc_orthogonal_loss = 'orthogonal_loss' in loss_schedule_config

        # エンコーダ
        self.encoder = TokenPoolSeparationEncoder(
            input_dim=input_dim,
            patch_size=patch_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            dropout=dropout,
            style_latent_dim=style_latent_dim,
            skill_latent_dim=skill_latent_dim,
            max_patches=max_patches
        )

        # 事前学習済み重みのロード
        if pretrained_encoder_path is not None:
            self.load_pretrained_encoder(pretrained_encoder_path)

        # エンコーダの凍結
        if freeze_encoder:
            self.freeze_encoder_weights()

        # 拡散デコーダ
        self.decoder = ConditionalDiffusionDecoder(
            output_dim=input_dim,
            patch_size=patch_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            dropout=dropout,
            style_latent_dim=style_latent_dim,
            skill_latent_dim=skill_latent_dim,
            max_patches=max_patches
        )

        # サブタスクネットワーク
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

        # サンプリングに必要な係数を事前計算
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

    def load_pretrained_weights(self, checkpoint_path: str, load_partial: bool = False):
        """
        CLAUDE_ADDED: 事前学習済みモデルの重みをロード
        experiment_managerの読み込み処理に対応

        Args:
            checkpoint_path: チェックポイントファイルのパス
            load_partial: 部分的なロード（エンコーダのみなど）を許可するか
        """
        print(f"Loading pretrained weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model_state_dict = checkpoint.get('model_state_dict', checkpoint)

        if load_partial:
            # 部分的にロード（エンコーダのみなど）
            # 現在のモデルのstate_dictと一致する部分のみロード
            current_state_dict = self.state_dict()
            filtered_state_dict = {}

            for key, value in model_state_dict.items():
                if key in current_state_dict:
                    # 形状が一致する場合のみロード
                    if current_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                        print(f"  ✓ Loaded: {key}")
                    else:
                        print(f"  ✗ Skipped (shape mismatch): {key}")
                else:
                    print(f"  ✗ Skipped (not found): {key}")

            self.load_state_dict(filtered_state_dict, strict=False)
            print(f"✓ Partial weights loaded successfully ({len(filtered_state_dict)}/{len(model_state_dict)} layers)")
        else:
            # 完全にロード
            self.load_state_dict(model_state_dict, strict=True)
            print(f"✓ Pretrained weights loaded successfully")

    def load_pretrained_encoder(self, checkpoint_path: str):
        """CLAUDE_ADDED: 事前学習済みエンコーダの重みのみロード（後方互換性のため残す）"""
        print(f"Loading pretrained encoder from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # state_dictから encoder の重みのみ抽出
        encoder_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('encoder.'):
                # 'encoder.' プレフィックスを除去
                new_key = key[8:]
                encoder_state_dict[new_key] = value

        # エンコーダに重みをロード
        self.encoder.load_state_dict(encoder_state_dict, strict=True)
        print(f"✓ Pretrained encoder loaded successfully")

    def apply_freezing_config(self, freeze_modules: List[str] = None, unfreeze_modules: List[str] = None):
        """
        CLAUDE_ADDED: 設定に基づいてモジュールの重みをフリーズ/解凍する
        experiment_managerから呼び出される

        Args:
            freeze_modules: フリーズするモジュール名のプレフィックスリスト
            unfreeze_modules: 解凍するモジュール名のプレフィックスリスト
        """
        if freeze_modules is None:
            freeze_modules = []
        if unfreeze_modules is None:
            unfreeze_modules = []

        # パターン1: "unfreeze_modules" が指定されている場合 (Freeze all, except)
        # -> まず全パラメータをフリーズし、指定されたものだけ解凍
        if unfreeze_modules and not freeze_modules:
            print("モード: 'unfreeze_modules' (指定されたモジュール以外をフリーズ)")
            # 1. 全てフリーズ
            for param in self.parameters():
                param.requires_grad = False

            # 2. 指定されたモジュール（と、その子モジュール）を解凍
            for name, param in self.named_parameters():
                if any(name.startswith(prefix) for prefix in unfreeze_modules):
                    param.requires_grad = True

        # パターン2: "freeze_modules" が指定されている場合 (Unfreeze all, except)
        # -> （デフォルトは全解凍なので）指定されたモジュールだけフリーズ
        elif freeze_modules:
            print("モード: 'freeze_modules' (指定されたモジュールのみフリーズ)")
            for name, param in self.named_parameters():
                if any(name.startswith(prefix) for prefix in freeze_modules):
                    param.requires_grad = False

        # CLAUDE_ADDED: フリーズされたモジュールを評価モードに設定（Dropout/BatchNorm対策）
        for module_name in freeze_modules:
            if hasattr(self, module_name):
                module = getattr(self, module_name)
                module.eval()
                print(f"  → {module_name} を評価モードに設定（Dropout無効化）")

        # (デバッグ用に訓練対象のパラメータを表示)
        print("--- 訓練対象パラメータ ---")
        trainable_count = 0
        total_count = 0
        for name, param in self.named_parameters():
            total_count += param.numel()
            if param.requires_grad:
                print(f"  [TRAIN] {name}")
                trainable_count += param.numel()

        trainable_ratio = (trainable_count / total_count * 100) if total_count > 0 else 0
        print(f"訓練対象パラメータ数: {trainable_count} / {total_count} ({trainable_ratio:.2f}%)")
        print("--------------------------")

    def freeze_encoder_weights(self):
        """CLAUDE_ADDED: エンコーダの重みを凍結（後方互換性のため残す）"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        # CLAUDE_ADDED: Dropout/BatchNormを評価モードにする
        self.encoder.eval()
        print("✓ Encoder weights frozen and set to eval mode")

    def unfreeze_encoder_weights(self):
        """CLAUDE_ADDED: エンコーダの重みの凍結を解除（後方互換性のため残す）"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("✓ Encoder weights unfrozen")

    def _extract(self, a, t, x_shape):
        """バッファから拡散ステップtに対応する値を取得"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.to(a.device))
        return out.reshape(batch_size, *((1, ) * (len(x_shape) - 1)))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_process(self, x0, t):
        """
        CLAUDE_ADDED: x0にtステップ分のノイズを加える
        Args:
            x0: [B, Patches, PatchSize, Features]
            t: [B]
        Returns:
            xt: ノイズ付き軌道
            noise: 加えたノイズ
        """
        noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return xt, noise

    def forward(self, x: torch.Tensor,
                attention_mask: torch.Tensor,
                subject_ids: torch.Tensor = None,
                skill_factor: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, Patches, PatchSize, Features]
            attention_mask: [B, Patches] (True = パディング)
            subject_ids: [B] (未使用、互換性のため)
            skill_factor: [B, factor_num]
        """
        # CLAUDE_ADDED: エンコーダが凍結されている場合、評価モードを維持
        # （model.train()が呼ばれても、エンコーダは評価モードのままにする）
        if self.freeze_encoder and self.encoder.training:
            self.encoder.eval()

        # 1. エンコーダで潜在変数を計算
        encoded = self.encoder(x, src_key_padding_mask=attention_mask)
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])

        # 2. 拡散損失を計算
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        xt, noise = self.forward_process(x, t)
        predicted_noise = self.decoder(xt, t, z_style, z_skill)

        # 3. 損失を計算
        losses = self.compute_losses(x, attention_mask, encoded, z_style, z_skill,
                                      noise, predicted_noise, skill_factor)

        return {
            'predicted_noise': predicted_noise,
            'z_style': z_style,
            'z_skill': z_skill,
            **losses
        }

    def compute_losses(self, x, attention_mask, encoded, z_style, z_skill,
                       noise, predicted_noise, skill_factors):
        """
        CLAUDE_ADDED: 拡散デコーダ学習用の損失計算
        エンコーダ凍結時は拡散損失のみを使用
        """
        weights = self.loss_scheduler.get_weights()
        losses = {}

        # マスク処理: パディング部分を除外
        # attention_mask: [B, Patches] -> [B, Patches, 1, 1]
        loss_mask = (~attention_mask).unsqueeze(-1).unsqueeze(-1).float()

        # ========================================
        # 拡散損失（デコーダ学習の主損失）
        # ========================================
        # CLAUDE_ADDED: 標準的な拡散損失計算（平均二乗誤差）
        # パディング部分を除外して平均を取る
        if attention_mask is not None and attention_mask.any():
            # マスクがある場合: 有効な要素のみで平均
            # loss_mask: [B, Patches, 1, 1] -> [B, Patches, PatchSize, Features] にブロードキャスト
            loss_mask = (~attention_mask).unsqueeze(-1).unsqueeze(-1).float()  # [B, Patches, 1, 1]
            masked_diff = (predicted_noise - noise) * loss_mask  # [B, Patches, PatchSize, Features]

            # 有効な要素数 = パッチ数 × パッチサイズ × 特徴量数
            # loss_maskを正しく展開してから総和を計算
            num_valid_elements = (loss_mask * torch.ones_like(predicted_noise)).sum()

            if num_valid_elements > 0:
                losses['diffusion_loss'] = masked_diff.pow(2).sum() / num_valid_elements
            else:
                losses['diffusion_loss'] = torch.tensor(0.0, device=x.device)
        else:
            # マスクがない場合: 全要素で平均（標準的なMSE）
            losses['diffusion_loss'] = F.mse_loss(predicted_noise, noise)

        # ========================================
        # エンコーダ関連損失（記録のみ、学習には使用しない）
        # ========================================
        # KL損失（記録のみ、total_lossには含めない）
        style_kl = -0.5 * torch.mean(
            torch.sum(1 + encoded['style_logvar'] - encoded['style_mu'].pow(2) -
                      encoded['style_logvar'].exp(), dim=1))
        skill_kl = -0.5 * torch.mean(
            torch.sum(1 + encoded['skill_logvar'] - encoded['skill_mu'].pow(2) -
                      encoded['skill_logvar'].exp(), dim=1))
        losses['kl_style_loss'] = style_kl
        losses['kl_skill_loss'] = skill_kl

        # 直交性損失（記録のみ、total_lossには含めない）
        if self.calc_orthogonal_loss:
            z_style_norm = (z_style - z_style.mean(dim=0)) / (z_style.std(dim=0) + 1e-8)
            z_skill_norm = (z_skill - z_skill.mean(dim=0)) / (z_skill.std(dim=0) + 1e-8)
            cross_corr = torch.matmul(z_style_norm.T, z_skill_norm) / z_style.shape[0]
            losses['orthogonal_loss'] = torch.mean(cross_corr ** 2)
        else:
            losses['orthogonal_loss'] = torch.tensor(0.0, device=x.device)

        # サブタスク損失（記録のみ、total_lossには含めない）
        if skill_factors is not None and self.calc_factor_subtask:
            skill_factor_pred = self.factor_regressor(z_skill)
            losses['factor_regression_loss'] = F.mse_loss(skill_factor_pred, skill_factors)
        else:
            losses['factor_regression_loss'] = torch.tensor(0.0, device=x.device)

        # ========================================
        # 総合損失: 拡散損失のみ
        # ========================================
        # エンコーダが凍結されている場合、エンコーダ関連の損失は無意味
        # 拡散損失のみでデコーダを学習
        total_loss = losses['diffusion_loss']

        losses['total_loss'] = total_loss

        return losses

    @torch.no_grad()
    def ddim_sample(self, z_style, z_skill, num_patches, ddim_steps=None, eta=None):
        """
        CLAUDE_ADDED: DDIM sampling
        Args:
            z_style: [B, style_dim]
            z_skill: [B, skill_dim]
            num_patches: 生成するパッチ数
            ddim_steps: サンプリングステップ数
            eta: 確率性パラメータ (0=決定的, 1=確率的)
        Returns:
            trajectory: [B, Patches, PatchSize, Features]
        """
        device = z_style.device
        batch_size = z_style.shape[0]

        if ddim_steps is None:
            ddim_steps = self.ddim_steps
        if eta is None:
            eta = self.ddim_sampling_eta

        # DDIMタイムステップの作成
        c = self.num_timesteps // ddim_steps
        ddim_timesteps = torch.arange(0, self.num_timesteps, c, device=device).long()

        # 純粋なノイズから開始
        trajectory = torch.randn(batch_size, num_patches, self.patch_size,
                                 self.input_dim, device=device)

        # 逆拡散プロセス
        for i, t in enumerate(reversed(ddim_timesteps)):
            time = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # 前のタイムステップ
            t_prev = ddim_timesteps[-(i+2)] if i < len(ddim_timesteps) - 1 else torch.tensor(-1, device=device)

            # ノイズ予測
            predicted_noise = self.decoder(trajectory, time, z_style, z_skill)

            # alpha値
            alpha_t = self._extract(self.alphas_cumprod, time, trajectory.shape)

            if t_prev >= 0:
                time_prev = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
                alpha_t_prev = self._extract(self.alphas_cumprod, time_prev, trajectory.shape)
            else:
                alpha_t_prev = torch.ones_like(alpha_t)

            # x0予測
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1. - alpha_t)
            pred_x0 = (trajectory - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t

            # DDIM更新
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
            dir_xt = torch.sqrt(1. - alpha_t_prev - sigma_t**2) * predicted_noise
            noise = torch.randn_like(trajectory) if eta > 0 else torch.zeros_like(trajectory)
            trajectory = sqrt_alpha_t_prev * pred_x0 + dir_xt + sigma_t * noise

        return trajectory

    @torch.no_grad()
    def sample(self, z_style, z_skill, num_patches=None):
        """デフォルトのサンプリング（DDIM使用）"""
        if num_patches is None:
            num_patches = self.max_patches
        return self.ddim_sample(z_style, z_skill, num_patches)

    def on_epoch_start(self, epoch: int):
        """エポック開始時の処理"""
        self.loss_scheduler.step(epoch)
        self.current_epoch = epoch

    def training_step(self, batch, optimizers, device: torch.device, max_norm=None) -> Dict[str, torch.Tensor]:
        """1バッチ分の学習処理"""
        trajectory = batch['trajectory'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        skill_factors = batch['skill_factor'].to(device)
        subject_ids = batch['subject_id']

        optimizer = optimizers[0] if isinstance(optimizers, (list, tuple)) else optimizers

        optimizer.zero_grad()
        outputs = self.forward(trajectory, attention_mask, subject_ids, skill_factors)

        loss = outputs['total_loss']
        loss.backward()

        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm)

        optimizer.step()

        loss_dict = {}
        for key, value in outputs.items():
            if 'loss' in key.lower():
                if torch.is_tensor(value):
                    loss_dict[key] = value.item() if value.numel() == 1 else value.mean().item()
                else:
                    loss_dict[key] = float(value)

        return loss_dict

    def validation_step(self, batch, device: torch.device) -> Dict[str, torch.Tensor]:
        """1バッチ分の検証処理"""
        trajectory = batch['trajectory'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        skill_factors = batch['skill_factor'].to(device)
        subject_ids = batch['subject_id']

        with torch.no_grad():
            outputs = self.forward(trajectory, attention_mask, subject_ids, skill_factors)

        loss_dict = {}
        for key, value in outputs.items():
            if 'loss' in key.lower():
                if torch.is_tensor(value):
                    loss_dict[key] = value.item() if value.numel() == 1 else value.mean().item()
                else:
                    loss_dict[key] = float(value)

        return loss_dict

    def configure_optimizers(self, training_config: Dict[str, Any]):
        """オプティマイザとスケジューラの設定"""
        super().configure_optimizers(training_config)

        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self._create_optimizer(trainable_params)
        scheduler = self._create_scheduler(optimizer)

        if scheduler is None:
            return optimizer, None
        else:
            return optimizer, scheduler

    def encode(self, x, attention_mask=None):
        """エンコードのみ"""
        encoded = self.encoder(x, src_key_padding_mask=attention_mask)
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])
        return {'z_style': z_style, 'z_skill': z_skill}

    def decode(self, z_style, z_skill, num_patches=None):
        """デコードのみ（評価器との互換性）"""
        trajectory = self.sample(z_style, z_skill, num_patches)
        return {'trajectory': trajectory}