"""
- BERTライクに潜在変数の平均と分散を表すトークンを系列データの先頭に付加して学習するモデル
- 分布トークン計算後圧縮する
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
        batch_size, num_patches, _, _ = x.shape

        # パッチをフラット化 [B, Patches, PatchSize * Features]
        x_flat = x.flatten(start_dim=2)

        # 入力射影
        encoded = self.input_proj(x_flat)

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


class TokenPoolSeparationDecoder(nn.Module):
    """分布トークンで潜在空間を学習するデコーダ"""

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
        self.d_model = d_model

        # スタイル変数とスキル変数を統合して単一のコンテキストに変換する層
        self.latent_fusion = nn.Linear(style_latent_dim + skill_latent_dim, d_model)

        # 出力系列を生成するためのクエリ
        self.queries = nn.Parameter(torch.randn(1, max_patches, d_model))

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, max_patches)

        # TransformerDecoderを定義
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=n_layers)

        # 出力層
        self.output_proj = nn.Linear(d_model, output_dim*patch_size)
        self.output_dim = output_dim

    def forward(self, z_style, z_skill, num_patches):
        batch_size, _ = z_style.shape

        # スタイルとスキルを結合してコンテキストを生成
        z_combined = torch.cat([z_style, z_skill], dim=1)
        memory = self.latent_fusion(z_combined)
        memory = memory.unsqueeze(1)

        # クエリを準備
        queries = self.queries[:, :num_patches, :].expand(batch_size, -1, -1)
        pos_encoded_queries = self.pos_encoding(queries)

        transformed = self.transformer_decoder(tgt=pos_encoded_queries, memory=memory)

        # [B, Patches, d_model] -> [B, Patches, Features * PatchSize]
        reconstructed_flat = self.output_proj(transformed)
        # [B, Patches, Features * PatchSize] -> [B, Patches, PatchSize, Features]
        reconstructed = reconstructed_flat.view(batch_size, num_patches, self.patch_size, self.output_dim)

        return reconstructed


class PatchedTokenPoolCompressedPhysicalSeparationNet(BaseExperimentModel):
    def __init__(self,
                 input_dim=6,
                 patch_size=20,
                 max_patches=100,
                 d_model=256,
                 n_heads=4,
                 n_encoder_layers=4,
                 n_decoder_layers=4,
                 dropout=0.1,
                 style_latent_dim=16,
                 skill_latent_dim=16,
                 factor_num=2,
                 n_subjects=6,
                 loss_schedule_config: Dict[str, Any] = None,
                 dt: float = 0.01,  # CLAUDE_ADDED: サンプリング周期（秒）
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
                         loss_schedule_config=loss_schedule_config,
                         **kwargs)

        self.patch_size = patch_size
        self.d_model = d_model
        self.style_latent_dim = style_latent_dim
        self.skill_latent_dim = skill_latent_dim
        self.dt = dt  # CLAUDE_ADDED: サンプリング周期

        print(f"PatchedTokenPoolSeparationNet instantiated:")
        print(f"  d_model: {d_model}, n_heads: {n_heads}")
        print(f"  encoder_layers: {n_encoder_layers}, decoder_layers: {n_decoder_layers}")
        print(f"  latent_dims: style={style_latent_dim}, skill={skill_latent_dim}")
        print(f"  dt: {dt} (sampling period)")  # CLAUDE_ADDED

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
        self.calc_physical_consistency = 'physical_consistency_loss' in loss_schedule_config  # CLAUDE_ADDED: physical_consistency

        # エンコーダ・デコーダ
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

        self.decoder = TokenPoolSeparationDecoder(
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

    def forward(self, x: torch.Tensor,
                attention_mask: torch.Tensor,
                subject_ids: torch.Tensor = None,
                skill_factor: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # xの形状: [B, S_max, P, F]
        # attention_maskの形状: [B, S_max]

        batch_size = x.shape[0]

        # エンコード
        encoded = self.encoder(x, src_key_padding_mask=attention_mask)

        # 潜在変数サンプリング
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])

        # デコード
        num_patches = x.size(1)
        reconstructed = self.decoder(z_style, z_skill, num_patches)

        # サブタスク: スキル空間から因子スコアを回帰
        skill_factor_pred = None
        if self.calc_factor_subtask:
            skill_factor_pred = self.factor_regressor(z_skill)

        # 損失計算
        losses = self.compute_losses(
            x,
            reconstructed,
            attention_mask,
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

    def compute_physical_consistency_loss(self, trajectory, attention_mask):
        """
        CLAUDE_ADDED: 物理的整合性損失を計算
        位置、速度、加速度が微分関係を満たすように制約する

        Args:
            trajectory: [B, Patches, PatchSize, Features]
                        Features = [pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
            attention_mask: [B, Patches] パディングマスク (Trueがパディング)

        Returns:
            physical_loss: スカラーテンソル
        """
        # データ形状を取得
        batch_size, num_patches, patch_size, num_features = trajectory.shape

        # 特徴量を分解 (input_dim=6: x, y, vx, vy, ax, ay)
        pos = trajectory[..., 0:2]  # [B, Patches, PatchSize, 2]
        vel = trajectory[..., 2:4]  # [B, Patches, PatchSize, 2]
        acc = trajectory[..., 4:6]  # [B, Patches, PatchSize, 2]

        # 方法: 前進差分法 (forward difference) + スケール正規化
        # d(pos)/dt ≈ vel[t] を検証
        # pos[t+1] - pos[t] = vel[t] * dt
        # [B, Patches, PatchSize-1, 2]
        pos_diff = pos[:, :, 1:, :] - pos[:, :, :-1, :]  # 位置差分
        vel_expected = vel[:, :, :-1, :] * self.dt  # 速度から期待される位置変化

        # d(vel)/dt ≈ acc[t] を検証
        # vel[t+1] - vel[t] = acc[t] * dt
        vel_diff = vel[:, :, 1:, :] - vel[:, :, :-1, :]  # 速度差分
        acc_expected = acc[:, :, :-1, :] * self.dt  # 加速度から期待される速度変化

        # CLAUDE_ADDED: スケール正規化された損失計算
        # 相対誤差を使うことで、異なるスケールの量を公平に扱う
        # 位置-速度の整合性: |Δpos - vel*dt| / |vel*dt|
        pos_vel_consistency = (pos_diff - vel_expected).pow(2) / (vel_expected.pow(2) + 1e-6)

        # 速度-加速度の整合性: |Δvel - acc*dt| / |acc*dt|
        # 加速度の寄与を強調するため、より強い正規化を適用
        vel_acc_consistency = (vel_diff - acc_expected).pow(2) / (acc_expected.pow(2) + 1e-6)

        # パッチ方向のマスク処理
        # attention_mask: [B, Patches] -> [B, Patches, 1, 1]に拡張
        patch_mask = (~attention_mask).unsqueeze(-1).unsqueeze(-1).float()
        # PatchSize-1に対応するマスク (最後のフレームは差分計算に使えない)
        patch_mask_diff = patch_mask.expand(-1, -1, patch_size-1, 2)

        # マスクを適用して損失を計算
        pos_vel_loss = (pos_vel_consistency * patch_mask_diff).sum()
        vel_acc_loss = (vel_acc_consistency * patch_mask_diff).sum()

        # 有効な要素数で正規化
        num_valid = patch_mask_diff.sum()
        if num_valid > 0:
            physical_loss = (pos_vel_loss + vel_acc_loss) / num_valid
        else:
            physical_loss = torch.tensor(0.0, device=trajectory.device)

        # CLAUDE_ADDED: デバッグ用ロギング（初回エポックと物理損失導入時）
        if self.training and (self.current_epoch <= 1 or (41 <= self.current_epoch <= 42)) and torch.rand(1).item() < 0.02:
            print(f"\n[Physical Loss Debug] epoch={self.current_epoch}")
            print(f"  dt={self.dt}")
            print(f"  pos_diff range: [{pos_diff.min().item():.6f}, {pos_diff.max().item():.6f}]")
            print(f"  vel_expected range: [{vel_expected.min().item():.6f}, {vel_expected.max().item():.6f}]")
            print(f"  vel_diff range: [{vel_diff.min().item():.6f}, {vel_diff.max().item():.6f}]")
            print(f"  acc_expected range: [{acc_expected.min().item():.6f}, {acc_expected.max().item():.6f}]")
            print(f"  pos_vel_consistency (normalized) mean: {pos_vel_consistency.mean().item():.6f}")
            print(f"  vel_acc_consistency (normalized) mean: {vel_acc_consistency.mean().item():.6f}")
            print(f"  pos_vel_loss: {(pos_vel_loss/num_valid).item():.6f}")
            print(f"  vel_acc_loss: {(vel_acc_loss/num_valid).item():.6f}")
            print(f"  physical_loss: {physical_loss.item():.6f}")
            print(f"  loss_weight: {self.loss_scheduler.get_weights().get('physical_consistency_loss', 0.0):.6f}\n")

        return physical_loss

    def compute_losses(self, x, reconstructed, attention_mask, encoded, z_style, z_skill,
                       skill_factor_pred, skill_factor):
        """CLAUDE_ADDED: 相関行列ベースの直交性損失を使用"""
        weights = self.loss_scheduler.get_weights()

        losses = {}
        # 基本損失
        # マスクを損失計算用に変形: [B, S] -> [B, S, 1, 1]
        loss_mask = (~attention_mask).unsqueeze(-1).unsqueeze(-1)

        # マスクされた要素は0になる
        masked_diff = (reconstructed - x) * loss_mask

        # マスクされていない要素の数で割る
        num_valid_elements = loss_mask.sum()
        if num_valid_elements > 0:
            losses['reconstruction_loss'] = (masked_diff.pow(2).sum()) / num_valid_elements
        else:
            losses['reconstruction_loss'] = torch.tensor(0.0, device=x.device)

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

        # CLAUDE_ADDED: 物理的整合性損失 (位置・速度・加速度の微分関係)
        if self.calc_physical_consistency:
            losses['physical_consistency_loss'] = self.compute_physical_consistency_loss(reconstructed, attention_mask)

        # 総合損失計算
        total_loss = (losses['reconstruction_loss']
                      + weights.get('beta_style', weights.get('beta', 0.0)) * losses['kl_style_loss']
                      + weights.get('beta_skill', weights.get('beta', 0.0)) * losses['kl_skill_loss']
                      + weights.get('orthogonal_loss', 0.0) * losses.get('orthogonal_loss',
                                                                         torch.tensor(0.0, device=x.device))
                      + weights.get('factor_regression_loss', 0.0) * losses.get('factor_regression_loss',
                                                                                torch.tensor(0.0, device=x.device))
                      + weights.get('physical_consistency_loss', 0.0) * losses.get('physical_consistency_loss',
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
        trajectory = batch['trajectory'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        skill_factors = batch['skill_factor'].to(device)
        subject_ids = batch['subject_id']  # これはCPU上のリストのまま

        # 単一オプティマイザの場合
        if isinstance(optimizers, tuple):
            optimizer = optimizers[0]
        else:
            optimizer = optimizers

        # 順伝播
        optimizer.zero_grad()
        outputs = self.forward(trajectory, attention_mask, subject_ids, skill_factors)

        # 逆伝播
        loss = outputs['total_loss']
        loss.backward()

        # 勾配クリッピング
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm)

        optimizer.step()

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
        trajectory = batch['trajectory'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        skill_factors = batch['skill_factor'].to(device)
        subject_ids = batch['subject_id']  # これはCPU上のリストのまま

        # 順伝播のみ
        with torch.no_grad():
            outputs = self.forward(trajectory, attention_mask, subject_ids, skill_factors)

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

    def encode(self, x, attention_mask=None):
        """エンコードのみ"""
        encoded = self.encoder(x, src_key_padding_mask=attention_mask)
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])
        return {'z_style': z_style, 'z_skill': z_skill}

    def decode(self, z_style, z_skill, num_patches=None):
        """デコードのみ"""
        if num_patches is None:
            num_patches = self.decoder.queries.shape[1]  # max_patches
        trajectory = self.decoder(z_style, z_skill, num_patches)
        return {'trajectory': trajectory}
