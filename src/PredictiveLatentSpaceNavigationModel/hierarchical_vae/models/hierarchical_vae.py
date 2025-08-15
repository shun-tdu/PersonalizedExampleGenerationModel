import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PredictionErrorModule(nn.Module):
    """予測誤差を計算するモジュール(自由エネルギー原理)"""
    def __init__(self, dim: int):
        super().__init__()
        self.precision = nn.Parameter(torch.ones(dim))  # 精度パラメータ

    def forward(self, prediction, target):
        """予測誤差を精度で重み付けして計算"""
        error = target - prediction
        weighted_error = self.precision * (error ** 2)
        return weighted_error.sum(dim=-1)


class MotorPrimitiveEncoder(nn.Module):
    """レベル1: 運動プリミティブをエンコード"""
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, hidden_dim//2, kernel_size=3, padding=1)
        self.conv2d = nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=3, padding=1)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.prediction_error = PredictionErrorModule(input_dim)

    def forward(self, x: torch.Tensor, top_down_pred: torch.Tensor=None):
        """
        :param x: [batch, seq_len, input_dim] 入力軌道
        :param top_down_pred: [batch, seq_len, input_dim] 上位からの予測
        """
        batch_size, seq_len, input_dim = x.shape

        # conv1D処理のための次元入れ替え [batch, input_dim, seq_len]
        x_conv = x.transpose(1, 2)
        conv_out = F.relu(self.conv1d(x_conv))
        conv_out = F.relu(self.conv1d(conv_out))
        x_conv = x.transpose(1, 2)

        # LSTM処理
        lstm_out, (h_n, c_n) = self.lstm(conv_out)

        # Self Attention処理
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # 時系列全体をアテンションプール
        feature = attn_out.mean(dim=1) # [batch, hidden_dim]

        mu = self.mu_layer(feature)
        logvar = self.logvar_layer(feature)

        # 予測誤差の計算
        pred_error = torch.zeros(batch_size, device=x.device)
        if top_down_pred is not None:
            pred_error = self.prediction_error(top_down_pred, x).mean(dim=1)

        return mu, logvar, pred_error

class SkillEncoder(nn.Module):
    """レベル2: スキル・技能レベルをエンコード"""
    def __init__(self, level1_latent_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        self.skill_processor = nn.Sequential(
            nn.Linear(level1_latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.prediction_error = PredictionErrorModule(level1_latent_dim)

    def forward(self, z1: torch.Tensor, top_down_pred: torch.Tensor=None):
        """
        :param z1: [batch, level1_latent_dim] 下位レベルの潜在変数
        :param top_down_pred: [batch, level1_latent_dim] 上位からの予測
        """
        feature = self.skill_processor(z1)

        mu = self.mu_layer(feature)
        logvar = self.logvar_layer(feature)

        # 予測誤差の計算
        pred_error = torch.zeros(z1.shape[0], device=z1.device)
        if top_down_pred is not None:
            pred_error = self.prediction_error(top_down_pred, z1)

        return mu, logvar, pred_error


class StyleEncoder(nn.Module):
    """レベル3: 個人のスタイル・特性をエンコード"""
    def __init__(self, level2_latent_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        self.style_processor = nn.Sequential(
            nn.Linear(level2_latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),    # スタイルは過学習しやすいので軽くDropout
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU()
        )

        self.mu_layer = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim // 2, latent_dim)

        self.prediction_error = PredictionErrorModule(level2_latent_dim)

    def forward(self, z2: torch.Tensor, top_down_pred: torch.Tensor=None):
        """
        :param z2: [batch, level2_latent_dim] 下位レベルの潜在変数
        :param top_down_pred: [batch, level2_latent_dim] 上位からの予測(現状だとNone)
        """
        feature = self.style_processor(z2)

        mu = self.mu_layer(feature)
        logvar = self.logvar_layer(feature)

        # 最上位レベルだから予測誤差は0
        pred_error = torch.zeros(z2.shape[0], device=z2.device)

        return mu, logvar, pred_error


class StyleDecoder(nn.Module):
    """レベル3デコーダ: スタイルからスキルレベルを予測"""
    def __init__(self,style_latent_dim: int, hidden_dim: int, skill_latent_dim: int):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(style_latent_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, skill_latent_dim)
        )

    def forward(self, z_style: torch.Tensor):
        """スタイルからスキルの期待値を予測"""
        return self.decoder(z_style)


class SkillDecoder(nn.Module):
    """レベル2デコーダ: スキルから運動プリミティブを予測"""
    def __init__(self, skill_latent_dim: int, hidden_dim: int, primitive_latent_dim: int):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(skill_latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, primitive_latent_dim)
        )

    def forward(self, z_skill: torch.Tensor, style_pred:torch.Tensor=None):
        """
        スキルから運動プリミティブを予測
        :param z_skill: [batch, skill_latent_dim]
        :param style_pred: [batch, skill_latent_dim] 上位からの予測（条件付け用）
        """
        if style_pred is not None:
            # 上位からの予測で条件付け
            combined = z_skill + style_pred
        else:
            combined = z_skill

        return self.decoder(combined)


class MotorDecoder(nn.Module):
    """レベル1デコーダ: 運動プリミティブから軌道を生成"""
    def __init__(self, primitive_latent_dim: int, hidden_dim: int, output_dim: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim

        # 潜在変数をLSTMの初期状態に変換
        self.init_hidden = nn.Linear(primitive_latent_dim, hidden_dim)
        self.init_cell = nn.Linear(primitive_latent_dim, hidden_dim)

        # 軌道生成LSTM
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers=2, batch_first=True)

        # 出力層
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim)
        )

    def forward(self, z_primitive: torch.Tensor, skill_pred: torch.Tensor=None):
        """
        運動プリミティブから軌道を生成

        :param z_primitive: [batch, primitive_latent_dim]
        :param skill_pred: [batch, primitive_latent_dim] 上位からの予測
        """
        batch_size = z_primitive.shape[0]

        if skill_pred is not None:
            combined = z_primitive + skill_pred
        else:
            combined = z_primitive

        # LSTMの初期状態を設定
        h_0 = self.init_hidden(combined).unsqueeze(0).repeat(2, 1, 1)   # [2, batch, hidden]
        c_0 = self.init_cell(combined).unsqueeze(0).repeat(2, 1, 1)     # [2, batch, hidden]

        # 自己回帰的に軌道を生成
        outputs = []
        input_t = torch.zeros(batch_size, 1, self.output_dim, device=z_primitive.device)
        hidden = (h_0, c_0)

        for t in range(self.seq_len):
            output, hidden = self.lstm(input_t, hidden)
            output = self.output_layer(output)
            outputs.append(output)
            input_t = output

        return torch.cat(outputs, dim=1)    # [batch, seq_len, output_dim]


class HierarchicalVAE(nn.Module):
    """階層型VAE - 自由エネルギー原理に基づく予測符号化"""
    def __init__(
            self,
            input_dim: int,
            seq_len: int,
            hidden_dim: int = 128,
            primitive_latent_dim: int = 32,
            skill_latent_dim: int = 16,
            style_latent_dim: int = 8,
            beta_primitive: float = 1.0,
            beta_skill: float = 2.0,
            beta_style: float = 4.0,
            precision_lr: float = 0.1
    ):
        super().__init__()

        # ハイパーパラメータ
        self.beta_primitive = beta_primitive
        self.beta_skill = beta_skill
        self.beta_style = beta_style
        self.precision_lr = precision_lr

        # エンコーダ(ボトムアップ)
        self.primitive_encoder = MotorPrimitiveEncoder(input_dim, hidden_dim, primitive_latent_dim)
        self.skill_encoder = SkillEncoder(primitive_latent_dim, hidden_dim, skill_latent_dim)
        self.style_encoder = StyleEncoder(skill_latent_dim, hidden_dim, style_latent_dim)

        # デコーダ(トップダウン)
        self.style_decoder = StyleDecoder(style_latent_dim, hidden_dim, skill_latent_dim)
        self.skill_decoder = SkillDecoder(skill_latent_dim, hidden_dim, primitive_latent_dim)
        self.primitive_decoder = MotorDecoder(primitive_latent_dim, hidden_dim, input_dim, seq_len)

    def reparameterize(self, mu, logvar):
        """再パラメータ化トリック"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_hierarchical(self, x):
        """階層的エンコーディング（ボトムアップ推論）"""
        # Level 1: 運動プリミティブ
        mu1, logvar1, _ = self.primitive_encoder(x)
        z1 = self.reparameterize(mu1, logvar1)

        # Level 2: スキル
        mu2, logvar2, _ = self.skill_encoder(z1)
        z2 = self.reparameterize(mu2, logvar2)

        # Level 3: スタイル
        mu3, logvar3, _ = self.style_encoder(z2)
        z3 = self.reparameterize(mu3, logvar3)

        return {
            'z_primitive': z1, 'mu_primitive': mu1, 'logvar_primitive': logvar1,
            'z_skill': z2, 'mu_skill': mu2, 'logvar_skill': logvar2,
            'z_style': z3, 'mu_style': mu3, 'logvar_style': logvar3
        }

    def decode_hierarchically(self, z_style, z_skill=None, z_primitive=None):
        """階層的デコーディング（トップダウン予測）"""
        # Level 3 -> 2: スタイルからスキル予測
        skill_pred = self.style_decoder(z_style)

        # Level 2 -> 1: スキルから運動プリミティブ予測
        if z_skill is None:
            z_skill = skill_pred
        primitive_pred = self.skill_decoder(z_skill, skill_pred)

        # Level 1 -> 0: 運動プリミティブから軌道生成
        if z_primitive is None:
            z_primitive = primitive_pred
        trajectory = self.motor_decoder(z_primitive, primitive_pred)

        return trajectory

    def forward(self, x):
        """フォワードパス - 自由エネルギー最小化"""
        batch_size = x.shape[0]

        # === ボトムアップ推論 ===
        encoded = self.encode_hierarchically(x)
        z1, mu1, logvar1 = encoded['z_primitive'], encoded['mu_primitive'], encoded['logvar_primitive']
        z2, mu2, logvar2 = encoded['z_skill'], encoded['mu_skill'], encoded['logvar_skill']
        z3, mu3, logvar3 = encoded['z_style'], encoded['mu_style'], encoded['logvar_style']

        # === トップダウン予測 ===
        # Level 3 -> 2 予測
        skill_pred = self.style_decoder(z3)

        # Level 2 -> 1 予測
        primitive_pred = self.skill_decoder(z2, skill_pred)

        # Level 1 -> 0 予測（軌道再構成）
        reconstructed_x = self.motor_decoder(z1, primitive_pred)

        # === 予測誤差の計算 ===
        # Level 1での予測誤差（データとの比較）
        _, _, pred_error1 = self.primitive_encoder(x, reconstructed_x)

        # Level 2での予測誤差（スキル予測との比較）
        _, _, pred_error2 = self.skill_encoder(z1, skill_pred)

        # Level 3での予測誤差（最上位なので0）
        _, _, pred_error3 = self.style_encoder(z2)

        # === 損失計算 ===
        # 再構成損失（Level 1の予測誤差）
        recon_loss = F.mse_loss(reconstructed_x, x, reduction='sum')

        # KL損失（各レベルでの複雑性コスト）
        kl_primitive = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
        kl_skill = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
        kl_style = -0.5 * torch.sum(1 + logvar3 - mu3.pow(2) - logvar3.exp())

        # 予測誤差項
        prediction_error = pred_error1.sum() + pred_error2.sum() + pred_error3.sum()

        # 自由エネルギー = 精度項 + 複雑性項
        free_energy = (recon_loss + prediction_error +
                       self.beta_primitive * kl_primitive +
                       self.beta_skill * kl_skill +
                       self.beta_style * kl_style)

        return {
            'total_loss': free_energy,
            'reconstruct_loss': recon_loss,
            'prediction_error': prediction_error,
            'kl_primitive': kl_primitive,
            'kl_skill': kl_skill,
            'kl_style': kl_style,
            'reconstructed_x': reconstructed_x,
            'z_primitive': z1,
            'z_skill': z2,
            'z_style': z3
        }

    def encode(self, x):
        """評価用のエンコード関数"""
        encoded = self.encode_hierarchically(x)
        return {
            'z_style': encoded['z_style'],
            'z_skill': encoded['z_skill']
        }

    def decode(self, z_style, z_skill):
        """評価用のデコード関数"""
        return self.decode_hierarchically(z_style, z_skill)

    def generate_personalized_exemplar(self, learner_trajectory, skill_enhancement_factor=0.1):
        """個人最適化お手本生成"""
        with torch.no_grad():
            encoded = self.encode_hierarchically(learner_trajectory)

            # 個人のスタイルを保持
            z_style = encoded['z_style']
            current_skill = encoded['z_skill']

            # スキルを段階的に向上（自由エネルギー原理に基づく最適ステップ）
            skill_noise = torch.randn_like(current_skill) * 0.1
            enhanced_skill = current_skill + skill_enhancement_factor * skill_noise

            # 個人最適化されたお手本を生成
            exemplar = self.decode_hierarchically(z_style, enhanced_skill)

            return exemplar

    def update_epoch(self, epoch: int, max_epoch: int):
        """エポック更新（β-annealing）"""
        if not hasattr(self, 'initial_betas'):
            self.initial_betas = {
                'primitive': self.beta_primitive,
                'skill': self.beta_skill,
                'style': self.beta_style
            }

        # 階層ごとに異なるannealingスケジュール
        progress = epoch / max_epoch

        # 運動プリミティブは早めに学習
        self.beta_primitive = min(self.initial_betas['primitive'],
                                  0.1 + progress * self.initial_betas['primitive'])

        # スキルは中期で学習
        if progress > 0.3:
            skill_progress = (progress - 0.3) / 0.4
            self.beta_skill = min(self.initial_betas['skill'],
                                  0.1 + skill_progress * self.initial_betas['skill'])

        # スタイルは後期で学習（個人差は最後に）
        if progress > 0.6:
            style_progress = (progress - 0.6) / 0.4
            self.beta_style = min(self.initial_betas['style'],
                                  0.1 + style_progress * self.initial_betas['style'])

    def save_model(self, filepath: str):
        """モデル保存"""
        torch.save(self.state_dict(), filepath)