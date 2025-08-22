import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_style, subject_ids):
        batch_size = z_style.shape[0]
        unique_subjects = list(set(subject_ids))
        if len(unique_subjects) < 2:
            return torch.tensor(0.0, device=z_style.device)

        subject_to_idx = {subj: i for i, subj in enumerate(unique_subjects)}
        labels = torch.tensor([subject_to_idx[subj] for subj in subject_ids], device=z_style.device)

        z_style_norm = F.normalize(z_style, p=2, dim=1)
        sim_matrix = torch.mm(z_style_norm, z_style_norm.t()) / self.temperature
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask = mask - torch.eye(batch_size, device=z_style.device)

        exp_sim = torch.exp(sim_matrix)
        sum_exp = torch.sum(exp_sim * (1 - torch.eye(batch_size, device=z_style.device)), dim=1)
        pos_sum = torch.sum(exp_sim * mask, dim=1)
        pos_sum = torch.clamp(pos_sum, min=1e-8)

        loss = -torch.log(pos_sum / (sum_exp + 1e-8))
        return loss.mean()


class BatchSignatureCalculator(nn.Module):
    """バッチ処理対応の高効率シグネチャー計算"""

    def forward(self, trajectories):
        """
        Args:
            trajectories: [batch_size, seq_len, 6]
        Returns:
            signatures: [batch_size, 5]
        """
        batch_size, seq_len, _ = trajectories.shape
        device = trajectories.device

        # バッチ全体で並列計算
        signatures = torch.zeros(batch_size, 5, device=device, dtype=torch.float32)

        # 1. 軌道曲率（バッチ処理）
        signatures[:, 0] = self._batch_path_curvature(trajectories[:, :, :2])

        # 2. 速度滑らかさ（バッチ処理）
        signatures[:, 1] = self._batch_velocity_smoothness(trajectories[:, :, 2:4])

        # 3. 加速度ジャーク（バッチ処理）
        signatures[:, 2] = self._batch_acceleration_jerk(trajectories[:, :, 4:6])

        # 4. 運動リズム（バッチ処理）
        signatures[:, 3] = self._batch_movement_rhythm(trajectories[:, :, 2:4])

        # 5. 力調節（バッチ処理）
        signatures[:, 4] = self._batch_force_modulation(trajectories[:, :, 4:6])

        return signatures

    def _batch_path_curvature(self, pos_diffs):
        """バッチ処理版軌道曲率"""
        # pos_diffs: [batch_size, seq_len, 2]
        batch_size, seq_len, _ = pos_diffs.shape

        if seq_len < 3:
            return torch.zeros(batch_size, device=pos_diffs.device)

        # 位置復元
        positions = torch.cumsum(pos_diffs, dim=1)  # [batch_size, seq_len, 2]

        # 3点での曲率計算
        p1 = positions[:, :-2, :]  # [batch_size, seq_len-2, 2]
        p2 = positions[:, 1:-1, :]  # [batch_size, seq_len-2, 2]
        p3 = positions[:, 2:, :]  # [batch_size, seq_len-2, 2]

        v1 = p2 - p1  # [batch_size, seq_len-2, 2]
        v2 = p3 - p2  # [batch_size, seq_len-2, 2]

        # 外積
        cross = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]  # [batch_size, seq_len-2]

        # ノルム
        norms1 = torch.norm(v1, dim=2)  # [batch_size, seq_len-2]
        norms2 = torch.norm(v2, dim=2)  # [batch_size, seq_len-2]
        norms_product = norms1 * norms2  # [batch_size, seq_len-2]

        # ゼロ除算回避
        valid_mask = norms_product > 1e-6
        curvatures = torch.abs(cross) / (norms_product + 1e-6)

        # マスクを適用して平均
        curvatures = curvatures * valid_mask.float()
        valid_counts = valid_mask.float().sum(dim=1)
        valid_counts = torch.clamp(valid_counts, min=1.0)

        return curvatures.sum(dim=1) / valid_counts

    def _batch_velocity_smoothness(self, velocities):
        """バッチ処理版速度滑らかさ"""
        # velocities: [batch_size, seq_len, 2]
        if velocities.shape[1] < 2:
            return torch.ones(velocities.shape[0], device=velocities.device)

        vel_changes = torch.abs(torch.diff(velocities, dim=1))  # [batch_size, seq_len-1, 2]
        mean_changes = torch.mean(vel_changes, dim=(1, 2))  # [batch_size]

        return 1.0 / (1.0 + mean_changes)

    def _batch_acceleration_jerk(self, accelerations):
        """バッチ処理版加速度ジャーク"""
        # accelerations: [batch_size, seq_len, 2]
        if accelerations.shape[1] < 2:
            return torch.zeros(accelerations.shape[0], device=accelerations.device)

        jerk = torch.abs(torch.diff(accelerations, dim=1))  # [batch_size, seq_len-1, 2]
        return torch.mean(jerk, dim=(1, 2))  # [batch_size]

    def _batch_movement_rhythm(self, velocities):
        """バッチ処理版運動リズム"""
        # velocities: [batch_size, seq_len, 2]
        speeds = torch.norm(velocities, dim=2)  # [batch_size, seq_len]

        mean_speeds = torch.mean(speeds, dim=1)  # [batch_size]
        std_speeds = torch.std(speeds, dim=1)  # [batch_size]

        return std_speeds / (mean_speeds + 1e-6)

    def _batch_force_modulation(self, accelerations):
        """バッチ処理版力調節"""
        # accelerations: [batch_size, seq_len, 2]
        forces = torch.norm(accelerations, dim=2)  # [batch_size, seq_len]

        mean_forces = torch.mean(forces, dim=1)  # [batch_size]
        std_forces = torch.std(forces, dim=1)  # [batch_size]

        return std_forces / (mean_forces + 1e-6)

class PositionalEncoding(nn.Module):
    """時系列用位置エンコーディング"""

    def __init__(self, d_model, max_len=200):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch, d_model]
        return x + self.pe[:x.size(0), :]


class MultiHeadTemporalAttention(nn.Module):
    """運動特化型マルチヘッドアテンション"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

        # 運動特化：時間的近接性バイアス
        self.temporal_bias = nn.Parameter(torch.zeros(1, n_heads, 200, 200))
        self.register_temporal_bias()

    def register_temporal_bias(self):
        """時間的近接性に基づくアテンションバイアスを初期化"""
        with torch.no_grad():
            max_len = self.temporal_bias.size(-1)
            for i in range(max_len):
                for j in range(max_len):
                    # 時間的距離に基づく重み
                    temporal_distance = abs(i - j)
                    bias_value = -0.1 * temporal_distance / max_len
                    self.temporal_bias[0, :, i, j] = bias_value

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Q, K, V計算
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # アテンション計算
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 時間的バイアス追加
        temporal_bias = self.temporal_bias[:, :, :seq_len, :seq_len]
        attention_scores = attention_scores + temporal_bias

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 重み付き和
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)

        output = self.w_o(context)

        return output, attention_weights


class MotionTransformerBlock(nn.Module):
    """運動解析特化Transformerブロック"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadTemporalAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # フィードフォワード（運動特徴に特化）
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # ReLUよりもスムーズ
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # マルチヘッドアテンション
        attn_output, attention_weights = self.attention(x, mask)
        x = self.norm1(x + attn_output)

        # フィードフォワード
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)

        return x, attention_weights


class HierarchicalMotionEncoder(nn.Module):
    """階層的運動エンコーダー（Transformer版）"""

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 n_heads,
                 n_layers,
                 style_latent_dim,
                 skill_latent_dim,
                 max_len=200
                 ):
        super().__init__()

        self.d_model = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len)

        # マルチレベル Transformer
        self.transformer_blocks = nn.ModuleList([
            MotionTransformerBlock(hidden_dim, n_heads, hidden_dim * 4)
            for _ in range(n_layers)
        ])

        # 階層的特徴抽出
        self.local_pooling = nn.AdaptiveAvgPool1d(50)  # 局所特徴
        self.global_pooling = nn.AdaptiveAvgPool1d(1)  # 大域特徴

        # スタイル・スキル分離ヘッド
        combined_dim = hidden_dim * 2  # 局所 + 大域

        self.style_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, style_latent_dim * 2)  # mu + logvar
        )

        self.skill_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, skill_latent_dim * 2)  # mu + logvar
        )

        # 統合シグネチャー予測
        self.signature_predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 5)  # 5つのシグネチャー
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            dict: エンコーダー出力
        """
        batch_size, seq_len, _ = x.shape

        # 入力プロジェクション
        x = self.input_projection(x) * math.sqrt(self.d_model)

        # 位置エンコーディング
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch, seq_len, d_model]

        # Transformerブロック適用
        all_attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attention_weights = transformer_block(x)
            all_attention_weights.append(attention_weights)

        # 階層的特徴抽出
        x_transposed = x.transpose(1, 2)  # [batch, d_model, seq_len]

        # 局所特徴（中解像度）
        local_features = self.local_pooling(x_transposed)  # [batch, d_model, 50]
        local_features = torch.mean(local_features, dim=2)  # [batch, d_model]

        # 大域特徴（低解像度）
        global_features = self.global_pooling(x_transposed)  # [batch, d_model, 1]
        global_features = global_features.squeeze(2)  # [batch, d_model]

        # 特徴結合
        combined_features = torch.cat([local_features, global_features], dim=1)

        # 潜在変数パラメータ
        style_params = self.style_head(combined_features)
        skill_params = self.skill_head(combined_features)

        style_mu = style_params[:, :style_params.size(1) // 2]
        style_logvar = style_params[:, style_params.size(1) // 2:]
        skill_mu = skill_params[:, :skill_params.size(1) // 2]
        skill_logvar = skill_params[:, skill_params.size(1) // 2:]

        # シグネチャー予測
        predicted_signatures = self.signature_predictor(combined_features)

        return {
            'style_mu': style_mu,
            'style_logvar': style_logvar,
            'skill_mu': skill_mu,
            'skill_logvar': skill_logvar,
            'predicted_signatures': predicted_signatures,
            'attention_weights': all_attention_weights,
            'local_features': local_features,
            'global_features': global_features,
            'transformer_output': x  # [batch, seq_len, d_model]
        }


class MotionTransformerDecoder(nn.Module):
    """Transformer-based デコーダー"""

    def __init__(self, style_latent_dim, skill_latent_dim, d_model, n_heads,
                 n_layers, output_dim, max_len=200):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.output_dim = output_dim

        latent_dim = style_latent_dim + skill_latent_dim

        # 潜在変数から初期シーケンス生成
        self.latent_to_sequence = nn.Sequential(
            nn.Linear(latent_dim, d_model * max_len),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Decoderブロック
        self.decoder_blocks = nn.ModuleList([
            MotionTransformerBlock(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ])

        # 出力プロジェクション
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, output_dim)
        )

        # シグネチャー再構成
        self.signature_reconstructor = nn.Sequential(
            nn.Linear(latent_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 5)
        )

    def forward(self, z_style, z_skill, target_length=None):
        batch_size = z_style.size(0)
        if target_length is None:
            target_length = self.max_len

        z = torch.cat([z_style, z_skill], dim=1)

        # 潜在変数からシーケンス生成
        sequence = self.latent_to_sequence(z)  # [batch, d_model * max_len]
        sequence = sequence.view(batch_size, self.max_len, self.d_model)

        # 必要な長さに調整
        if target_length != self.max_len:
            if target_length < self.max_len:
                sequence = sequence[:, :target_length, :]
            else:
                # パディング
                padding = torch.zeros(batch_size, target_length - self.max_len,
                                      self.d_model, device=sequence.device)
                sequence = torch.cat([sequence, padding], dim=1)

        # 位置エンコーディング
        sequence = sequence.transpose(0, 1)  # [seq_len, batch, d_model]
        sequence = self.pos_encoding(sequence)
        sequence = sequence.transpose(0, 1)  # [batch, seq_len, d_model]

        # Decoderブロック適用
        for decoder_block in self.decoder_blocks:
            sequence, _ = decoder_block(sequence)

        # 出力プロジェクション
        reconstructed_trajectory = self.output_projection(sequence)

        # シグネチャー再構成
        reconstructed_signatures = self.signature_reconstructor(z)

        return {
            'trajectory': reconstructed_trajectory,
            'signatures': reconstructed_signatures
        }


class MotionTransformerVAE(nn.Module):
    """Transformer-based 運動 VAE"""

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 style_latent_dim,
                 skill_latent_dim,
                 seq_len,
                 n_layers=6,
                 beta=0.001,
                 contrastive_weight=0.1,
                 signature_weight=0.5,
                 n_heads=8,
                 max_len=100,
                 use_triplet=False):
        super().__init__()

        self.beta = beta
        self.contrastive_weight = contrastive_weight
        self.signature_weight = signature_weight

        self.encoder = HierarchicalMotionEncoder(
            input_dim, hidden_dim, n_heads, n_layers,
            style_latent_dim, skill_latent_dim, max_len
        )

        self.decoder = MotionTransformerDecoder(
            style_latent_dim, skill_latent_dim, hidden_dim, n_heads,
            n_layers, input_dim, max_len
        )

        self.signature_calculator = BatchSignatureCalculator()
        self.contrastive_loss = ContrastiveLoss()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, subject_ids=None):
        batch_size, seq_len, _ = x.shape

        # エンコード
        encoder_output = self.encoder(x)

        # 潜在変数サンプリング
        z_style = self.reparameterize(encoder_output['style_mu'], encoder_output['style_logvar'])
        z_skill = self.reparameterize(encoder_output['skill_mu'], encoder_output['skill_logvar'])

        # デコード
        decoder_output = self.decoder(z_style, z_skill, target_length=seq_len)

        # 真のシグネチャー計算
        true_signatures = self.signature_calculator(x)

        # 損失計算
        recon_loss = F.mse_loss(decoder_output['trajectory'], x)

        kl_style = -0.5 * torch.mean(torch.sum(
            1 + encoder_output['style_logvar'] - encoder_output['style_mu'].pow(2)
            - encoder_output['style_logvar'].exp(), dim=1))
        kl_skill = -0.5 * torch.mean(torch.sum(
            1 + encoder_output['skill_logvar'] - encoder_output['skill_mu'].pow(2)
            - encoder_output['skill_logvar'].exp(), dim=1))

        kl_style = torch.clamp(kl_style, min=0.0)
        kl_skill = torch.clamp(kl_skill, min=0.0)

        # シグネチャー損失（エンコーダー予測 + デコーダー再構成）
        encoder_sig_loss = F.mse_loss(encoder_output['predicted_signatures'], true_signatures)
        decoder_sig_loss = F.mse_loss(decoder_output['signatures'], true_signatures)
        signature_loss = encoder_sig_loss + decoder_sig_loss

        contrastive_loss_val = torch.tensor(0.0, device=x.device)
        if subject_ids is not None and len(set(subject_ids)) > 1:
            contrastive_loss_val = self.contrastive_loss(z_style, subject_ids)

        total_loss = (recon_loss +
                      self.beta * (kl_style + kl_skill) +
                      self.signature_weight * signature_loss +
                      self.contrastive_weight * contrastive_loss_val)

        return {
            'total_loss': total_loss,
            'trajectory_recon_loss': recon_loss,
            'kl_style': kl_style,
            'kl_skill': kl_skill,
            'signature_loss': signature_loss,
            'encoder_signature_loss': encoder_sig_loss,
            'decoder_signature_loss': decoder_sig_loss,
            'contrastive_loss': contrastive_loss_val,
            'reconstructed': decoder_output['trajectory'],
            'reconstructed_signatures': decoder_output['signatures'],
            'predicted_signatures': encoder_output['predicted_signatures'],
            'true_signatures': true_signatures,
            'z_style': z_style,
            'z_skill': z_skill,
            # 'attention_weights': encoder_output['attention_weights']
        }

    def encode(self, x):
        encoder_output = self.encoder(x)
        z_style = self.reparameterize(encoder_output['style_mu'], encoder_output['style_logvar'])
        z_skill = self.reparameterize(encoder_output['skill_mu'], encoder_output['skill_logvar'])
        return {'z_style': z_style, 'z_skill': z_skill, **encoder_output}

    def decode(self, z_style, z_skill, target_length=None):
        return self.decoder(z_style, z_skill)

    def save_model(self, filepath: str):
        """モデル保存"""
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath: str, device=None):
        """モデル読み込み"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        state_dict = torch.load(filepath, map_location=device)
        self.load_state_dict(state_dict)
        self.to(device)