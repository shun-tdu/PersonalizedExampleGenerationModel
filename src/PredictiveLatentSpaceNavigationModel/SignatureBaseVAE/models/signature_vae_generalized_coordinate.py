"""
🎯 統合シグネチャーVAE全体構造

SimpleDecoderWithSignature版での完全なモデル構造
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==========================================
# 1. 統合シグネチャー計算モジュール
# ==========================================
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

# ==========================================
# 2. 統合シグネチャー学習エンコーダー
# ==========================================
class SignatureGuidedEncoder(nn.Module):
    """統合シグネチャー学習機能付きエンコーダー"""

    def __init__(self, input_dim, hidden_dim, style_latent_dim, skill_latent_dim, n_layers=2):
        super().__init__()

        # 基本LSTM（従来通り）
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.1)

        # スタイル・スキル潜在変数用ヘッド（従来通り）
        self.fc_mu_style = nn.Linear(hidden_dim, style_latent_dim)
        self.fc_logvar_style = nn.Linear(hidden_dim, style_latent_dim)
        self.fc_mu_skill = nn.Linear(hidden_dim, skill_latent_dim)
        self.fc_logvar_skill = nn.Linear(hidden_dim, skill_latent_dim)

        # 📍 新機能: 統合シグネチャー予測ヘッド
        self.signature_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 5)  # 5つのシグネチャー特徴
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim] 軌道データ
        Returns:
            dict: エンコーダー出力（潜在変数パラメータ + 予測シグネチャー）
        """
        # LSTM処理
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]  # [batch_size, hidden_dim]

        # 潜在変数のパラメータ
        mu_style = self.fc_mu_style(last_hidden)
        logvar_style = self.fc_logvar_style(last_hidden)
        mu_skill = self.fc_mu_skill(last_hidden)
        logvar_skill = self.fc_logvar_skill(last_hidden)

        # 📍 新機能: 統合シグネチャーの予測
        predicted_signatures = self.signature_predictor(last_hidden)

        return {
            'mu_style': mu_style,
            'logvar_style': logvar_style,
            'mu_skill': mu_skill,
            'logvar_skill': logvar_skill,
            'predicted_signatures': predicted_signatures  # ← 追加出力
        }


# ==========================================
# 3. シグネチャー対応デコーダー
# ==========================================
class SimpleDecoderWithSignature(nn.Module):
    """軌道 + 統合シグネチャーの両方を出力するデコーダー"""

    def __init__(self, output_dim, hidden_dim, style_latent_dim, skill_latent_dim, seq_len, n_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim

        latent_dim = style_latent_dim + skill_latent_dim

        # 📍 軌道再構成部分（SimpleDecoderと同じ）
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, seq_len * output_dim)
        )

        # 📍 新機能: 統合シグネチャー再構成部分
        self.signature_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 5)  # 5つのシグネチャー特徴
        )

    def forward(self, z_style, z_skill):
        """
        Args:
            z_style: [batch_size, style_latent_dim] スタイル潜在変数
            z_skill: [batch_size, skill_latent_dim] スキル潜在変数
        Returns:
            dict: デコーダー出力（軌道 + シグネチャー）
        """
        batch_size = z_style.size(0)
        z = torch.cat([z_style, z_skill], dim=1)  # 潜在変数結合

        # 📍 軌道再構成
        trajectory_output = self.trajectory_decoder(z)
        reconstructed_trajectory = trajectory_output.view(batch_size, self.seq_len, self.output_dim)

        # 📍 統合シグネチャー再構成
        reconstructed_signatures = self.signature_decoder(z)

        return {
            'trajectory': reconstructed_trajectory,  # [batch_size, seq_len, output_dim]
            'signatures': reconstructed_signatures  # [batch_size, 5]
        }


# ==========================================
# 4. 統合VAEモデル
# ==========================================
class SignatureGuidedVAE(nn.Module):
    """統合シグネチャー学習機能付きVAE（完全版）"""

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 style_latent_dim,
                 skill_latent_dim,
                 seq_len,
                 n_layers=2,
                 beta=0.001,
                 contrastive_weight=0.1,
                 signature_weight=0.5,
                 use_triplet=False,
                 ):
        super().__init__()

        # ハイパーパラメータ
        self.beta = beta
        self.contrastive_weight = contrastive_weight
        self.signature_weight = signature_weight

        # 📍 モデル構成要素
        self.encoder = SignatureGuidedEncoder(input_dim, hidden_dim, style_latent_dim, skill_latent_dim, n_layers)
        self.decoder = SimpleDecoderWithSignature(input_dim, hidden_dim, style_latent_dim, skill_latent_dim, seq_len,
                                                  n_layers)
        self.signature_calculator = BatchSignatureCalculator()

        # 対照学習（従来通り）
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)

    def reparameterize(self, mu, logvar):
        """再パラメータ化トリック"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, subject_ids=None):
        """
        Args:
            x: [batch_size, seq_len, input_dim] 入力軌道
            subject_ids: list 被験者ID（対照学習用）
        Returns:
            dict: 全ての損失と出力
        """
        # 📍 1. エンコード
        encoder_output = self.encoder(x)

        # 📍 2. 潜在変数サンプリング
        z_style = self.reparameterize(encoder_output['mu_style'], encoder_output['logvar_style'])
        z_skill = self.reparameterize(encoder_output['mu_skill'], encoder_output['logvar_skill'])

        # 📍 3. デコード（軌道 + シグネチャー）
        decoder_output = self.decoder(z_style, z_skill)
        reconstructed_trajectory = decoder_output['trajectory']
        reconstructed_signatures = decoder_output['signatures']

        # 📍 4. 真の統合シグネチャーを計算
        true_signatures = self.signature_calculator(x)

        # 📍 5. 損失計算
        # 5-1. 軌道再構成損失（従来通り）
        trajectory_recon_loss = F.mse_loss(reconstructed_trajectory, x)

        # 5-2. KL損失（従来通り）
        kl_style = -0.5 * torch.mean(torch.sum(
            1 + encoder_output['logvar_style'] - encoder_output['mu_style'].pow(2) - encoder_output[
                'logvar_style'].exp(), dim=1))
        kl_skill = -0.5 * torch.mean(torch.sum(
            1 + encoder_output['logvar_skill'] - encoder_output['mu_skill'].pow(2) - encoder_output[
                'logvar_skill'].exp(), dim=1))
        kl_style = torch.clamp(kl_style, min=0.0)
        kl_skill = torch.clamp(kl_skill, min=0.0)

        # 📍 5-3. 統合シグネチャー関連損失（新機能）
        # エンコーダーのシグネチャー予測損失
        encoder_signature_loss = F.mse_loss(encoder_output['predicted_signatures'], true_signatures)
        # デコーダーのシグネチャー再構成損失
        decoder_signature_loss = F.mse_loss(reconstructed_signatures, true_signatures)

        # 5-4. 対照学習損失（従来通り）
        contrastive_loss_val = torch.tensor(0.0, device=x.device)
        if subject_ids is not None and len(set(subject_ids)) > 1:
            contrastive_loss_val = self.contrastive_loss(z_style, subject_ids)

        # 📍 6. 総合損失
        total_loss = (
                trajectory_recon_loss +  # 軌道再構成
                self.beta * (kl_style + kl_skill) +  # KL正則化
                self.signature_weight * encoder_signature_loss +  # エンコーダーシグネチャー学習
                self.signature_weight * decoder_signature_loss +  # デコーダーシグネチャー再構成
                self.contrastive_weight * contrastive_loss_val  # 対照学習
        )

        return {
            'total_loss': total_loss,
            'trajectory_recon_loss': trajectory_recon_loss,
            'kl_style': kl_style,
            'kl_skill': kl_skill,
            'encoder_signature_loss': encoder_signature_loss,
            'decoder_signature_loss': decoder_signature_loss,
            'contrastive_loss': contrastive_loss_val,
            'reconstructed_trajectory': reconstructed_trajectory,
            'reconstructed_signatures': reconstructed_signatures,
            'predicted_signatures': encoder_output['predicted_signatures'],
            'true_signatures': true_signatures,
            'z_style': z_style,
            'z_skill': z_skill
        }

    def encode(self, x):
        """評価用エンコード関数"""
        encoder_output = self.encoder(x)
        z_style = self.reparameterize(encoder_output['mu_style'], encoder_output['logvar_style'])
        z_skill = self.reparameterize(encoder_output['mu_skill'], encoder_output['logvar_skill'])

        return {
            'z_style': z_style,
            'z_skill': z_skill,
            'predicted_signatures': encoder_output['predicted_signatures'],
            **encoder_output
        }

    def decode(self, z_style, z_skill):
        """評価用デコード関数"""
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


# ==========================================
# 5. 対照学習モジュール（従来通り）
# ==========================================
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


# ==========================================
# 6. 使用例とデバッグ
# ==========================================
def create_signature_vae_model(config):
    """統合シグネチャーVAEモデルの作成"""
    return SignatureGuidedVAE(
        input_dim=config['input_dim'],  # 6 (位置差分2 + 速度差分2 + 加速度差分2)
        hidden_dim=config['hidden_dim'],  # 128
        style_latent_dim=config['style_latent_dim'],  # 16
        skill_latent_dim=config['skill_latent_dim'],  # 8
        seq_len=config['seq_len'],  # 100
        n_layers=config['n_layers'],  # 2
        beta=config['beta'],  # 0.001
        contrastive_weight=config['contrastive_weight'],  # 0.1
        signature_weight=config.get('signature_weight', 0.5)  # 0.5
    )


def debug_model_flow():
    """モデルの処理フローをデバッグ"""
    print("🔍 統合シグネチャーVAE処理フロー")
    print("=" * 50)

    print("入力: [batch_size, seq_len, 6] 軌道データ")
    print("  ↓")
    print("📍 SignatureGuidedEncoder:")
    print("  ├─ LSTM → [batch_size, hidden_dim]")
    print("  ├─ 潜在変数ヘッド → mu_style, logvar_style, mu_skill, logvar_skill")
    print("  └─ シグネチャー予測ヘッド → predicted_signatures [batch_size, 5]")
    print("  ↓")
    print("📍 再パラメータ化:")
    print("  └─ z_style [batch_size, style_latent_dim], z_skill [batch_size, skill_latent_dim]")
    print("  ↓")
    print("📍 SimpleDecoderWithSignature:")
    print("  ├─ 軌道デコーダー → reconstructed_trajectory [batch_size, seq_len, 6]")
    print("  └─ シグネチャーデコーダー → reconstructed_signatures [batch_size, 5]")
    print("  ↓")
    print("📍 SignatureCalculator:")
    print("  └─ 真の統合シグネチャー → true_signatures [batch_size, 5]")
    print("  ↓")
    print("📍 損失計算:")
    print("  ├─ 軌道再構成損失: MSE(reconstructed_trajectory, input)")
    print("  ├─ エンコーダーシグネチャー損失: MSE(predicted_signatures, true_signatures)")
    print("  ├─ デコーダーシグネチャー損失: MSE(reconstructed_signatures, true_signatures)")
    print("  ├─ KL損失: KL(q(z|x) || p(z))")
    print("  └─ 対照学習損失: InfoNCE(z_style, subject_ids)")


if __name__ == "__main__":
    debug_model_flow()

    # 設定例
    config = {
        'input_dim': 6,
        'hidden_dim': 128,
        'style_latent_dim': 16,
        'skill_latent_dim': 8,
        'seq_len': 100,
        'n_layers': 2,
        'beta': 0.001,
        'contrastive_weight': 0.1,
        'signature_weight': 0.5
    }

    # モデル作成例
    model = create_signature_vae_model(config)
    print(f"\n✅ モデル作成完了")
    print(f"総パラメータ数: {sum(p.numel() for p in model.parameters()):,}")