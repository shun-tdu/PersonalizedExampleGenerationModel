from typing import  Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

# CLAUDE_ADDED: 動的インポート用に絶対パス使用
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_model import BaseExperimentModel


class MotionTransformerVAE(BaseExperimentModel):
    """既存のVAEモデルを実験システム対応に改良"""

    def __init__(self,
                 input_dim,
                 seq_len,
                 hidden_dim,
                 style_latent_dim,
                 skill_latent_dim,
                 beta=1.0,
                 n_layers=4,
                 contrastive_weight=1.0,
                 use_triplet=False,
                 **kwargs
                 ):

        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            style_latent_dim=style_latent_dim,
            skill_latent_dim=skill_latent_dim,
            beta=beta,
            n_layers=n_layers,
            contrastive_weight=contrastive_weight,
            use_triplet=use_triplet,
            **kwargs
        )

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.style_latent_dim = style_latent_dim
        self.skill_latent_dim = skill_latent_dim
        self.beta = beta
        self.n_layers = n_layers
        self.contrastive_weight = contrastive_weight
        self.use_triplet = use_triplet

        # 実際のネットワーク層を定義
        self._build_network()

    def _build_network(self):
        """ネットワーク構造を構築"""
        # エンコーダー
        # CLAUDE_ADDED: シーケンス全体をフラット化した入力次元に対応
        flattened_input_dim = self.seq_len * self.input_dim
        self.encoder = nn.Sequential(
            nn.Linear(flattened_input_dim, self.hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU())
              for _ in range(self.n_layers - 1)]
        )

        # スタイル潜在空間
        self.style_mu = nn.Linear(self.hidden_dim, self.style_latent_dim)
        self.style_logvar = nn.Linear(self.hidden_dim, self.style_latent_dim)

        # スキル潜在空間
        self.skill_mu = nn.Linear(self.hidden_dim, self.skill_latent_dim)
        self.skill_logvar = nn.Linear(self.hidden_dim, self.skill_latent_dim)

        # デコーダー
        # CLAUDE_ADDED: フラット化した出力次元に対応
        self.decoder = nn.Sequential(
            nn.Linear(self.style_latent_dim + self.skill_latent_dim, self.hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU())
              for _ in range(self.n_layers - 1)],
            nn.Linear(self.hidden_dim, flattened_input_dim)
        )

    def encode(self, x):
        """エンコード"""
        # バッチ次元を考慮した処理
        batch_size, seq_len, features = x.shape
        x_flat = x.reshape(batch_size, -1)  # フラット化

        h = self.encoder(x_flat)

        style_mu = self.style_mu(h)
        style_logvar = self.style_logvar(h)
        skill_mu = self.skill_mu(h)
        skill_logvar = self.skill_logvar(h)

        # リパラメータ化トリック
        style_z = self.reparameterize(style_mu, style_logvar)
        skill_z = self.reparameterize(skill_mu, skill_logvar)

        return {
            'z_style': style_z,
            'z_skill': skill_z,
            'style_mu': style_mu,
            'style_logvar': style_logvar,
            'skill_mu': skill_mu,
            'skill_logvar': skill_logvar
        }

    def decode(self, z_style, z_skill):
        """デコード"""
        z_combined = torch.cat([z_style, z_skill], dim=-1)
        output = self.decoder(z_combined)

        # 元の形状に復元
        batch_size = z_style.shape[0]
        output = output.reshape(batch_size, self.seq_len, self.input_dim)

        return {'trajectory': output}

    def reparameterize(self, mu, logvar):
        """リパラメータ化トリック"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, trajectories, subject_ids=None):
        """
        統一インターフェースでの順伝播
        実験システムが期待する形式で損失を返す
        """
        # エンコード
        encoded = self.encode(trajectories)
        z_style = encoded['z_style']
        z_skill = encoded['z_skill']

        # デコード
        decoded = self.decode(z_style, z_skill)
        reconstructed = decoded['trajectory']

        # 損失計算
        losses = self._compute_all_losses(
            trajectories, reconstructed, encoded, subject_ids
        )

        # 出力辞書（実験システムが期待する形式）
        outputs = {
            'trajectory': reconstructed,
            'z_style': z_style,
            'z_skill': z_skill,
            **losses  # 全ての損失を含める
        }

        return outputs

    def _compute_all_losses(self, original, reconstructed, encoded, subject_ids):
        """全ての損失を計算"""
        losses = {}

        # 再構成損失
        recon_loss = nn.MSELoss()(reconstructed, original)
        losses['trajectory_recon_loss'] = recon_loss

        # KL発散（スタイル）
        style_kl = self._kl_divergence(encoded['style_mu'], encoded['style_logvar'])
        losses['kl_style_loss'] = style_kl

        # KL発散（スキル）
        skill_kl = self._kl_divergence(encoded['skill_mu'], encoded['skill_logvar'])
        losses['kl_skill_loss'] = skill_kl

        # コントラスト損失（被験者IDがある場合）
        if subject_ids is not None:
            contrastive_loss = self._contrastive_loss(encoded['z_style'], subject_ids)
            losses['contrastive_loss'] = contrastive_loss
        else:
            losses['contrastive_loss'] = torch.tensor(0.0)

        # シグネチャー損失（プレースホルダー）
        losses['encoder_signature_loss'] = torch.tensor(0.0)
        losses['decoder_signature_loss'] = torch.tensor(0.0)

        # 総損失
        total_loss = (recon_loss +
                      self.beta * style_kl +
                      self.beta * skill_kl +
                      self.contrastive_weight * losses['contrastive_loss'])

        losses['total_loss'] = total_loss

        return losses

    def _kl_divergence(self, mu, logvar):
        """KL発散の計算"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]

    def _contrastive_loss(self, z_style, subject_ids):
        """コントラスト損失の計算"""
        # CLAUDE_ADDED: subject_idsが文字列リストの場合の処理
        if isinstance(subject_ids, (list, tuple)):
            # 文字列IDを数値に変換
            unique_str_subjects = list(set(subject_ids))
            if len(unique_str_subjects) <= 1:
                return torch.tensor(0.0, device=z_style.device)
                
            subject_to_idx = {subj: i for i, subj in enumerate(unique_str_subjects)}
            subject_tensor = torch.tensor([subject_to_idx[subj] for subj in subject_ids], 
                                        device=z_style.device)
        else:
            subject_tensor = subject_ids
            unique_str_subjects = torch.unique(subject_tensor).tolist()
            if len(unique_str_subjects) <= 1:
                return torch.tensor(0.0, device=z_style.device)

        loss = 0.0
        count = 0
        for subject_idx in unique_str_subjects:
            if isinstance(subject_ids, (list, tuple)):
                # リストの場合の適切な処理
                mask = torch.tensor([subject_to_idx[subj] == subject_idx for subj in subject_ids], 
                                  device=z_style.device)
            else:
                mask = (subject_tensor == subject_idx)
                
            if torch.sum(mask) <= 1:
                continue

            subject_embeddings = z_style[mask]
            # 被験者内の類似性を高める
            center = subject_embeddings.mean(dim=0, keepdim=True)
            intra_loss = nn.MSELoss()(subject_embeddings, center.expand_as(subject_embeddings))
            loss += intra_loss
            count += 1

        return loss / count if count > 0 else torch.tensor(0.0, device=z_style.device)

    def get_evaluation_dict(self, test_data) -> Dict[str, Any]:
        """評価指標を計算（オプション）"""
        # 実際の評価ロジックをここに実装
        return {
            'reconstruction_quality': 0.95,  # 例
            'style_separation_score': 0.85,  # 例
            'skill_correlation': 0.75  # 例
        }