import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """軌道シーケンスを受け取り、潜在変数z_styleとz_skillを生成する"""
    def __init__(self, input_dim: int, hidden_dim: int, style_latent_dim: int, skill_latent_dim: int, n_layers: int=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

        # スタイル用のヘッド
        self.fc_mu_style = nn.Linear(hidden_dim, style_latent_dim)
        self.fc_logvar_style = nn.Linear(hidden_dim, style_latent_dim)

        # スキル用のヘッド
        self.fc_mu_skill = nn.Linear(hidden_dim, skill_latent_dim)
        self.fc_logvar_skill = nn.Linear(hidden_dim, skill_latent_dim)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch, seq_len, input_dim]
        :return: (mu_style, logvar_style, mu_skill, logvar_skill)
        """
        _, (hidden, _ ) = self.lstm(x)

        # 最終層の隠れ状態を使用 [batch, hidden_dim]
        last_hidden = hidden[-1]

        # スタイル潜在変数のパラメータ
        mu_style = self.fc_mu_style(last_hidden)
        logvar_style = self.fc_logvar_style(last_hidden)

        # スキル潜在変数のパラメータ
        mu_skill = self.fc_mu_skill(last_hidden)
        logvar_skill = self.fc_logvar_skill(last_hidden)

        return mu_style, logvar_style, mu_skill, logvar_skill


# -----------------------------------------------
# 2. デコーダ (Decoder)
# -----------------------------------------------
class Decoder(nn.Module):
    """潜在変数z_styleとz_skillから軌道シーケンスを生成する"""

    def __init__(self, output_dim, hidden_dim, style_latent_dim, skill_latent_dim, seq_len, n_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim

        # 潜在変数をLSTMの初期隠れ状態に変換する層
        self.fc_init = nn.Linear(style_latent_dim + skill_latent_dim, hidden_dim * n_layers)

        self.lstm = nn.LSTM(output_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z_style, z_skill):
        # z_styleとz_skillを結合
        z = torch.cat([z_style, z_skill], dim=1)

        # LSTMの初期隠れ状態を生成
        initial_hidden_flat = self.fc_init(z)
        # 形状を(n_layers, batch_size, hidden_dim)に整形
        h_0 = initial_hidden_flat.view(self.lstm.num_layers, -1, self.lstm.hidden_size)
        c_0 = torch.zeros_like(h_0)  # 初期セル状態はゼロ

        # 自己回帰的に軌道を生成
        # 最初の入力はゼロベクトル
        batch_size = z.size(0)
        decoder_input = torch.zeros(batch_size, 1, self.output_dim, device=z.device)

        outputs = []
        hidden = (h_0, c_0)
        for _ in range(self.seq_len):
            output, hidden = self.lstm(decoder_input, hidden)
            output = self.fc_out(output)
            outputs.append(output)
            decoder_input = output  # 次の入力は現在の出力

        return torch.cat(outputs, dim=1)


# -----------------------------------------------
# 3. β-VAEモデル (統合)
# -----------------------------------------------
class BetaVAE(nn.Module):
    """エンコーダとデコーダを組み合わせたβ-VAEモデル"""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            style_latent_dim: int,
            skill_latent_dim: int,
            seq_len: int,
            n_layers: int=2,
            beta: float=1.0
    ):
        super().__init__()
        self.beta = beta
        self.encoder = Encoder(input_dim, hidden_dim, style_latent_dim, skill_latent_dim, n_layers)
        self.decoder = Decoder(input_dim, hidden_dim, style_latent_dim, skill_latent_dim, seq_len, n_layers)

    def reparameterize(self, mu, logvar):
        """再パラメータ化トリック"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """エンコードのみを実行(評価用)"""
        mu_style, logvar_style, mu_skill, logvar_skill =self.encoder(x)
        z_style = self.reparameterize(mu_style, logvar_style)
        z_skill = self.reparameterize(mu_skill, logvar_skill)
        return {
            'z_style': z_style,
            'z_skill': z_skill,
            'mu_style': mu_style,
            'logvar_style': logvar_style,
            'mu_skill': mu_skill,
            'logvar_skill': logvar_skill
        }

    def decode(self, z_style, z_skill):
        """デコードのみを実行（評価用）"""
        return self.decoder(z_style, z_skill)

    def forward(self, x):
        # エンコード
        mu_style, logvar_style, mu_skill, logvar_skill = self.encoder(x)

        # サンプリング
        z_style = self.reparameterize(mu_style, logvar_style)
        z_skill = self.reparameterize(mu_skill, logvar_skill)

        # デコード
        reconstructed_x = self.decoder(z_style, z_skill)

        total_loss, reconstruct_loss, kl_style,kl_skill = loss_function(reconstructed_x, x, mu_style, logvar_style, mu_skill, logvar_skill, self.beta)

        return {
            'total_loss': total_loss,
            'reconstruct_loss': reconstruct_loss,
            'kl_style': kl_style,
            'kl_skill': kl_skill,
            'reconstructed_x': reconstructed_x,
            'z_style': z_style,
            'z_skill': z_skill
        }

    def reconstruct(self, x):
        """再構成のみを実行（評価用）"""
        encoded = self.encode(x)
        reconstructed = self.decode(encoded['z_style'], encoded['z_skill'])
        return reconstructed

    def generate_trajectory(self, z_style, z_skill):
        """潜在変数から軌道を生成（評価用）"""
        with torch.no_grad():
            return self.decode(z_style, z_skill)

    def interpolate_skill(self, z_style, z_skill_start, z_skill_end, num_steps=10):
        """スキル軸での補間軌道生成"""
        trajectories = []
        with torch.no_grad():
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                z_skill_interp = (1 - alpha) * z_skill_start + alpha * z_skill_end
                traj = self.decode(z_style, z_skill_interp)
                trajectories.append(traj)
        return torch.stack(trajectories)

    def update_epoch(self, epoch: int, max_epoch: int):
        """エポックを更新(β-annealing)"""
        if not hasattr(self,'initial_beta'):
            self.initial_beta = self.beta

        self.beta = min(self.initial_beta, 1e-5 + (epoch / max_epoch) * self.initial_beta)

    def save_model(self, filepath: str):
        torch.save(self.state_dict(), filepath)


# -----------------------------------------------
# 4. 損失関数
# -----------------------------------------------
def loss_function(reconstructed_x, x, mu_style, logvar_style, mu_skill, logvar_skill, beta):
    # 再構成損失 (MSE)
    recon_loss = nn.functional.mse_loss(reconstructed_x, x, reduction='sum')

    # KLダイバージェンス損失
    kl_style = -0.5 * torch.sum(1 + logvar_style - mu_style.pow(2) - logvar_style.exp())
    kl_skill = -0.5 * torch.sum(1 + logvar_skill - mu_skill.pow(2) - logvar_skill.exp())

    # 合計損失
    total_loss = recon_loss + beta * (kl_style + kl_skill)

    return total_loss, recon_loss, kl_style, kl_skill