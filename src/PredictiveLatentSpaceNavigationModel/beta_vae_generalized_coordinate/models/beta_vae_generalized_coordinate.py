import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_style, subject_ids):
        batch_size = z_style.shape[0]

        # 被験者IDを数値化
        unique_subjects = list(set(subject_ids))
        if len(unique_subjects) < 2:
            return torch.tensor(0.0, device=z_style.device)

        subject_to_idx = {subj: i for i, subj in enumerate(unique_subjects)}
        labels = torch.tensor([subject_to_idx[subj] for subj in subject_ids], device=z_style.device)

        # L2正規化
        z_style_norm = F.normalize(z_style, p=2, dim=1)

        # 類似度行列
        sim_matrix = torch.mm(z_style_norm, z_style_norm.t()) / self.temperature

        # 正例・負例マスク
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask = mask - torch.eye(batch_size, device=z_style.device)  # 自分自身を除外

        # InfoNCE損失
        exp_sim = torch.exp(sim_matrix)
        sum_exp = torch.sum(exp_sim * (1 - torch.eye(batch_size, device=z_style.device)), dim=1)
        pos_sum = torch.sum(exp_sim * mask, dim=1)

        # ゼロ除算回避
        pos_sum = torch.clamp(pos_sum, min=1e-8)

        loss = -torch.log(pos_sum / (sum_exp + 1e-8))
        return loss.mean()

class Encoder(nn.Module):
    """軌道シーケンスを受け取り、潜在変数z_styleとz_skillを生成する"""
    def __init__(self, input_dim: int, hidden_dim:int, style_latent_dim: int, skill_latent_dim, n_layers: int=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

        # スタイル用のヘッド
        self.fc_mu_style = nn.Linear(hidden_dim, style_latent_dim)
        self.fc_logvar_style = nn.Linear(hidden_dim, style_latent_dim)

        # スキル用のヘッド
        self.fc_mu_skill = nn.Linear(hidden_dim, skill_latent_dim)
        self.fc_logvar_skill = nn.Linear(hidden_dim, skill_latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch, seq_len, input_dim]
        :return: (mu_style, logvar_style, mu_skill, logvar_skill)
        """
        _, (hidden, _) = self.lstm(x)

        # 最終層の隠れ状態を使用 [batch, hidden_dim]
        last_hidden = hidden[-1]

        # スタイル潜在変数のパラメータ
        mu_style = self.fc_mu_style(last_hidden)
        logvar_style = self.fc_logvar_style(last_hidden)

        # スキル潜在変数のパラメータ
        mu_skill = self.fc_mu_skill(last_hidden)
        logvar_skill = self.fc_logvar_skill(last_hidden)

        return mu_style, logvar_style, mu_skill, logvar_skill


class SimpleDecoder(nn.Module):
    """自己回帰を避けたシンプルなDecoder"""

    def __init__(self, output_dim, hidden_dim, style_latent_dim, skill_latent_dim, seq_len, n_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim

        latent_dim = style_latent_dim + skill_latent_dim

        # 潜在変数を全系列に展開するためのMLP
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, seq_len * output_dim)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, z_style, z_skill):
        """直接的な系列生成"""
        batch_size = z_style.size(0)
        z = torch.cat([z_style, z_skill], dim=1)

        # MLPで系列全体を生成
        x = self.activation(self.fc1(z))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        # [batch, seq_len * output_dim] -> [batch, seq_len, output_dim]
        output = x.view(batch_size, self.seq_len, self.output_dim)

        return output

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


class BetaVAEGeneralizedCoordinate(nn.Module):
    """対照学習付きβ-VAEモデル"""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            style_latent_dim: int,
            skill_latent_dim: int,
            seq_len: int,
            n_layers: int = 2,
            beta: float = 1.0,
            contrastive_weight: float = 5.0,
            use_triplet:bool = False
    ):
        super().__init__()
        self.beta = beta
        self.contrastive_weight = contrastive_weight

        self.encoder = Encoder(input_dim, hidden_dim, style_latent_dim, skill_latent_dim, n_layers)
        self.decoder = SimpleDecoder(input_dim, hidden_dim, style_latent_dim, skill_latent_dim, seq_len, n_layers)

        # 対照学習
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        self.use_triplete = use_triplet

    def reparameterize(self, mu, logvar):
        """再パラメータ化トリック"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """エンコードのみを実行(評価用)"""
        mu_style, logvar_style, mu_skill, logvar_skill = self.encoder(x)
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

    def forward(self, x, subject_ids=None):
        # エンコード
        mu_style, logvar_style, mu_skill, logvar_skill = self.encoder(x)

        # サンプリング
        z_style = self.reparameterize(mu_style, logvar_style)
        z_skill = self.reparameterize(mu_skill, logvar_skill)

        # デコード
        reconstructed_x = self.decoder(z_style, z_skill)

        # 基本損失
        total_loss, reconstruct_loss, kl_style, kl_skill = self.loss_function(
            reconstructed_x, x, mu_style, logvar_style, mu_skill, logvar_skill, self.beta
        )

        # 対照学習損失（スタイル潜在変数のみ使用）
        contrastive_loss_val = torch.tensor(0.0, device=x.device)
        if subject_ids is not None and len(set(subject_ids)) > 1:
            contrastive_loss_val = self.contrastive_loss(z_style, subject_ids)
            total_loss += self.contrastive_weight * contrastive_loss_val

        return {
            'total_loss': total_loss,
            'reconstruct_loss': reconstruct_loss,
            'kl_style': kl_style,
            'kl_skill': kl_skill,
            'contrastive_loss': contrastive_loss_val,
            'reconstructed_x': reconstructed_x,
            'z_style': z_style,
            'z_skill': z_skill
        }

    def loss_function(self, reconstructed_x, x, mu_style, logvar_style, mu_skill, logvar_skill, beta):
        """安全なKL損失計算"""
        # 再構成損失 (MSE)
        recon_loss = F.mse_loss(reconstructed_x, x, reduction='mean')

        # KLダイバージェンス損失（安全版）
        kl_style = -0.5 * torch.mean(torch.sum(1 + logvar_style - mu_style.pow(2) - logvar_style.exp(), dim=1))
        kl_skill = -0.5 * torch.mean(torch.sum(1 + logvar_skill - mu_skill.pow(2) - logvar_skill.exp(), dim=1))

        # KL損失を非負に制限
        kl_style = torch.clamp(kl_style, min=0.0)
        kl_skill = torch.clamp(kl_skill, min=0.0)

        # 合計損失
        total_loss = recon_loss + beta * (kl_style + kl_skill)

        return total_loss, recon_loss, kl_style, kl_skill

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
        if not hasattr(self, 'initial_beta'):
            self.initial_beta = self.beta
            self.initial_contrastive_weight = self.contrastive_weight

        # β値は徐々に増加（KL損失を後半で強化）
        self.beta = min(self.initial_beta, 1e-5 + (epoch / max_epoch) * self.initial_beta)

        # 対照学習は早期から強化
        if epoch > max_epoch * 0.1:  # 10%経過後から対照学習開始
            self.contrastive_weight = self.initial_contrastive_weight
        else:
            self.contrastive_weight = 0.0

    def save_model(self, filepath: str):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath: str, device=None):
        """モデル読み込み"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        state_dict = torch.load(filepath, map_location=device)
        self.load_state_dict(state_dict)
        self.to(device)


# # 推奨設定
# def create_lstm_vae_with_contrastive(config):
#     """推奨設定でモデル作成"""
#     return BetaVAEWithContrastive(
#         input_dim=config['model']['input_dim'],
#         hidden_dim=config['model']['hidden_dim'],
#         style_latent_dim=32,  # スタイル次元
#         skill_latent_dim=16,  # スキル次元
#         seq_len=config['model']['seq_len'],
#         n_layers=2,
#         beta=0.1,  # 軽いKL正則化
#         contrastive_weight=10.0  # 強い対照学習
#     )