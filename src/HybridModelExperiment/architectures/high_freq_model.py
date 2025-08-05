# CLAUDE_ADDED
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from HybridModelExperiment.modules.embedding import TimeEmbedding
from HybridModelExperiment.modules.blocks import ResidualBlock
from HybridModelExperiment.modules.embedding import SinusoidalPositionalEmbedding


class UNet1DForTrajectory(nn.Module):
    def __init__(
            self,
            input_dim: int = 2,
            condition_dim: int = 5,
            time_embed_dim: int = 128,
            base_channels: int = 32,  # 小さくする
            num_resolutions: int = 3  # 減らす
    ):
        super().__init__()

        self.channels = [base_channels * (2 ** i) for i in range(num_resolutions)]
        self.num_resolutions = num_resolutions

        # 時間埋め込み
        self.time_embedding = SinusoidalPositionalEmbedding(time_embed_dim)

        # 入力を適切な次元に変換（2次元座標を保持）
        self.input_proj = nn.Conv1d(input_dim, base_channels, kernel_size=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        in_c = base_channels
        for i, out_c in enumerate(self.channels):
            self.encoder_blocks.append(
                ResidualBlock(
                    in_c,out_c,time_embed_dim,condition_dim,use_attention=(1 >= num_resolutions - 2)
                )
            )

            # 最後以外はダウンサンプリング
            if i < self.num_resolutions - 1:
                self.downsample_layers.append(nn.Conv1d(out_c, out_c, kernel_size=3, stride=2, padding=1))
            else:
                self.downsample_layers.append(nn.Identity())
            in_c = out_c

        # Bottleneck
        self.bottleneck = ResidualBlock(
            self.channels[-1], self.channels[-1],
            time_embed_dim, condition_dim,
            use_attention=True
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        decoder_channels = list(reversed(self.channels[:-1]))
        in_c = self.channels[-1]

        for i, out_c in enumerate(decoder_channels):
            self.upsample_layers.append(
                nn.ConvTranspose1d(in_c, out_c, kernel_size=3, stride=2, padding=1,output_padding=1)
            )
            # スキップ接続を考慮: encoder特徴量はout_cチャンネル、アップサンプル後もout_cチャンネル
            self.decoder_blocks.append(
                ResidualBlock(
                    out_c * 2, out_c, time_embed_dim, condition_dim,
                    use_attention=(i == 0)
                )
            )
            in_c = out_c

        # 出力層
        self.output_proj = nn.Conv1d(base_channels, input_dim, kernel_size=1)


    def forward(
            self,
            x: torch.Tensor,
            time_steps: torch.Tensor,
            condition: torch.Tensor
            ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, 2] 高周波成分
            time_steps: [batch] 時間ステップ
            condition: [batch, condition_dim] 条件

        Returns:
            [batch, seq_len, 2] 予測ノイズ
        """
        batch_size, seq_len, _ = x.shape

        x = x.permute(0, 2, 1)

        # 時間埋め込み
        time_embed = self.time_embedding(time_steps)

        # 入力プロジェクション
        x = self.input_proj(x)  # [batch, seq_len, base_channels]

        # Encoder
        encoder_features = []
        for i, (block, downsample) in enumerate(zip(self.encoder_blocks,self.downsample_layers)):
            x = block(x, time_embed, condition)
            
            # ボトルネック以外のみスキップ接続として保存
            if i < self.num_resolutions - 1:
                encoder_features.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.bottleneck(x, time_embed, condition)

        # Decoder
        for i, (upsample, block) in enumerate(
                zip(self.upsample_layers, self.decoder_blocks)
        ):
            # アップサンプリング
            x = upsample(x)

            # スキップ接続
            skip = encoder_features.pop()

            # サイズ調整
            if x.size(-1) != skip.size(-1):
                x = F.interpolate(x, size=skip.size(-1), mode='linear', align_corners=False)

            # 結合
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_embed, condition)

        # 出力プロジェクション
        output = self.output_proj(x)
        output = output.permute(0, 2, 1)

        return output

class UNet1D(nn.Module):
    def __init__(self,
                 input_dim: int = 2,
                 condition_dim: int = 5,
                 time_embed_dim: int = 128,
                 base_channels: int = 64,
                 num_resolutions = 4):
        super().__init__()

        self.channels = [base_channels * (2 ** i) for i in range(num_resolutions)]
        self.num_resolutions = num_resolutions

        # 時間埋め込み
        self.time_embedding = SinusoidalPositionalEmbedding(time_embed_dim)

        # 入力データを最初のチャンネル数に変換
        self.input_proj = nn.Conv1d(input_dim, base_channels, kernel_size=1)

        # Encoder (Down Sampling)
        self.encoder_block = nn.ModuleList()
        self.downsample_layer = nn.ModuleList()

        in_c = base_channels
        for i, out_c in enumerate(self.channels):
            self.encoder_block.append(ResidualBlock(in_c, out_c, time_embed_dim, condition_dim))

            # 最後以外はダウンサンプリング
            if i < self.num_resolutions - 1:
                self.downsample_layer.append(nn.Conv1d(out_c, out_c, kernel_size=3, stride=2, padding=1))
            else:
                self.downsample_layer.append(nn.Identity())
            in_c = out_c

        # Bottleneck
        self.bottleneck = ResidualBlock(self.channels[-1], self.channels[-1], time_embed_dim, condition_dim)

        # Decoder (Up Sampling)
        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        decoder_channels = list(reversed(self.channels[:-1]))
        in_c = self.channels[-1]

        for i, out_c in enumerate(decoder_channels):
            # アップサンプリング層
            self.upsample_layers.append(nn.ConvTranspose1d(in_c, out_c, kernel_size=3, stride=2, padding=1, output_padding=1))
            # スキップ接続
            concat_channels = out_c + out_c
            self.decoder_blocks.append(ResidualBlock(concat_channels, out_c, time_embed_dim, condition_dim))
            in_c = out_c

        # 出力層
        self.output_proj = nn.Conv1d(base_channels, input_dim, kernel_size=1)

    def forward(self,
                x: torch.Tensor,
                time_steps: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch, seq_len, ind_dim] ノイズ軌道
        :param time_steps: [batch, ] 時間ステップ
        :param condition: [batch, condition_dim] 個人特性ベクトル
        :return: [batch, in_dim, seq_len] 予測されたノイズ
        """
        x = x.permute(0, 2, 1)

        time_embed = self.time_embedding(time_steps)
        x = self.input_proj(x)

        # エンコーダからの接続を保存
        encoder_features = []

        # Encoder
        for i, (block, downsample) in enumerate(zip(self.encoder_block, self.downsample_layer)):
            x = block(x, time_embed, condition)
            # ボトルネック以外のみスキップ接続として保存
            if i < self.num_resolutions - 1:
                encoder_features.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.bottleneck(x, time_embed, condition)

        # Decoder
        for i, (upsample, block) in enumerate(zip(self.upsample_layers, self.decoder_blocks)):
            # アップサンプリング
            x = upsample(x)

            # スキップ接続
            skip = encoder_features.pop()

            # サイズ調整
            if x.size(-1) != skip.size(-1):
                x = F.interpolate(x, size=skip.size(-1), mode='linear', align_corners=False)

            # チャンネルを結合
            x = torch.cat([x, skip], dim=1)

            # ResidualBlock
            x = block(x, time_embed, condition)

        output_permuted = self.output_proj(x)
        output = output_permuted.permute(0, 2, 1)

        return output


class SimpleDiffusionMLP(nn.Module):
    """
    高周波成分用のシンプルな拡散モデル（MLPベース）
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        condition_dim: int = 3,
        time_embedding_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 入力の特徴次元数
            hidden_dim: 隠れ層の次元数
            num_layers: MLPのレイヤー数
            condition_dim: 条件ベクトルの次元数
            time_embedding_dim: 時刻埋め込みの次元数
            dropout: ドロップアウト率
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim

        # 時刻埋め込み
        self.time_embedding = TimeEmbedding(time_embedding_dim, hidden_dim)

        # 条件ベクトルの処理
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)

        # 入力プロジェクション
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # MLPレイヤー
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout)
                )
            )

        # 出力層
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        # スケール・シフト層（時刻と条件の情報を統合）
        self.scale_shift_layers = nn.ModuleList()
        for i in range(num_layers):
            self.scale_shift_layers.append(
                nn.Linear(hidden_dim * 2, hidden_dim * 2)  # scale and shift
            )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        ノイズ予測を行う

        Args:
            x: ノイズが加えられたデータ [batch_size, sequence_length, input_dim]
            timesteps: 時刻ステップ [batch_size]
            condition: 条件ベクトル [batch_size, condition_dim]

        Returns:
            predicted_noise: 予測されたノイズ [batch_size, sequence_length, input_dim]
        """
        batch_size, seq_len, _ = x.shape

        # 時刻埋め込みを取得
        time_emb = self.time_embedding(timesteps)  # [batch_size, hidden_dim]

        # 条件ベクトルを処理
        cond_emb = self.condition_proj(condition)  # [batch_size, hidden_dim]

        # 時刻と条件を結合
        time_cond_emb = torch.cat([time_emb, cond_emb], dim=-1)  # [batch_size, hidden_dim * 2]

        # 入力を平坦化してプロジェクション
        x_flat = x.view(batch_size * seq_len, -1)  # [batch_size * seq_len, input_dim]
        h = self.input_proj(x_flat)  # [batch_size * seq_len, hidden_dim]

        # MLPレイヤーを通す
        for i, layer in enumerate(self.layers):
            # スケール・シフト変換を計算
            scale_shift = self.scale_shift_layers[i](time_cond_emb)  # [batch_size, hidden_dim * 2]
            scale, shift = scale_shift.chunk(2, dim=-1)  # each [batch_size, hidden_dim]

            # バッチ次元を拡張して適用
            scale = scale.unsqueeze(1).expand(-1, seq_len, -1).contiguous().view(-1, self.hidden_dim)
            shift = shift.unsqueeze(1).expand(-1, seq_len, -1).contiguous().view(-1, self.hidden_dim)

            # レイヤーを通す
            h = layer(h)

            # スケール・シフト変換を適用
            h = h * (1 + scale) + shift

        # 出力プロジェクション
        output = self.output_proj(h)  # [batch_size * seq_len, input_dim]

        # 元の形状に戻す
        output = output.view(batch_size, seq_len, -1)

        return output


class HighFreqDiffusion:
    """
    高周波成分用の拡散プロセス管理クラス
    """

    def __init__(self,
                 num_timesteps: int = 100,
                 beta_start: float = 0.00001,
                 beta_end: float = 0.001,
                 schedule: str = 'cosine'
                 ):
        """
        Args:
            num_timesteps: 拡散ステップ数
            beta_start: βの開始値
            beta_end: βの終了値
        """
        self.num_timesteps = num_timesteps

        if schedule == 'cosine':
            s = 0.008
            steps = num_timesteps + 1
            t = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((t / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
            
            # α = 1 - β
            self.alphas = 1.0 - self.betas
            
            # α の累積積
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
            
            # 逆プロセス用の分散
            self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        else:
            # β スケジュール（線形）
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

            # α = 1 - β
            self.alphas = 1.0 - self.betas

            # α の累積積
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

            # 逆プロセス用の分散
            self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        データにノイズを追加

        Args:
            x_0: 元データ [batch_size, sequence_length, input_dim]
            t: 時刻ステップ [batch_size]
            noise: 追加するノイズ（オプション）

        Returns:
            x_t: ノイズが追加されたデータ
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # デバイスを同じにする
        device = x_0.device
        self.alphas_cumprod = self.alphas_cumprod.to(device)

        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt()

        # バッチ次元を調整
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        ランダムな時刻ステップをサンプリング

        Args:
            batch_size: バッチサイズ
            device: デバイス

        Returns:
            timesteps: サンプリングされた時刻ステップ [batch_size]
        """
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

    @torch.no_grad()
    def denoise_step(
        self,
        model: UNet1DForTrajectory,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        デノイジングステップ

        Args:
            model: ノイズ予測モデル
            x_t: 現在のノイズ付きデータ
            t: 現在の時刻ステップ
            condition: 条件ベクトル

        Returns:
            x_prev: 前の時刻のデータ
        """
        # デバイスを同じにする
        device = x_t.device
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.betas = self.betas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)

        # ノイズを予測
        predicted_noise = model(x_t, t, condition)

        # デノイジング計算
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1)

        # 平均を計算
        x_0_pred = (x_t - predicted_noise * (1 - alpha_cumprod_t).sqrt()) / alpha_cumprod_t.sqrt()

        if t[0] > 0:
            # t > 0 の場合、ノイズを追加
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1)
            noise = torch.randn_like(x_t)
            x_prev = (x_t - beta_t * predicted_noise / (1 - alpha_cumprod_t).sqrt()) / alpha_t.sqrt()
            x_prev = x_prev + posterior_variance_t.sqrt() * noise
        else:
            # t = 0 の場合、ノイズなし
            x_prev = x_0_pred

        return x_prev

    @torch.no_grad()
    def generate(self, model, shape: tuple, condition: torch.Tensor, device: torch.device) -> torch.Tensor:
        batch_size = shape[0]

        # より小さな初期ノイズ（標準正規分布）
        x = torch.randn(shape, device=device)

        # プログレスバー（オプション）
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.denoise_step(model, x, t, condition)

            # ガイダンス（オプション：より安定した生成のため）
            if i > 0:
                # 予測されたx_0に向かって少し引き寄せる
                alpha_t = self.alphas[t[0]]
                x = x * 0.95 + torch.randn_like(x) * 0.05 * torch.sqrt(1 - alpha_t)

        return x