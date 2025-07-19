import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SinusoidalPositionalEmbedding(nn.Module):
    """
    時間ステップtをベクトルに変換するためのSinusoidal Positional Embedding
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time_steps: torch.Tensor) -> torch.Tensor:
        """
        :param time_steps:[Batch, ]の時間ステップ
        :return: [Batch, dim]の埋め込みベクトル
        """
        device = time_steps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time_steps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CrossAttention(nn.Module):
    """
    軌道データ(Query)と個人特性(Context)を結びつけるためのCross-Attention層。
    """

    def __init__(self, query_dim: int, context_dim: int, heads: int = 8, dim_head: int = 32):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** - 0.5
        self.heads = heads

        # 入力(x)からQueryを、条件(context)からKeyとValueを生成するための線形層
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch, seq_len, query_dim]の軌道特徴量
        :param context: [Batch, context_dim]の個人特性ベクトル
        :return: [batch, seq_len, query_dim] アテンション適用後の軌道特徴量
        """
        batch_size, seq_len, _ = x.shape

        # 各線形層でq, k, vを計算
        q = self.to_q(x)
        k = self.to_k(context.unsqueeze(1))
        v = self.to_v(context.unsqueeze(1))

        # Multihead Attention のためにヘッド数で分割、次元を並べ替え
        q = q.view(batch_size, seq_len, self.heads, -1).transpose(1, 2)
        k = k.view(batch_size, 1, self.heads, -1).transpose(1, 2)
        v = v.view(batch_size, 1, self.heads, -1).transpose(1, 2)

        # QとKの内積でAttentionを計算
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = F.softmax(attention_scores, dim=-1)

        # AttentionをVに適用
        out = torch.matmul(attention, v)

        # 元の形状に戻す
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)

        return self.to_out(out)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 time_embed_dim: int,
                 condition_dim: int,
                 kernel_size: int = 3,
                 num_group: int = 8):
        super().__init__()

        # 1つ目の畳み込み層
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.norm1 = nn.GroupNorm(num_group, out_channels)

        # 2つ目の畳み込み層
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.norm2 = nn.GroupNorm(num_group, out_channels)

        # 時間埋め込みの次元変換層
        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        # 条件付けのためのCross-Attention層
        self.cross_attention = CrossAttention(out_channels, condition_dim)

        # 入力と出力のチャンネル数が違う場合、次元を合わせるためのスキップ接続
        self.skip_connection = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor,
                time_embed: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """
        :param x: [Batch, in_channels, seq_len]
        :param time_embed: [Batch, time_emb_dim]
        :param condition: [Batch, condition_dim]
        :return: [Batch, out_channels, seq_len]
        """
        # スキップ接続
        residual = self.skip_connection(x)

        # 1.畳み込み -> 正規化 -> 時間埋め込み -> 活性化
        x = self.conv1(x)
        x = self.norm1(x)
        time_embed = self.time_proj(time_embed)[:, :, None]  # (B, C) -> (B, C, 1)に拡張
        x = x + time_embed
        x = F.silu(x)

        # 2. 畳み込み -> 正規化
        x = self.conv2(x)
        x = self.norm2(x)

        # 3. Cross-Attentionによる条件付け
        x_transposed = x.transpose(1, 2)  # Attentionは[batch, seq_len, channel]を受け付けるので、次元を入れ替える
        x_attended = self.cross_attention(x_transposed, condition)
        x = x_attended.transpose(1, 2)  # 元の形状に戻す

        # 4. Residual Connection
        return x + residual


class UNet1D(nn.Module):
    """
    軌道データのノイズ除去を行うための、条件付き1D U-Netモデル。
    """

    def __init__(self,
                 input_dim: int = 2,
                 condition_dim: int = 5,
                 time_embed_dim: int = 128,
                 base_channels: int = 64):
        super().__init__()

        self.channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.num_resolutions = len(self.channels)

        # 時間埋め込み
        self.time_embedding = SinusoidalPositionalEmbedding(time_embed_dim)

        # 入力データを最初のチャンネル数に変換
        self.input_proj = nn.Conv1d(input_dim, base_channels, kernel_size=1)

        # Encoder(Down-sampling)
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        in_c = base_channels
        for i, out_c in enumerate(self.channels):
            self.encoder_blocks.append(ResidualBlock(in_c, out_c, time_embed_dim, condition_dim))
            # 最後以外はダウンサンプリング
            if i < self.num_resolutions - 1:
                self.downsample_layers.append(nn.Conv1d(out_c, out_c, kernel_size=3, stride=2, padding=1))
            else:
                self.downsample_layers.append(nn.Identity())
            in_c = out_c

        # Bottleneck
        self.bottleneck = ResidualBlock(self.channels[-1], self.channels[-1], time_embed_dim, condition_dim)

        # Decoder(Up-sampling)
        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        # デコーダーは逆順（最後の3つのチャンネルを逆順に）
        decoder_channels = list(reversed(self.channels[:-1]))  # [256, 128, 64]
        in_c = self.channels[-1]  # 512

        for i, out_c in enumerate(decoder_channels):
            # アップサンプリング層
            self.upsample_layers.append(
                nn.ConvTranspose1d(in_c, out_c, kernel_size=3, stride=2,
                                   padding=1, output_padding=1)
            )
            # スキップ接続後のチャンネル数: アップサンプル後 + スキップ接続
            concat_channels = out_c + out_c  # 両方とも同じチャンネル数
            self.decoder_blocks.append(ResidualBlock(concat_channels, out_c, time_embed_dim, condition_dim))
            in_c = out_c

        # 出力層
        self.output_proj = nn.Conv1d(base_channels, input_dim, kernel_size=1)

    def forward(self,
                x: torch.Tensor,
                time_steps: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch, 2, seq_len] ノイズ軌道
        :param time_steps: [batch, ] 時間ステップ
        :param condition: [batch, condition_dim] 個人特性ベクトル
        :return: [batch, 2, seq_len] 予測されたノイズ
        """
        time_embed = self.time_embedding(time_steps)
        x = self.input_proj(x)

        # エンコーダーからのスキップ接続を保存（ボトルネック以外）
        encoder_features = []

        # Encoder
        for i, (block, downsample) in enumerate(zip(self.encoder_blocks, self.downsample_layers)):
            x = block(x, time_embed, condition)
            # ボトルネック以外のみスキップ接続として保存
            if i < self.num_resolutions - 1:
                encoder_features.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.bottleneck(x, time_embed, condition)

        # Decoder（スキップ接続を逆順で使用）
        for i, (upsample, block) in enumerate(zip(self.upsample_layers, self.decoder_blocks)):
            # アップサンプリング
            x = upsample(x)

            # スキップ接続（逆順で取得）
            skip = encoder_features.pop()  # 最後から取り出す

            # サイズ調整
            if x.size(-1) != skip.size(-1):
                x = F.interpolate(x, size=skip.size(-1), mode='linear', align_corners=False)

            # チャンネル結合
            x = torch.cat([x, skip], dim=1)

            # ResidualBlock
            x = block(x, time_embed, condition)

        return self.output_proj(x)


if __name__ == '__main__':
    # ダミーデータでモデルの動作確認
    batch_size = 4
    seq_len = 101
    condition_dim = 5

    # モデルのインスタンス化
    model = UNet1D(condition_dim=condition_dim)

    # ダミー入力テンソルを作成
    dummy_trajectory = torch.randn(batch_size, 2, seq_len)
    dummy_timesteps = torch.randint(0, 1000, (batch_size,))
    dummy_condition = torch.randn(batch_size, condition_dim)

    print("Testing model...")
    try:
        # モデルにデータを入力
        predicted_noise = model(dummy_trajectory, dummy_timesteps, dummy_condition)

        # 出力形状を確認
        print(f"Input trajectory shape: {dummy_trajectory.shape}")
        print(f"Predicted noise shape: {predicted_noise.shape}")

        # 形状が一致していればOK
        assert dummy_trajectory.shape == predicted_noise.shape
        print("\n✅ Model forward pass successful!")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback

        traceback.print_exc()