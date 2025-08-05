import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CrossAttention


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 time_embed_dim: int,
                 condition_dim: int,
                 kernel_size:int = 3,
                 num_group:int = 8,
                 use_attention: bool = True):
        super().__init__()

        self.use_attention = use_attention

        # 1つ目の畳み込み層
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.norm1 = nn.GroupNorm(num_group, out_channels)

        # 2つ目の畳み込み層
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.norm2 = nn.GroupNorm(num_group, out_channels)

        # 時間埋め込みの次元変換層
        # self.time_proj = nn.Linear(time_embed_dim, out_channels)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, out_channels),
            nn.SiLU()
        )

        if use_attention:
            # 条件付けのためのCross-Attention層
            self.norm_attn = nn.GroupNorm(num_group, out_channels)
            self.cross_attention = CrossAttention(out_channels, condition_dim)

         # 入力と出力のチャンネル数を合わせるための、スキップ接続
        self.skip_connection = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self,
                x: torch.Tensor,
                time_embed: torch.Tensor,
                condition:torch.Tensor) -> torch.Tensor:
        """
        :param x: [Batch, in_channels, seq_len]     入力テンソル
        :param time_embed: [Batch, time_emb_dim]    拡散ステップの埋め込み
        :param condition: [Batch, condition_dim]    条件の埋め込み
        :return: [Batch, out_channels, seq_len]     ノイズの推測値
        """
        # スキップ接続
        residual = self.skip_connection(x)

        # 1.畳み込み -> 正規化 -> 時間埋め込み -> 活性化
        x = self.conv1(x)
        x = self.norm1(x)
        time_embed = self.time_proj(time_embed)[:, :, None] # (B, C) -> (B, C, 1)に拡張
        x = x + time_embed
        x = F.silu(x)

        # 2.畳み込み -> 正規化
        x = self.conv2(x)
        x = self.norm2(x)

        if self.use_attention:
            # 3. Cross Attentionによる条件付け
            x_transposed = x.transpose(1, 2)
            x_transposed = self.norm_attn(x_transposed.transpose(1, 2)).transpose(1, 2)
            x_attended = self.cross_attention(x_transposed, condition)
            x = x + x_attended.transpose(1, 2)

        # 4. Residual Connection
        return x + residual

class TransformerBlock(nn.Module):
    """Transformerベースのブロック"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            time_embed_dim: int,
            condition_dim: int
    ):
        super().__init__()

        # 入力と出力のチャンネル数を保存
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Self-attention（入力チャンネル数で正規化）
        self.norm1 = nn.LayerNorm(in_channels)
        self.self_attn = nn.MultiheadAttention(
            in_channels,
            num_heads=4,
            batch_first=True
        )

        # Cross-attention with condition（入力チャンネル数で正規化）
        self.norm2 = nn.LayerNorm(in_channels)
        self.cross_attn = CrossAttention(in_channels, condition_dim)

        # チャンネル数変換層
        self.channel_proj = nn.Linear(in_channels, out_channels)

        # Feedforward（出力チャンネル数で正規化）
        self.norm3 = nn.LayerNorm(out_channels)
        self.ff = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Linear(out_channels * 4, out_channels)
        )

        # Time embedding
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, out_channels),
            nn.SiLU()
        )

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Linear(in_channels, out_channels)
        else:
            self.skip = nn.Identity()

    def forward(self, x, time_embed, condition):
        # Skip connection
        residual = self.skip(x)

        # Self-attention（入力チャンネル数のまま処理）
        x_norm = self.norm1(x)
        x_attn, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + x_attn  # 残差接続

        # Cross-attention（入力チャンネル数のまま処理）
        x_norm = self.norm2(x)
        x_cross = self.cross_attn(x_norm, condition)
        x = x + x_cross  # 残差接続

        # チャンネル数を変換
        x = self.channel_proj(x)

        # Time modulation（出力チャンネル数で処理）
        time_scale = self.time_proj(time_embed).unsqueeze(1)
        x = x * (1 + time_scale)

        # Feedforward（出力チャンネル数で処理）
        x_norm = self.norm3(x)
        x_ff = self.ff(x_norm)
        x = x + x_ff  # 残差接続

        # 残差接続を安全に適用
        if x.shape == residual.shape:
            return x + residual
        else:
            return x