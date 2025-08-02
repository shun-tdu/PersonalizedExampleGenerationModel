# CLAUDE_ADDED
import math

import torch
import torch.nn as nn
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch, seq_len, embedding_dim]
        :return: [batch, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1), :]



class LowFreqTransformer(nn.Module):
    def __init__(self,
                 input_dim: int = 2,
                 dim_head: int = 32,
                 heads: int = 8,
                 num_layers: int = 4,
                 condition_dim:int = 5,
                 dropout: float = 0.1,
                 max_len: int = 256
                 ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        # 入力層: 入力次元を潜在次元に埋め込み
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
            nn.ReLU()
        )

        # 条件ベクトルの次元を潜在次元に埋め込み
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, self.inner_dim),
            nn.ReLU(),
            nn.Linear(self.inner_dim, self.inner_dim)
        )

        # 正弦波位置エンコーディング
        self.pos_encoder = PositionalEncoding(self.inner_dim, max_len)

        # 4. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.inner_dim,
            nhead=heads,
            dim_feedforward=self.inner_dim*4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # 5.出力層 潜在次元を入力次元に戻す
        self.output_proj = nn.Sequential(
            nn.Linear(self.inner_dim, self.inner_dim // 2),
            nn.ReLU(),
            nn.Linear(self.inner_dim//2, input_dim)
        )

        # 開始トークン(ゴール位置の影響を受けやすくする)
        self.start_token_proj = nn.Linear(condition_dim, input_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        学習時のフォワードパス
        :param x: [batch, seq_len, in_dim]
        :param condition: [batch, 5]
        :return: [batch, seq_len-1, 2]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # 条件埋め込み(メモリとして使用)
        memory = self.condition_proj(condition).unsqueeze(1) # [B, 1, inner_dim]

        # Scheduled sampling: 確率的に ground truthか生成を使う
        use_teacher_forcing = self.training and torch.rand(1).item() > 0.1

        if use_teacher_forcing:
            # Teacher forcing
            # 入力を1つシフト(最初に開始トークン)
            start_pos = self.start_token_proj(condition).unsqueeze(1)
            shifted_x = torch.cat([start_pos, x[:,:-1]], dim=1)

            # 埋め込み
            tgt = self.input_proj(shifted_x)
            tgt = self.pos_encoder(tgt)

            # Causal mask
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(device)

            # Decode
            output = self.transformer(tgt, memory, tgt_mask=tgt_mask)
            return self.output_proj(output)
        else:
            # 自己回帰生成
            return self._autoregressive_forward(x, condition, seq_len)

    def _autoregressive_forward(self, x: torch.Tensor, condition: torch.Tensor, seq_len: int):
        """学習時の自己回帰生成"""
        batch_size = condition.shape[0]
        device = condition.device

        # 条件埋め込み
        memory = self.condition_proj(condition).unsqueeze(1)

        # 開始位置
        start_pos = self.start_token_proj(condition)
        generated = [start_pos]

        # 各ステップを生成
        for i in range(seq_len - 1):
            # これまでの生成結果
            current_seq = torch.stack(generated, dim=1)

            # 埋め込みと位置エンコーディング
            tgt = self.input_proj(current_seq)
            tgt = self.pos_encoder(tgt)

            # マスク
            current_len = len(generated)
            tgt_mask = self.generate_square_subsequent_mask(current_len).to(device)

            # デコード
            output = self.transformer(tgt, memory, tgt_mask=tgt_mask)
            next_pos = self.output_proj(output[:, -1])

            # 50%の確率でground truthを使用
            if self.training and x is not None and i < seq_len - 1 and torch.rand(1).item() < 0.5:
                next_pos = x[:, i]

            generated.append(next_pos)

        return torch.stack(generated[1: ], dim=1) # 開始トークンを除く

    def generate(self,condition: torch.Tensor,seq_len: int) -> torch.Tensor:

        self.eval()
        with torch.no_grad():
            return self._autoregressive_forward(None, condition, seq_len)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        未来の情報を隠すためのAttention Maskを生成
        """
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
