# CLAUDE_ADDED
import math

import torch
import torch.nn as nn
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 max_len: int = 100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch, seq_len, embedding_dim]
        :return: [batch, seq_len, embedding_dim]
        """
        return x + self.pe[:,x.size(0),:]



class LowFreqTransformer(nn.Module):
    def __init__(self,
                 input_dim: int = 2,
                 dim_head: int = 32,
                 heads: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 512,
                 condition_dim:int = 5,
                 dropout: float = 0.1
                 ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        # 1.入力層: 入力次元を潜在次元に変換
        self.input_proj = nn.Linear(input_dim, self.inner_dim)

        # 2.正弦波位置エンコーディング
        self.pos_encoder = PositionalEncoding(self.inner_dim, dropout)

        # 3.条件ベクトルの次元を潜在次元に変換
        self.condition_proj = nn.Linear(condition_dim, self.inner_dim)

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.inner_dim,
            nhead=heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # 5.出力層 潜在次元を入力次元に戻す
        self.output_proj = nn.Linear(self.inner_dim, input_dim)

        # 開始トークン(形状を(1, 1, inner_dim)に変更)
        self.start_token = nn.Parameter(torch.randn(1, 1, self.inner_dim))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        学習時のフォワードパス
        :param x: [batch, seq_len, in_dim]
        :param condition: [batch, 5]
        :return: [batch, seq_len-1, 2]
        """
        batch_size = x.shape[0]
        device = x.device

        src = x[:, :-1, :]

        # 1. 入力を潜在次元に変換
        src_embedded = self.input_proj(src)

        # 2. 条件を潜在次元に変換し、系列の各ステップに加算
        condition_emb = self.condition_proj(condition).unsqueeze(1)

        # 3. Transformerへの入力を作成
        # [条件，軌道点1, 軌道点2, ...]の順でシーケンスを構成
        transformer_input = torch.cat([condition_emb, src_embedded], dim=1)
        transformer_input = self.pos_encoder(transformer_input)

        # 4. Transformerで次の点を予測
        src_mask = self.generate_square_subsequent_mask(transformer_input.size(1))
        output = self.transformer_encoder(transformer_input, mask=src_mask)

        # 5. 出力次元に変換
        return self.output_proj(output[:, 1:, :])   # 条件ベクトルに対応する出力を除く

    def generate(self,
                 condition: torch.Tensor,
                 seq_len: int,
                 ) -> torch.Tensor:

        self.eval()
        batch_size = condition.shape[0]
        device = condition.device

        # 1. 条件ベクトルを埋め込み(B, 1, inner_dim)
        condition_emb = self.condition_proj(condition).unsqueeze(1)

        # 2. 開始トークンと条件を結合して，最初の入力シーケンスを作成
        # generated_sequenceの形状は (B, current_len, d_model)
        generated_sequence_embedded = self.start_token.expand(batch_size, -1, -1)

        # 3. 1点ずつ自己回帰的に生成
        for _ in range(seq_len):
            transformer_input = torch.cat([condition_emb, generated_sequence_embedded], dim=1)
            transformer_input = self.pos_encoder(transformer_input)

            src_mask = self.generate_square_subsequent_mask(transformer_input.size(1)).to(device)
            output = self.transformer_encoder(transformer_input, mask=src_mask)

            # 最後の点の出力(inner_model次元ベクトル)を次の入力として使用
            next_point_emb = output[:, -1, :].unsqueeze(1)
            generated_sequence_embedded = torch.cat([generated_sequence_embedded, next_point_emb], dim=1)

        # 4. 出力層で座標に変換
        # start_tokenを除き，軌道部分だけを取り出す
        final_trajectory_emb = generated_sequence_embedded[:, 1:, :]
        generated_trajectory = self.output_proj(final_trajectory_emb)

        self.train()
        return generated_trajectory

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        未来の情報を隠すためのAttention Maskを生成
        """
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
