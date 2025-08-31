import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """時系列用位置エンコーディング"""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class SimpleTransformerAE(nn.Module):
    def __init__(self, input_dim=6, seq_len=100, d_model=128, n_head=4, num_layers=2):
        super().__init__()

        # 入力射影
        self.input_proj = nn.Linear(input_dim, d_model)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len)

        # Transformerエンコーダ
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 出力射影
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x, subject_ids=None, is_expert=None):
        # 入力射影
        encoded = self.input_proj(x)
        pos_encoded = self.pos_encoding(encoded)
        transformed = self.transformer(pos_encoded)
        reconstructed = self.output_proj(transformed)

        reconstructed_loss = F.mse_loss(reconstructed, x)

        result = {
            'reconstructed': reconstructed,
            'reconstructed_loss': reconstructed_loss,
            'total_loss': reconstructed_loss
        }

        return result


