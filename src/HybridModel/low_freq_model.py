# CLAUDE_ADDED
import math

import torch
import torch.nn as nn
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [seq_len, batch_size, embedding_dim]
        :return: [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LowFreqTransformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 dim_head: int = 32,
                 heads: int = 8,
                 num_encoder_layers: int = 3,
                 dim_feedforward: int = 512,
                 condition_dim:int = 3,
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

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # 1. 入力を潜在次元に変換
        x = self.input_proj(x) & math.sqrt(self.inner_dim)

        # 2. 条件を潜在次元に変換し、系列の各ステップに加算
        condition_emb = self.condition_proj(condition).unsqueeze(1)
        x = x + condition_emb

        # 3. 正弦波位置エンコーディング
        x = x.permute(1, 0, 2)  # [seq_len, batch, d_model]
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # [batch, seq_len d_model]

        # 4. Transformer Encoderで処理
        output = self.transformer_encoder(x)

        # 5. 出力次元に変換
        return self.output_proj(x)

    def generate(self,
                 condition: torch.Tensor,
                 seq_len: int,
                 start_token: Optional[torch.Tensor] = None
                 ) -> torch.Tensor:
        batch_size = condition.shape[0]
        device = condition.device
        input_dim = self.output_proj.out_features

        #　開始トークンを設定
        if start_token is None:
            generated_sequence = torch.zeros(batch_size, 1, input_dim, device=device)
        else:
            generated_sequence = start_token

        for _ in range(seq_len - 1):
            # これまで生成した系列を全て入力
            output = self.forward(generated_sequence, condition)

            # 最後のステップの予測だけを取り出す
            next_token = output[:, -1:, :]  # [batch, 1, input_dim]

            # 生成した系列に結合
            generated_sequence = torch.cat([generated_sequence, next_token], dim=1)

        return generated_sequence


class LowFreqLSTM(nn.Module):
    """
    低周波成分を学習・生成するLSTMモデル
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 128, 
        num_layers: int = 2,
        condition_dim: int = 3,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 入力の特徴次元数
            hidden_dim: LSTMの隠れ層次元数
            num_layers: LSTMのレイヤー数
            condition_dim: 条件ベクトルの次元数
            dropout: ドロップアウト率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.condition_dim = condition_dim
        
        # 条件ベクトルを処理する層
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        
        # LSTM層
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 出力層
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        condition: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        順方向計算
        
        Args:
            x: 入力データ [batch_size, sequence_length, input_dim]
            condition: 条件ベクトル [batch_size, condition_dim]
            hidden: 初期隠れ状態（オプション）
            
        Returns:
            output: 出力 [batch_size, sequence_length, input_dim]
            hidden: 最終隠れ状態
        """
        batch_size, seq_len, _ = x.shape
        
        # 初期隠れ状態を設定
        if hidden is None:
            # 条件ベクトルから初期隠れ状態を生成
            condition_emb = self.condition_proj(condition)  # [batch_size, hidden_dim]
            h_0 = condition_emb.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, batch_size, hidden_dim]
            c_0 = torch.zeros_like(h_0)
            hidden = (h_0, c_0)
        
        # LSTM計算
        lstm_out, hidden = self.lstm(x, hidden)
        
        # ドロップアウト
        lstm_out = self.dropout(lstm_out)
        
        # 出力プロジェクション
        output = self.output_proj(lstm_out)
        
        return output, hidden
    
    def generate(
        self, 
        condition: torch.Tensor, 
        sequence_length: int,
        start_token: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        低周波成分の軌道を生成
        
        Args:
            condition: 条件ベクトル [batch_size, condition_dim]
            sequence_length: 生成する系列長
            start_token: 開始トークン [batch_size, 1, input_dim]（オプション）
            
        Returns:
            generated: 生成された軌道 [batch_size, sequence_length, input_dim]
        """
        batch_size = condition.shape[0]
        device = condition.device
        
        # 初期隠れ状態を設定
        condition_emb = self.condition_proj(condition)
        h_0 = condition_emb.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)
        hidden = (h_0, c_0)
        
        # 開始トークンを設定
        if start_token is None:
            current_input = torch.zeros(batch_size, 1, self.input_dim, device=device)
        else:
            current_input = start_token
        
        # 自回帰的に生成
        generated_sequence = []
        
        for _ in range(sequence_length):
            output, hidden = self.forward(current_input, condition, hidden)
            generated_sequence.append(output)
            current_input = output  # 次の入力として使用
        
        # 結果を結合
        generated = torch.cat(generated_sequence, dim=1)
        
        return generated