# CLAUDE_ADDED
import math

import torch
import torch.nn as nn
from typing import Optional, Tuple
from HybridModelExperiment.modules.embedding import AdaptiveStartToken


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
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


class CurveLowFreqTransformer(nn.Module):
    """曲線対応改良型Transformerモデル"""
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

        # 入力埋め込み(位置情報のみ)
        self.position_emb = nn.Sequential(
            nn.Linear(input_dim, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 速度情報も含めた埋め込み
        self.velocity_aware_emb = nn.Sequential(
            nn.Linear(input_dim * 2, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
            nn.ReLU()
        )

        # 条件エンコーダ
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, self.inner_dim),
            nn.ReLU(),
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.LayerNorm(self.inner_dim)
        )

        # 曲線特徴抽出器
        self.curve_feature_extractor = nn.Sequential(
            nn.Linear(condition_dim, self.inner_dim),
            nn.ReLU(),
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.LayerNorm(self.inner_dim)
        )

        # 中間点予測器(曲線の形状をガイド)
        self.waypoint_predictor = nn.Sequential(
            nn.Linear(condition_dim, self.inner_dim),
            nn.ReLU(),
            nn.Linear(self.inner_dim, 3 * input_dim)   # 3つの中間点
        )

        # 正弦波位置エンコーディング
        self.pos_encoder = PositionalEncoding(self.inner_dim, max_len)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.inner_dim,
            nhead=heads,
            dim_feedforward=self.inner_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )

        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # 出力プロジェクション(位置と速度の変化を出力)
        self.output_proj = nn.Sequential(
            nn.Linear(self.inner_dim, self.inner_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.inner_dim // 2, input_dim * 2)   # position_delta + velocity_delta
        )

        # 適応的な開始トークン
        self.adaptive_start = AdaptiveStartToken(condition_dim, self.inner_dim, input_dim)

        # Scheduled Samplingの確率
        self.register_buffer('sampling_prob', torch.tensor(0.0))

    def create_curved_memory(self, condition: torch.Tensor) -> torch.Tensor:
        """曲線の特徴を含むメモリを生成"""
        batch_size = condition.shape[0]

        # 基本的な条件埋め込み
        base_memory = self.condition_encoder(condition).unsqueeze(1)

        # 曲線特徴
        curve_features = self.curve_feature_extractor(condition).unsqueeze(1)

        # 中間点の予測(曲線のガイド)
        waypoints = self.waypoint_predictor(condition)
        waypoints = waypoints.view(batch_size, 3, 2)
        waypoint_emb = self.position_emb(waypoints)

        # 全てを結合
        return torch.cat([base_memory, curve_features, waypoint_emb], dim=1) #[batch, 5, inner_dim]

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        学習時のフォワードパス
        :param x: [batch, seq_len, in_dim]  軌道データ
        :param condition: [batch, 5]        条件データ
        :return: [batch, seq_len, in_dim]   予測軌道
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # メモリ生成
        memory = self.create_curved_memory(condition)

        # Scheduled sampling
        use_teacher_forcing = self.training and torch.rand(1).item() > self.sampling_prob.item()

        if use_teacher_forcing:
            # Teacher forcing
            return self._teacher_forcing_forward(x, condition, memory)
        else:
            # 自己回帰生成
            return self._autoregressive_forward(x, condition, seq_len, memory)

    def _teacher_forcing_forward(
            self,
            x: torch.Tensor,
            condition: torch.Tensor,
            memory:torch.Tensor,
            ) -> torch.Tensor:
        """Teacher forcingによる学習"""
        batch_size, seq_len, _ = x.shape
        device = x.device

        # 開始位置と速度
        start_pos, start_vel = self.adaptive_start(condition)

        # 速度の計算
        velocities = torch.zeros_like(x)
        velocities[:, 0] = start_vel
        velocities[:, 1:] = x[:, 1:] - x[:, :-1]

        # 入力をシフト
        shifted_pos = torch.cat([start_pos.unsqueeze(1), x[:, :-1]], dim=1)
        shifted_vel = torch.cat([start_vel.unsqueeze(1), velocities[:, :-1]], dim=1)

        # 位置と速度を結合して埋め込み
        combined = torch.cat([shifted_pos, shifted_vel], dim=-1)
        tgt = self.velocity_aware_emb(combined)
        tgt = self.pos_encoder(tgt)

        # Causal mask
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(device)

        # デコード
        output = self.transformer(tgt, memory, tgt_mask=tgt_mask)

        # 位置と速度の変化を予測
        deltas = self.output_proj(output)
        pos_deltas, vel_deltas = deltas.chunk(2, dim=-1)

        # 位置の更新
        predicted_position = shifted_pos + pos_deltas * 0.1

        return predicted_position

    def _autoregressive_forward(
            self,
            x: Optional[torch.Tensor],
            condition: torch.Tensor,
            seq_len: int,
            memory: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """自己回帰的な生成"""
        batch_size = condition.shape[0]
        device = condition.device

        if memory is None:
            memory = self.create_curved_memory(condition)


        # 適応的な開始位置と速度
        current_pos, current_vel = self.adaptive_start(condition)

        generated_positions = [current_pos]
        generated_velocities = [current_vel]

        # 各ステップを生成
        for i in range(seq_len - 1):
            # 現在までの軌道
            positions = torch.stack(generated_positions, dim=1)
            velocities = torch.stack(generated_velocities, dim=1)

            # 位置と速度を結合
            combined = torch.cat([positions, velocities], dim=-1)
            tgt = self.velocity_aware_emb(combined)
            tgt = self.pos_encoder(tgt)

            # マスク
            current_len = len(generated_positions)
            tgt_mask = self.generate_square_subsequent_mask(current_len).to(device)

            # デコード
            output = self.transformer(tgt, memory, tgt_mask=tgt_mask)

            # 変化量を予測
            deltas = self.output_proj(output[:, -1])
            pos_delta, vel_delta = deltas.chunk(2, dim=-1)

            # 速度の更新(慣性を考慮)
            current_vel = current_vel * 0.9 + vel_delta

            # 位置の更新
            next_pos = current_pos + current_vel * 0.1

            # カリキュラム学習(学習時のみ)
            if self.training and x is not None and i < seq_len - 1:
                if torch.rand(1).item() < 0.3:
                    next_pos = x[:, i]
                    if i > 0:
                        current_vel = (x[:, i] - x[:, i-1]) * 10

            current_pos = next_pos
            generated_positions.append(next_pos)
            generated_velocities.append(current_vel)

        return torch.stack(generated_positions, dim=1) # 開始トークンを除く

    def generate(self, condition: torch.Tensor,seq_len: int) -> torch.Tensor:
        """推論時の生成"""
        self.eval()
        with torch.no_grad():
            return self._autoregressive_forward(None, condition, seq_len)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Causal Maskの生成"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def update_sampling_prob(self, epoch: int, max_epochs: int):
        """Scheduled samplingの確率を更新"""
        self.sampling_prob = torch.tensor(min(0.5, epoch / (max_epochs * 0.5)))

