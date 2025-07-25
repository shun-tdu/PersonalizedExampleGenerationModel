# CLAUDE_ADDED
import math

import torch
import torch.nn as nn
from typing import Optional, Tuple


class DifferentiableBSpline(nn.Module):
    """微分可能なB-スプライン基底関数（簡易版）"""

    def __init__(self, num_control_points: int, degree: int = 3):
        super().__init__()
        self.num_control_points = num_control_points
        self.degree = degree

    def forward(self, control_points: torch.Tensor, num_output_points: int) -> torch.Tensor:
        """
        簡易的な3次スプライン補間（Catmull-Rom風）

        Args:
            control_points: [batch_size, num_control_points, 2]
            num_output_points: 出力する点の数

        Returns:
            trajectory: [batch_size, num_output_points, 2]
        """
        batch_size, n_cp, dim = control_points.shape
        device = control_points.device

        # 出力する軌道点
        trajectory = []

        # 各セグメント間を補間
        segments = n_cp - 1
        points_per_segment = num_output_points // segments + 1

        for i in range(segments):
            # セグメントの両端
            p0 = control_points[:, i]
            p1 = control_points[:, i + 1]

            # 前後の点（境界処理）
            if i > 0:
                pm1 = control_points[:, i - 1]
            else:
                pm1 = 2 * p0 - p1

            if i < segments - 1:
                p2 = control_points[:, i + 2]
            else:
                p2 = 2 * p1 - p0

            # このセグメントでの補間点数
            if i == segments - 1:
                n_points = num_output_points - len(trajectory)
            else:
                n_points = points_per_segment

            # パラメータt（0から1）
            t = torch.linspace(0, 1, n_points, device=device)

            # Catmull-Rom補間
            for tj in t[:-1] if i < segments - 1 else t:
                # 基底関数
                t2 = tj * tj
                t3 = t2 * tj

                h00 = 2 * t3 - 3 * t2 + 1
                h10 = t3 - 2 * t2 + tj
                h01 = -2 * t3 + 3 * t2
                h11 = t3 - t2

                # 接線ベクトル
                m0 = 0.5 * (p1 - pm1)
                m1 = 0.5 * (p2 - p0)

                # 補間点
                point = h00.unsqueeze(-1) * p0 + \
                        h10.unsqueeze(-1) * m0 + \
                        h01.unsqueeze(-1) * p1 + \
                        h11.unsqueeze(-1) * m1

                trajectory.append(point)

        # リストをテンソルに変換
        trajectory = torch.stack(trajectory, dim=1)

        # 正確にnum_output_pointsになるよう調整
        if trajectory.shape[1] > num_output_points:
            trajectory = trajectory[:, :num_output_points]
        elif trajectory.shape[1] < num_output_points:
            # 最後の点を繰り返して埋める
            last_point = trajectory[:, -1:].repeat(1, num_output_points - trajectory.shape[1], 1)
            trajectory = torch.cat([trajectory, last_point], dim=1)

        return trajectory


class LowFreqSpline(nn.Module):
    """
    低周波成分を学習・生成するスプラインベースモデル
    """

    def __init__(
            self,
            input_dim: int = 2,
            condition_dim: int = 5,
            num_control_points: int = 8,
            hidden_dim: int = 256,
            spline_degree: int = 3,
            dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.num_control_points = num_control_points

        # B-スプライン補間器
        self.spline = DifferentiableBSpline(num_control_points, degree=spline_degree)

        # 条件から制御点を生成するネットワーク
        self.control_point_generator = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_control_points * input_dim)
        )

        # 制約を生成するネットワーク
        self.constraint_net = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # start_x, start_y, scale_x, scale_y
        )

        # 制御点の基準位置（学習可能）
        self.register_parameter(
            'base_control_points',
            nn.Parameter(torch.randn(num_control_points, input_dim) * 0.1)
        )

        # 時間方向の分布を制御
        time_distribution = torch.linspace(0, 1, num_control_points)
        self.register_buffer('time_distribution', time_distribution)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        順方向計算（学習時）

        Args:
            x: 入力軌道 [batch_size, sequence_length, input_dim]
            condition: 条件ベクトル [batch_size, condition_dim]

        Returns:
            output: 予測軌道 [batch_size, sequence_length, input_dim]
        """
        batch_size, seq_len, _ = x.shape

        # 制御点を生成
        control_points = self._generate_control_points(condition)

        # スプライン補間で軌道を生成
        output = self.spline(control_points, seq_len)

        return output

    def _generate_control_points(self, condition: torch.Tensor) -> torch.Tensor:
        """
        条件から制御点を生成
        """
        batch_size = condition.shape[0]
        device = condition.device

        # 制御点のオフセットを生成
        control_offsets = self.control_point_generator(condition)
        control_offsets = control_offsets.view(batch_size, self.num_control_points, self.input_dim)

        # 基準制御点にオフセットを加算
        control_points = self.base_control_points.unsqueeze(0) + control_offsets

        # 制約を適用
        constraints = self.constraint_net(condition)
        start_pos = constraints[:, :2]
        scale = torch.sigmoid(constraints[:, 2:]) * 2 + 0.1  # 0.1 ~ 2.1

        # 始点を原点に固定
        control_points[:, 0, :] = torch.zeros(batch_size, 2, device=device)

        # 終点を条件から設定（GoalX, GoalY）
        if condition.shape[1] >= 5:
            goal_pos = condition[:, 3:5]  # GoalX, GoalY
            control_points[:, -1, :] = goal_pos

            # 中間の制御点を調整
            for i in range(1, self.num_control_points - 1):
                t = self.time_distribution[i]
                # 始点と終点の間に配置
                baseline = start_pos * (1 - t) + goal_pos * t
                offset = control_points[:, i, :] - baseline
                control_points[:, i, :] = baseline + offset * 0.5

        return control_points

    def generate(
            self,
            condition: torch.Tensor,
            sequence_length: int,
            start_token: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        低周波成分の軌道を生成
        """
        # 制御点を生成
        control_points = self._generate_control_points(condition)

        # スプライン補間で軌道を生成
        generated = self.spline(control_points, sequence_length)

        return generated

    def get_control_points(self, condition: torch.Tensor) -> torch.Tensor:
        """
        デバッグ用：条件から生成される制御点を取得
        """
        with torch.no_grad():
            return self._generate_control_points(condition)


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
        x = self.input_proj(x) * math.sqrt(self.inner_dim)

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