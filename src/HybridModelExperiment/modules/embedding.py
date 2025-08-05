# CLAUDE_ADDED
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SinusoidalPositionalEmbedding(nn.Module):
    """
    拡散モデル用の正弦波位置埋め込み
    """
    
    def __init__(self, embedding_dim: int):
        """
        Args:
            embedding_dim: 埋め込み次元数
        """
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        時刻ステップを正弦波埋め込みに変換
        
        Args:
            timesteps: 時刻ステップ [batch_size] or [batch_size, 1]
            
        Returns:
            embeddings: 時刻埋め込み [batch_size, embedding_dim]
        """
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(-1)
        
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        
        # 周波数を計算
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # 正弦波と余弦波を計算
        emb = timesteps * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # 奇数次元の場合、最後の次元を削除
        if self.embedding_dim % 2 == 1:
            emb = emb[:, :-1]
        
        return emb


class TimeEmbedding(nn.Module):
    """
    時刻埋め込みを処理するモジュール
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int):
        """
        Args:
            embedding_dim: 正弦波埋め込みの次元数
            hidden_dim: 出力隠れ層の次元数
        """
        super().__init__()
        
        self.sinusoidal_embedding = SinusoidalPositionalEmbedding(embedding_dim)
        
        # MLPで時刻埋め込みを変換
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        時刻ステップを埋め込みベクトルに変換
        
        Args:
            timesteps: 時刻ステップ [batch_size]
            
        Returns:
            time_emb: 時刻埋め込み [batch_size, hidden_dim]
        """
        # 正弦波埋め込みを取得
        sinusoidal_emb = self.sinusoidal_embedding(timesteps)
        
        # MLPで変換
        time_emb = self.mlp(sinusoidal_emb)
        
        return time_emb


class AdaptiveStartToken(nn.Module):
    """条件に応じた開始位置と初期速度を生成"""
    def __init__(self, condition_dim = 5, inner_dim = 256, output_dim = 2):
        super().__init__()

        # 開始位置に加えて初期速度も予測
        self.start_predictor = nn.Sequential(
            nn.Linear(condition_dim, inner_dim),
            nn.ReLU(),
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, output_dim * 2) # position + velocity
        )

        # ゴール方向を考慮
        self.goal_direction_net = nn.Sequential(
            nn.Linear(2, inner_dim // 2),    # goal_x, goal_y
            nn.ReLU(),
            nn.Linear(inner_dim // 2, output_dim)
        )

    def forward(self, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 基本的な開始状態
        start_state = self.start_predictor(condition)
        start_pos, start_vel = start_state.chunk(2, dim=-1)

        # ゴール方向の影響
        goal_pos = condition[:, 3:5]
        goal_direction = self.goal_direction_net(goal_pos)

        # 初期速度をゴール方向に調整
        start_vel = start_vel + 0.1 * goal_direction

        # 開始位置は原点付近
        start_pos = start_pos * 0.1

        return start_pos, start_vel
