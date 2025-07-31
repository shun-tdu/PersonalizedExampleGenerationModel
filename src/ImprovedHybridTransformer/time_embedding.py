# CLAUDE_ADDED
import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbedding(nn.Module):
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
        
        self.sinusoidal_embedding = SinusoidalPositionEmbedding(embedding_dim)
        
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