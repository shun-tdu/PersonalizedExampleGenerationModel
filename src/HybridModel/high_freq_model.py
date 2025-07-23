# CLAUDE_ADDED
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from time_embedding import TimeEmbedding
import math


class SimpleDiffusionMLP(nn.Module):
    """
    高周波成分用のシンプルな拡散モデル（MLPベース）
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        condition_dim: int = 3,
        time_embedding_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 入力の特徴次元数
            hidden_dim: 隠れ層の次元数
            num_layers: MLPのレイヤー数
            condition_dim: 条件ベクトルの次元数
            time_embedding_dim: 時刻埋め込みの次元数
            dropout: ドロップアウト率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        
        # 時刻埋め込み
        self.time_embedding = TimeEmbedding(time_embedding_dim, hidden_dim)
        
        # 条件ベクトルの処理
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        
        # 入力プロジェクション
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # MLPレイヤー
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout)
                )
            )
        
        # 出力層
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # スケール・シフト層（時刻と条件の情報を統合）
        self.scale_shift_layers = nn.ModuleList()
        for i in range(num_layers):
            self.scale_shift_layers.append(
                nn.Linear(hidden_dim * 2, hidden_dim * 2)  # scale and shift
            )
    
    def forward(
        self, 
        x: torch.Tensor, 
        timesteps: torch.Tensor, 
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        ノイズ予測を行う
        
        Args:
            x: ノイズが加えられたデータ [batch_size, sequence_length, input_dim]
            timesteps: 時刻ステップ [batch_size]
            condition: 条件ベクトル [batch_size, condition_dim]
            
        Returns:
            predicted_noise: 予測されたノイズ [batch_size, sequence_length, input_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 時刻埋め込みを取得
        time_emb = self.time_embedding(timesteps)  # [batch_size, hidden_dim]
        
        # 条件ベクトルを処理
        cond_emb = self.condition_proj(condition)  # [batch_size, hidden_dim]
        
        # 時刻と条件を結合
        time_cond_emb = torch.cat([time_emb, cond_emb], dim=-1)  # [batch_size, hidden_dim * 2]
        
        # 入力を平坦化してプロジェクション
        x_flat = x.view(batch_size * seq_len, -1)  # [batch_size * seq_len, input_dim]
        h = self.input_proj(x_flat)  # [batch_size * seq_len, hidden_dim]
        
        # MLPレイヤーを通す
        for i, layer in enumerate(self.layers):
            # スケール・シフト変換を計算
            scale_shift = self.scale_shift_layers[i](time_cond_emb)  # [batch_size, hidden_dim * 2]
            scale, shift = scale_shift.chunk(2, dim=-1)  # each [batch_size, hidden_dim]
            
            # バッチ次元を拡張して適用
            scale = scale.unsqueeze(1).expand(-1, seq_len, -1).contiguous().view(-1, self.hidden_dim)
            shift = shift.unsqueeze(1).expand(-1, seq_len, -1).contiguous().view(-1, self.hidden_dim)
            
            # レイヤーを通す
            h = layer(h)
            
            # スケール・シフト変換を適用
            h = h * (1 + scale) + shift
        
        # 出力プロジェクション
        output = self.output_proj(h)  # [batch_size * seq_len, input_dim]
        
        # 元の形状に戻す
        output = output.view(batch_size, seq_len, -1)
        
        return output


class HighFreqDiffusion:
    """
    高周波成分用の拡散プロセス管理クラス
    """
    
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        """
        Args:
            num_timesteps: 拡散ステップ数
            beta_start: βの開始値
            beta_end: βの終了値
        """
        self.num_timesteps = num_timesteps
        
        # β スケジュール（線形）
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # α = 1 - β
        self.alphas = 1.0 - self.betas
        
        # α の累積積
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 逆プロセス用の分散
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        データにノイズを追加
        
        Args:
            x_0: 元データ [batch_size, sequence_length, input_dim]
            t: 時刻ステップ [batch_size]
            noise: 追加するノイズ（オプション）
            
        Returns:
            x_t: ノイズが追加されたデータ
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # デバイスを同じにする
        device = x_0.device
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt()
        
        # バッチ次元を調整
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        ランダムな時刻ステップをサンプリング
        
        Args:
            batch_size: バッチサイズ
            device: デバイス
            
        Returns:
            timesteps: サンプリングされた時刻ステップ [batch_size]
        """
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)
    
    @torch.no_grad()
    def denoise_step(
        self, 
        model: SimpleDiffusionMLP, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        デノイジングステップ
        
        Args:
            model: ノイズ予測モデル
            x_t: 現在のノイズ付きデータ
            t: 現在の時刻ステップ
            condition: 条件ベクトル
            
        Returns:
            x_prev: 前の時刻のデータ
        """
        # デバイスを同じにする
        device = x_t.device
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.betas = self.betas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        
        # ノイズを予測
        predicted_noise = model(x_t, t, condition)
        
        # デノイジング計算
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1)
        
        # 平均を計算
        x_0_pred = (x_t - predicted_noise * (1 - alpha_cumprod_t).sqrt()) / alpha_cumprod_t.sqrt()
        
        if t[0] > 0:
            # t > 0 の場合、ノイズを追加
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1)
            noise = torch.randn_like(x_t)
            x_prev = (x_t - beta_t * predicted_noise / (1 - alpha_cumprod_t).sqrt()) / alpha_t.sqrt()
            x_prev = x_prev + posterior_variance_t.sqrt() * noise
        else:
            # t = 0 の場合、ノイズなし
            x_prev = x_0_pred
        
        return x_prev
    
    @torch.no_grad()
    def generate(
        self, 
        model: SimpleDiffusionMLP, 
        shape: tuple, 
        condition: torch.Tensor, 
        device: torch.device
    ) -> torch.Tensor:
        """
        高周波成分を生成
        
        Args:
            model: ノイズ予測モデル
            shape: 生成するデータの形状 (batch_size, sequence_length, input_dim)
            condition: 条件ベクトル
            device: デバイス
            
        Returns:
            generated: 生成された高周波成分
        """
        batch_size = shape[0]
        
        # 初期ノイズ
        x = torch.randn(shape, device=device)
        
        # 逆拡散プロセス
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.denoise_step(model, x, t, condition)
        
        return x