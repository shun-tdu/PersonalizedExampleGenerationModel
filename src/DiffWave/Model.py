# CLAUDE_ADDED
"""
DiffWaveベースの軌道生成モデル
軌道データの時系列特性を活かした1D拡散モデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class TimeEmbedding(nn.Module):
    """
    時間埋め込み層（DiffWaveスタイル）
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: [batch_size] のタイムステップ
        Returns:
            [batch_size, dim] の時間埋め込み
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """
    DiffWave スタイルの残差ブロック
    """
    def __init__(self, 
                 residual_channels: int,
                 skip_channels: int,
                 condition_channels: int,
                 dilation: int = 1,
                 kernel_size: int = 3):
        super().__init__()
        
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        
        # Dilated convolution
        self.conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation
        )
        
        # Condition projection
        self.condition_proj = nn.Conv1d(condition_channels, 2 * residual_channels, 1)
        
        # Output projections
        self.residual_proj = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_proj = nn.Conv1d(residual_channels, skip_channels, 1)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, residual_channels, seq_len]
            condition: [batch_size, condition_channels, seq_len]
        Returns:
            residual: [batch_size, residual_channels, seq_len]
            skip: [batch_size, skip_channels, seq_len]
        """
        # Dilated convolution
        h = self.conv(x)
        
        # Add condition
        condition_h = self.condition_proj(condition)
        h = h + condition_h
        
        # Gated activation (tanh + sigmoid)
        h1, h2 = h.chunk(2, dim=1)
        h = torch.tanh(h1) * torch.sigmoid(h2)
        
        # Residual and skip connections
        residual = self.residual_proj(h)
        skip = self.skip_proj(h)
        
        return (x + residual) / math.sqrt(2), skip


class ConditionEncoder(nn.Module):
    """
    個人特性をエンコードして各時間ステップで利用可能にする
    """
    def __init__(self, 
                 condition_dim: int,
                 condition_channels: int,
                 seq_len: int):
        super().__init__()
        self.condition_dim = condition_dim
        self.condition_channels = condition_channels
        self.seq_len = seq_len
        
        # 条件ベクトルを高次元に投影
        self.condition_proj = nn.Linear(condition_dim, condition_channels)
        
        # 時間方向に展開するための学習可能なパラメータ
        self.time_expand = nn.Parameter(torch.randn(condition_channels, seq_len))
        
    def forward(self, condition: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Args:
            condition: [batch_size, condition_dim]
            seq_len: 系列長
        Returns:
            [batch_size, condition_channels, seq_len]
        """
        batch_size = condition.shape[0]
        
        # 条件ベクトルを投影
        h = self.condition_proj(condition)  # [batch_size, condition_channels]
        
        # 時間方向に展開
        h = h.unsqueeze(-1)  # [batch_size, condition_channels, 1]
        
        # 学習可能な時間展開パラメータを使用
        time_pattern = self.time_expand[:, :seq_len].unsqueeze(0)  # [1, condition_channels, seq_len]
        h = h * time_pattern  # [batch_size, condition_channels, seq_len]
        
        return h


class DiffWave1D(nn.Module):
    """
    DiffWaveベースの1D軌道生成モデル
    """
    def __init__(self,
                 input_dim: int = 2,
                 condition_dim: int = 5,
                 residual_channels: int = 64,
                 skip_channels: int = 64,
                 condition_channels: int = 128,
                 num_layers: int = 20,
                 cycles: int = 4,
                 time_embed_dim: int = 128):
        super().__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.num_layers = num_layers
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, condition_channels)
        
        # Condition encoder
        self.condition_encoder = ConditionEncoder(
            condition_dim, condition_channels, seq_len=101  # デフォルト系列長
        )
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, residual_channels, 1)
        
        # Residual blocks with exponentially increasing dilation
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** (i % cycles)
            self.residual_blocks.append(
                ResidualBlock(
                    residual_channels=residual_channels,
                    skip_channels=skip_channels,
                    condition_channels=condition_channels,
                    dilation=dilation
                )
            )
        
        # Output layers
        self.skip_proj = nn.Conv1d(skip_channels, skip_channels, 1)
        self.output_proj = nn.Conv1d(skip_channels, input_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """重み初期化"""
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                x: torch.Tensor, 
                time: torch.Tensor, 
                condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [batch_size, input_dim, seq_len] ノイズ付き軌道
            time: [batch_size] タイムステップ
            condition: [batch_size, condition_dim] 個人特性ベクトル
            
        Returns:
            [batch_size, input_dim, seq_len] 予測ノイズ
        """
        batch_size, _, seq_len = x.shape
        
        # Time embedding
        time_embed = self.time_embedding(time)  # [batch_size, time_embed_dim]
        time_proj = self.time_proj(time_embed)  # [batch_size, condition_channels]
        
        # Condition encoding
        condition_embed = self.condition_encoder(condition, seq_len)  # [batch_size, condition_channels, seq_len]
        
        # Add time information to condition
        time_proj_expanded = time_proj.unsqueeze(-1).expand(-1, -1, seq_len)
        condition_total = condition_embed + time_proj_expanded
        
        # Input projection
        h = self.input_proj(x)  # [batch_size, residual_channels, seq_len]
        
        # Residual blocks
        skip_connections = []
        for block in self.residual_blocks:
            h, skip = block(h, condition_total)
            skip_connections.append(skip)
        
        # Sum skip connections
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0)
        skip_sum = skip_sum / math.sqrt(len(skip_connections))
        
        # Output projection
        output = F.relu(self.skip_proj(skip_sum))
        output = self.output_proj(output)
        
        return output


class DiffWaveTrainer:
    """
    DiffWaveモデル用のトレーナー
    """
    def __init__(self, 
                 model: DiffWave1D,
                 device: torch.device,
                 learning_rate: float = 1e-4):
        self.model = model
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def compute_loss(self, 
                    trajectories: torch.Tensor,
                    conditions: torch.Tensor,
                    scheduler) -> torch.Tensor:
        """
        損失計算
        """
        batch_size = trajectories.shape[0]
        
        # Random timesteps
        timesteps = torch.randint(
            0, scheduler.num_timesteps, (batch_size,), device=self.device
        )
        
        # Add noise
        noise = torch.randn_like(trajectories)
        noisy_trajectories = scheduler.add_noise(trajectories, noise, timesteps)
        
        # Predict noise
        predicted_noise = self.model(noisy_trajectories, timesteps, conditions)
        
        # Compute loss
        loss = self.criterion(predicted_noise, noise)
        
        return loss
    
    def train_step(self, batch, scheduler) -> float:
        """
        Training step
        """
        trajectories, conditions = batch
        trajectories = trajectories.to(self.device)
        conditions = conditions.to(self.device)
        
        # Compute loss
        loss = self.compute_loss(trajectories, conditions, scheduler)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader, scheduler) -> float:
        """
        Validation
        """
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                trajectories, conditions = batch
                trajectories = trajectories.to(self.device)
                conditions = conditions.to(self.device)
                
                loss = self.compute_loss(trajectories, conditions, scheduler)
                val_loss += loss.item()
        
        self.model.train()
        return val_loss / len(val_loader)


if __name__ == '__main__':
    # テスト用コード
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデル作成
    model = DiffWave1D(
        input_dim=2,
        condition_dim=5,
        residual_channels=64,
        skip_channels=64,
        condition_channels=128,
        num_layers=20,
        cycles=4
    ).to(device)
    
    # テストデータ
    batch_size = 4
    seq_len = 101
    
    x = torch.randn(batch_size, 2, seq_len).to(device)
    time = torch.randint(0, 1000, (batch_size,)).to(device)
    condition = torch.randn(batch_size, 5).to(device)
    
    # Forward pass
    output = model(x, time, condition)
    
    print(f"Input shape: {x.shape}")
    print(f"Time shape: {time.shape}")
    print(f"Condition shape: {condition.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("DiffWave1D model test passed!")