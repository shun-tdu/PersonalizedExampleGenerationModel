# CLAUDE_ADDED
"""
Transformerベースの軌道生成モデル
自己回帰的な軌道生成と条件付き生成を行う
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import numpy as np


class PositionalEncoding(nn.Module):
    """
    位置エンコーディング（時系列データ用）
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :]


class TrajectoryEmbedding(nn.Module):
    """
    軌道データ埋め込み層
    """
    def __init__(self, input_dim: int = 2, d_model: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 2D座標を高次元ベクトルに変換
        self.embedding = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            [seq_len, batch_size, d_model]
        """
        # [batch_size, seq_len, input_dim] -> [batch_size, seq_len, d_model]
        embedded = self.embedding(x)
        embedded = self.layer_norm(embedded)
        
        # [batch_size, seq_len, d_model] -> [seq_len, batch_size, d_model]
        return embedded.transpose(0, 1)


class ConditionEncoder(nn.Module):
    """
    個人特性条件をエンコード
    """
    def __init__(self, condition_dim: int, d_model: int = 256):
        super().__init__()
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            condition: [batch_size, condition_dim]
        Returns:
            [batch_size, d_model]
        """
        return self.condition_proj(condition)


class TransformerTrajectoryGenerator(nn.Module):
    """
    Transformerベースの軌道生成モデル
    """
    def __init__(self,
                 input_dim: int = 2,
                 condition_dim: int = 5,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 max_seq_len: int = 101,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 軌道埋め込み
        self.trajectory_embedding = TrajectoryEmbedding(input_dim, d_model)
        
        # 条件エンコーダ
        self.condition_encoder = ConditionEncoder(condition_dim, d_model)
        
        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer本体
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # seq_len first
        )
        
        # 出力層
        self.output_projection = nn.Linear(d_model, input_dim)
        
        # スタートトークン
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 重み初期化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """重み初期化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        自己回帰用のマスク生成
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, 
                src_trajectories: torch.Tensor,
                conditions: torch.Tensor,
                tgt_trajectories: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            src_trajectories: [batch_size, src_seq_len, input_dim] ソース軌道
            conditions: [batch_size, condition_dim] 個人特性
            tgt_trajectories: [batch_size, tgt_seq_len, input_dim] ターゲット軌道（訓練時）
            
        Returns:
            [batch_size, tgt_seq_len, input_dim] 予測軌道
        """
        batch_size = src_trajectories.shape[0]
        device = src_trajectories.device
        
        # ソース軌道の埋め込み
        src_embedded = self.trajectory_embedding(src_trajectories)  # [src_seq, batch, d_model]
        src_embedded = self.pos_encoding(src_embedded)
        
        # 条件エンコーディング
        condition_encoded = self.condition_encoder(conditions)  # [batch, d_model]
        condition_encoded = condition_encoded.unsqueeze(0)  # [1, batch, d_model]
        
        # 条件をソースに追加
        src_with_condition = torch.cat([condition_encoded, src_embedded], dim=0)  # [1+src_seq, batch, d_model]
        
        if tgt_trajectories is not None:
            # 訓練時: Teacher forcing
            tgt_embedded = self.trajectory_embedding(tgt_trajectories)  # [tgt_seq, batch, d_model]
            tgt_embedded = self.pos_encoding(tgt_embedded)
            
            # マスク生成
            tgt_seq_len = tgt_embedded.shape[0]
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(device)
            
            # Transformer forward
            output = self.transformer(
                src=src_with_condition,
                tgt=tgt_embedded,
                tgt_mask=tgt_mask
            )  # [tgt_seq, batch, d_model]
            
        else:
            # 推論時: 自己回帰生成
            tgt_seq_len = self.max_seq_len
            
            # スタートトークンで開始
            tgt_tokens = self.start_token.expand(1, batch_size, -1)  # [1, batch, d_model]
            
            for i in range(tgt_seq_len):
                # 現在の予測シーケンス
                current_tgt = tgt_tokens
                
                # マスク生成
                current_len = current_tgt.shape[0]
                tgt_mask = self.generate_square_subsequent_mask(current_len).to(device)
                
                # Transformer forward
                output = self.transformer(
                    src=src_with_condition,
                    tgt=current_tgt,
                    tgt_mask=tgt_mask
                )  # [current_len, batch, d_model]
                
                # 最後のトークンを取得
                next_token = output[-1:, :, :]  # [1, batch, d_model]
                
                # 次のトークンを追加
                tgt_tokens = torch.cat([tgt_tokens, next_token], dim=0)
            
            output = tgt_tokens[1:]  # スタートトークンを除く
        
        # 出力投影
        # [seq_len, batch, d_model] -> [seq_len, batch, input_dim] -> [batch, seq_len, input_dim]
        predictions = self.output_projection(output).transpose(0, 1)
        
        return predictions
    
    @torch.no_grad()
    def generate_trajectory(self, 
                          start_points: torch.Tensor,
                          conditions: torch.Tensor,
                          max_length: int = 101) -> torch.Tensor:
        """
        軌道生成（推論専用）
        
        Args:
            start_points: [batch_size, num_start_points, input_dim] 開始点
            conditions: [batch_size, condition_dim] 個人特性
            max_length: 生成する軌道の最大長
            
        Returns:
            [batch_size, max_length, input_dim] 生成された軌道
        """
        self.eval()
        batch_size = start_points.shape[0]
        device = start_points.device
        
        # 条件エンコーディング
        condition_encoded = self.condition_encoder(conditions)  # [batch, d_model]
        condition_encoded = condition_encoded.unsqueeze(0)  # [1, batch, d_model]
        
        # 開始点の埋め込み
        src_embedded = self.trajectory_embedding(start_points)  # [num_start, batch, d_model]
        src_embedded = self.pos_encoding(src_embedded)
        
        # 条件をソースに追加
        src_with_condition = torch.cat([condition_encoded, src_embedded], dim=0)
        
        # 生成された軌道を格納
        generated_points = []
        
        # スタートトークンで開始
        current_input = self.start_token.expand(1, batch_size, -1)
        
        for step in range(max_length):
            # マスク生成
            current_len = current_input.shape[0]
            tgt_mask = self.generate_square_subsequent_mask(current_len).to(device)
            
            # Transformer forward
            output = self.transformer(
                src=src_with_condition,
                tgt=current_input,
                tgt_mask=tgt_mask
            )  # [current_len, batch, d_model]
            
            # 最新の出力を軌道点に変換
            latest_output = output[-1:, :, :]  # [1, batch, d_model]
            next_point = self.output_projection(latest_output)  # [1, batch, input_dim]
            
            generated_points.append(next_point)
            
            # 次のステップの入力として使用
            next_embedded = self.trajectory_embedding(next_point.transpose(0, 1))  # [1, batch, d_model]
            next_embedded = self.pos_encoding(next_embedded)
            current_input = torch.cat([current_input, next_embedded], dim=0)
        
        # [max_length, batch, input_dim] -> [batch, max_length, input_dim]
        generated_trajectory = torch.cat(generated_points, dim=0).transpose(0, 1)
        
        return generated_trajectory


class TransformerTrainer:
    """
    Transformerモデル用のトレーナー
    """
    def __init__(self, 
                 model: TransformerTrajectoryGenerator,
                 device: torch.device,
                 learning_rate: float = 1e-4):
        self.model = model
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def compute_loss(self, 
                    trajectories: torch.Tensor,
                    conditions: torch.Tensor) -> torch.Tensor:
        """
        損失計算（Teacher forcing）
        """
        batch_size, seq_len, _ = trajectories.shape
        
        # 入力と出力を分割
        src_trajectories = trajectories[:, :-1, :]  # 最後を除く
        tgt_trajectories = trajectories[:, :-1, :]  # 最後を除く（input）
        target_trajectories = trajectories[:, 1:, :]  # 最初を除く（target）
        
        # 予測
        predictions = self.model(src_trajectories, conditions, tgt_trajectories)
        
        # 損失計算
        loss = self.criterion(predictions, target_trajectories)
        
        return loss
    
    def train_step(self, batch) -> float:
        """
        Training step
        """
        trajectories, conditions = batch
        trajectories = trajectories.to(self.device)
        conditions = conditions.to(self.device)
        
        # Compute loss
        loss = self.compute_loss(trajectories, conditions)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader) -> float:
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
                
                loss = self.compute_loss(trajectories, conditions)
                val_loss += loss.item()
        
        self.model.train()
        return val_loss / len(val_loader)


if __name__ == '__main__':
    # テスト用コード
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデル作成
    model = TransformerTrajectoryGenerator(
        input_dim=2,
        condition_dim=5,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        max_seq_len=101
    ).to(device)
    
    # テストデータ
    batch_size = 4
    seq_len = 101
    
    trajectories = torch.randn(batch_size, seq_len, 2).to(device)
    conditions = torch.randn(batch_size, 5).to(device)
    
    # Forward pass (training)
    src_traj = trajectories[:, :-1, :]
    tgt_traj = trajectories[:, :-1, :]
    predictions = model(src_traj, conditions, tgt_traj)
    
    print(f"Input trajectories shape: {trajectories.shape}")
    print(f"Conditions shape: {conditions.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generation test
    start_points = trajectories[:, :1, :]  # 最初の点のみ
    generated = model.generate_trajectory(start_points, conditions, max_length=10)
    print(f"Generated trajectory shape: {generated.shape}")
    
    print("Transformer Trajectory Generator test passed!")