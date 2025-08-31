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


class SimpleTransformerVAE(nn.Module):
    def __init__(self, input_dim=6, seq_len=100, d_model=128, n_head=4, num_layers=2, latent_dim=64, 
                 beta_annealing_config=None):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.latent_dim = latent_dim
        
        # CLAUDE_ADDED: β-アニーリングの設定を初期化（設定ファイルまたはデフォルト値）
        default_beta_config = {
            'enabled': True,
            'schedule': 'linear',  # 'linear', 'exponential', 'sigmoid'
            'start_epoch': 0,
            'end_epoch': 100,
            'min_beta': 0.0,
            'max_beta': 1.0,
            'current_beta': 0.01  # デフォルト値
        }
        
        # 設定ファイルからの値で更新（提供されている場合）
        if beta_annealing_config:
            default_beta_config.update(beta_annealing_config)
        
        self.beta_annealing_config = default_beta_config
        
        # CLAUDE_ADDED: 訓練コンテキスト（エポック情報など）
        self.training_context = {
            'current_epoch': 0
        }

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
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 潜在変数変換
        self.to_latent = nn.Linear(d_model * seq_len, 2 * latent_dim)

        self.from_latent = nn.Linear(latent_dim, d_model * seq_len)

        # Transformerデコーダ
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x, subject_ids=None, is_expert=None):
        batch_size, _, _ = x.shape

        # 入力射影
        encoded = self.input_proj(x)

        # 位置エンコード
        pos_encoded = self.pos_encoding(encoded)

        # 特徴量変換
        transformed = self.transformer_encoder(pos_encoded)

        # Transformerの出力を集約する (シーケンス次元で平均を取る)
        transformed_flat = transformed.view(batch_size, -1)

        # 潜在変数変換
        latent = self.to_latent(transformed_flat)  # 形状: (batch, d_model * seq_len)

        # muとlogvarを最後の次元で正しく分割する
        mu = latent[:, :self.latent_dim]
        logvar = latent[:, self.latent_dim:]
        z = self.reparameterize(mu, logvar)  # 形状: (batch, latent_dim)

        # デコーダ
        decoder_flat = self.from_latent(z)
        decoded = decoder_flat.view(batch_size, self.seq_len, self.d_model)

        # 位置エンコード
        decoded_with_pos = self.pos_encoding(decoded)

        #  形状をシーケンスに戻す
        refined = self.transformer_decoder(decoded_with_pos)

        reconstructed = self.output_proj(refined)  # 形状: (batch, seq_len, input_dim)

        # 損失計算 (引数を修正)
        losses = self.compute_losses(x, reconstructed, mu, logvar)

        result = {
            'reconstructed': reconstructed,
            'z': z,
        }

        for loss_name, loss_value in losses.items():
            result[f'{loss_name}_loss'] = loss_value

        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    # CLAUDE_ADDED: 訓練コンテキスト（エポック情報など）を設定
    def set_training_context(self, current_epoch=None, **kwargs):
        """訓練コンテキスト情報を設定（ExperimentRunnerから呼び出される）"""
        if current_epoch is not None:
            self.training_context['current_epoch'] = current_epoch
        
        # 他の訓練コンテキストも設定可能
        self.training_context.update(kwargs)
        
        # β値を更新
        self._update_beta()
    
    # CLAUDE_ADDED: β-アニーリング設定を更新
    def configure_beta_annealing(self, enabled=True, schedule='linear', start_epoch=0, end_epoch=100, 
                                min_beta=0.0, max_beta=1.0):
        """β-アニーリングの設定を変更"""
        self.beta_annealing_config.update({
            'enabled': enabled,
            'schedule': schedule,
            'start_epoch': start_epoch,
            'end_epoch': end_epoch,
            'min_beta': min_beta,
            'max_beta': max_beta
        })
        self._update_beta()
    
    # CLAUDE_ADDED: 現在のβ値を計算・更新
    def _update_beta(self):
        """現在のエポックに基づいてβ値を更新"""
        if not self.beta_annealing_config['enabled']:
            self.beta_annealing_config['current_beta'] = self.beta_annealing_config['max_beta']
            return
        
        current_epoch = self.training_context['current_epoch']
        start_epoch = self.beta_annealing_config['start_epoch']
        end_epoch = self.beta_annealing_config['end_epoch']
        min_beta = self.beta_annealing_config['min_beta']
        max_beta = self.beta_annealing_config['max_beta']
        schedule = self.beta_annealing_config['schedule']
        
        # エポックが範囲外の場合の処理
        if current_epoch <= start_epoch:
            beta = min_beta
        elif current_epoch >= end_epoch:
            beta = max_beta
        else:
            # アニーリング進行度を計算 (0.0 to 1.0)
            progress = (current_epoch - start_epoch) / (end_epoch - start_epoch)
            
            if schedule == 'linear':
                beta = min_beta + progress * (max_beta - min_beta)
            elif schedule == 'exponential':
                # 指数的に増加
                beta = min_beta + (max_beta - min_beta) * (progress ** 2)
            elif schedule == 'sigmoid':
                # シグモイド関数による滑らかな変化
                sigmoid_input = (progress - 0.5) * 12  # -6 to 6 の範囲
                sigmoid_value = 1 / (1 + np.exp(-sigmoid_input))
                beta = min_beta + sigmoid_value * (max_beta - min_beta)
            else:
                # デフォルトはlinear
                beta = min_beta + progress * (max_beta - min_beta)
        
        self.beta_annealing_config['current_beta'] = beta

    def compute_losses(self, x, reconstructed, mu, logvar):
        losses = {}

        # 再構築誤差
        losses['reconstruction'] = F.mse_loss(x, reconstructed)

        # KLDiv
        logvar_clamped = torch.clamp(logvar, min=-10, max=10)
        losses['kl'] = -0.5 * torch.mean(
            torch.sum(1 + logvar_clamped - mu.pow(2)
                      - logvar_clamped.exp(), dim=1))

        # CLAUDE_ADDED: β-アニーリングを使用してKL項の重みを調整
        current_beta = self.beta_annealing_config['current_beta']
        total_loss = losses['reconstruction'] + current_beta * losses['kl']

        losses['total'] = total_loss
        
        # CLAUDE_ADDED: β値も損失情報として記録
        losses['beta'] = current_beta
        
        return losses


