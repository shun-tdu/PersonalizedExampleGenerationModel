# CLAUDE_ADDED
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from jupyterlab.semver import inc

from time_embedding import TimeEmbedding
import math



class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time_step:torch.Tensor) -> torch.Tensor:
        """

        :param time_step: [Batch, ] 時間ステップ
        :return: [Batch, dim] 時間埋め込みベクトル
        """
        device = time_step.device
        half_dim  = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time_step[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings



class CrossAttention(nn.Module):
    """
    軌道データ(Query)と個人特性(Context)を結びつけるためのCross Attention層
    """
    def __init__(self, query_dim:int, context_dim:int, heads:int = 8, dim_head:int = 32):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        # 入力(x)からQueryを、条件(Context)からKeyとValueを生成するための線形層
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # 出力への線形写像
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x:torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch, seq_len, query_dim]の軌道特徴量
        :param context: [Batch, context_dim]の個人特性ベクトル
        :return: [batch, seq_len, query_dim] アテンション適用後の軌道特徴量
        """
        batch_size, seq_len, _ = x.shape

        # 各線形層でq, k, vを計算
        q = self.to_q(x)
        k = self.to_k(context.unsqueeze(1))
        v = self.to_v(context.unsqueeze(1))

        # Multihead Attentionのためにヘッド数で分割、次元を並べ替え
        q = q.view(batch_size, seq_len, self.heads, -1).transpose(1, 2)
        k = k.view(batch_size, 1, self.heads, -1).transpose(1, 2)
        v = v.view(batch_size, 1, self.heads, -1).transpose(1, 2)

        # QとKの内積でAttentionを計算
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = F.softmax(attention_scores, dim=1)

        # AttentionをVに適用
        out = torch.matmul(attention, v)

        # 元の形状に戻す
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)

        return self.to_out(out)


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 time_embed_dim: int,
                 condition_dim: int,
                 kernel_size:int = 3,
                 num_group:int = 8):
        super().__init__()

        # 1つ目の畳み込み層
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.norm1 = nn.GroupNorm(num_group, out_channels)

        # 2つ目の畳み込み層
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.norm2 = nn.GroupNorm(num_group, out_channels)

        # 時間埋め込みの次元変換層
        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        # 条件付けのためのCross-Attention層
        self.cross_attention = CrossAttention(out_channels, condition_dim)

        # 入力と出力のチャンネル数を合わせるための、スキップ接続
        self.skip_connection = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self,
                x: torch.Tensor,
                time_embed: torch.Tensor,
                condition:torch.Tensor) -> torch.Tensor:
        """
        :param x: [Batch, in_channels, seq_len]     入力テンソル
        :param time_embed: [Batch, time_emb_dim]    拡散ステップの埋め込み
        :param condition: [Batch, condition_dim]    条件の埋め込み
        :return: [Batch, out_channels, seq_len]     ノイズの推測値
        """
        # スキップ接続
        residual = self.skip_connection(x)

        # 1.畳み込み -> 正規化 -> 時間埋め込み -> 活性化
        x = self.conv1(x)
        x = self.norm1(x)
        time_embed = self.time_proj(time_embed)[:, :, None] # (B, C) -> (B, C, 1)に拡張
        x = x + time_embed
        x = F.silu(x)

        # 2.畳み込み -> 正規化
        x = self.conv2(x)
        x = self.norm2(x)

        # 3. Cross Attentionによる条件付け
        x_transposed = x.transpose(1, 2)
        x_attended = self.cross_attention(x_transposed, condition)
        x = x_attended.transpose(1, 2)

        # 4. Residual Connection
        return x + residual


class UNet1D(nn.Module):
    def __init__(self,
                 input_dim: int = 2,
                 condition_dim: int = 5,
                 time_embed_dim: int = 128,
                 base_channels: int = 64,
                 num_resolutions = 4):
        super().__init__()

        self.channels = [base_channels * (2 ** i) for i in range(num_resolutions)]
        self.num_resolutions = num_resolutions

        # 時間埋め込み
        self.time_embedding = SinusoidalPositionalEmbedding(time_embed_dim)

        # 入力データを最初のチャンネル数に変換
        self.input_proj = nn.Conv1d(input_dim, base_channels, kernel_size=1)

        # Encoder (Down Sampling)
        self.encoder_block = nn.ModuleList()
        self.downsample_layer = nn.ModuleList()

        in_c = base_channels
        for i, out_c in enumerate(self.channels):
            self.encoder_block.append(ResidualBlock(in_c, out_c, time_embed_dim, condition_dim))

            # 最後以外はダウンサンプリング
            if i < self.num_resolutions - 1:
                self.downsample_layer.append(nn.Conv1d(out_c, out_c, kernel_size=3, stride=2, padding=1))
            else:
                self.downsample_layer.append(nn.Identity())
            in_c = out_c

        # Bottleneck
        self.bottleneck = ResidualBlock(self.channels[-1], self.channels[-1], time_embed_dim, condition_dim)

        # Decoder (Up Sampling)
        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        decoder_channels = list(reversed(self.channels[-1]))
        in_c = self.channels[-1]

        for i, out_c in enumerate(decoder_channels):
            # アップサンプリング層
            self.upsample_layers.append(nn.ConvTranspose1d(in_c, out_c, kernel_size=3, stride=2, padding=1, output_padding=1))
            # スキップ接続
            concat_channels = out_c + out_c
            self.decoder_blocks.append(ResidualBlock(concat_channels, out_c, time_embed_dim, condition_dim))
            in_c = out_c

        # 出力層
        self.output_proj = nn.Conv1d(base_channels, input_dim, kernel_size=1)

    def forward(self,
                x: torch.Tensor,
                time_steps: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch, 2, seq_len] ノイズ軌道
        :param time_steps: [batch, ] 時間ステップ
        :param condition: [batch, condition_dim] 個人特性ベクトル
        :return: [batch, 2, seq_len] 予測されたノイズ
        """
        time_embed = self.time_embedding(time_steps)
        x = self.input_proj(x)

        # エンコーダからの接続を保存
        encoder_features = []

        # Encoder
        for i, (block, downsample) in enumerate(zip(self.encoder_block, self.downsample_layer)):
            x = block(x, time_embed, condition)
            # ボトルネック以外のみスキップ接続として保存
            if i < self.num_resolutions - 1:
                encoder_features.append(x)
            x = downsample(x)

            # Bottleneck
            x = self.bottleneck(x, time_embed, condition)

            # Decoder
            for i, (upsample, block) in enumerate(zip(self.upsample_layers, self.decoder_blocks)):
                # アップサンプリング
                x = upsample(x)

                # スキップ接続
                skip = encoder_features.pop()

                # サイズ調整
                if x.size(-1) != skip.size(-1):
                    x = F.interpolate(x, size=skip.size(-1), mode='linear', align_corners=False)

                # チャンネルを結合
                x = torch.cat([x, skip], dim=1)

                # ResidualBlock
                x = block(x, time_embed, condition)

            return self.output_proj(x)


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