import torch
import torch.nn as nn
import torch.nn.functional as F

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
        attention = F.softmax(attention_scores, dim=-1)

        # AttentionをVに適用
        out = torch.matmul(attention, v)

        # 元の形状に戻す
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)

        return self.to_out(out)
