from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.base_model import BaseExperimentModel
from models.components.loss_weight_scheduler import LossWeightScheduler


class ImprovedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, margin=1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, features, labels):
        # ハードネガティブマイニングを追加
        batch_size = features.shape[0]

        # 特徴量正規化
        features = F.normalize(features, p=2, dim=1)

        # ペアワイズ距離計算
        distances = torch.cdist(features, features, p=2)

        # ラベルマスク
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        neg_mask = 1 - pos_mask

        # トリプレットマージン損失的アプローチ
        pos_distances = distances * pos_mask
        neg_distances = distances * neg_mask + pos_mask * 1e9  # 正例ペアを除外

        # 各サンプルに対して最も近い負例を選択（ハードネガティブ）
        hardest_negatives, _ = torch.min(neg_distances, dim=1)

        # 各サンプルに対する正例の平均距離
        pos_count = pos_mask.sum(dim=1) - 1  # 自分自身を除く
        avg_positives = pos_distances.sum(dim=1) / (pos_count + 1e-8)

        # マージン損失
        loss = F.relu(avg_positives - hardest_negatives + self.margin)

        return loss.mean()

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon) の実装。
    各アンカーに対して複数の正例・負例を考慮し、数値的安定性のためにlogsumexpを使用する。
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features (torch.Tensor): 潜在特徴量 [batch_size, feature_dim]
            labels (torch.Tensor): ラベル [batch_size]

        Returns:
            torch.Tensor: 計算された損失値
        """
        batch_size = features.shape[0]
        # ラベルの形状を [batch_size, 1] に統一
        labels = labels.contiguous().view(-1, 1)

        # バッチサイズが1以下の場合は損失を計算できない
        if batch_size <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # 正例ペアを識別するためのマスクを作成
        # torch.eq(labels, labels.T) は、同じラベルを持つペアでTrueとなる行列を生成
        mask = torch.eq(labels, labels.T).float().to(features.device)

        # 特徴量をL2正規化（コサイン類似度を計算するため）
        features = F.normalize(features, p=2, dim=1)

        # 類似度行列を計算
        # features と features.T の内積を計算し、温度でスケーリング
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 対角要素（自分自身とのペア）は損失計算から除外する
        # `logits_mask` は対角成分が0、その他が1の行列
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=features.device)
        mask = mask * logits_mask

        # 各アンカーに対して正例が1つも存在しない場合のエッジケース処理
        # (例: バッチ内の全サンプルが異なるラベルを持つ場合)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # log(exp(sim)) の分母部分を計算
        # 自分自身を除外した全ペアの類似度でlogsumexpを計算
        # 対角成分に大きな負の値を加算し、exp計算時に0に近づけることで除外する
        logits = similarity_matrix - (1 - logits_mask) * 1e9
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        # 正例ペアに対するlog確率の平均を計算
        # (mask * log_prob).sum(1) で各アンカーの正例ペアに対するlog確率の合計を計算
        # mask.sum(1) で各アンカーの正例の数で割り、平均を求める
        mean_log_prob_pos = (mask * log_prob).sum(1) / torch.clamp(mask.sum(1), min=1.0)

        # 損失を計算（負の対数尤度）し、バッチ全体で平均化
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss

# class ContrastiveLoss(nn.Module):
#     """対比学習損失"""
#
#     def __init__(self, temperature=0.07):
#         super().__init__()
#         self.temperature = temperature
#
#     def forward(self, features, labels):
#         batch_size = features.shape[0]
#         labels = labels.contiguous().view(-1, 1)
#
#         # CLAUDE_ADDED: バッチサイズが小さい場合の対処
#         if batch_size <= 1:
#             return torch.tensor(0.0, device=features.device, requires_grad=True)
#
#         # 正例・負例マスク
#         mask = torch.eq(labels, labels.T).float()
#
#         # 特徴量正規化 - CLAUDE_ADDED: 数値安定性向上
#         features = F.normalize(features, p=2, dim=1, eps=1e-8)
#
#         # 類似度計算 - CLAUDE_ADDED: 温度パラメータをクリップ
#         temperature_clamped = torch.clamp(torch.tensor(self.temperature), min=0.01)
#         similarity_matrix = torch.matmul(features, features.T) / temperature_clamped
#
#         # CLAUDE_ADDED: 類似度をクリップして数値安定性を向上
#         similarity_matrix = torch.clamp(similarity_matrix, min=-10, max=10)
#
#         # 対角成分除去
#         mask = mask - torch.eye(batch_size, device=mask.device)
#
#         # 損失計算 - CLAUDE_ADDED: 数値安定性の改善
#         exp_sim = torch.exp(similarity_matrix)
#         pos_sum = torch.sum(exp_sim * mask, dim=1)
#         neg_sum = torch.sum(exp_sim * (1 - torch.eye(batch_size, device=mask.device)), dim=1)
#
#         # CLAUDE_ADDED: ゼロ除算とlog(0)を防ぐ
#         pos_sum = torch.clamp(pos_sum, min=1e-8)
#         neg_sum = torch.clamp(neg_sum, min=1e-8)
#
#         loss = -torch.log(pos_sum / neg_sum)
#
#         # CLAUDE_ADDED: NaNチェック
#         if torch.any(torch.isnan(loss)):
#             return torch.tensor(0.0, device=features.device, requires_grad=True)
#
#         return loss.mean()


# class ContrastiveLoss(nn.Module):
#     """
#     Supervised Contrastive Lossの正しい実装 - CLAUDE_ADDED: より数学的に正確で安定
#     """
#     def __init__(self, temperature=0.07):
#         super().__init__()
#         self.temperature = temperature
#         # CLAUDE_ADDED: デバッグ用の統計情報追跡
#         self.register_buffer('debug_stats', torch.zeros(5))  # [loss, sim_max, sim_min, pos_pairs, neg_pairs]
#
#     def forward(self, features, labels):
#         batch_size = features.shape[0]
#         if batch_size <= 1:
#             return torch.tensor(0.0, device=features.device, requires_grad=True)
#
#         # 特徴量をL2正規化
#         features = F.normalize(features, p=2, dim=1)
#
#         # 類似度行列を計算し、温度でスケール
#         similarity_matrix = torch.matmul(features, features.T) / self.temperature
#
#         # 自分自身とのペアを除外するためのマスク
#         logits_mask = torch.ones_like(similarity_matrix) - torch.eye(batch_size, device=features.device)
#
#         # 正例ペアを定義するためのマスク（自分自身は含まない）
#         labels = labels.contiguous().view(-1, 1)
#         positive_mask = torch.eq(labels, labels.T).float()
#         positive_mask = positive_mask * logits_mask # 対角成分を除外
#
#         # 正例が存在しないサンプルの場合、損失は0とする
#         if positive_mask.sum() == 0:
#             return torch.tensor(0.0, device=features.device, requires_grad=True)
#
#         # logsumexpを計算する際、自分自身を含めないように対角成分をマスク
#         # 非常に大きな負の値を加算することでexpの結果が0に近くなる
#         logits = similarity_matrix - (1 - logits_mask) * 1e9
#
#         # 分母部分: log(sum(exp)) を安定的に計算
#         log_denominator = torch.logsumexp(logits, dim=1, keepdim=True)
#
#         # log(exp(pos)/sum(exp)) = pos - log(sum(exp))
#         log_prob = similarity_matrix - log_denominator
#
#         # 各サンプルについて、正例ペアに対する損失の平均を計算
#         # positive_mask.sum(1) で各サンプルの正例数で割る
#         mean_log_prob_pos = (positive_mask * log_prob).sum(1) / torch.clamp(positive_mask.sum(1), min=1.0)
#
#         # 損失を計算（負の対数尤度）し、バッチ全体で平均
#         loss = -mean_log_prob_pos
#         final_loss = loss.mean()
#
#         # CLAUDE_ADDED: デバッグ統計の更新
#         with torch.no_grad():
#             sim_max = similarity_matrix[logits_mask.bool()].max().item()
#             sim_min = similarity_matrix[logits_mask.bool()].min().item()
#             n_pos_pairs = positive_mask.sum().item()
#             n_neg_pairs = (logits_mask - positive_mask).sum().item()
#             self.debug_stats[0] = final_loss.item()
#             self.debug_stats[1] = sim_max
#             self.debug_stats[2] = sim_min
#             self.debug_stats[3] = n_pos_pairs
#             self.debug_stats[4] = n_neg_pairs
#
#         return final_loss


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


class ResidualConnection(nn.Module):
    """残差接続付きの線形層"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 2)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return residual + x


class AdaptiveSkipGate(nn.Module):
    """CLAUDE_ADDED: Attention-based Adaptive Gating Mechanism"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # ゲート決定ネットワーク
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),  # 潜在+スキップ特徴量
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, 1)
        )

        # 初期バイアスを0に設定（50%の確率でスタート）
        self.gate_network[-1].bias.data.fill_(0.0)

        # ゲート統計追跡用
        self.register_buffer('gate_history', torch.zeros(1000))  # 最新1000回の履歴
        self.register_buffer('gate_ptr', torch.zeros(1, dtype=torch.long))

    def forward(self, latent_features, skip_features, temperature=1.0):
        """
        Args:
            latent_features: [batch, seq_len, d_model] - 潜在空間からの特徴量
            skip_features: [batch, seq_len, d_model] - スキップ接続からの特徴量
            temperature: ゲートの決断力制御（低いほど決断的）
        """
        batch_size, seq_len = latent_features.shape[:2]

        # グローバル特徴量を抽出（平均プーリング）
        latent_global = latent_features.mean(dim=1)  # [batch, d_model]
        skip_global = skip_features.mean(dim=1)      # [batch, d_model]

        # 特徴量結合
        combined_features = torch.cat([latent_global, skip_global], dim=-1)  # [batch, 2*d_model]

        # ゲート重み計算
        gate_logits = self.gate_network(combined_features)  # [batch, 1]

        # 温度調整Sigmoid（Gumbel-Sigmoidの簡易版）
        gate_weights = torch.sigmoid(gate_logits / temperature)  # [batch, 1]

        # ゲート統計の更新（学習時のみ）
        if self.training:
            with torch.no_grad():
                avg_gate = gate_weights.mean().item()
                ptr = self.gate_ptr.item()
                self.gate_history[ptr] = avg_gate
                self.gate_ptr[0] = (ptr + 1) % 1000

        return gate_weights.unsqueeze(1)  # [batch, 1, 1]

    def get_gate_statistics(self):
        """ゲートの統計情報を取得"""
        valid_history = self.gate_history[self.gate_history != 0]
        if len(valid_history) == 0:
            return {"mean": 0.5, "std": 0.0, "entropy": 0.0}

        mean_gate = valid_history.mean().item()
        std_gate = valid_history.std().item()

        # エントロピー計算（0または1に偏っていないかの指標）
        p = torch.clamp(valid_history, 1e-8, 1-1e-8)
        entropy = -(p * torch.log(p) + (1-p) * torch.log(1-p)).mean().item()

        return {
            "mean": mean_gate,
            "std": std_gate,
            "entropy": entropy,
            "samples": len(valid_history)
        }


class InformationBottleneck(nn.Module):
    """CLAUDE_ADDED: Information Bottleneck for Skip Connections"""

    def __init__(self, d_model, bottleneck_dim=None, dropout=0.1):
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = d_model // 2

        self.encoder = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        # 情報量推定用
        self.bottleneck_dim = bottleneck_dim

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            compressed: 圧縮された特徴量
            info_loss: 情報ボトルネック損失
        """
        # エンコード（情報圧縮）
        compressed = self.encoder(x)  # [batch, seq_len, bottleneck_dim]

        # デコード（情報復元）
        restored = self.decoder(compressed)  # [batch, seq_len, d_model]

        # 情報損失（復元誤差）
        info_loss = F.mse_loss(restored, x, reduction='mean')

        return restored, info_loss


class AdaptiveGatedSkipEncoder(nn.Module):
    """CLAUDE_ADDED: Adaptive Gated Skip Connection Encoder"""

    def __init__(self, input_dim, seq_len, d_model, n_heads, n_layers, dropout,
                 style_latent_dim, skill_latent_dim):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len

        # 基本コンポーネント
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, seq_len)
        self.layer_norm_after_proj = nn.LayerNorm(d_model)
        self.layer_norm_before_heads = nn.LayerNorm(d_model)

        # Transformer層
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4*d_model,
                batch_first=True,
                dropout=dropout,
                activation='gelu'
            ) for _ in range(n_layers)
        ])

        # アテンション付きプーリング
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout
        )
        self.pooling_query = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # 潜在空間ヘッド
        self.style_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, style_latent_dim * 2)
        )

        self.skill_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, skill_latent_dim * 2)
        )

        # 重み初期化
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier初期化で重みを安定化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        batch_size = x.size(0)

        # スキップ接続収集
        skip_connections = []

        # 入力処理
        encoded = self.input_proj(x)
        encoded = self.layer_norm_after_proj(encoded)
        pos_encoded = self.pos_encoding(encoded)

        # Transformer層
        current_input = pos_encoded
        for layer in self.encoder_layers:
            current_input = layer(current_input)
            skip_connections.append(current_input)

        # プーリング
        query = self.pooling_query.unsqueeze(0).expand(batch_size, -1, -1)
        pooled_features, _ = self.attention_pooling(query, current_input, current_input)
        pooled_features = pooled_features.squeeze(1)
        pooled_features = self.layer_norm_before_heads(pooled_features)

        # 潜在空間ヘッド
        style_params = self.style_head(pooled_features)
        skill_params = self.skill_head(pooled_features)

        return {
            'style_mu': style_params[:, :style_params.size(1) // 2],
            'style_logvar': style_params[:, style_params.size(1) // 2:],
            'skill_mu': skill_params[:, :skill_params.size(1) // 2],
            'skill_logvar': skill_params[:, skill_params.size(1) // 2:],
            'skip_connections': skip_connections
        }


class AdaptiveGatedSkipDecoder(nn.Module):
    """CLAUDE_ADDED: Adaptive Gated Skip Connection Decoder"""

    def __init__(self, output_dim, seq_len, d_model, n_heads, n_layers, dropout,
                 style_latent_dim, skill_latent_dim, skip_dropout=0.3):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layers = n_layers

        # 潜在変数プロジェクター
        self.style_projector = nn.Sequential(
            nn.Linear(style_latent_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            ResidualConnection(d_model, dropout)
        )

        self.skill_projector = nn.Sequential(
            nn.Linear(skill_latent_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            ResidualConnection(d_model, dropout)
        )

        # 融合ネットワーク
        self.fusion_layers = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            ResidualConnection(d_model, dropout)
        )

        # CLAUDE_ADDED: Adaptive Skip Gating
        self.adaptive_gates = nn.ModuleList([
            AdaptiveSkipGate(d_model, dropout) for _ in range(n_layers)
        ])

        # CLAUDE_ADDED: Information Bottleneck
        self.information_bottlenecks = nn.ModuleList([
            InformationBottleneck(d_model, d_model // 2, skip_dropout)
            for _ in range(n_layers)
        ])

        # スキップ融合層
        self.skip_fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                ResidualConnection(d_model, dropout)
            ) for _ in range(n_layers)
        ])

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        # デコーダー層
        self.decoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(n_layers)
        ])

        # 出力層
        self.output_layers = nn.Sequential(
            ResidualConnection(d_model, dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim)
        )

        # ゲート損失記録
        self.register_buffer('bottleneck_losses', torch.zeros(n_layers))

    def forward(self, z_style, z_skill, skip_connections, epoch=0, max_epochs=200):
        batch_size = z_style.size(0)

        # 温度スケジュール（学習進行に応じてゲートを決断的に）
        temperature = max(0.1, 2.0 * (1 - epoch / max_epochs))

        # 潜在変数処理
        style_features = self.style_projector(z_style)
        skill_features = self.skill_projector(z_skill)
        fused_features = self.fusion_layers(
            torch.cat([style_features, skill_features], dim=-1)
        )

        # 時系列展開
        memory_from_z = fused_features.unsqueeze(1).expand(-1, self.seq_len, -1)
        decoded_features = self.pos_encoding(memory_from_z)

        # 各層でのAdaptive Gated Skip Connection
        bottleneck_losses = []
        gate_weights_list = []

        for i, (layer, gate, bottleneck) in enumerate(
            zip(self.decoder_layers, self.adaptive_gates, self.information_bottlenecks)
        ):
            # スキップ接続取得
            skip_idx = len(skip_connections) - i - 1
            skip_idx = max(0, min(skip_idx, len(skip_connections) - 1))
            skip = skip_connections[skip_idx]

            # Information Bottleneck適用
            processed_skip, info_loss = bottleneck(skip)
            bottleneck_losses.append(info_loss)

            # Adaptive Gate計算
            gate_weights = gate(decoded_features, processed_skip, temperature)
            gate_weights_list.append(gate_weights.mean().item())

            # ゲート適用
            gated_skip = gate_weights * processed_skip

            # 特徴量融合
            combined_memory = torch.cat([decoded_features, gated_skip], dim=2)
            fused_input = self.skip_fusion_layers[i](combined_memory)

            # Transformer処理
            decoded_features = layer(fused_input)

        # ボトルネック損失記録
        self.bottleneck_losses = torch.stack(bottleneck_losses)

        # 出力
        output = self.output_layers(decoded_features)

        return output, {
            'bottleneck_losses': self.bottleneck_losses,
            'gate_weights': gate_weights_list
        }


class AdaptiveGatedSkipConnectionNet(BaseExperimentModel):
    """CLAUDE_ADDED: Adaptive Gated Skip Connection Network"""

    def __init__(self,
                 input_dim=6,
                 seq_len=100,
                 d_model=128,
                 n_heads=4,
                 n_encoder_layers=2,
                 n_decoder_layers=2,
                 dropout=0.1,
                 skip_dropout=0.3,
                 style_latent_dim=16,
                 skill_latent_dim=16,
                 n_subjects=6,
                 loss_schedule_config: Dict[str, Any] = None,
                 **kwargs):

        super().__init__(input_dim=input_dim, seq_len=seq_len, d_model=d_model,
                         n_heads=n_heads, n_encoder_layers=n_encoder_layers,
                         n_decoder_layers=n_decoder_layers, dropout=dropout,
                         skip_dropout=skip_dropout, style_latent_dim=style_latent_dim,
                         skill_latent_dim=skill_latent_dim, n_subjects=n_subjects,
                         loss_schedule_config=loss_schedule_config, **kwargs)

        self.seq_len = seq_len
        self.d_model = d_model
        self.style_latent_dim = style_latent_dim
        self.skill_latent_dim = skill_latent_dim

        # CLAUDE_ADDED: パラメータログ
        print(f"AdaptiveGatedSkipConnectionNet instantiated:")
        print(f"  d_model: {d_model}, n_heads: {n_heads}")
        print(f"  n_encoder_layers: {n_encoder_layers}, n_decoder_layers: {n_decoder_layers}")
        print(f"  style_latent_dim: {style_latent_dim}, skill_latent_dim: {skill_latent_dim}")
        print(f"  skip_dropout: {skip_dropout}")

        # 損失スケジューラ初期化
        if loss_schedule_config is None:
            loss_schedule_config = {
                'beta_style': {'schedule': 'linear', 'start_epoch': 40, 'end_epoch': 70, 'start_val': 0.0, 'end_val': 0.001},
                'beta_skill': {'schedule': 'linear', 'start_epoch': 40, 'end_epoch': 70, 'start_val': 0.0, 'end_val': 0.0001},
                'orthogonal_loss': {'schedule': 'constant', 'val': 0.1},
                'contrastive_loss': {'schedule': 'constant', 'val': 0.1},
                'manifold_loss': {'schedule': 'constant', 'val': 0.1},
                'style_classification_loss': {'schedule': 'constant', 'val': 0.1},
                'skill_regression_loss': {'schedule': 'constant', 'val': 0.1},
                # CLAUDE_ADDED: ゲート学習用損失
                'gate_entropy_loss': {'schedule': 'constant', 'val': 0.1},
                'gate_balance_loss': {'schedule': 'constant', 'val': 0.05},
                'bottleneck_loss': {'schedule': 'constant', 'val': 0.01}
            }
        self.loss_scheduler = LossWeightScheduler(loss_schedule_config)

        # スケジューラ設定からlossの計算フラグを受け取り
        self.calc_orthogonal_loss = 'orthogonal_loss' in loss_schedule_config
        self.calc_contrastive_loss = 'contrastive_loss' in loss_schedule_config
        self.calc_manifold_loss = 'manifold_loss' in loss_schedule_config
        self.calc_style_subtask = 'style_classification_loss' in loss_schedule_config
        self.calc_skill_subtask = 'skill_regression_loss' in loss_schedule_config

        # エンコーダ・デコーダ
        self.encoder = AdaptiveGatedSkipEncoder(
            input_dim, seq_len, d_model, n_heads, n_encoder_layers,
            dropout, style_latent_dim, skill_latent_dim
        )

        self.decoder = AdaptiveGatedSkipDecoder(
            input_dim, seq_len, d_model, n_heads, n_decoder_layers,
            dropout, style_latent_dim, skill_latent_dim, skip_dropout
        )

        # サブタスクネットワーク
        if self.calc_style_subtask:
            self.style_prototype_network = nn.Sequential(
                nn.Linear(style_latent_dim, style_latent_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(style_latent_dim, style_latent_dim)
            )
            self.register_buffer('style_prototypes', torch.zeros(n_subjects, style_latent_dim))
            self.register_buffer('prototype_counts', torch.zeros(n_subjects))

        if self.calc_skill_subtask:
            self.skill_score_regressor = nn.Sequential(
                nn.Linear(skill_latent_dim, skill_latent_dim//2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(skill_latent_dim//2, 1)
            )

        # 対照学習
        self.contrastive_loss = ImprovedContrastiveLoss()

        # エポック追跡
        self.current_epoch = 0
        self.max_epochs = 200

    def on_epoch_start(self, epoch: int):
        """学習ループからエポックの開始時呼び出されるメソッド"""
        self.loss_scheduler.step(epoch)
        self.current_epoch = epoch

        # ゲート統計をログ出力
        if epoch % 10 == 0:
            self._log_gate_statistics()

    def _log_gate_statistics(self):
        """ゲート統計をログ出力"""
        print(f"\nEpoch {self.current_epoch} Gate Statistics:")
        for i, gate in enumerate(self.decoder.adaptive_gates):
            stats = gate.get_gate_statistics()
            print(f"  Layer {i}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, entropy={stats['entropy']:.3f}")

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, subject_ids: str = None, skill_scores: float = None) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]

        # エンコード
        encoded = self.encoder(x)

        # 潜在変数サンプリング
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])

        # デコード（Adaptive Gating付き）
        reconstructed, decoder_info = self.decoder(
            z_style, z_skill, encoded['skip_connections'],
            self.current_epoch, self.max_epochs
        )

        # サブタスク
        subject_pred = None
        skill_score_pred = None
        if self.calc_style_subtask:
            subject_pred = self._prototype_based_classification(z_style, subject_ids)
        if self.calc_skill_subtask:
            skill_score_pred = self.skill_score_regressor(z_skill)

        # 損失計算
        losses = self.compute_losses(
            x, reconstructed, encoded, z_style, z_skill,
            subject_pred, skill_score_pred, subject_ids, skill_scores,
            decoder_info
        )

        result = {
            'reconstructed': reconstructed,
            'z_style': z_style,
            'z_skill': z_skill,
        }

        return result | losses

    def compute_losses(self, x, reconstructed, encoded, z_style, z_skill,
                      subject_pred, skill_score_pred, subject_ids, skill_scores, decoder_info):
        weights = self.loss_scheduler.get_weights()

        def safe_tensor(tensor, name="unknown"):
            if self.training and hasattr(self, '_debug_nan_check') and self._debug_nan_check:
                if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
                    print(f"Warning: {name} contains NaN or inf, replacing with zeros")
                    return torch.zeros_like(tensor)
            return tensor

        # 基本損失
        losses = {'reconstruction_loss': F.mse_loss(reconstructed, x)}

        # KL損失
        style_mu_clamped = torch.clamp(encoded['style_mu'], min=-5, max=5)
        skill_mu_clamped = torch.clamp(encoded['skill_mu'], min=-5, max=5)
        style_logvar_clamped = torch.clamp(encoded['style_logvar'], min=-8, max=8)
        skill_logvar_clamped = torch.clamp(encoded['skill_logvar'], min=-8, max=8)

        style_kl_terms = 1 + style_logvar_clamped - style_mu_clamped.pow(2) - style_logvar_clamped.exp()
        skill_kl_terms = 1 + skill_logvar_clamped - skill_mu_clamped.pow(2) - skill_logvar_clamped.exp()

        losses['kl_style_loss'] = -0.5 * torch.mean(torch.sum(style_kl_terms, dim=1))
        losses['kl_skill_loss'] = -0.5 * torch.mean(torch.sum(skill_kl_terms, dim=1))

        # CLAUDE_ADDED: Multi-Objective Gating Losses
        # ボトルネック損失
        losses['bottleneck_loss'] = decoder_info['bottleneck_losses'].mean()

        # ゲートエントロピー損失（多様性促進）
        gate_weights = torch.tensor(decoder_info['gate_weights'], device=x.device, requires_grad=True)
        # CLAUDE_ADDED: より厳密な数値安定性のためのクランプ
        gate_weights = torch.clamp(gate_weights, min=1e-6, max=1-1e-6)

        # CLAUDE_ADDED: NaN回避のための安全なエントロピー計算
        log_p = torch.log(gate_weights + 1e-8)
        log_1_minus_p = torch.log(1 - gate_weights + 1e-8)
        gate_entropy = -(gate_weights * log_p + (1-gate_weights) * log_1_minus_p).mean()

        # CLAUDE_ADDED: NaNチェックとフォールバック
        if torch.isnan(gate_entropy) or torch.isinf(gate_entropy):
            gate_entropy = torch.tensor(0.0, device=x.device, requires_grad=True)

        losses['gate_entropy_loss'] = -gate_entropy  # エントロピーを最大化

        # ゲートバランス損失（50%付近を目標）
        losses['gate_balance_loss'] = torch.abs(gate_weights.mean() - 0.5)

        # 直交性損失
        if self.calc_orthogonal_loss:
            z_style_norm = (z_style - z_style.mean(dim=0)) / (z_style.std(dim=0) + 1e-8)
            z_skill_norm = (z_skill - z_skill.mean(dim=0)) / (z_skill.std(dim=0) + 1e-8)
            cross_correlation_matrix = torch.matmul(z_style_norm.T, z_skill_norm) / z_style.shape[0]
            losses['orthogonal_loss'] = torch.mean(cross_correlation_matrix ** 2)

        # サブタスク損失
        if subject_ids is not None:
            all_subjects = ['h.nakamura', 'r.morishita', 'r.yanase', 's.miyama', 's.tahara', 't.hasegawa']
            subject_to_idx = {subj: i for i, subj in enumerate(all_subjects)}
            subject_indices = torch.tensor([subject_to_idx[subj] for subj in subject_ids], device=z_style.device)

            if self.calc_contrastive_loss:
                losses['contrastive_loss'] = self.contrastive_loss(z_style, subject_indices)

            if self.calc_style_subtask and subject_pred is not None:
                losses['style_classification_loss'] = F.cross_entropy(subject_pred, subject_indices)

        if skill_scores is not None:
            if self.calc_skill_subtask and skill_score_pred is not None:
                skill_score_pred_flat = skill_score_pred.squeeze(-1)
                losses['skill_regression_loss'] = F.mse_loss(skill_score_pred_flat, skill_scores)

            if self.calc_manifold_loss:
                losses['manifold_loss'] = self.compute_manifold_loss(z_skill, skill_scores)

        # 総合損失計算
        beta_style_weight = weights.get('beta_style', weights.get('beta', 0.0))
        beta_skill_weight = weights.get('beta_skill', weights.get('beta', 0.0))

        total_loss = (losses['reconstruction_loss']
                      + beta_style_weight * losses['kl_style_loss']
                      + beta_skill_weight * losses['kl_skill_loss']
                      + weights.get('orthogonal_loss', 0.0) * losses.get('orthogonal_loss', torch.tensor(0.0, device=x.device))
                      + weights.get('contrastive_loss', 0.0) * losses.get('contrastive_loss', torch.tensor(0.0, device=x.device))
                      + weights.get('manifold_loss', 0.0) * losses.get('manifold_loss', torch.tensor(0.0, device=x.device))
                      + weights.get('style_classification_loss', 0.0) * losses.get('style_classification_loss', torch.tensor(0.0, device=x.device))
                      + weights.get('skill_regression_loss', 0.0) * losses.get('skill_regression_loss', torch.tensor(0.0, device=x.device))
                      + weights.get('gate_entropy_loss', 0.0) * losses['gate_entropy_loss']
                      + weights.get('gate_balance_loss', 0.0) * losses['gate_balance_loss']
                      + weights.get('bottleneck_loss', 0.0) * losses['bottleneck_loss'])

        losses['total_loss'] = total_loss

        return losses

    def compute_manifold_loss(self, z_skill: torch.Tensor, skill_score: torch.Tensor, min_separation: float = 1.0):
        """熟達多様体形成損失"""
        batch_size = z_skill.size(0)
        device = z_skill.device

        expert_mask = skill_score > 0.0
        novice_mask = skill_score <= 0.0

        n_experts = expert_mask.sum()
        n_novices = novice_mask.sum()

        loss_separation = torch.tensor(0.0, device=device, requires_grad=True)
        loss_expert_cohesion = torch.tensor(0.0, device=device, requires_grad=True)
        loss_novice_cohesion = torch.tensor(0.0, device=device, requires_grad=True)

        # 重心分離損失
        if n_experts > 0 and n_novices > 0:
            expert_centroid = z_skill[expert_mask].mean(dim=0)
            novice_centroid = z_skill[novice_mask].mean(dim=0)
            centroid_distance = torch.norm(expert_centroid - novice_centroid, p=2)
            loss_separation = torch.clamp(min_separation - centroid_distance, min=0.0) ** 2

        # グループ内凝集性
        if n_experts > 1:
            z_skill_experts = z_skill[expert_mask]
            expert_scores = skill_score[expert_mask]
            expert_centroid = z_skill_experts.mean(dim=0)
            expert_distances_to_centroid = torch.norm(z_skill_experts - expert_centroid.unsqueeze(0), p=2, dim=1)
            expert_weights = torch.sigmoid(expert_scores * 2.0)
            loss_expert_cohesion = torch.mean(expert_weights * expert_distances_to_centroid)

        if n_novices > 1:
            z_skill_novices = z_skill[novice_mask]
            novice_scores = skill_score[novice_mask]
            novice_centroid = z_skill_novices.mean(dim=0)
            novice_distances_to_centroid = torch.norm(z_skill_novices - novice_centroid.unsqueeze(0), p=2, dim=1)
            novice_weights = torch.sigmoid(-novice_scores * 1.0)
            loss_novice_cohesion = torch.mean(novice_weights * novice_distances_to_centroid)

        # スキルベース分離
        loss_skill_based_separation = torch.tensor(0.0, device=device, requires_grad=True)
        if batch_size > 1:
            skill_diff_matrix = torch.abs(skill_score.unsqueeze(1) - skill_score.unsqueeze(0))
            z_skill_distance_matrix = torch.cdist(z_skill, z_skill, p=2)

            mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            skill_diffs = skill_diff_matrix[mask]
            z_distances = z_skill_distance_matrix[mask]

            if skill_diffs.numel() > 0:
                target_distances = skill_diffs * 0.5
                distance_violations = torch.clamp(target_distances - z_distances, min=0.0)
                loss_skill_based_separation = torch.mean(distance_violations ** 2)

        # 重み付き総合損失
        alpha_separation = 2.0
        alpha_expert_cohesion = 1.0
        alpha_novice_cohesion = 0.3
        alpha_skill_separation = 1.5

        total_manifold_loss = (
                alpha_separation * loss_separation +
                alpha_expert_cohesion * loss_expert_cohesion +
                alpha_novice_cohesion * loss_novice_cohesion +
                alpha_skill_separation * loss_skill_based_separation
        )

        return total_manifold_loss

    def encode(self, x):
        """エンコードのみ"""
        encoded = self.encoder(x)
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])
        return {'z_style': z_style, 'z_skill': z_skill, 'skip_connections': encoded['skip_connections']}

    def decode(self, z_style, z_skill, skip_connections=None):
        """デコードのみ"""
        if skip_connections is None:
            batch_size = z_style.size(0)
            dummy_skip = torch.zeros(batch_size, self.seq_len, self.d_model, device=z_style.device)
            skip_connections = [dummy_skip.clone() for _ in range(self.decoder.n_layers)]

        trajectory, _ = self.decoder(z_style, z_skill, skip_connections, self.current_epoch, self.max_epochs)
        return {'trajectory': trajectory}

    def _prototype_based_classification(self, z_style: torch.Tensor, subject_ids: list = None):
        """プロトタイプベースのスタイル識別"""
        style_features = self.style_prototype_network(z_style)
        style_features = F.normalize(style_features, p=2, dim=1)

        batch_size = z_style.size(0)

        if self.training and subject_ids is not None:
            return self._update_prototypes_and_classify(style_features, subject_ids)
        else:
            return self._distance_based_classification(style_features)

    def _update_prototypes_and_classify(self, style_features: torch.Tensor, subject_ids: list):
        """学習時：プロトタイプ更新と分類を同時実行"""
        batch_size = style_features.size(0)
        device = style_features.device

        all_subjects = ['h.nakamura', 'r.morishita', 'r.yanase', 's.miyama', 's.tahara', 't.hasegawa']
        subject_to_idx = {subj: i for i, subj in enumerate(all_subjects)}
        subject_indices = [subject_to_idx[subj] for subj in subject_ids]

        similarities = torch.zeros(batch_size, self.style_prototypes.size(0), device=device)

        for i, (feature, subj_idx, subj_id) in enumerate(zip(style_features, subject_indices, subject_ids)):
            if subj_idx < self.style_prototypes.size(0):
                momentum = 0.9
                current_count = self.prototype_counts[subj_idx].item()

                if current_count == 0:
                    self.style_prototypes[subj_idx] = feature.detach()
                    self.prototype_counts[subj_idx] = 1.0
                else:
                    self.style_prototypes[subj_idx] = (
                            momentum * self.style_prototypes[subj_idx] +
                            (1 - momentum) * feature.detach()
                    )
                    self.prototype_counts[subj_idx] += 1.0

        normalized_prototypes = F.normalize(self.style_prototypes, p=2, dim=1)
        similarities = torch.mm(style_features, normalized_prototypes.T)

        temperature = 0.5
        logits = similarities / temperature

        return logits

    def _distance_based_classification(self, style_features: torch.Tensor):
        """テスト時：距離ベース分類"""
        normalized_prototypes = F.normalize(self.style_prototypes, p=2, dim=1)
        similarities = torch.mm(style_features, normalized_prototypes.T)

        temperature = 0.5
        logits = similarities / temperature

        return logits