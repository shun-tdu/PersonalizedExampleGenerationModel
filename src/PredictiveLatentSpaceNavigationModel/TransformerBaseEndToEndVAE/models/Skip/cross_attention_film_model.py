# CLAUDE_ADDED: Cross Attention FiLM Skip Connection Model
"""
Cross Attentionベースの特徴変調を持つスキップ接続モデル
- FiLM処理をCross Attentionメカニズムで置き換え
- より柔軟で表現力豊かな潜在変数による特徴変調
- simple_film_adaptive_gate_model.pyをベースに改良
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.base_model import BaseExperimentModel
from models.components.loss_weight_scheduler import LossWeightScheduler


class ImprovedContrastiveLoss(nn.Module):
    """改良された対照学習損失"""
    def __init__(self, temperature=0.07, margin=1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, features, labels):
        batch_size = features.shape[0]
        features = F.normalize(features, p=2, dim=1)
        distances = torch.cdist(features, features, p=2)

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        neg_mask = 1 - pos_mask

        pos_distances = distances * pos_mask
        neg_distances = distances * neg_mask + pos_mask * 1e9

        hardest_negatives, _ = torch.min(neg_distances, dim=1)
        pos_count = pos_mask.sum(dim=1) - 1
        avg_positives = pos_distances.sum(dim=1) / (pos_count + 1e-8)

        loss = F.relu(avg_positives - hardest_negatives + self.margin)
        return loss.mean()


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


class CrossAttentionModulation(nn.Module):
    """Cross Attentionベースの特徴変調層"""
    def __init__(self, d_model, condition_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        # 条件変数（潜在変数）用のプロジェクション
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Cross Attention用のQuery, Key, Value射影
        self.q_proj = nn.Linear(d_model, d_model)  # Query: 特徴量から
        self.k_proj = nn.Linear(d_model, d_model)  # Key: 条件変数から
        self.v_proj = nn.Linear(d_model, d_model)  # Value: 条件変数から

        # 出力射影
        self.out_proj = nn.Linear(d_model, d_model)

        # 残差接続とレイヤー正規化
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # アテンション重み正規化のための温度パラメータ
        self.temperature = nn.Parameter(torch.ones(1) * math.sqrt(self.head_dim))

    def forward(self, features, condition):
        """
        Args:
            features: [batch, seq_len, d_model] - 変調対象の特徴量
            condition: [batch, condition_dim] - 条件変数（潜在変数）
        Returns:
            modulated_features: [batch, seq_len, d_model] - 変調された特徴量
        """
        batch_size, seq_len, _ = features.shape
        residual = features

        # 条件変数を特徴量空間に射影
        condition_features = self.condition_proj(condition)  # [batch, d_model]
        condition_features = condition_features.unsqueeze(1)  # [batch, 1, d_model]

        # Query, Key, Value計算
        Q = self.q_proj(features)  # [batch, seq_len, d_model]
        K = self.k_proj(condition_features)  # [batch, 1, d_model]
        V = self.v_proj(condition_features)  # [batch, 1, d_model]

        # Multi-head reshaping
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
        K = K.view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)      # [batch, n_heads, 1, head_dim]
        V = V.view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)      # [batch, n_heads, 1, head_dim]

        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.temperature  # [batch, n_heads, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # アテンション適用
        attended_output = torch.matmul(attention_weights, V)  # [batch, n_heads, seq_len, head_dim]

        # Multi-head結合
        attended_output = attended_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )  # [batch, seq_len, d_model]

        # 出力射影
        output = self.out_proj(attended_output)

        # 残差接続とレイヤー正規化
        modulated_features = self.layer_norm(residual + output)

        return modulated_features


class SimpleAdaptiveGate(nn.Module):
    """シンプルなAdaptive Gate機構"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # シンプルなゲートネットワーク（軽量化）
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

        # ゲート統計追跡用
        self.register_buffer('gate_history', torch.zeros(100))
        self.register_buffer('gate_ptr', torch.zeros(1, dtype=torch.long))

    def forward(self, latent_features, skip_features):
        """
        Args:
            latent_features: [batch, seq_len, d_model] - 潜在空間からの特徴量
            skip_features: [batch, seq_len, d_model] - スキップ接続からの特徴量
        """
        # グローバル特徴量を抽出（平均プーリング）
        latent_global = latent_features.mean(dim=1)  # [batch, d_model]
        skip_global = skip_features.mean(dim=1)      # [batch, d_model]

        # 特徴量結合
        combined_features = torch.cat([latent_global, skip_global], dim=-1)  # [batch, 2*d_model]

        # ゲート重み計算
        gate_weights = self.gate_network(combined_features)  # [batch, 1]

        # ゲート統計の更新（学習時のみ）
        if self.training:
            with torch.no_grad():
                avg_gate = gate_weights.mean().item()
                ptr = self.gate_ptr.item()
                self.gate_history[ptr] = avg_gate
                self.gate_ptr[0] = (ptr + 1) % 100

        return gate_weights.unsqueeze(1)  # [batch, 1, 1]

    def get_gate_statistics(self):
        """ゲートの統計情報を取得"""
        valid_history = self.gate_history[self.gate_history != 0]
        if len(valid_history) == 0:
            return {"mean": 0.5, "std": 0.0}

        return {
            "mean": valid_history.mean().item(),
            "std": valid_history.std().item(),
            "samples": len(valid_history)
        }


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


class CrossAttentionFilmEncoder(nn.Module):
    """Cross Attention FiLM機構付きエンコーダ"""
    def __init__(self,
                 input_dim,
                 seq_len,
                 d_model,
                 n_heads,
                 n_layers,
                 dropout,
                 style_latent_dim,
                 skill_latent_dim):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len

        # 入力射影
        self.input_proj = nn.Linear(input_dim, d_model)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        # スキップ接続用のレイヤーリスト
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                batch_first=True,
                dropout=dropout,
                activation='gelu'
            ) for _ in range(n_layers)
        ])

        # スタイル潜在空間用ヘッド
        self.style_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, style_latent_dim * 2)
        )

        # スキル潜在空間用ヘッド
        self.skill_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, skill_latent_dim * 2)
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
        # スキップ接続リスト
        skip_connections = []

        # 入力射影
        encoded = self.input_proj(x)

        # 位置エンコーディング
        pos_encoded = self.pos_encoding(encoded)

        # Transformer層（スキップ接続用）
        current_input = pos_encoded
        for layer in self.encoder_layers:
            current_input = layer(current_input)
            skip_connections.append(current_input)

        # シーケンス次元で平均プーリング
        pooled_features = torch.mean(current_input, dim=1)  # [batch_size, d_model]

        # スタイル・スキル射影
        style_params = self.style_head(pooled_features)
        skill_params = self.skill_head(pooled_features)

        return {
            'style_mu': style_params[:, :style_params.size(1) // 2],
            'style_logvar': style_params[:, style_params.size(1) // 2:],
            'skill_mu': skill_params[:, :skill_params.size(1) // 2],
            'skill_logvar': skill_params[:, skill_params.size(1) // 2:],
            'skip_connections': skip_connections
        }


class CrossAttentionFilmDecoder(nn.Module):
    """TransformerDecoderLayerベースのCross Attention FiLM機構付きデコーダ"""
    def __init__(self,
                 output_dim,
                 seq_len,
                 d_model,
                 n_heads,
                 n_layers,
                 dropout,
                 style_latent_dim,
                 skill_latent_dim,
                 skip_layers=1):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.skip_layers = min(skip_layers, n_layers)

        # 潜在変数サンプリング層（simple_film_adaptive_gate_model.pyと同じ）
        self.from_style_latent = nn.Sequential(
            nn.Linear(style_latent_dim, d_model),
            nn.ReLU(),
            ResidualConnection(d_model, dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * seq_len)
        )
        self.from_skill_latent = nn.Sequential(
            nn.Linear(skill_latent_dim, d_model),
            nn.ReLU(),
            ResidualConnection(d_model, dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * seq_len)
        )

        # スタイル・スキル合成層
        self.fusion_proj = nn.Linear(2 * d_model, d_model)

        # 潜在変数をメモリとして使用するための射影層
        self.style_memory_proj = nn.Sequential(
            nn.Linear(style_latent_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.skill_memory_proj = nn.Sequential(
            nn.Linear(skill_latent_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # SimpleAdaptiveGates（最後のskip_layers層のみ）
        self.adaptive_gates = nn.ModuleList([
            SimpleAdaptiveGate(d_model, dropout) for _ in range(self.skip_layers)
        ])

        # スキップ接続融合層
        self.skip_fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model)
            ) for _ in range(self.skip_layers)
        ])

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        # TransformerDecoderLayer（Cross Attentionを含む）
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
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
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, z_style, z_skill, skip_connections):
        batch_size = z_style.size(0)
        gate_weights_list = []

        # 潜在変数をメモリとして準備
        style_memory = self.style_memory_proj(z_style).unsqueeze(1)  # [batch, 1, d_model]
        skill_memory = self.skill_memory_proj(z_skill).unsqueeze(1)  # [batch, 1, d_model]
        # スタイル・スキルメモリを結合
        latent_memory = torch.cat([style_memory, skill_memory], dim=1)  # [batch, 2, d_model]

        if self.skip_layers == 0:
            # skip_layers=0: 純粋なTransformerDecoderによるCross Attention
            # スタイル・スキル潜在変数を系列データに変換
            style_seq = self.from_style_latent(z_style).view(batch_size, self.seq_len, self.d_model)
            skill_seq = self.from_skill_latent(z_skill).view(batch_size, self.seq_len, self.d_model)

            # スタイル・スキル系列データを合成
            concatenated_seq = torch.concat([style_seq, skill_seq], dim=2)
            fusion_seq = self.fusion_proj(concatenated_seq)

            # 位置エンコーディング
            decoded_features = self.pos_encoding(fusion_seq)

            # TransformerDecoderLayer使用（Cross Attentionで潜在変数参照）
            for layer in self.decoder_layers:
                decoded_features = layer(
                    tgt=decoded_features,           # Target: デコード中の特徴量
                    memory=latent_memory,           # Memory: 潜在変数（Style + Skill）
                    tgt_mask=None,                  # Self-attentionマスクなし
                    memory_mask=None,               # Cross-attentionマスクなし
                    tgt_key_padding_mask=None,      # パディングマスクなし
                    memory_key_padding_mask=None    # メモリパディングマスクなし
                )

        else:
            # skip_layers>0: TransformerDecoder + Adaptive Gate処理
            # 潜在変数を系列データに変換
            style_seq = self.from_style_latent(z_style).view(batch_size, self.seq_len, self.d_model)
            skill_seq = self.from_skill_latent(z_skill).view(batch_size, self.seq_len, self.d_model)

            # スタイル・スキル系列データを合成
            concatenated_seq = torch.concat([style_seq, skill_seq], dim=2)
            decoded_features = self.fusion_proj(concatenated_seq)

            # 位置エンコーディング
            decoded_features = self.pos_encoding(decoded_features)

            # 各層でのAdaptive Gated Skip Connection + TransformerDecoder
            for i, layer in enumerate(self.decoder_layers):
                # TransformerDecoderLayer処理（Cross Attentionで潜在変数参照）
                transformed_features = layer(
                    tgt=decoded_features,           # Target: デコード中の特徴量
                    memory=latent_memory,           # Memory: 潜在変数（Style + Skill）
                    tgt_mask=None,                  # Self-attentionマスクなし
                    memory_mask=None,               # Cross-attentionマスクなし
                    tgt_key_padding_mask=None,      # パディングマスクなし
                    memory_key_padding_mask=None    # メモリパディングマスクなし
                )

                # スキップ接続適用判定（最後のskip_layers層のみ）
                skip_layer_idx = i - (self.n_layers - self.skip_layers)
                if skip_layer_idx >= 0 and self.skip_layers > 0:
                    # 対応するエンコーダ層のスキップ接続を取得
                    encoder_skip_idx = len(skip_connections) - 1 - skip_layer_idx
                    encoder_skip_idx = max(0, min(encoder_skip_idx, len(skip_connections) - 1))
                    skip = skip_connections[encoder_skip_idx]

                    # Adaptive Gate計算
                    gate = self.adaptive_gates[skip_layer_idx]
                    gate_weights = gate(transformed_features, skip)
                    gate_weights_list.append(gate_weights.mean().item())

                    # ゲート適用
                    gated_skip = gate_weights * skip

                    # 特徴量融合
                    combined_memory = torch.cat([transformed_features, gated_skip], dim=2)
                    decoded_features = self.skip_fusion_layers[skip_layer_idx](combined_memory)
                else:
                    # スキップ接続なしの場合：TransformerDecoder結果をそのまま使用
                    decoded_features = transformed_features

        # 共通の出力層
        output = self.output_layers(decoded_features)

        return output, {'gate_weights': gate_weights_list}


class CrossAttentionFilmNet(BaseExperimentModel):
    """CLAUDE_ADDED: Cross Attention FiLM Skip Connection Network"""

    def __init__(self,
                 input_dim=6,
                 seq_len=100,
                 d_model=128,
                 n_heads=4,
                 n_encoder_layers=4,
                 n_decoder_layers=4,
                 dropout=0.1,
                 style_latent_dim=16,
                 skill_latent_dim=16,
                 n_subjects=6,
                 skip_layers=1,
                 loss_schedule_config: Dict[str, Any] = None,
                 **kwargs):

        super().__init__(input_dim=input_dim, seq_len=seq_len, d_model=d_model,
                         n_heads=n_heads, n_encoder_layers=n_encoder_layers,
                         n_decoder_layers=n_decoder_layers, dropout=dropout,
                         style_latent_dim=style_latent_dim, skill_latent_dim=skill_latent_dim,
                         n_subjects=n_subjects, skip_layers=skip_layers,
                         loss_schedule_config=loss_schedule_config, **kwargs)

        self.seq_len = seq_len
        self.d_model = d_model
        self.style_latent_dim = style_latent_dim
        self.skill_latent_dim = skill_latent_dim
        self.skip_layers = skip_layers

        print(f"CrossAttentionFilmNet instantiated:")
        print(f"  d_model: {d_model}, n_heads: {n_heads}")
        print(f"  encoder_layers: {n_encoder_layers}, decoder_layers: {n_decoder_layers}")
        print(f"  skip_layers: {skip_layers} (from deepest layers)")
        print(f"  latent_dims: style={style_latent_dim}, skill={skill_latent_dim}")
        print(f"  Cross Attention: FiLM処理をCross Attentionで置き換え")

        # 損失スケジューラ初期化
        if loss_schedule_config is None:
            loss_schedule_config = {
                'beta_style': {'schedule': 'linear', 'start_epoch': 21, 'end_epoch': 50, 'start_val': 0.0, 'end_val': 0.0001},
                'beta_skill': {'schedule': 'linear', 'start_epoch': 21, 'end_epoch': 50, 'start_val': 0.0, 'end_val': 0.0001},
                'orthogonal_loss': {'schedule': 'linear', 'start_epoch': 30, 'end_epoch': 60, 'start_val': 0.0, 'end_val': 0.8},
                'contrastive_loss': {'schedule': 'constant', 'val': 0.1},
                'style_classification_loss': {'schedule': 'linear', 'start_epoch': 41, 'end_epoch': 70, 'start_val': 0.0, 'end_val': 1.5},
                'skill_regression_loss': {'schedule': 'linear', 'start_epoch': 20, 'end_epoch': 60, 'start_val': 0.0, 'end_val': 0.2},
                'manifold_loss': {'schedule': 'constant', 'val': 0.1}
            }

        self.loss_scheduler = LossWeightScheduler(loss_schedule_config)

        # 損失計算フラグ
        self.calc_orthogonal_loss = 'orthogonal_loss' in loss_schedule_config
        self.calc_contrastive_loss = 'contrastive_loss' in loss_schedule_config
        self.calc_manifold_loss = 'manifold_loss' in loss_schedule_config
        self.calc_style_subtask = 'style_classification_loss' in loss_schedule_config
        self.calc_skill_subtask = 'skill_regression_loss' in loss_schedule_config

        # エンコーダ・デコーダ
        self.encoder = CrossAttentionFilmEncoder(
            input_dim, seq_len, d_model, n_heads, n_encoder_layers,
            dropout, style_latent_dim, skill_latent_dim
        )

        self.decoder = CrossAttentionFilmDecoder(
            input_dim, seq_len, d_model, n_heads, n_decoder_layers,
            dropout, style_latent_dim, skill_latent_dim, skip_layers
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
                nn.Linear(skill_latent_dim, skill_latent_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(skill_latent_dim // 2, 1)
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
            print(f"  Skip Layer {i}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

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

        # デコード（Cross Attention + Adaptive Gating付き）
        reconstructed, decoder_info = self.decoder(z_style, z_skill, encoded['skip_connections'])

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
            subject_pred, skill_score_pred, subject_ids, skill_scores
        )

        result = {
            'reconstructed': reconstructed,
            'z_style': z_style,
            'z_skill': z_skill,
        }

        return result | losses

    def compute_losses(self, x, reconstructed, encoded, z_style, z_skill,
                      subject_pred, skill_score_pred, subject_ids, skill_scores):
        weights = self.loss_scheduler.get_weights()

        # 基本損失
        losses = {'reconstruction_loss': F.mse_loss(reconstructed, x)}

        # KL損失
        style_kl_terms = 1 + encoded['style_logvar'] - encoded['style_mu'].pow(2) - encoded['style_logvar'].exp()
        skill_kl_terms = 1 + encoded['skill_logvar'] - encoded['skill_mu'].pow(2) - encoded['skill_logvar'].exp()

        losses['kl_style_loss'] = -0.5 * torch.mean(torch.sum(style_kl_terms, dim=1))
        losses['kl_skill_loss'] = -0.5 * torch.mean(torch.sum(skill_kl_terms, dim=1))

        # 直交性損失
        if self.calc_orthogonal_loss:
            z_style_norm = (z_style - z_style.mean(dim=0)) / (z_style.std(dim=0) + 1e-8)
            z_skill_norm = (z_skill - z_skill.mean(dim=0)) / (z_skill.std(dim=0) + 1e-8)
            cross_correlation_matrix = torch.matmul(z_style_norm.T, z_skill_norm) / z_style.shape[0]
            losses['orthogonal_loss'] = torch.mean(cross_correlation_matrix ** 2)

        # サブタスク損失
        if subject_ids is not None:
            all_subjects = ['h.nakamura', 'r.morishita', 'r.yanase', 's.miyama', 's.tahara', 't.hasegawa', 'k.shindo', 'm.kashiwagi']
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
                      + weights.get('skill_regression_loss', 0.0) * losses.get('skill_regression_loss', torch.tensor(0.0, device=x.device)))

        losses['total_loss'] = total_loss
        return losses

    def compute_manifold_loss(self, z_skill: torch.Tensor, skill_score: torch.Tensor, min_separation: float = 1.0):
        """熟達多様体形成損失"""
        batch_size = z_skill.size(0)
        device = z_skill.device

        skill_score_threshold = 0.9

        expert_mask = skill_score > skill_score_threshold
        novice_mask = skill_score <= skill_score_threshold

        n_experts = expert_mask.sum()
        n_novices = novice_mask.sum()

        loss_separation = torch.tensor(0.0, device=device, requires_grad=True)
        loss_expert_cohesion = torch.tensor(0.0, device=device, requires_grad=True)

        # 重心分離損失
        if n_experts > 0 and n_novices > 0:
            expert_centroid = z_skill[expert_mask].mean(dim=0)
            novice_centroid = z_skill[novice_mask].mean(dim=0)
            centroid_distance = torch.norm(expert_centroid - novice_centroid, p=2)
            loss_separation = torch.clamp(min_separation - centroid_distance, min=0.0) ** 2

        # グループ内凝集性
        if n_experts > 1:
            z_skill_experts = z_skill[expert_mask]
            expert_centroid = z_skill_experts.mean(dim=0)
            expert_distances_to_centroid = torch.norm(z_skill_experts - expert_centroid.unsqueeze(0), p=2, dim=1)
            loss_expert_cohesion = torch.mean(expert_distances_to_centroid)

        total_manifold_loss = 2.0 * loss_separation + 1.0 * loss_expert_cohesion
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

        trajectory, _ = self.decoder(z_style, z_skill, skip_connections)
        return {'trajectory': trajectory}

    def _prototype_based_classification(self, z_style: torch.Tensor, subject_ids: list = None):
        """プロトタイプベースのスタイル識別"""
        style_features = self.style_prototype_network(z_style)
        style_features = F.normalize(style_features, p=2, dim=1)

        if self.training and subject_ids is not None:
            return self._update_prototypes_and_classify(style_features, subject_ids)
        else:
            return self._distance_based_classification(style_features)

    def _update_prototypes_and_classify(self, style_features: torch.Tensor, subject_ids: list):
        """学習時：プロトタイプ更新と分類を同時実行"""
        batch_size = style_features.size(0)
        device = style_features.device

        all_subjects = ['h.nakamura', 'r.morishita', 'r.yanase', 's.miyama', 's.tahara', 't.hasegawa', 'k.shindo', 'm.kashiwagi']
        subject_to_idx = {subj: i for i, subj in enumerate(all_subjects)}
        subject_indices = [subject_to_idx[subj] for subj in subject_ids]

        for i, (feature, subj_idx) in enumerate(zip(style_features, subject_indices)):
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