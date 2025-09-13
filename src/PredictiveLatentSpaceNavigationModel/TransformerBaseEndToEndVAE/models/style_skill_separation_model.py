from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.base_model import BaseExperimentModel
from models.components.loss_weight_scheduler import LossWeightScheduler



class ContrastiveLoss(nn.Module):
    """対比学習損失"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)

        # CLAUDE_ADDED: バッチサイズが小さい場合の対処
        if batch_size <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # 正例・負例マスク
        mask = torch.eq(labels, labels.T).float()

        # 特徴量正規化 - CLAUDE_ADDED: 数値安定性向上
        features = F.normalize(features, p=2, dim=1, eps=1e-8)

        # 類似度計算 - CLAUDE_ADDED: 温度パラメータをクリップ
        temperature_clamped = torch.clamp(torch.tensor(self.temperature), min=0.01)
        similarity_matrix = torch.matmul(features, features.T) / temperature_clamped

        # CLAUDE_ADDED: 類似度をクリップして数値安定性を向上
        similarity_matrix = torch.clamp(similarity_matrix, min=-10, max=10)

        # 対角成分除去
        mask = mask - torch.eye(batch_size, device=mask.device)

        # 損失計算 - CLAUDE_ADDED: 数値安定性の改善
        exp_sim = torch.exp(similarity_matrix)
        pos_sum = torch.sum(exp_sim * mask, dim=1)
        neg_sum = torch.sum(exp_sim * (1 - torch.eye(batch_size, device=mask.device)), dim=1)

        # CLAUDE_ADDED: ゼロ除算とlog(0)を防ぐ
        pos_sum = torch.clamp(pos_sum, min=1e-8)
        neg_sum = torch.clamp(neg_sum, min=1e-8)

        loss = -torch.log(pos_sum / neg_sum)

        # CLAUDE_ADDED: NaNチェック
        if torch.any(torch.isnan(loss)):
            return torch.tensor(0.0, device=features.device, requires_grad=True)

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


class StyleSkillSeparationNetEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 seq_len,
                 d_model,
                 n_heads,
                 n_layers,
                 style_latent_dim,
                 skill_latent_dim
                 ):
        super().__init__()

        # 入力射影
        self.input_proj = nn.Linear(input_dim, d_model)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model)

        # 特徴量抽出Transformer層
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)

        # スタイル潜在空間用ヘッド
        self.style_head = nn.Linear(d_model * seq_len, style_latent_dim * 2)

        # スキル潜在空間用ヘッド
        self.skill_head = nn.Linear(d_model * seq_len, skill_latent_dim * 2)

    def forward(self, x):
        batch_size, _, _  = x.shape

        # 入力射影
        encoded = self.input_proj(x)

        # 位置エンコーディング
        pos_encoded = self.pos_encoding(encoded)

        # 特徴量変換
        attended = self.transformer_encoder(pos_encoded)

        # Transformerの出力をシーケンス次元で平均を取る
        transformed_flat = attended.view(batch_size, -1)

        # スタイル・スキル射影
        style_params = self.style_head(transformed_flat)
        skill_params = self.skill_head(transformed_flat)

        return {
            'style_mu': style_params[:,:style_params.size(1) // 2],
            'style_logvar': style_params[:,style_params.size(1) // 2:],
            'skill_mu': skill_params[:,:skill_params.size(1) // 2],
            'skill_logvar': skill_params[:,skill_params.size(1) // 2:],
        }


class StyleSkillSeparationNetDecoder(nn.Module):
    def __init__(self,
                 output_dim,
                 seq_len,
                 d_model,
                 n_heads,
                 n_layers,
                 style_latent_dim,
                 skill_latent_dim):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model

        # 潜在変数サンプリング層
        self.from_style_latent = nn.Linear(style_latent_dim, d_model * seq_len)
        self.from_skill_latent = nn.Linear(skill_latent_dim, d_model * seq_len)

        # スタイル・スキル合成層
        self.fusion_proj = nn.Linear(2 * d_model, d_model)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformerデコーダ層
        decoder_layers = nn.TransformerEncoderLayer(d_model=d_model,
                                                         nhead=n_heads,
                                                         batch_first=True
                                                         )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layers, num_layers=n_layers)

        # 出力射影
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, z_style, z_skill):
        batch_size, _ = z_style.shape

        # スタイル・スキル潜在変数を系列データに変換
        style_seq = self.from_style_latent(z_style).view(batch_size, self.seq_len, self.d_model)
        skill_seq = self.from_skill_latent(z_skill).view(batch_size, self.seq_len, self.d_model)

        # スタイル・スキル系列データを合成
        concatenated_seq = torch.concat([style_seq, skill_seq], dim=2)
        fusion_seq = self.fusion_proj(concatenated_seq)

        # 位置エンコーディング
        pos_encoded = self.pos_encoding(fusion_seq)

        # Transformer Decode
        transformed = self.transformer_decoder(pos_encoded)

        # 出力射影
        reconstructed = self.output_proj(transformed)

        return reconstructed


class StyleSkillSeparationNet(BaseExperimentModel):
    def __init__(self,
                 input_dim=6,
                 seq_len=100,
                 d_model=128,
                 n_heads=4,
                 n_layers = 2,
                 style_latent_dim = 8,
                 skill_latent_dim = 8,
                 n_subjects = 6,
                 loss_schedule_config: Dict[str, Any] = None,
                 **kwargs
                 ):
        super().__init__(input_dim=input_dim,
                         seq_len=seq_len,
                         d_model=d_model,
                         n_heads=n_heads,
                         n_layers = n_layers,
                         style_latent_dim = style_latent_dim,
                         skill_latent_dim = skill_latent_dim,
                         n_subjects = n_subjects,
                         loss_schedule_config = loss_schedule_config,
                         **kwargs
                         )
        self.seq_len = seq_len
        self.d_model = d_model
        self.style_latent_dim = style_latent_dim
        self.skill_latent_dim = skill_latent_dim

        # 損失関数の重みスケジューラの初期化
        if loss_schedule_config is None:
            loss_schedule_config = {
                'beta':{'schedule': 'linear', 'start_epoch': 0, 'end_epoch': 100, 'start_val': 0.0, 'end_val': 0.01},
                'orthogonal_loss' : {'schedule': 'constant', 'val': 0.1},
                'contrastive_loss': {'schedule': 'constant', 'val': 0.1},
                'style_classification_loss': {'schedule': 'constant', 'val': 0.1},
                'skill_regression_loss': {'schedule': 'constant', 'val': 0.1},
            }
        self.loss_scheduler = LossWeightScheduler(loss_schedule_config)

        # スケジューラ設定からlossの計算フラグを受け取り
        self.calc_orthogonal_loss = 'orthogonal_loss' in loss_schedule_config
        self.calc_contrastive_loss = 'contrastive_loss' in loss_schedule_config
        self.calc_style_subtask = 'style_classification_loss' in loss_schedule_config
        self.calc_skill_subtask = 'skill_regression_loss' in loss_schedule_config



        # エンコーダ定義
        self.encoder = StyleSkillSeparationNetEncoder(input_dim,
                                                      seq_len,
                                                      d_model,
                                                      n_heads,
                                                      n_layers,
                                                      style_latent_dim,
                                                      skill_latent_dim
                                                      )
        # デコーダ定義
        self.decoder = StyleSkillSeparationNetDecoder(input_dim,
                                                      seq_len,
                                                      d_model,
                                                      n_heads,
                                                      n_layers,
                                                      style_latent_dim,
                                                      skill_latent_dim)

        # CLAUDE_ADDED: プロトタイプベースのスタイル識別 (被験者数に依存しない)
        if self.calc_style_subtask:
            self.style_prototype_network = nn.Sequential(
                nn.Linear(style_latent_dim, style_latent_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(style_latent_dim, style_latent_dim)  # プロトタイプ空間への写像
            )
            # プロトタイプを保存するバッファ（学習中に更新）
            self.register_buffer('style_prototypes', torch.zeros(n_subjects, style_latent_dim))
            self.register_buffer('prototype_counts', torch.zeros(n_subjects))

        # 補助タスク用ネット (スキルスコア回帰)
        if self.calc_skill_subtask:
            self.skill_score_regressor = nn.Sequential(
                nn.Linear(skill_latent_dim, skill_latent_dim//2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(skill_latent_dim//2, 1)
            )

        # 対照学習の損失
        self.contrastive_loss = ContrastiveLoss()

    def on_epoch_start(self, epoch: int):
        """学習ループからエポックの開始時呼び出されるメソッド"""
        self.loss_scheduler.step(epoch)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, subject_ids: str = None, skill_scores: float=None) -> Dict[str, torch.Tensor]:
        batch_size, _, _ = x.shape

        # エンコード
        encoded = self.encoder(x)

        # 潜在変数サンプリング
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])

        # デコード
        reconstructed = self.decoder(z_style, z_skill)

        # 補助タスク
        subject_pred = None
        skill_score_pred = None
        if self.calc_style_subtask:
            subject_pred = self._prototype_based_classification(z_style, subject_ids)
        if self.calc_skill_subtask:
            skill_score_pred = self.skill_score_regressor(z_skill)

        # 損失計算
        losses = self.compute_losses(
            x, reconstructed, encoded, z_style, z_skill,
            subject_pred, skill_score_pred,
            subject_ids, skill_scores
        )

        result  = {
            'reconstructed': reconstructed,
            'z_style': z_style,
            'z_skill': z_skill,
        }

        return result | losses


    def compute_losses(self, x, reconstructed, encoded, z_style, z_skill, subject_pred, skill_score_pred, subject_ids, skill_scores):
        # スケジューラから現在の重みを取得
        weights = self.loss_scheduler.get_weights()

        # 再構成損失
        losses = {'reconstruction_loss': F.mse_loss(reconstructed, x)}

        # KLダイバージェンス
        style_logvar_clamped = torch.clamp(encoded['style_logvar'], min=-10, max=10)
        skill_logvar_clamped = torch.clamp(encoded['skill_logvar'], min=-10, max=10)

        losses['kl_style_loss'] = -0.5 * torch.mean(
            torch.sum(1 + style_logvar_clamped - encoded['style_mu'].pow(2)
                      - style_logvar_clamped.exp(), dim=1)
        )
        losses['kl_skill_loss'] = -0.5 * torch.mean(
            torch.sum(1 + skill_logvar_clamped - encoded['skill_mu'].pow(2)
                      - skill_logvar_clamped.exp(), dim=1)
        )

        # 直交性損失(z_styleとz_skillの独立性を促進)
        if self.calc_orthogonal_loss:
            cos_sim = F.cosine_similarity(z_style, z_skill, dim=1)
            losses['orthogonal_loss'] = torch.mean(cos_sim ** 2)

        if subject_ids is not None:
            # CLAUDE_ADDED: 全被験者を固定的なインデックスにマッピング（一貫性を保つ）
            all_subjects = ['h.nakamura', 'r.morishita', 'r.yanase', 's.miyama', 's.tahara', 't.hasegawa']
            subject_to_idx = {subj: i for i, subj in enumerate(all_subjects)}
            subject_indices = torch.tensor([subject_to_idx[subj] for subj in subject_ids],
                                            device=z_style.device)
            # 対照学習
            if self.calc_contrastive_loss:
                losses['contrastive_loss'] = self.contrastive_loss(z_style, subject_indices)

            # スタイル分類サブタスク
            losses['style_classification_loss'] = F.cross_entropy(subject_pred, subject_indices)

        if skill_scores is not None:
            # CLAUDE_ADDED: スキルスコア回帰サブタスク（次元を合わせる）
            # skill_score_pred: [batch, 1] -> [batch] に変換
            skill_score_pred_flat = skill_score_pred.squeeze(-1)
            losses['skill_regression_loss'] = F.mse_loss(skill_score_pred_flat, skill_scores)
        else:
            # CLAUDE_ADDED: skill_scoresがNoneの場合は0の損失を設定
            losses['skill_regression_loss'] = torch.tensor(0.0, device=x.device)

        # 総合損失の計算
        total_loss = (losses['reconstruction_loss']
                      + weights.get('beta', 0.0) * (losses['kl_style_loss'] + losses['kl_skill_loss'])
                      + weights.get('orthogonal_loss', 0.0) * losses.get('orthogonal_loss', torch.tensor(0.0, device=x.device))
                      + weights.get('contrastive_loss', 0.0) * losses.get('contrastive_loss', torch.tensor(0.0, device=x.device))
                      + weights.get('style_classification_loss', 0.0) * losses.get('style_classification_loss', torch.tensor(0.0, device=x.device))
                      + weights.get('skill_regression_loss', 0.0) * losses['skill_regression_loss'])

        losses['total_loss'] = total_loss

        return losses

    def encode(self, x):
        """エンコードのみ"""
        encoded = self.encoder(x)
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])
        return {'z_style': z_style, 'z_skill':z_skill}

    def decode(self, z_style, z_skill):
        """デコードのみ"""
        trajectory = self.decoder(z_style, z_skill)
        return {'trajectory': trajectory}
    
    def _prototype_based_classification(self, z_style: torch.Tensor, subject_ids: list = None):
        """プロトタイプベースのスタイル識別"""
        # CLAUDE_ADDED: プロトタイプ空間への写像
        style_features = self.style_prototype_network(z_style)
        style_features = F.normalize(style_features, p=2, dim=1)  # L2正規化
        
        batch_size = z_style.size(0)
        
        if self.training and subject_ids is not None:
            # 学習時：プロトタイプを更新しながら分類
            return self._update_prototypes_and_classify(style_features, subject_ids)
        else:
            # テスト時：既存プロトタイプとの距離ベース分類
            return self._distance_based_classification(style_features)
    
    def _update_prototypes_and_classify(self, style_features: torch.Tensor, subject_ids: list):
        """学習時：プロトタイプ更新と分類を同時実行"""
        batch_size = style_features.size(0)
        device = style_features.device
        
        # CLAUDE_ADDED: 全被験者を固定的なインデックスにマッピング（一貫性を保つ）
        all_subjects = ['h.nakamura', 'r.morishita', 'r.yanase', 's.miyama', 's.tahara', 't.hasegawa']
        subject_to_idx = {subj: i for i, subj in enumerate(all_subjects)}
        subject_indices = [subject_to_idx[subj] for subj in subject_ids]
        
        # 予測用の類似度行列を計算
        similarities = torch.zeros(batch_size, self.style_prototypes.size(0), device=device)
        
        # バッチ内の各サンプルについてプロトタイプを更新
        for i, (feature, subj_idx, subj_id) in enumerate(zip(style_features, subject_indices, subject_ids)):
            if subj_idx < self.style_prototypes.size(0):
                # プロトタイプの移動平均更新 (momentum=0.9)
                momentum = 0.9
                current_count = self.prototype_counts[subj_idx].item()
                
                if current_count == 0:
                    # 初回: プロトタイプを直接設定
                    self.style_prototypes[subj_idx] = feature.detach()
                    self.prototype_counts[subj_idx] = 1.0
                else:
                    # 移動平均で更新
                    self.style_prototypes[subj_idx] = (
                        momentum * self.style_prototypes[subj_idx] + 
                        (1 - momentum) * feature.detach()
                    )
                    self.prototype_counts[subj_idx] += 1.0
        
        # 正規化されたプロトタイプとの類似度計算
        normalized_prototypes = F.normalize(self.style_prototypes, p=2, dim=1)
        similarities = torch.mm(style_features, normalized_prototypes.T)
        
        # 温度スケーリングで確率分布に変換
        temperature = 0.5
        logits = similarities / temperature
        
        return logits
    
    def _distance_based_classification(self, style_features: torch.Tensor):
        """テスト時：距離ベース分類"""
        device = style_features.device
        
        # 正規化されたプロトタイプとの類似度計算
        normalized_prototypes = F.normalize(self.style_prototypes, p=2, dim=1)
        similarities = torch.mm(style_features, normalized_prototypes.T)
        
        # 温度スケーリング
        temperature = 0.5
        logits = similarities / temperature
        
        return logits
        
    def get_prototype_info(self):
        """プロトタイプの情報を取得（デバッグ用）"""
        return {
            'prototypes': self.style_prototypes.cpu().numpy(),
            'counts': self.prototype_counts.cpu().numpy(),
            'active_prototypes': (self.prototype_counts > 0).sum().item()
        }






