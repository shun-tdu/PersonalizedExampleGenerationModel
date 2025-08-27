import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """時系列用位置エンコーディング"""

    def __init__(self, d_model, max_len=1000):
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


class StyleSkillAttentionLayer(nn.Module):
    """スタイル・スキル分離に特化した注意層"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        # 共通のTransformerLayer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )

        # スタイル・スキル分岐のための射影
        self.style_projection = nn.Linear(d_model, d_model)
        self.skill_projection = nn.Linear(d_model, d_model)

        # 動的重み学習（どちらの表現を重視するか）
        self.attention_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # 共通変換
        transformed = self.transformer_layer(x)

        # スタイル・スキル射影
        style_features = self.style_projection(transformed)
        skill_features = self.skill_projection(transformed)

        # 動的重み計算
        pooled = transformed.mean(dim=1)  # [batch, d_model]
        weights = self.attention_gate(pooled)  # [batch, 2]

        # 重み付き結合
        weighted_output = (weights[:, 0:1, None] * style_features +
                           weights[:, 1:2, None] * skill_features)

        return {
            'output': weighted_output,
            'style_features': style_features,
            'skill_features': skill_features,
            'attention_weights': weights
        }


class EndToEndTransformerEncoder(nn.Module):
    """純粋End-to-Endエンコーダ"""

    def __init__(self,
                 input_dim,
                 d_model,
                 n_heads,
                 n_layers,
                 style_latent_dim,
                 skill_latent_dim):
        super().__init__()

        # 入力射影（特徴量を事前定義しない）
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model)

        # スタイル・スキル分離Transformer層
        self.attention_layers = nn.ModuleList([
            StyleSkillAttentionLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])

        # 最終的な特徴集約
        self.style_aggregator = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.skill_aggregator = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 潜在変数パラメータ生成
        self.style_head = nn.Linear(d_model // 2, style_latent_dim * 2)
        self.skill_head = nn.Linear(d_model // 2, skill_latent_dim * 2)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim] - 生の軌道データ
        Returns:
            encoding結果
        """
        # 入力射影
        x = self.input_projection(x) * math.sqrt(x.size(-1))

        # 位置エンコーディング
        x = self.pos_encoding(x)

        # 段階的な特徴抽出と分離
        all_attention_weights = []
        style_features_list = []
        skill_features_list = []

        current = x
        for layer in self.attention_layers:
            layer_output = layer(current)
            current = layer_output['output']

            style_features_list.append(layer_output['style_features'])
            skill_features_list.append(layer_output['skill_features'])
            all_attention_weights.append(layer_output['attention_weights'])

        # 層間での特徴統合（各層の寄与を学習）
        # スタイル特徴の統合
        stacked_style = torch.stack(style_features_list, dim=0)  # [n_layers, batch, seq, d_model]
        style_pooled = stacked_style.mean(dim=(0, 2))  # [batch, d_model]
        style_aggregated = self.style_aggregator(style_pooled)

        # スキル特徴の統合
        stacked_skill = torch.stack(skill_features_list, dim=0)
        skill_pooled = stacked_skill.mean(dim=(0, 2))
        skill_aggregated = self.skill_aggregator(skill_pooled)

        # 潜在変数パラメータ
        style_params = self.style_head(style_aggregated)
        skill_params = self.skill_head(skill_aggregated)

        style_mu = style_params[:, :style_params.size(1) // 2]
        style_logvar = style_params[:, style_params.size(1) // 2:]
        skill_mu = skill_params[:, :skill_params.size(1) // 2]
        skill_logvar = skill_params[:, skill_params.size(1) // 2:]

        return {
            'style_mu': style_mu,
            'style_logvar': style_logvar,
            'skill_mu': skill_mu,
            'skill_logvar': skill_logvar,
            'attention_weights': all_attention_weights,
            'final_features': current
        }


class TransformerDecoder(nn.Module):
    """Transformer デコーダ"""

    def __init__(self,
                 style_latent_dim,
                 skill_latent_dim,
                 d_model,
                 n_heads,
                 n_layers,
                 output_dim):
        super().__init__()

        self.d_model = d_model

        # 潜在変数から初期メモリ生成
        self.latent_to_memory = nn.Sequential(
            nn.Linear(style_latent_dim + skill_latent_dim, d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Transformerデコーダ層
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(n_layers)
        ])

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model)

        # 出力射影
        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, z_style, z_skill, target_length):
        batch_size = z_style.size(0)

        # 潜在変数結合
        z_combined = torch.cat([z_style, z_skill], dim=1)

        # メモリ生成（全時刻で共通）
        memory_base = self.latent_to_memory(z_combined)  # [batch, d_model]
        memory = memory_base.unsqueeze(1).expand(-1, target_length, -1)

        # 初期ターゲットシーケンス（学習可能な埋め込み）
        tgt = torch.zeros(batch_size, target_length, self.d_model, device=z_style.device)

        # 位置エンコーディング適用
        tgt = self.pos_encoding(tgt)

        # 因果マスク（自己回帰的生成のため）
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_length).to(z_style.device)

        # Transformerデコーダ適用
        current = tgt
        for decoder_layer in self.decoder_layers:
            current = decoder_layer(current, memory, tgt_mask=tgt_mask)

        # 出力射影
        reconstructed = self.output_projection(current)

        return reconstructed


class ContrastiveLoss(nn.Module):
    """対比学習損失"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)

        # 正例・負例マスク
        mask = torch.eq(labels, labels.T).float()

        # 特徴量正規化
        features = F.normalize(features, p=2, dim=1)

        # 類似度計算
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 対角成分除去
        mask = mask - torch.eye(batch_size, device=mask.device)

        # 損失計算
        exp_sim = torch.exp(similarity_matrix)
        pos_sum = torch.sum(exp_sim * mask, dim=1)
        neg_sum = torch.sum(exp_sim * (1 - torch.eye(batch_size, device=mask.device)), dim=1)

        loss = -torch.log(pos_sum / (neg_sum + 1e-8))
        return loss.mean()


class EndToEndTransformerVAE(nn.Module):
    """純粋End-to-End Transformer VAE"""

    def __init__(self,
                 input_dim,
                 d_model=256,
                 n_heads=8,
                 n_encoder_layers=6,
                 n_decoder_layers=3,
                 style_latent_dim=64,
                 skill_latent_dim=64,
                 beta=0.001,
                 contrastive_weight=0.1):
        super().__init__()

        self.beta = beta
        self.contrastive_weight = contrastive_weight

        # エンコーダ・デコーダ
        self.encoder = EndToEndTransformerEncoder(
            input_dim, d_model, n_heads, n_encoder_layers,
            style_latent_dim, skill_latent_dim
        )

        self.decoder = TransformerDecoder(
            style_latent_dim, skill_latent_dim, d_model,
            n_heads, n_decoder_layers, input_dim
        )

        # 補助タスク（分離を促進）
        self.subject_classifier = nn.Sequential(
            nn.Linear(style_latent_dim, style_latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(style_latent_dim // 2, 50)  # 仮の被験者数
        )

        self.skill_regressor = nn.Sequential(
            nn.Linear(skill_latent_dim, skill_latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(skill_latent_dim // 2, 1)  # パフォーマンススコア
        )

        # 損失関数
        self.contrastive_loss = ContrastiveLoss()

        # 分離を促進するための敵対的損失
        self.style_discriminator = nn.Sequential(
            nn.Linear(style_latent_dim, style_latent_dim // 2),
            nn.ReLU(),
            nn.Linear(style_latent_dim // 2, skill_latent_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, subject_ids=None, is_expert=None):
        batch_size, seq_len, _ = x.shape

        # エンコード
        encoded = self.encoder(x)

        # 潜在変数サンプリング
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])

        # デコード
        reconstructed = self.decoder(z_style, z_skill, seq_len)

        # 補助タスク予測
        subject_pred = self.subject_classifier(z_style)
        skill_pred = self.skill_regressor(z_skill)

        # 敵対的予測（z_styleからz_skillを予測→できないほど良い）
        fake_skill = self.style_discriminator(z_style)

        # 損失計算
        losses = self.compute_losses(
            x, reconstructed, encoded, z_style, z_skill,
            subject_pred, skill_pred, fake_skill,
            subject_ids, is_expert
        )

        return {
            'reconstructed': reconstructed,
            'z_style': z_style,
            'z_skill': z_skill,
            'subject_pred': subject_pred,
            'skill_pred': skill_pred,
            'losses': losses,
            'attention_weights': encoded['attention_weights']
        }

    def compute_losses(self, x, reconstructed, encoded, z_style, z_skill,
                       subject_pred, skill_pred, fake_skill, subject_ids, is_expert):
        losses = {}

        # 再構成損失
        losses['reconstruction'] = F.mse_loss(reconstructed, x)

        # KL発散
        losses['kl_style'] = -0.5 * torch.mean(
            torch.sum(1 + encoded['style_logvar'] - encoded['style_mu'].pow(2)
                      - encoded['style_logvar'].exp(), dim=1))

        losses['kl_skill'] = -0.5 * torch.mean(
            torch.sum(1 + encoded['skill_logvar'] - encoded['skill_mu'].pow(2)
                      - encoded['skill_logvar'].exp(), dim=1))

        # 補助タスク損失
        if subject_ids is not None:
            # 文字列ラベルを数値に変換
            subject_to_idx = {subj: i for i, subj in enumerate(set(subject_ids))}
            subject_indices = torch.tensor([subject_to_idx[subj] for subj in subject_ids],
                                           device=z_style.device)
            losses['subject_classification'] = F.cross_entropy(subject_pred, subject_indices)

            # 対比学習（同じ被験者のスタイルは近く）
            losses['style_contrastive'] = self.contrastive_loss(z_style, subject_indices)

        if is_expert is not None:
            losses['skill_regression'] = F.mse_loss(skill_pred.squeeze(), is_expert.float())

        # 敵対的損失（スタイル-スキル分離促進）
        losses['adversarial'] = F.mse_loss(fake_skill, torch.zeros_like(fake_skill))

        # 直交性制約（z_style と z_skill が独立）
        correlation = torch.mean(z_style.unsqueeze(2) * z_skill.unsqueeze(1))
        losses['orthogonality'] = torch.abs(correlation)

        # 総損失
        total_loss = (losses['reconstruction'] +
                      self.beta * (losses['kl_style'] + losses['kl_skill']) +
                      0.3 * losses.get('subject_classification', 0) +
                      0.2 * losses.get('skill_regression', 0) +
                      self.contrastive_weight * losses.get('style_contrastive', 0) +
                      0.1 * losses['adversarial'] +
                      0.1 * losses['orthogonality'])

        losses['total'] = total_loss
        return losses

    def encode(self, x):
        """エンコードのみ"""
        encoded = self.encoder(x)
        z_style = self.reparameterize(encoded['style_mu'], encoded['style_logvar'])
        z_skill = self.reparameterize(encoded['skill_mu'], encoded['skill_logvar'])
        return {'z_style': z_style, 'z_skill': z_skill}

    def decode(self, z_style, z_skill, target_length):
        """デコードのみ"""
        return self.decoder(z_style, z_skill, target_length)


# Datasetとの統合例
def create_dataloader(dataset, batch_size=32, shuffle=True):
    """DataLoaderの作成"""
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn
    )


def collate_fn(batch):
    """バッチ処理用の関数"""
    trajectories, subject_ids, is_expert = zip(*batch)

    return {
        'trajectory': torch.stack(trajectories),
        'subject_ids': list(subject_ids),
        'is_expert': torch.stack(is_expert)
    }


# 段階的学習戦略
def staged_training(model, train_loader, val_loader, num_epochs=150):
    """段階的学習"""

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    for epoch in range(num_epochs):
        # 段階的な重み調整
        if epoch < 50:  # Stage 1: 基本表現学習
            model.contrastive_weight = 0.5
            focus_reconstruction = True
        elif epoch < 100:  # Stage 2: スタイル分離
            model.contrastive_weight = 0.2
            focus_reconstruction = False
        else:  # Stage 3: 統合最適化
            model.contrastive_weight = 0.1
            focus_reconstruction = False

        # 訓練ループ
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(
                batch['trajectory'],
                batch['subject_ids'],
                batch['is_expert']
            )

            loss = outputs['losses']['total']
            loss.backward()

            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        scheduler.step()

        # 検証
        if epoch % 10 == 0:
            val_metrics = validate_model(model, val_loader)
            print(f"Epoch {epoch}: Val Reconstruction = {val_metrics['reconstruction']:.4f}")
            print(f"  Style Separation = {val_metrics['style_purity']:.3f}")


def validate_model(model, val_loader):
    """モデル検証"""
    model.eval()
    total_recon_loss = 0
    all_z_style = []
    all_subject_ids = []

    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch['trajectory'], batch['subject_ids'], batch['is_expert'])
            total_recon_loss += outputs['losses']['reconstruction'].item()

            all_z_style.append(outputs['z_style'])
            all_subject_ids.extend(batch['subject_ids'])

    # スタイル純度計算
    all_z_style = torch.cat(all_z_style)
    style_purity = compute_style_purity(all_z_style, all_subject_ids)

    return {
        'reconstruction': total_recon_loss / len(val_loader),
        'style_purity': style_purity
    }


def compute_style_purity(z_styles, subject_ids):
    """スタイル表現の純度評価"""
    unique_subjects = list(set(subject_ids))
    if len(unique_subjects) < 2:
        return 0.0

    intra_distances = []
    inter_distances = []

    for subject in unique_subjects:
        subject_indices = [i for i, s in enumerate(subject_ids) if s == subject]
        if len(subject_indices) > 1:
            subject_styles = z_styles[subject_indices]
            intra_dist = torch.pdist(subject_styles).mean()
            intra_distances.append(intra_dist)

        other_indices = [i for i, s in enumerate(subject_ids) if s != subject]
        if other_indices:
            other_styles = z_styles[other_indices]
            subject_mean = z_styles[subject_indices].mean(dim=0, keepdim=True)
            inter_dist = torch.cdist(subject_mean, other_styles).mean()
            inter_distances.append(inter_dist)

    if intra_distances and inter_distances:
        avg_intra = torch.mean(torch.stack(intra_distances))
        avg_inter = torch.mean(torch.stack(inter_distances))
        purity = (avg_inter - avg_intra) / (avg_inter + avg_intra + 1e-8)
        return purity.item()

    return 0.0