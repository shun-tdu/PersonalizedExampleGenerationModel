import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Union, List
from scipy.signal import butter, filtfilt
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


class GeneralizedCoordinateExtractor:
    """一般化座標を軌道データから抽出するクラス"""

    def __init__(self, dt: float = 0.01):
        self.dt = dt  # サンプリング時間間隔

    def extract_coordinates(self, trajectory: np.ndarray) -> Dict[str, np.ndarray]:
        """
        軌道データから一般化座標を抽出

        Args:
            trajectory: [seq_len, 2] の軌道データ (x, y)

        Returns:
            一般化座標の辞書
        """
        seq_len = trajectory.shape[0]
        coords = {}

        # === 基本座標 (VAEで学習対象) ===
        # 0次: 位置 (q)
        coords['position'] = trajectory  # [seq_len, 2]

        # 1次: 速度 (dq/dt)
        velocity = np.gradient(trajectory, axis=0) / self.dt
        coords['velocity'] = velocity

        # 2次: 加速度 (d²q/dt²)
        acceleration = np.gradient(velocity, axis=0) / self.dt
        coords['acceleration'] = acceleration

        # 3次: ジャーク (d³q/dt³)
        jerk = np.gradient(acceleration, axis=0) / self.dt
        coords['jerk'] = jerk

        # === 合成特徴 (基本座標から計算) ===
        coords['speed'] = np.linalg.norm(velocity, axis=1, keepdims=True)
        coords['acceleration_magnitude'] = np.linalg.norm(acceleration, axis=1, keepdims=True)
        coords['jerk_magnitude'] = np.linalg.norm(jerk, axis=1, keepdims=True)
        coords['curvature'] = self._compute_curvature(trajectory)

        # === VAE学習用: 基本座標のみ (8次元) ===
        basic_coordinates = np.concatenate([
            coords['position'],  # 2次元
            coords['velocity'],  # 2次元
            coords['acceleration'],  # 2次元
            coords['jerk']  # 2次元
        ], axis=1)  # 合計8次元

        # === 完全な座標: 基本 + 合成 (12次元) ===
        full_coordinates = np.concatenate([
            basic_coordinates,  # 8次元
            coords['speed'],  # 1次元
            coords['acceleration_magnitude'],  # 1次元
            coords['jerk_magnitude'],  # 1次元
            coords['curvature']  # 1次元
        ], axis=1)  # 合計12次元

        return {
            'basic_coordinates': basic_coordinates,  # VAE学習用
            'full_coordinates': full_coordinates,  # 評価・可視化用
            'components': coords
        }

    def _compute_curvature(self, trajectory: np.ndarray) -> np.ndarray:
        """軌道の曲率を計算"""
        dx = np.gradient(trajectory[:, 0])
        dy = np.gradient(trajectory[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # 曲率の計算
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx ** 2 + dy ** 2) ** (3 / 2)

        # ゼロ除算を回避
        denominator = np.where(denominator < 1e-8, 1e-8, denominator)
        curvature = numerator / denominator

        return curvature.reshape(-1, 1)


class TemporalFilterModule(nn.Module):
    """時間スケール分離のためのフィルタモジュール"""

    def __init__(self, seq_len: int, coord_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.coord_dim = coord_dim

        # 学習可能な時間フィルタ
        self.short_term_filter = nn.Conv1d(coord_dim, coord_dim, kernel_size=3, padding=1, groups=coord_dim)
        self.medium_term_filter = nn.Conv1d(coord_dim, coord_dim, kernel_size=9, padding=4, groups=coord_dim)
        self.long_term_filter = nn.Conv1d(coord_dim, coord_dim, kernel_size=21, padding=10, groups=coord_dim)

        # 初期化: 異なる周波数特性を持つように
        self._initialize_filters()

    def _initialize_filters(self):
        """フィルタを異なる周波数特性で初期化"""
        # 短期フィルタ: 高周波通過特性
        with torch.no_grad():
            self.short_term_filter.weight.fill_(0)
            self.short_term_filter.weight[:, 0, 1] = 1.0  # 中心を強調

        # 中期フィルタ: 中周波通過特性
        with torch.no_grad():
            kernel_size = 9
            sigma = kernel_size / 6
            kernel = torch.exp(-0.5 * ((torch.arange(kernel_size) - kernel_size // 2) / sigma) ** 2)
            kernel = kernel / kernel.sum()
            self.medium_term_filter.weight[:, 0, :] = kernel.unsqueeze(0)

        # 長期フィルタ: 低周波通過特性
        with torch.no_grad():
            kernel_size = 21
            kernel = torch.ones(kernel_size) / kernel_size
            self.long_term_filter.weight[:, 0, :] = kernel.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, coord_dim]
        Returns:
            short_term, medium_term, long_term: それぞれ [batch, seq_len, coord_dim]
        """
        # Conv1Dのために次元を入れ替え [batch, coord_dim, seq_len]
        x_transposed = x.transpose(1, 2)

        short_term = self.short_term_filter(x_transposed).transpose(1, 2)
        medium_term = self.medium_term_filter(x_transposed).transpose(1, 2)
        long_term = self.long_term_filter(x_transposed).transpose(1, 2)

        return short_term, medium_term, long_term


class TemporalEncoder(nn.Module):
    """時間スケール特化エンコーダー"""

    def __init__(self, coord_dim: int, hidden_dim: int, latent_dim: int, temporal_scale: str):
        super().__init__()
        self.temporal_scale = temporal_scale

        # 時間スケールに応じた受容野
        if temporal_scale == 'short':
            self.conv1d = nn.Conv1d(coord_dim, hidden_dim, kernel_size=3, padding=1)
            self.attention_heads = 8
        elif temporal_scale == 'medium':
            self.conv1d = nn.Conv1d(coord_dim, hidden_dim, kernel_size=7, padding=3)
            self.attention_heads = 4
        else:  # long
            self.conv1d = nn.Conv1d(coord_dim, hidden_dim, kernel_size=15, padding=7)
            self.attention_heads = 2

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=self.attention_heads, batch_first=True
        )

        self.mu_layer = nn.Linear(hidden_dim * 2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim * 2, latent_dim)

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, coord_dim]
        Returns:
            mu, logvar: [batch, latent_dim]
        """
        batch_size, seq_len, coord_dim = x.shape

        # 1D畳み込み
        x_conv = x.transpose(1, 2)  # [batch, coord_dim, seq_len]
        conv_out = F.relu(self.conv1d(x_conv))
        conv_out = conv_out.transpose(1, 2)  # [batch, seq_len, hidden_dim]

        # LSTM
        lstm_out, _ = self.lstm(conv_out)  # [batch, seq_len, hidden_dim*2]
        lstm_out = self.dropout(lstm_out)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out)

        # 時系列全体をプール
        if self.temporal_scale == 'short':
            # 短期: 最大値プール（瞬間的な特徴を重視）
            pooled = torch.max(attn_out, dim=1)[0]
        elif self.temporal_scale == 'medium':
            # 中期: 平均プール（中期的傾向を重視）
            pooled = torch.mean(attn_out, dim=1)
        else:
            # 長期: 重み付き平均（長期的パターンを重視）
            weights = torch.softmax(torch.sum(attn_out, dim=2), dim=1)
            pooled = torch.sum(attn_out * weights.unsqueeze(2), dim=1)

        mu = self.mu_layer(pooled)
        logvar = self.logvar_layer(pooled)

        return mu, logvar


class HierarchicalDecoder(nn.Module):
    """階層的デコーダー"""

    def __init__(self, latent_dims: Dict[str, int], hidden_dim: int, coord_dim: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.coord_dim = coord_dim

        total_latent_dim = sum(latent_dims.values())

        # 潜在変数から初期状態を生成
        self.latent_projection = nn.Linear(total_latent_dim, hidden_dim * 2)

        # デコーダーLSTM
        self.decoder_lstm = nn.LSTM(coord_dim, hidden_dim, num_layers=2, batch_first=True)

        # 出力層
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, coord_dim)
        )

        # 初期入力
        self.register_buffer('initial_input', torch.zeros(1, 1, coord_dim))

    def forward(self, z_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            z_dict: {'primitive': z1, 'skill': z2, 'style': z3}
        Returns:
            reconstructed: [batch, seq_len, coord_dim]
        """
        batch_size = list(z_dict.values())[0].shape[0]

        # 全ての潜在変数を結合
        z_combined = torch.cat(list(z_dict.values()), dim=1)

        # LSTM初期状態を生成
        init_state = self.latent_projection(z_combined)  # [batch, hidden_dim*2]
        h_0 = init_state[:, :init_state.shape[1] // 2].unsqueeze(0).repeat(2, 1, 1)
        c_0 = init_state[:, init_state.shape[1] // 2:].unsqueeze(0).repeat(2, 1, 1)

        # 自己回帰的生成
        outputs = []
        hidden = (h_0, c_0)
        input_t = self.initial_input.repeat(batch_size, 1, 1)

        for t in range(self.seq_len):
            output, hidden = self.decoder_lstm(input_t, hidden)
            output = self.output_layer(output)
            outputs.append(output)
            input_t = output

        return torch.cat(outputs, dim=1)  # [batch, seq_len, coord_dim]


class PhysicsInformedLoss(nn.Module):
    """物理制約を考慮した損失関数"""

    def __init__(self, dt: float = 0.01):
        super().__init__()
        self.dt = dt

    def forward(self, predicted_coords: torch.Tensor, target_coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            predicted_coords, target_coords: [batch, seq_len, coord_dim]
        Returns:
            losses: 各種物理制約損失
        """
        losses = {}

        # 基本再構成損失
        losses['reconstruction'] = F.mse_loss(predicted_coords, target_coords)

        # 物理的一貫性損失
        losses['physics_consistency'] = self._compute_physics_consistency(predicted_coords)

        # 滑らかさ損失（最小ジャーク原理）
        losses['smoothness'] = self._compute_smoothness_loss(predicted_coords)

        # エネルギー保存損失
        losses['energy_conservation'] = self._compute_energy_loss(predicted_coords)

        return losses

    def _compute_physics_consistency(self, coords: torch.Tensor) -> torch.Tensor:
        """運動方程式の一貫性を検証"""
        # 位置、速度、加速度、ジャークを抽出
        pos = coords[:, :, :2]
        vel = coords[:, :, 2:4]
        acc = coords[:, :, 4:6]
        jerk = coords[:, :, 6:8]

        # 数値微分による一貫性チェック
        vel_diff = torch.diff(pos, dim=1) / self.dt
        acc_diff = torch.diff(vel, dim=1) / self.dt
        jerk_diff = torch.diff(acc, dim=1) / self.dt

        # 差分の二乗誤差
        vel_consistency = F.mse_loss(vel[:, 1:], vel_diff)
        acc_consistency = F.mse_loss(acc[:, 1:], acc_diff)
        jerk_consistency = F.mse_loss(jerk[:, 1:], jerk_diff)

        return vel_consistency + acc_consistency + jerk_consistency

    def _compute_smoothness_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """ジャーク最小化による滑らかさ損失"""
        jerk = coords[:, :, 6:8]  # ジャーク成分
        return torch.mean(torch.norm(jerk, dim=2))

    def _compute_energy_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """エネルギー保存の近似的チェック"""
        vel = coords[:, :, 2:4]
        kinetic_energy = 0.5 * torch.sum(vel ** 2, dim=2)

        # エネルギーの時間変化が滑らかであることを要求
        energy_diff = torch.diff(kinetic_energy, dim=1)
        return torch.mean(energy_diff ** 2)


class StyleSkillSeparationLoss(nn.Module):
    """スタイル・スキル分離を促進する損失"""

    def __init__(self):
        super().__init__()

    def forward(self,
                z_style: torch.Tensor,
                z_skill: torch.Tensor,
                z_primitive: torch.Tensor,
                subject_ids: torch.Tensor,
                skill_levels: torch.Tensor) -> Dict[str, torch.Tensor]:

        losses = {}

        # スタイル分離損失
        losses['style_clustering'] = self._compute_style_clustering_loss(z_style, subject_ids)

        # スキル構造損失
        losses['skill_structure'] = self._compute_skill_structure_loss(z_skill, skill_levels)

        # 直交性損失
        losses['orthogonality'] = self._compute_orthogonality_loss(z_style, z_skill, z_primitive)

        return losses

    def _compute_style_clustering_loss(self, z_style: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """同一人物のスタイルをクラスタリング"""
        unique_subjects = torch.unique(subject_ids)
        intra_cluster_loss = 0.0
        inter_cluster_loss = 0.0

        subject_centroids = []

        for subject_id in unique_subjects:
            mask = (subject_ids == subject_id)
            subject_styles = z_style[mask]

            if len(subject_styles) > 1:
                centroid = subject_styles.mean(dim=0)
                subject_centroids.append(centroid)

                # 同一人物内の分散を最小化
                intra_cluster_loss += torch.mean(torch.norm(subject_styles - centroid, dim=1))

        # 異なる人物間の距離を最大化
        if len(subject_centroids) > 1:
            centroids = torch.stack(subject_centroids)
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    inter_cluster_loss -= torch.norm(centroids[i] - centroids[j])

        return intra_cluster_loss + inter_cluster_loss

    def _compute_skill_structure_loss(self, z_skill: torch.Tensor, skill_levels: torch.Tensor) -> torch.Tensor:
        """スキル空間の構造を強制"""
        # 熟達者（skill_level=1）は密集
        expert_mask = (skill_levels == 1)
        if expert_mask.sum() > 1:
            expert_skills = z_skill[expert_mask]
            expert_centroid = expert_skills.mean(dim=0)
            expert_compactness = torch.mean(torch.norm(expert_skills - expert_centroid, dim=1))
        else:
            expert_compactness = 0.0

        # 初心者（skill_level=0）は広範囲
        novice_mask = (skill_levels == 0)
        if novice_mask.sum() > 1:
            novice_skills = z_skill[novice_mask]
            novice_spread = -torch.mean(torch.pdist(novice_skills))  # 負号で広がりを促進
        else:
            novice_spread = 0.0

        return expert_compactness + novice_spread

    def _compute_orthogonality_loss(self, z_style: torch.Tensor, z_skill: torch.Tensor,
                                    z_primitive: torch.Tensor) -> torch.Tensor:
        """階層間の直交性を促進"""

        # 正準相関を最小化
        def canonical_correlation(x, y):
            if x.shape[0] < 2:
                return torch.tensor(0.0, device=x.device)

            x_centered = x - x.mean(dim=0)
            y_centered = y - y.mean(dim=0)

            # 共分散行列
            cov_xy = torch.mm(x_centered.T, y_centered) / (x.shape[0] - 1)
            cov_xx = torch.mm(x_centered.T, x_centered) / (x.shape[0] - 1) + 1e-6 * torch.eye(x.shape[1],
                                                                                              device=x.device)
            cov_yy = torch.mm(y_centered.T, y_centered) / (y.shape[0] - 1) + 1e-6 * torch.eye(y.shape[1],
                                                                                              device=y.device)

            # 正準相関の近似（フロベニウスノルム）
            return torch.norm(cov_xy) / (torch.norm(cov_xx) * torch.norm(cov_yy)).sqrt()

        orthogonality = (
                canonical_correlation(z_style, z_skill) +
                canonical_correlation(z_skill, z_primitive) +
                canonical_correlation(z_style, z_primitive)
        )

        return orthogonality

class GeneralizedCoordinateHierarchicalVAE(nn.Module):
    """一般化座標を入力とする階層型VAE (データセット互換版)"""

    def __init__(self,
                 basic_coord_dim: int = 8,  # 基本座標次元 (pos+vel+acc+jerk)
                 derived_coord_dim: int = 4,  # 合成特徴次元 (speed+acc_mag+jerk_mag+curvature)
                 seq_len: int = 100,
                 hidden_dim: int = 128,
                 latent_dims: Optional[Dict[str, int]] = None,
                 beta_weights: Optional[Dict[str, float]] = None,
                 physics_weight: float = 0.1,
                 separation_weight: float = 0.5):
        super().__init__()

        if latent_dims is None:
            latent_dims = {'primitive': 32, 'skill': 16, 'style': 8}
        if beta_weights is None:
            beta_weights = {'primitive': 1.0, 'skill': 2.0, 'style': 4.0}

        self.basic_coord_dim = basic_coord_dim
        self.derived_coord_dim = derived_coord_dim
        self.full_coord_dim = basic_coord_dim + derived_coord_dim
        self.latent_dims = latent_dims
        self.beta_weights = beta_weights
        self.physics_weight = physics_weight
        self.separation_weight = separation_weight

        # Subject ID → 数値ID の変換マップ
        self.subject_id_map = {}
        self.next_subject_id = 0

        # 時間スケール分離モジュール (基本座標のみ)
        self.temporal_filter = TemporalFilterModule(seq_len, basic_coord_dim)

        # 階層的エンコーダー (基本座標のみ)
        self.primitive_encoder = TemporalEncoder(basic_coord_dim, hidden_dim, latent_dims['primitive'], 'short')
        self.skill_encoder = TemporalEncoder(basic_coord_dim, hidden_dim, latent_dims['skill'], 'medium')
        self.style_encoder = TemporalEncoder(basic_coord_dim, hidden_dim, latent_dims['style'], 'long')

        # 階層的デコーダー (基本座標のみ)
        self.decoder = HierarchicalDecoder(latent_dims, hidden_dim, basic_coord_dim, seq_len)

        # 損失関数 (完全な座標で評価)
        self.physics_loss = PhysicsInformedLoss()
        self.separation_loss = StyleSkillSeparationLoss()

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """再パラメータ化トリック"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _convert_subject_ids(self, subject_ids: Union[List[str], Tuple[str, ...], torch.Tensor]) -> torch.Tensor:
        """Subject IDを数値に変換"""
        if isinstance(subject_ids, torch.Tensor):
            # 既に数値の場合はそのまま
            return subject_ids

        numeric_ids = []
        for subject_id in subject_ids:
            if subject_id not in self.subject_id_map:
                self.subject_id_map[subject_id] = self.next_subject_id
                self.next_subject_id += 1
            numeric_ids.append(self.subject_id_map[subject_id])

        return torch.tensor(numeric_ids, dtype=torch.long)

    def compute_derived_features(self, basic_coords: torch.Tensor) -> torch.Tensor:
        """基本座標から合成特徴を物理的に正しく計算"""
        batch_size, seq_len = basic_coords.shape[:2]

        # 基本座標を分解
        pos = basic_coords[:, :, :2]  # 位置 [batch, seq, 2]
        vel = basic_coords[:, :, 2:4]  # 速度 [batch, seq, 2]
        acc = basic_coords[:, :, 4:6]  # 加速度 [batch, seq, 2]
        jerk = basic_coords[:, :, 6:8]  # ジャーク [batch, seq, 2]

        # スカラー特徴量を計算
        speed = torch.norm(vel, dim=2, keepdim=True)  # [batch, seq, 1]
        acc_magnitude = torch.norm(acc, dim=2, keepdim=True)  # [batch, seq, 1]
        jerk_magnitude = torch.norm(jerk, dim=2, keepdim=True)  # [batch, seq, 1]
        curvature = self._compute_curvature_torch(pos)  # [batch, seq, 1]

        # 合成特徴を結合
        derived_features = torch.cat([
            speed, acc_magnitude, jerk_magnitude, curvature
        ], dim=2)  # [batch, seq, 4]

        return derived_features

    def _compute_curvature_torch(self, pos: torch.Tensor) -> torch.Tensor:
        """PyTorchでの曲率計算"""
        # 数値微分による1次・2次導関数
        dx = torch.gradient(pos[:, :, 0], dim=1)[0]  # x方向速度
        dy = torch.gradient(pos[:, :, 1], dim=1)[0]  # y方向速度
        ddx = torch.gradient(dx, dim=1)[0]  # x方向加速度
        ddy = torch.gradient(dy, dim=1)[0]  # y方向加速度

        # 曲率公式: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        numerator = torch.abs(dx * ddy - dy * ddx)
        denominator = (dx ** 2 + dy ** 2) ** (3 / 2)

        # ゼロ除算回避
        denominator = torch.clamp(denominator, min=1e-8)
        curvature = numerator / denominator

        return curvature.unsqueeze(2)  # [batch, seq, 1]

    def split_coordinates(self, full_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """完全な座標を基本座標と合成特徴に分割"""
        basic_coords = full_coords[:, :, :self.basic_coord_dim]
        derived_coords = full_coords[:, :, self.basic_coord_dim:]
        return basic_coords, derived_coords

    def combine_coordinates(self, basic_coords: torch.Tensor, derived_coords: torch.Tensor) -> torch.Tensor:
        """基本座標と合成特徴を結合"""
        return torch.cat([basic_coords, derived_coords], dim=2)

    def encode_hierarchically(self, basic_coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """階層的エンコーディング (基本座標のみ)"""
        # 時間スケール分離
        short_term, medium_term, long_term = self.temporal_filter(basic_coords)

        # 各階層でエンコード
        mu_primitive, logvar_primitive = self.primitive_encoder(short_term)
        mu_skill, logvar_skill = self.skill_encoder(medium_term)
        mu_style, logvar_style = self.style_encoder(long_term)

        # 再パラメータ化
        z_primitive = self.reparameterize(mu_primitive, logvar_primitive)
        z_skill = self.reparameterize(mu_skill, logvar_skill)
        z_style = self.reparameterize(mu_style, logvar_style)

        return {
            'z_primitive': z_primitive, 'mu_primitive': mu_primitive, 'logvar_primitive': logvar_primitive,
            'z_skill': z_skill, 'mu_skill': mu_skill, 'logvar_skill': logvar_skill,
            'z_style': z_style, 'mu_style': mu_style, 'logvar_style': logvar_style
        }

    def decode_hierarchically(self, z_style: torch.Tensor, z_skill: torch.Tensor = None,
                              z_primitive: torch.Tensor = None) -> torch.Tensor:
        """階層的デコーディング (基本座標のみ)"""
        z_dict = {'style': z_style}
        if z_skill is not None:
            z_dict['skill'] = z_skill
        if z_primitive is not None:
            z_dict['primitive'] = z_primitive

        # 基本座標のみデコード
        basic_coords = self.decoder(z_dict)
        return basic_coords

    def forward(self, full_coords: torch.Tensor,
                subject_ids: Union[List[str], Tuple[str, ...], torch.Tensor] = None,
                skill_levels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """フォワードパス (データセット互換版)"""
        batch_size = full_coords.shape[0]

        # Subject IDを数値に変換
        if subject_ids is not None:
            subject_ids_numeric = self._convert_subject_ids(subject_ids).to(full_coords.device)
        else:
            subject_ids_numeric = None

        # 完全な座標を基本座標と合成特徴に分割
        basic_coords, target_derived = self.split_coordinates(full_coords)

        # 基本座標のみで階層的エンコーディング
        encoded = self.encode_hierarchically(basic_coords)
        z_primitive = encoded['z_primitive']
        z_skill = encoded['z_skill']
        z_style = encoded['z_style']

        # 基本座標のみデコーディング
        reconstructed_basic = self.decode_hierarchically(z_style, z_skill, z_primitive)

        # 合成特徴を物理的に正しく計算
        reconstructed_derived = self.compute_derived_features(reconstructed_basic)

        # 完全な座標を再構成
        reconstructed_full = self.combine_coordinates(reconstructed_basic, reconstructed_derived)

        # === 損失計算 ===
        losses = {}

        # 物理制約損失 (完全な座標で評価)
        physics_losses = self.physics_loss(reconstructed_full, full_coords)
        for key, loss in physics_losses.items():
            losses[f'physics_{key}'] = loss

        # KL発散損失
        def kl_divergence(mu, logvar):
            return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        losses['kl_primitive'] = kl_divergence(encoded['mu_primitive'], encoded['logvar_primitive'])
        losses['kl_skill'] = kl_divergence(encoded['mu_skill'], encoded['logvar_skill'])
        losses['kl_style'] = kl_divergence(encoded['mu_style'], encoded['logvar_style'])

        # スタイル・スキル分離損失
        if subject_ids_numeric is not None and skill_levels is not None:
            try:
                separation_losses = self.separation_loss(z_style, z_skill, z_primitive,
                                                        subject_ids_numeric, skill_levels)
                for key, loss in separation_losses.items():
                    losses[f'separation_{key}'] = loss
            except Exception as e:
                # 分離損失でエラーが発生した場合はスキップ
                print(f"Warning: Separation loss computation failed: {e}")
                losses['separation_style_clustering'] = torch.tensor(0.0, device=full_coords.device)
                losses['separation_skill_structure'] = torch.tensor(0.0, device=full_coords.device)
                losses['separation_orthogonality'] = torch.tensor(0.0, device=full_coords.device)

        # 総損失
        total_loss = (
                losses['physics_reconstruction'] +
                self.physics_weight * (
                        losses['physics_physics_consistency'] +
                        losses['physics_smoothness'] +
                        losses['physics_energy_conservation']
                ) +
                self.beta_weights['primitive'] * losses['kl_primitive'] +
                self.beta_weights['skill'] * losses['kl_skill'] +
                self.beta_weights['style'] * losses['kl_style']
        )

        if subject_ids_numeric is not None and skill_levels is not None:
            total_loss += self.separation_weight * (
                    losses.get('separation_style_clustering', 0) +
                    losses.get('separation_skill_structure', 0) +
                    losses.get('separation_orthogonality', 0)
            )

        return {
            'total_loss': total_loss,
            'reconstructed': reconstructed_full,
            'reconstructed_basic': reconstructed_basic,
            'reconstructed_derived': reconstructed_derived,
            'encoded': encoded,
            'individual_losses': losses
        }

    def generate_personalized_exemplar(self,
                                       learner_trajectory: torch.Tensor,
                                       skill_enhancement_factor: float = 0.15,
                                       preserve_style: bool = True) -> torch.Tensor:
        """個人最適化されたお手本を生成"""
        with torch.no_grad():
            # 完全な座標から基本座標を抽出
            basic_coords, _ = self.split_coordinates(learner_trajectory)

            # 学習者の現在のスタイル・スキルを抽出
            encoded = self.encode_hierarchically(basic_coords)
            learner_style = encoded['z_style']
            learner_skill = encoded['z_skill']

            # スキルを熟達者方向に向上
            if hasattr(self, 'expert_skill_centroid'):
                # 事前に計算された熟達者の重心に向かって移動
                skill_direction = self.expert_skill_centroid - learner_skill
                skill_direction = F.normalize(skill_direction, dim=1)
                enhanced_skill = learner_skill + skill_enhancement_factor * skill_direction
            else:
                # フォールバック: ランダムな改善
                skill_noise = torch.randn_like(learner_skill) * 0.1
                enhanced_skill = learner_skill + skill_enhancement_factor * skill_noise

            # 個人スタイルを保持しつつ基本座標を生成
            enhanced_basic = self.decode_hierarchically(learner_style, enhanced_skill)

            # 合成特徴を物理的に正しく計算
            enhanced_derived = self.compute_derived_features(enhanced_basic)

            # 完全な座標を結合
            enhanced_full = self.combine_coordinates(enhanced_basic, enhanced_derived)

            return enhanced_full

    def cache_expert_representations(self, expert_dataloader, device: str = 'cpu'):
        """熟達者の表現をキャッシュ"""
        self.eval()
        expert_skills = []

        with torch.no_grad():
            for batch in expert_dataloader:
                coords = batch[0].to(device)
                basic_coords, _ = self.split_coordinates(coords)
                encoded = self.encode_hierarchically(basic_coords)
                expert_skills.append(encoded['z_skill'].cpu())

        if expert_skills:
            expert_skills = torch.cat(expert_skills, dim=0)
            self.expert_skill_centroid = expert_skills.mean(dim=0).to(device)
        else:
            self.expert_skill_centroid = torch.zeros(self.latent_dims['skill'], device=device)

    def save_model(self, filepath: str):
        """モデル保存"""
        torch.save({
            'state_dict': self.state_dict(),
            'latent_dims': self.latent_dims,
            'beta_weights': self.beta_weights,
            'physics_weight': self.physics_weight,
            'separation_weight': self.separation_weight,
            'subject_id_map': self.subject_id_map,
            'next_subject_id': self.next_subject_id,
            'expert_skill_centroid': getattr(self, 'expert_skill_centroid', None)
        }, filepath)

    def load_model(self, filepath: str, device: str = 'cpu'):
        """モデル読み込み"""
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['state_dict'])

        if 'subject_id_map' in checkpoint:
            self.subject_id_map = checkpoint['subject_id_map']
            self.next_subject_id = checkpoint['next_subject_id']

        if 'expert_skill_centroid' in checkpoint and checkpoint['expert_skill_centroid'] is not None:
            self.expert_skill_centroid = checkpoint['expert_skill_centroid'].to(device)

        self.to(device)