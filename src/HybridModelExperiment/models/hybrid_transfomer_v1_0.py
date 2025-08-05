import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from HybridModelExperiment.architectures.data_decomposition import ImprovedDataDecomposer
from HybridModelExperiment.architectures.low_freq_model import CurveLowFreqTransformer
from HybridModelExperiment.architectures.high_freq_model import UNet1DForTrajectory, SimpleDiffusionMLP, HighFreqDiffusion


class HybridModel(nn.Module):
    def __init__(self,
                 architecture: dict,
                 decomposer_params: dict,
                 low_freq_params: dict,
                 high_freq_params: dict,
                 input_dim: int,
                 condition_dim: int,
                 **kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hparams = {
            **kwargs,
            'architecture': architecture,
            'decomposer_params': decomposer_params,
            'low_freq_params': low_freq_params,
            'high_freq_params': high_freq_params,
            'input_dim': input_dim,
            'condition_dim': condition_dim
        }

        # 1. Decomposerモデルの動的読み込み
        decomposer_name = architecture.get('decomposer_model', 'ImprovedDataDecomposer')
        if decomposer_name == 'ImprovedDataDecomposer':
            self.decomposer = ImprovedDataDecomposer(**decomposer_params)
        else:
            raise ValueError(f"Unknown decomposer_model: {decomposer_name}")

        # 2. 低周波モデルの動的読み込み
        low_freq_model_name = architecture.get('low_freq_model', 'CurveLowFreqTransformer')
        if low_freq_model_name == 'CurveLowFreqTransformer':
            self.low_freq_model = CurveLowFreqTransformer(
                input_dim = input_dim,
                condition_dim=condition_dim,
                **low_freq_params
            )
        else:
            raise ValueError(f"Unknown low_freq_model: {low_freq_model_name}")

        # 3. 高周波モデルの動的読み込み
        high_freq_model_name = architecture.get('high_freq_model', 'UNet1DForTrajectory')
        if high_freq_model_name == 'UNet1DForTrajectory':
            self.high_freq_model =UNet1DForTrajectory(
                input_dim = input_dim,
                condition_dim=condition_dim,
                time_embed_dim=kwargs.get('time_embed_dim', 128),
                **high_freq_params
            )
        elif high_freq_model_name == 'SimpleDiffusionMLP':
            self.high_freq_model = SimpleDiffusionMLP(
                input_dim=input_dim,
                condition_dim=condition_dim,
                time_embedding_dim=kwargs.get('time_embed_dim', 128),
                **high_freq_params
            )
        else:
            raise ValueError(f"Unknown high_freq_model: {high_freq_model_name}")

        self.diffusion = HighFreqDiffusion(
            num_timesteps=kwargs.get('num_diffusion_steps', 1000),
            schedule=kwargs.get('diffusion_schedule', 'cosine')
        )

        # 正規化パラメータ（EMA）
        self.register_buffer('high_freq_mean', torch.zeros(1))
        self.register_buffer('high_freq_std', torch.ones(1))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        学習時のフォワードパス
        :param x: [batch, seq_len, 2]   軌道データ
        :param condition:   [batch, 5]  条件データ
        :return:                        各種損失を含む辞書
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # データを分解
        low_freq, high_freq = self.decomposer.decompose(x)

        # ============ 低周波域の損失計算 ============
        low_freq_pred = self.low_freq_model(low_freq, condition)

        # 基本的なMSE損失
        low_freq_mse = F.mse_loss(low_freq_pred, low_freq)

        # 1. ゴール到達損失
        if low_freq_pred.shape[1] > 0:
            final_pos = low_freq_pred[:, -1, :]
            goal_pos = condition[:, 3:5]
            goal_loss = F.mse_loss(final_pos, goal_pos)
        else:
            goal_loss = torch.tensor(0.0, device=device)

        # 2. 滑らかさ損失(2階微分)
        low_freq_smoothness = self._compute_smoothness_loss(low_freq_pred)

        # 3. 曲率一貫性損失
        curvature_loss = self._compute_curvature_consistency_loss(low_freq_pred)

        # 4. 経路効率損失
        path_efficiency_loss = self._compute_path_efficiency_loss(low_freq_pred, condition)

        # 5. 速度一貫性損失
        velocity_consistency_loss = self._compute_velocity_consistency_loss(low_freq_pred)

        # 低周波の統合損失
        low_freq_loss = (
                low_freq_mse +
                goal_loss * 10.0 +
                low_freq_smoothness * 0.1 +
                curvature_loss * 0.05 +
                path_efficiency_loss * 1.0 +
                velocity_consistency_loss * 0.1
        )

        # ============ 高周波域の損失計算 ============
        # 正規化
        high_freq_mean = high_freq.mean()
        high_freq_std = high_freq.std() + 1e-6
        high_freq_normalized = (high_freq - high_freq_mean) / high_freq_std

        # EMA更新
        if self.training:
            with torch.no_grad():
                self.high_freq_mean = 0.95 * self.high_freq_mean + 0.05 * high_freq_mean
                self.high_freq_std = 0.95 * self.high_freq_std + 0.05 * high_freq_std

        # 拡散モデルの学習
        timesteps = self.diffusion.sample_timesteps(batch_size, device)
        noise = torch.randn_like(high_freq_normalized)
        high_freq_noisy = self.diffusion.add_noise(high_freq_normalized, timesteps, noise)
        predicted_noise = self.high_freq_model(high_freq_noisy, timesteps, condition)

        # 高周波のMSE損失
        high_freq_mse = F.mse_loss(predicted_noise, noise)

        # 高周波の滑らかさペナルティ
        high_freq_smoothness = self._compute_smoothness_loss(predicted_noise)

        # 高周波の統合損失
        high_freq_loss = high_freq_mse + high_freq_smoothness * 0.001

        # ============ 統合損失 ============
        total_loss = low_freq_loss * 100 + high_freq_loss + low_freq_smoothness * 0.01 + high_freq_smoothness * 0.001

        return {
            # 損失関連
            'total_loss': total_loss,
            'low_freq_loss': low_freq_loss,
            'high_freq_loss': high_freq_loss,
            # ログ出力や分析用の詳細な損失
            'low_freq_mse': low_freq_mse,
            'goal_loss': goal_loss,
            'low_freq_smoothness': low_freq_smoothness,
            'path_efficiency_loss': path_efficiency_loss,
            'velocity_consistency_loss': velocity_consistency_loss,
            'high_freq_mse': high_freq_mse,
            'high_freq_smoothness': high_freq_smoothness,
            # デバッグ可視化用
            'low_freq_pred': low_freq_pred,
            'high_freq_target': high_freq,
            'predicted_noise': predicted_noise,
            'actual_noise': noise
        }

    def _compute_smoothness_loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        """軌道の滑らかさを評価"""
        if trajectory.shape[1] < 3:
            return torch.tensor(0.0, device=trajectory.device)

        # 速度
        vel = trajectory[:, 1:] - trajectory[:, :-1]
        # 加速度
        acc = vel[:, 1:] - vel[:, :-1]

        # 加速度のL2ノルム
        return torch.mean(torch.sum(acc ** 2, dim=-1))

    def _compute_curvature_consistency_loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        """曲率の一貫性を評価"""
        if trajectory.shape[1] < 3:
            return torch.tensor(0.0, device=trajectory.device)

        # 速度と加速度
        vel = trajectory[:, 1:] - trajectory[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]

        # 曲率 (2D cross product)
        vel_norm = torch.norm(vel[:, :-1], dim=-1, keepdim=True) + 1e-6

        cross_product = torch.abs(
            vel[:, :-1, 0] * acc[:, :, 1] - vel[:, :-1, 1] * acc[:, :, 0]
        )
        curvature = cross_product / (vel_norm.squeeze(-1) ** 3 + 1e-8)

        # 曲率の分散(一貫性)
        if curvature.shape[1] > 1:
            curvature_var = torch.var(curvature, dim=1).mean()
        else:
            curvature_var = torch.tensor(0.0, device=trajectory.device)

        return curvature_var

    def _compute_path_efficiency_loss(self, trajectory: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """経路効率を評価(過度に長い経路にペナルティ)"""
        if trajectory.shape[1] < 2:
            return torch.tensor(0.0, device=trajectory.device)

        # 総経路長
        segment_length = torch.norm(
            trajectory[:, 1:] - trajectory[:, :-1],
            dim=-1
        )
        path_length = torch.sum(segment_length, dim=1)

        # 視点から終点への直線距離
        start_pos = trajectory[:, 0]
        goal_pos = condition[:, 3:5]
        straight_distance = torch.norm(goal_pos - start_pos, dim=-1)

        # 効率性(1.0が直線，大きいほど非効率的)
        efficiency = path_length / (straight_distance + 1e-6)

        # 2.0を超える非効率な経路にペナルティ
        penalty = torch.relu(efficiency - 2.0)

        return penalty.mean()

    def _compute_velocity_consistency_loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        """速度の一貫性を評価"""
        if trajectory.shape[1] < 2:
            return torch.tensor(0.0, device=trajectory.device)

        # 速度
        velocities = trajectory[:, 1:] - trajectory[:, :-1]

        # 速度の大きさ
        speed = torch.norm(velocities, dim=-1)

        # 速度の変化率
        if speed.shape[1] > 1:
            speed_changes = torch.abs(speed[:, 1:] - speed[:, :-1])
            consistency_loss = torch.mean(speed_changes)
        else:
            consistency_loss = torch.tensor(0.0, device=trajectory.device)

        return consistency_loss

    @torch.no_grad()
    def generate(self, condition: torch.Tensor, sequence_length: int, num_samples: int = 1) -> torch.Tensor:
        batch_size = condition.shape[0]
        device = condition.device

        # 条件を拡張
        if num_samples > 1:
            condition_expanded = condition.unsqueeze(1).expand(-1, num_samples, -1)
            condition_expanded = condition_expanded.contiguous().view(-1, self.condition_dim)
        else:
            condition_expanded = condition

        total_batch_size = condition_expanded.shape[0]

        # 低周波成分を生成
        low_freq_generated = self.low_freq_model.generate(
            condition_expanded,
            sequence_length
        )

        # 高周波成分を生成（正規化された空間で）
        high_freq_shape = (total_batch_size, sequence_length, self.input_dim)
        high_freq_normalized = self.diffusion.generate(
            self.high_freq_model,
            high_freq_shape,
            condition_expanded,
            device
        )

        # 正規化を逆変換
        high_freq_generated = high_freq_normalized * self.high_freq_std + self.high_freq_mean

        # 低周波成分と高周波成分を結合
        generated_trajectories = self.decomposer.reconstruct(
            low_freq_generated,
            high_freq_generated
        )

        return generated_trajectories

    def update_epoch(self, epoch: int, max_epoch: int):
        """エポックを更新(Schedule Sampling用)"""
        self.epoch = torch.tensor(epoch)
        self.low_freq_model.update_sampling_prob(epoch, max_epoch)

    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        low_freq_params = sum(p.numel() for p in self.low_freq_model.parameters())
        high_freq_params = sum(p.numel() for p in self.high_freq_model.parameters())

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'low_freq_parameters': low_freq_params,
            'high_freq_parameters': high_freq_params,
            'input_dim': self.input_dim,
            'condition_dim': self.condition_dim,
            'moving_average_window': self.decomposer.window_size,
            'diffusion_steps': self.diffusion.num_timesteps
        }

    def save_model(self, filepath: str):
        torch.save({'model_state_dict': self.state_dict(), 'hparams': self.hparams}, filepath)

    @classmethod
    def load_model(cls, filepath: str, device: str = 'cpu'):
        """
        モデルを読み込み
        """
        checkpoint = torch.load(filepath, map_location=device)
        hparams = checkpoint['hparams']

        # モデルを初期化
        model = cls(**hparams)

        # 状態を読み込み
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model