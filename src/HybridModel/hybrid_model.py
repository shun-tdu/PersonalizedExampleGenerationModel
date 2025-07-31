# CLAUDE_ADDED
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
from data_decomposition import DataDecomposer
from low_freq_model import LowFreqTransformer
from high_freq_model import SimpleDiffusionMLP, HighFreqDiffusion, UNet1D,UNet1DForTrajectory


class HybridTrajectoryModel(nn.Module):
    def __init__(
            self,
            input_dim: int = 2,
            condition_dim: int = 5,  # 5次元に統一
            # スプラインモデルのパラメータ
            num_control_points: int = 8,
            spline_hidden_dim: int = 256,
            # 拡散モデルのパラメータ
            high_freq_model_type: str = 'unet',  # 'unet' or 'mlp'
            diffusion_hidden_dim: int = 256,
            diffusion_num_layers: int = 4,
            # その他のパラメータ
            moving_average_window: int = 10,
            num_diffusion_steps: int = 100,
            diffusion_schedule: str = 'cosine',
            **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.high_freq_model_type = high_freq_model_type

        # パラメータを保存
        self.hparams = {
            "input_dim": input_dim, "condition_dim": condition_dim, "num_control_points": num_control_points,
            "spline_hidden_dim": spline_hidden_dim, "high_freq_model_type": high_freq_model_type,
            "diffusion_hidden_dim": diffusion_hidden_dim, "diffusion_num_layers": diffusion_num_layers,
            "moving_average_window": moving_average_window, "num_diffusion_steps": num_diffusion_steps,
            "diffusion_schedule": diffusion_schedule
        }

        # データ分解器
        self.decomposer = DataDecomposer(window_size=moving_average_window)

        # 低周波モデル（Transformer）
        self.low_freq_model = LowFreqTransformer(
            input_dim=input_dim,
            condition_dim=condition_dim,
        )

        # 高周波モデル
        if high_freq_model_type == 'unet':
            self.high_freq_model = UNet1DForTrajectory(
                input_dim=input_dim,
                condition_dim=condition_dim,
            )
        else:
            self.high_freq_model = SimpleDiffusionMLP(
                input_dim=input_dim,
                hidden_dim=diffusion_hidden_dim,
                num_layers=diffusion_num_layers,
                condition_dim=condition_dim
            )

        # 拡散プロセス
        self.diffusion = HighFreqDiffusion(
            num_timesteps=num_diffusion_steps,
            schedule=diffusion_schedule
        )

        # 正規化パラメータ（EMA）
        self.register_buffer('high_freq_mean', torch.zeros(1))
        self.register_buffer('high_freq_std', torch.ones(1))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        device = x.device

        # データを分解
        low_freq, high_freq = self.decomposer.decompose(x)

        # 低周波成分の学習
        low_freq_pred = self.low_freq_model(low_freq, condition)
        low_freq_loss = F.mse_loss(low_freq_pred, low_freq)

        # 低周波の滑らかさ損失
        low_freq_smoothness = self._compute_smoothness_loss(low_freq_pred)

        # 高周波成分の正規化
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

        high_freq_loss = F.mse_loss(predicted_noise, noise)

        # 高周波の滑らかさペナルティ
        high_freq_smoothness = self._compute_smoothness_loss(predicted_noise)

        return {
            'low_freq_loss': low_freq_loss,
            'high_freq_loss': high_freq_loss,
            'low_freq_smoothness': low_freq_smoothness,
            'high_freq_smoothness': high_freq_smoothness,
            'total_loss': (
                    low_freq_loss * 100 +
                    high_freq_loss +
                    low_freq_smoothness * 0.01 +
                    high_freq_smoothness * 0.001
            ),
            'low_freq_pred': low_freq_pred,
            'high_freq_target': high_freq,
            'predicted_noise': predicted_noise,
            'actual_noise': noise
        }

    def _compute_smoothness_loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        """軌道の滑らかさを評価"""
        if trajectory.shape[1] < 3:
            return torch.tensor(0.0, device=trajectory.device)

        # 2階微分（加速度）
        vel = trajectory[:, 1:] - trajectory[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]

        return torch.mean(torch.sum(acc ** 2, dim=-1))

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

    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報を取得
        
        Returns:
            model_info: モデルの詳細情報
        """
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