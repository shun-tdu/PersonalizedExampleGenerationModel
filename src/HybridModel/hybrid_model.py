# CLAUDE_ADDED
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
from data_decomposition import DataDecomposer
from low_freq_model import LowFreqLSTM
from low_freq_model import LowFreqTransformer
from low_freq_model import LowFreqSpline
from high_freq_model import SimpleDiffusionMLP, HighFreqDiffusion


class HybridTrajectoryModel(nn.Module):
    """
    低周波LSTM + 高周波拡散モデルのハイブリッド軌道生成モデル
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        condition_dim: int = 3,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        diffusion_hidden_dim: int = 256,
        diffusion_num_layers: int = 4,
        moving_average_window: int = 10,
        num_diffusion_steps: int = 1000,
        **kwargs
    ):
        """
        Args:
            input_dim: 入力軌道の次元数（x, y座標）
            condition_dim: 条件ベクトルの次元数（動作時間、終点誤差、ジャーク）
            lstm_hidden_dim: LSTM隠れ層の次元数
            lstm_num_layers: LSTMのレイヤー数
            diffusion_hidden_dim: 拡散モデル隠れ層の次元数
            diffusion_num_layers: 拡散モデルのレイヤー数
            moving_average_window: 移動平均のウィンドウサイズ
            num_diffusion_steps: 拡散ステップ数
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        
        # データ分解器
        self.decomposer = DataDecomposer(window_size=moving_average_window)
        
        # 低周波モデル（LSTM）
        # self.low_freq_model = LowFreqLSTM(
        #     input_dim=input_dim,
        #     hidden_dim=lstm_hidden_dim,
        #     num_layers=lstm_num_layers,
        #     condition_dim=condition_dim
        # )
        # self.low_freq_model = LowFreqTransformer(
        #     input_dim=input_dim,
        #     condition_dim=condition_dim
        # )
        self.low_freq_model = LowFreqSpline(
            input_dim=input_dim,
            condition_dim=condition_dim,
            num_control_points=8,
            hidden_dim=256
        )

        # 高周波モデル（拡散モデル）
        self.high_freq_model = SimpleDiffusionMLP(
            input_dim=input_dim,
            hidden_dim=diffusion_hidden_dim,
            num_layers=diffusion_num_layers,
            condition_dim=condition_dim
        )
        
        # 拡散プロセス
        self.diffusion = HighFreqDiffusion(num_timesteps=num_diffusion_steps)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        学習用の順方向計算
        
        Args:
            x: 入力軌道 [batch_size, sequence_length, input_dim]
            condition: 条件ベクトル [batch_size, condition_dim]
            
        Returns:
            losses: 各成分の損失を含む辞書
        """
        batch_size = x.shape[0]
        device = x.device
        
        # データを分解
        low_freq, high_freq = self.decomposer.decompose(x)
        
        # 低周波成分の学習
        low_freq_pred = self.low_freq_model(low_freq, condition)
        low_freq_loss = F.mse_loss(low_freq_pred, low_freq)
        
        # 高周波成分の学習（拡散モデル）
        # ランダムな時刻ステップをサンプリング
        timesteps = self.diffusion.sample_timesteps(batch_size, device)
        
        # ノイズを追加
        noise = torch.randn_like(high_freq)
        high_freq_noisy = self.diffusion.add_noise(high_freq, timesteps, noise)
        
        # ノイズを予測
        predicted_noise = self.high_freq_model(high_freq_noisy, timesteps, condition)
        high_freq_loss = F.mse_loss(predicted_noise, noise)
        
        return {
            'low_freq_loss': low_freq_loss,
            'high_freq_loss': high_freq_loss,
            'total_loss': low_freq_loss*10 + high_freq_loss,
            'low_freq_pred': low_freq_pred,
            'high_freq_target': high_freq,
            'predicted_noise': predicted_noise,
            'actual_noise': noise
        }
    
    @torch.no_grad()
    def generate(
        self, 
        condition: torch.Tensor, 
        sequence_length: int,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        軌道を生成
        
        Args:
            condition: 条件ベクトル [batch_size, condition_dim]
            sequence_length: 生成する系列長
            num_samples: 各条件に対して生成するサンプル数
            
        Returns:
            generated_trajectories: 生成された軌道 [batch_size * num_samples, sequence_length, input_dim]
        """
        batch_size = condition.shape[0]
        device = condition.device
        
        # 条件を拡張（複数サンプル対応）
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
        
        # 高周波成分を生成
        high_freq_shape = (total_batch_size, sequence_length, self.input_dim)
        high_freq_generated = self.diffusion.generate(
            self.high_freq_model,
            high_freq_shape,
            condition_expanded,
            device
        )
        
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
        """
        モデルを保存
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info()
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'cpu'):
        """
        モデルを読み込み
        """
        checkpoint = torch.load(filepath, map_location=device)
        model_info = checkpoint['model_info']
        
        # モデルを初期化
        model = cls(
            input_dim=model_info['input_dim'],
            condition_dim=model_info['condition_dim'],
            moving_average_window=model_info['moving_average_window'],
            num_diffusion_steps=model_info['diffusion_steps']
        )
        
        # 状態を読み込み
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model