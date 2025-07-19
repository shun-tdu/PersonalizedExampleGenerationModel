# CLAUDE_ADDED
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, Any
import json
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.pytorch
from Model import UNet1D
from TrajectoryDataset import TrajectoryDataset


class DDPMScheduler:
    """
    DDPM (Denoising Diffusion Probabilistic Models) のノイズスケジューラ
    """
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        
        # βのスケジュール（線形）
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # αの計算
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # 逆プロセス用のパラメータ
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
        """
        x0にノイズを追加してxtを生成
        """
        device = x0.device
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps.cpu()].to(device).view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps.cpu()].to(device).view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise


class DiffusionTrainer:
    """
    拡散モデルの訓練クラス
    """
    def __init__(self, 
                 model: UNet1D,
                 scheduler: DDPMScheduler,
                 device: torch.device,
                 learning_rate: float = 1e-4):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        
        # オプティマイザ
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        # 損失関数
        self.criterion = nn.MSELoss()
        
        # 訓練ログ用リスト
        self.train_losses = []
        self.val_losses = []
        
    def train_step(self, batch: tuple) -> float:
        """
        1ステップの訓練
        """
        trajectories, conditions = batch
        trajectories = trajectories.to(self.device)
        conditions = conditions.to(self.device)
        
        batch_size = trajectories.shape[0]
        
        # ランダムなタイムステップをサンプル
        timesteps = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=self.device)
        
        # ガウシアンノイズを生成
        noise = torch.randn_like(trajectories)
        
        # ノイズを追加した軌道を生成
        noisy_trajectories = self.scheduler.add_noise(trajectories, noise, timesteps)
        
        # モデルにノイズ予測をさせる
        predicted_noise = self.model(noisy_trajectories, timesteps, conditions)
        
        # 損失を計算
        loss = self.criterion(predicted_noise, noise)
        
        # 逆伝播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader = None,
              num_epochs: int = 100,
              save_interval: int = 10,
              checkpoint_dir: str = 'checkpoints'):
        """
        モデルの訓練
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            # 訓練ループ
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
                for batch in pbar:
                    loss = self.train_step(batch)
                    epoch_loss += loss
                    
                    pbar.set_postfix({'Loss': f'{loss:.4f}'})
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_epoch_loss)
            
            # バリデーション
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                print(f'Epoch {epoch+1}: Train Loss = {avg_epoch_loss:.4f}, Val Loss = {val_loss:.4f}')
            else:
                print(f'Epoch {epoch+1}: Train Loss = {avg_epoch_loss:.4f}')
            
            # チェックポイント保存
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1, checkpoint_dir)
        
        # 訓練完了後に損失曲線を保存
        self.save_training_curves(checkpoint_dir)
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        バリデーション
        """
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                trajectories, conditions = batch
                trajectories = trajectories.to(self.device)
                conditions = conditions.to(self.device)
                
                batch_size = trajectories.shape[0]
                timesteps = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=self.device)
                noise = torch.randn_like(trajectories)
                noisy_trajectories = self.scheduler.add_noise(trajectories, noise, timesteps)
                
                predicted_noise = self.model(noisy_trajectories, timesteps, conditions)
                loss = self.criterion(predicted_noise, noise)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def save_checkpoint(self, epoch: int, checkpoint_dir: str):
        """
        チェックポイント保存
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')
    
    def save_training_curves(self, checkpoint_dir: str):
        """
        訓練曲線を画像として保存
        """
        if len(self.train_losses) == 0:
            return
            
        plt.figure(figsize=(12, 4))
        
        # 訓練損失
        plt.subplot(1, 2, 1)
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        if len(self.val_losses) > 0:
            plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 損失改善率
        if len(self.train_losses) > 1:
            plt.subplot(1, 2, 2)
            train_improvement = [(self.train_losses[i-1] - self.train_losses[i]) / self.train_losses[i-1] * 100 
                               for i in range(1, len(self.train_losses))]
            plt.plot(epochs[1:], train_improvement, 'b-', label='Training Improvement')
            
            if len(self.val_losses) > 1:
                val_improvement = [(self.val_losses[i-1] - self.val_losses[i]) / self.val_losses[i-1] * 100 
                                 for i in range(1, len(self.val_losses))]
                plt.plot(epochs[1:], val_improvement, 'r-', label='Validation Improvement')
            
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.xlabel('Epoch')
            plt.ylabel('Improvement (%)')
            plt.title('Loss Improvement Rate')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        curve_path = os.path.join(checkpoint_dir, 'training_curves.png')
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Training curves saved: {curve_path}')


def create_dummy_data(num_samples: int = 1000, seq_len: int = 101, condition_dim: int = 5):
    """
    ダミーデータの生成（実際のプロジェクトでは実データに置き換える）
    """
    # ランダムな軌道データ（正規化されていると仮定）
    trajectories = np.random.randn(num_samples, 2, seq_len).astype(np.float32)
    
    # ランダムな個人特性データ（動作時間、終点誤差、ジャークなど）
    conditions = np.random.randn(num_samples, condition_dim).astype(np.float32)
    
    return trajectories, conditions


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra統合メイン関数
    """
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # デバイス設定
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # MLFlowセットアップ
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        # 設定をMLFlowにログ
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        
        if cfg.training.use_dummy:
            # ダミーデータを使用
            print('Creating dummy data...')
            train_trajectories, train_conditions = create_dummy_data(1000, 101, cfg.model.condition_dim)
            val_trajectories, val_conditions = create_dummy_data(200, 101, cfg.model.condition_dim)
            
            # ダミーTrajectoryDatasetクラスを内部定義
            class DummyTrajectoryDataset(Dataset):
                def __init__(self, trajectory_data: np.ndarray, condition_data: np.ndarray):
                    self.trajectories = torch.FloatTensor(trajectory_data)
                    self.conditions = torch.FloatTensor(condition_data)
                    
                def __len__(self):
                    return len(self.trajectories)
                
                def __getitem__(self, idx):
                    return self.trajectories[idx], self.conditions[idx]
            
            train_dataset = DummyTrajectoryDataset(train_trajectories, train_conditions)
            val_dataset = DummyTrajectoryDataset(val_trajectories, val_conditions)
            condition_dim = cfg.model.condition_dim
        else:
            # 実データを使用
            print(f'Loading training data from: {cfg.data.train_data}')
            train_dataset = TrajectoryDataset(cfg.data.train_data)
            
            if cfg.data.val_data:
                print(f'Loading validation data from: {cfg.data.val_data}')
                val_dataset = TrajectoryDataset(cfg.data.val_data)
            else:
                # バリデーションデータが指定されていない場合は訓練データから分割
                print('Splitting training data for validation...')
                train_size = int((1 - cfg.data.val_split_ratio) * len(train_dataset))
                val_size = len(train_dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            
            # データセットから条件次元を取得
            sample_trajectory, sample_condition = train_dataset[0]
            condition_dim = sample_condition.shape[0]
            print(f'Condition dimension detected: {condition_dim}')
        
        # データローダー作成
        train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)
        
        # モデル作成
        model = UNet1D(
            input_dim=cfg.model.input_dim,
            condition_dim=condition_dim,
            time_embed_dim=cfg.model.time_embed_dim,
            base_channels=cfg.model.base_channels
        ).to(device)
        
        # モデル情報をログ
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_params({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "actual_condition_dim": condition_dim
        })
        
        # スケジューラ作成
        scheduler = DDPMScheduler(
            num_timesteps=cfg.scheduler.num_timesteps,
            beta_start=cfg.scheduler.beta_start,
            beta_end=cfg.scheduler.beta_end
        )
        
        # トレーナー作成（MLFlow統合）
        trainer = MLFlowDiffusionTrainer(
            model, scheduler, device, cfg.training.learning_rate
        )
        
        # 訓練開始
        print('Starting training...')
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=cfg.training.epochs,
            save_interval=cfg.training.save_interval,
            checkpoint_dir=cfg.output.checkpoint_dir
        )
        
        # モデルをMLFlowに保存
        if cfg.mlflow.log_model:
            mlflow.pytorch.log_model(
                model, 
                "model",
                registered_model_name=f"{cfg.mlflow.experiment_name}_unet"
            )
        
        print('Training completed!')


class MLFlowDiffusionTrainer(DiffusionTrainer):
    """
    MLFlow統合拡散モデル訓練クラス
    """
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader = None,
              num_epochs: int = 100,
              save_interval: int = 10,
              checkpoint_dir: str = 'checkpoints'):
        """
        MLFlow統合訓練
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            # 訓練ループ
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
                for batch in pbar:
                    loss = self.train_step(batch)
                    epoch_loss += loss
                    
                    pbar.set_postfix({'Loss': f'{loss:.4f}'})
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_epoch_loss)
            
            # バリデーション
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                print(f'Epoch {epoch+1}: Train Loss = {avg_epoch_loss:.4f}, Val Loss = {val_loss:.4f}')
            else:
                print(f'Epoch {epoch+1}: Train Loss = {avg_epoch_loss:.4f}')
            
            # MLFlowにメトリクスをログ
            mlflow.log_metrics({
                "train_loss": avg_epoch_loss,
                "val_loss": val_loss if val_loss is not None else 0.0,
                "epoch": epoch + 1
            }, step=epoch)
            
            # チェックポイント保存
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = self.save_checkpoint(epoch + 1, checkpoint_dir)
                
                # MLFlowにアーティファクトとして保存
                mlflow.log_artifact(checkpoint_path, "checkpoints")
        
        # 訓練完了後に損失曲線を保存
        curve_path = self.save_training_curves(checkpoint_dir)
        if curve_path:
            mlflow.log_artifact(curve_path, "plots")
    
    def save_checkpoint(self, epoch: int, checkpoint_dir: str) -> str:
        """
        チェックポイント保存（パス返却版）
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')
        return checkpoint_path
    
    def save_training_curves(self, checkpoint_dir: str) -> str:
        """
        訓練曲線を画像として保存（パス返却版）
        """
        if len(self.train_losses) == 0:
            return None
            
        plt.figure(figsize=(12, 4))
        
        # 訓練損失
        plt.subplot(1, 2, 1)
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        if len(self.val_losses) > 0:
            plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 損失改善率
        if len(self.train_losses) > 1:
            plt.subplot(1, 2, 2)
            train_improvement = [(self.train_losses[i-1] - self.train_losses[i]) / self.train_losses[i-1] * 100 
                               for i in range(1, len(self.train_losses))]
            plt.plot(epochs[1:], train_improvement, 'b-', label='Training Improvement')
            
            if len(self.val_losses) > 1:
                val_improvement = [(self.val_losses[i-1] - self.val_losses[i]) / self.val_losses[i-1] * 100 
                                 for i in range(1, len(self.val_losses))]
                plt.plot(epochs[1:], val_improvement, 'r-', label='Validation Improvement')
            
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.xlabel('Epoch')
            plt.ylabel('Improvement (%)')
            plt.title('Loss Improvement Rate')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        curve_path = os.path.join(checkpoint_dir, 'training_curves.png')
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Training curves saved: {curve_path}')
        return curve_path


if __name__ == '__main__':
    main()