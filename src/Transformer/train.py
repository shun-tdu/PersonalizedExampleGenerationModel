# CLAUDE_ADDED
"""
Transformerベースモデルの訓練スクリプト
"""

import os
import sys
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

from Model import TransformerTrajectoryGenerator, TransformerTrainer


class EarlyStopping:
    """
    アーリーストップ機能クラス
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        バリデーション損失をチェックして早期終了判定
        
        Args:
            val_loss: 現在のバリデーション損失
            model: モデル
            
        Returns:
            bool: 早期終了するかどうか
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict({k: v.to(model.device if hasattr(model, 'device') else 'cpu') 
                                     for k, v in self.best_weights.items()})
                print(f"Restored best weights with loss: {self.best_loss:.6f}")
            return True
            
        return False


class TrajectoryDataset(Dataset):
    """
    軌道データセット（UNetのものと同じ）
    """
    def __init__(self, data_path: str):
        data = np.load(data_path)
        self.trajectories = torch.FloatTensor(data['trajectories'])  # [N, 2, seq_len]
        self.conditions = torch.FloatTensor(data['conditions'])      # [N, condition_dim]
        
        # Transformerでは [N, seq_len, 2] の形状が必要
        self.trajectories = self.trajectories.transpose(1, 2)  # [N, seq_len, 2]
        
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return self.trajectories[idx], self.conditions[idx]


class MLFlowTransformerTrainer(TransformerTrainer):
    """
    MLFlow統合Transformerトレーナー
    """
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader = None,
              num_epochs: int = 100,
              save_interval: int = 10,
              checkpoint_dir: str = 'checkpoints',
              early_stopping_config: dict = None):
        """
        MLFlow統合訓練（アーリーストップ対応）
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # アーリーストップ機能の初期化
        early_stopping = None
        if early_stopping_config and early_stopping_config.get('enabled', False):
            early_stopping = EarlyStopping(
                patience=early_stopping_config.get('patience', 10),
                min_delta=early_stopping_config.get('min_delta', 1e-6),
                restore_best_weights=early_stopping_config.get('restore_best_weights', True)
            )
            print(f"Early stopping enabled with patience: {early_stopping.patience}")
        
        self.model.train()
        
        for epoch in range(num_epochs):
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
                
                # アーリーストップチェック
                if early_stopping is not None:
                    if early_stopping(val_loss, self.model):
                        print(f'Early stopping at epoch {epoch+1}')
                        # MLFlowにアーリーストップ情報をログ
                        mlflow.log_params({
                            "early_stopped": True,
                            "early_stop_epoch": epoch + 1,
                            "best_val_loss": early_stopping.best_loss
                        })
                        break
                    elif early_stopping.counter > 0:
                        print(f'Early stopping counter: {early_stopping.counter}/{early_stopping.patience}')
            else:
                print(f'Epoch {epoch+1}: Train Loss = {avg_epoch_loss:.4f}')
                # バリデーションデータがない場合は訓練損失でアーリーストップ
                if early_stopping is not None:
                    if early_stopping(avg_epoch_loss, self.model):
                        print(f'Early stopping at epoch {epoch+1} (using training loss)')
                        mlflow.log_params({
                            "early_stopped": True,
                            "early_stop_epoch": epoch + 1,
                            "best_train_loss": early_stopping.best_loss
                        })
                        break
                    elif early_stopping.counter > 0:
                        print(f'Early stopping counter: {early_stopping.counter}/{early_stopping.patience}')
            
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
        
        checkpoint_path = os.path.join(checkpoint_dir, f'transformer_checkpoint_epoch_{epoch}.pth')
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
        plt.title('Transformer Training Curves')
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
        curve_path = os.path.join(checkpoint_dir, 'transformer_training_curves.png')
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Training curves saved: {curve_path}')
        return curve_path


def create_dummy_data(num_samples: int = 1000, seq_len: int = 101, condition_dim: int = 5):
    """
    ダミーデータの生成（実際のプロジェクトでは実データに置き換える）
    """
    # ランダムな軌道データ（正規化されていると仮定）
    trajectories = np.random.randn(num_samples, seq_len, 2).astype(np.float32)
    
    # ランダムな個人特性データ（動作時間、終点誤差、ジャークなど）
    conditions = np.random.randn(num_samples, condition_dim).astype(np.float32)
    
    return trajectories, conditions


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra統合メイン関数
    """
    print(f"Transformer Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
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
        model = TransformerTrajectoryGenerator(
            input_dim=cfg.model.input_dim,
            condition_dim=condition_dim,
            d_model=cfg.model.d_model,
            nhead=cfg.model.nhead,
            num_encoder_layers=cfg.model.num_encoder_layers,
            num_decoder_layers=cfg.model.num_decoder_layers,
            dim_feedforward=cfg.model.dim_feedforward,
            max_seq_len=cfg.model.max_seq_len,
            dropout=cfg.model.dropout
        ).to(device)
        
        # モデル情報をログ
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_params({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "actual_condition_dim": condition_dim
        })
        
        print(f'Model created with {total_params:,} parameters')
        
        # トレーナー作成（MLFlow統合）
        trainer = MLFlowTransformerTrainer(
            model, device, cfg.training.learning_rate
        )
        
        # 訓練開始
        print('Starting Transformer training...')
        early_stopping_config = cfg.training.get('early_stopping', None)
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=cfg.training.epochs,
            save_interval=cfg.training.save_interval,
            checkpoint_dir=cfg.output.checkpoint_dir,
            early_stopping_config=early_stopping_config
        )
        
        # モデルをMLFlowに保存
        if cfg.mlflow.log_model:
            mlflow.pytorch.log_model(
                model, 
                "model",
                registered_model_name=f"{cfg.mlflow.experiment_name}_transformer"
            )
        
        print('Transformer training completed!')


if __name__ == '__main__':
    main()