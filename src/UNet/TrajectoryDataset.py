import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    .npz形式のデータセットを読み込み、PyTorchの学習に利用するためのDatasetクラス。
    """

    def __init__(self, data_path: str):
        """
        クラスの初期化時に、データファイルを読み込んで準備します。

        Args:
            data_path (str): .npzデータセットファイルのパス。
        """
        try:
            # .npzファイルを読み込む
            data = np.load(data_path)

            # 各データをPyTorchのTensorに変換してクラスのメンバー変数として保持
            # 軌道データ: (N, 101, 2) -> (N, 2, 101) に次元を入れ替えておく
            self.trajectories = torch.from_numpy(data['trajectories']).float().permute(0, 2, 1)

            # 条件ベクトル (標準化済み)
            self.conditions = torch.from_numpy(data['conditions']).float()

            # 後で新しいデータを正規化するために、スケーラーの情報を保持
            self.condition_mean = torch.from_numpy(data['condition_scaler_mean']).float()
            self.condition_std = torch.from_numpy(data['condition_scaler_scale']).float()

            print(f"データセットを正常に読み込みました。")
            print(f"  - 軌道データ数: {len(self.trajectories)}")
            print(f"  - 軌道データの形状: {self.trajectories.shape}")
            print(f"  - 条件ベクトルの形状: {self.conditions.shape}")

        except FileNotFoundError:
            print(f"エラー: データファイルが見つかりません: {data_path}")
            # エラー発生時は、ダミーデータを作成してプログラムが停止しないようにする
            self.trajectories = torch.randn(100, 2, 101)
            self.conditions = torch.randn(100, 3)
            self.condition_mean = torch.zeros(3)
            self.condition_std = torch.ones(3)
            print("代わりにダミーデータセットを作成しました。")
        except KeyError as e:
            print(f"エラー: .npzファイルに必要なキーがありません: {e}")
            # プログラムを停止させる
            raise

    def __len__(self):
        """
        データセットに含まれるデータの総数を返します。
        DataLoaderが、データセットの大きさを把握するために使います。
        """
        return len(self.trajectories)

    def __getitem__(self, idx):
        """
        指定されたインデックス(idx)のデータを1つ取り出して返します。
        DataLoaderが、バッチを作成するために使います。
        """
        trajectory = self.trajectories[idx]
        condition = self.conditions[idx]

        return trajectory, condition


