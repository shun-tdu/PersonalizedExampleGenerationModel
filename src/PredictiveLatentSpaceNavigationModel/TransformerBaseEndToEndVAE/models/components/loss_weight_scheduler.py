from typing import Dict, Any

class LossWeightScheduler:
    def __init__(self, config: Dict[str, Any]):
        """

        :param config: config (Dict[str, Any]): 各損失の重みの設定。
                例:
                {
                    'beta': {'schedule': 'linear', 'start_epoch': 0, 'end_epoch': 100, 'start_val': 0.0, 'end_val': 1.0},
                    'orthogonal_loss_weight': {'schedule': 'constant', 'val': 0.5}
                }
        """
        self.config = config
        self.current_weights = {}
        self.step(0)

    def _calculate_weight(self, epoch: int, schedule_config: Dict[str, Any]) -> float:
        """単一の重みを計算する"""
        schedule_type = schedule_config.get('schedule', 'constant')

        if schedule_type == 'constant':
            return schedule_config.get('val', 1.0)

        if schedule_type == 'linear':
            start_epoch = schedule_config.get('start_epoch', 0)
            end_epoch = schedule_config.get('end_epoch', 1)
            start_val = schedule_config.get('start_val', 0.0)
            end_val = schedule_config.get('end_val', 1.0)

            if epoch < start_epoch:
                return start_val
            if epoch > end_epoch:
                return end_val

            # 線形補間
            progress = (epoch - start_epoch) / (end_epoch - start_epoch)
            return start_val + progress*(end_val - start_val)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    def step(self, epoch: int):
        """
        現在のエポックに基づいて、すべての重みを更新する。
        学習ループの各エポックの開始時に呼び出す。
        """
        for weight_name, schedule_config in self.config.items():
            self.current_weights[weight_name] = self._calculate_weight(epoch, schedule_config)

    def get_weights(self) -> Dict[str, float]:
        return self.current_weights
