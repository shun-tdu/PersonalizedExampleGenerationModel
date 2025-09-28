from typing import Dict, Any
import math

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

        # CLAUDE_ADDED: Sigmoid delayed schedule for progressive skip weighting
        elif schedule_type == 'sigmoid_delayed':
            start_epoch = schedule_config.get('start_epoch', 0)
            end_epoch = schedule_config.get('end_epoch', 100)
            start_val = schedule_config.get('start_val', 0.0)
            end_val = schedule_config.get('end_val', 1.0)

            if epoch < start_epoch:
                return start_val

            # シグモイド関数での段階的増加（遅延開始）
            progress = (epoch - start_epoch) / max(end_epoch - start_epoch, 1)
            # シグモイド曲線: ゆっくり開始→急激な変化→ゆっくり収束
            sigmoid_progress = 1 / (1 + math.exp(-6 * (progress - 0.5)))

            return start_val + sigmoid_progress * (end_val - start_val)

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


class ImprovedLossWeightScheduler(LossWeightScheduler):
    """改良された損失重みスケジューラ（適応的重み計算付き）"""

    def __init__(self, loss_schedule_config, structure_priority_epochs=60):
        # KL損失の重複を避けるため設定を修正
        self.structure_priority_epochs = structure_priority_epochs

        # 修正されたconfig（KL損失の重複を解決）
        corrected_config = self._fix_kl_duplication(loss_schedule_config)
        super().__init__(corrected_config)

    def _fix_kl_duplication(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """KL損失の重複問題を修正"""
        corrected_config = config.copy()

        # beta_style_lateとbeta_skill_lateを統合版に変更
        if 'beta_style_late' in corrected_config:
            # 後期のKL損失は早期の値を上書きする形に変更
            late_config = corrected_config.pop('beta_style_late')

            # 元のbeta_styleの終了エポック以降に適用
            if 'beta_style' in corrected_config:
                early_end_epoch = corrected_config['beta_style']['end_epoch']
                late_start_epoch = max(late_config['start_epoch'], early_end_epoch + 1)

                # 段階的KL損失スケジュール（重複なし）
                corrected_config['beta_style'] = {
                    'schedule': 'piecewise_linear',
                    'segments': [
                        {
                            'start_epoch': corrected_config['beta_style']['start_epoch'],
                            'end_epoch': early_end_epoch,
                            'start_val': corrected_config['beta_style']['start_val'],
                            'end_val': corrected_config['beta_style']['end_val']
                        },
                        {
                            'start_epoch': late_start_epoch,
                            'end_epoch': late_config['end_epoch'],
                            'start_val': corrected_config['beta_style']['end_val'],  # 連続性確保
                            'end_val': late_config['end_val']
                        }
                    ]
                }

        # 同様にbeta_skill_lateも修正
        if 'beta_skill_late' in corrected_config:
            late_config = corrected_config.pop('beta_skill_late')

            if 'beta_skill' in corrected_config:
                early_end_epoch = corrected_config['beta_skill']['end_epoch']
                late_start_epoch = max(late_config['start_epoch'], early_end_epoch + 1)

                corrected_config['beta_skill'] = {
                    'schedule': 'piecewise_linear',
                    'segments': [
                        {
                            'start_epoch': corrected_config['beta_skill']['start_epoch'],
                            'end_epoch': early_end_epoch,
                            'start_val': corrected_config['beta_skill']['start_val'],
                            'end_val': corrected_config['beta_skill']['end_val']
                        },
                        {
                            'start_epoch': late_start_epoch,
                            'end_epoch': late_config['end_epoch'],
                            'start_val': corrected_config['beta_skill']['end_val'],
                            'end_val': late_config['end_val']
                        }
                    ]
                }

        return corrected_config

    def _calculate_weight(self, epoch: int, schedule_config: Dict[str, Any]) -> float:
        """区分線形スケジュールを含む重み計算"""
        schedule_type = schedule_config.get('schedule', 'constant')

        # 基本スケジュールは親クラスに委譲
        if schedule_type != 'piecewise_linear':
            return super()._calculate_weight(epoch, schedule_config)

        # 区分線形スケジュールの処理
        segments = schedule_config.get('segments', [])

        for segment in segments:
            start_epoch = segment['start_epoch']
            end_epoch = segment['end_epoch']

            if start_epoch <= epoch <= end_epoch:
                start_val = segment['start_val']
                end_val = segment['end_val']

                if epoch < start_epoch:
                    return start_val
                if epoch > end_epoch:
                    return end_val

                # 線形補間
                progress = (epoch - start_epoch) / max(end_epoch - start_epoch, 1)
                return start_val + progress * (end_val - start_val)

        # どのセグメントにも該当しない場合は最後のセグメントの終了値
        if segments:
            return segments[-1]['end_val']
        return 0.0

    def get_adaptive_weights(self, epoch, reconstruction_loss, structure_losses):
        """適応的重み計算"""
        base_weights = self.get_weights()

        # 構造損失の平均（NaN/inf チェック付き）
        valid_losses = []
        for loss in structure_losses.values():
            if isinstance(loss, torch.Tensor):
                if torch.isfinite(loss):
                    valid_losses.append(loss.item())
            else:
                valid_losses.append(float(loss))

        avg_structure_loss = sum(valid_losses) / max(len(valid_losses), 1)

        if epoch < self.structure_priority_epochs:
            # 構造優先期間：再構成損失を適度に制限
            if isinstance(reconstruction_loss, torch.Tensor):
                recon_loss_val = reconstruction_loss.item() if torch.isfinite(reconstruction_loss) else 1.0
            else:
                recon_loss_val = float(reconstruction_loss)

            recon_weight = min(1.5, max(0.5, recon_loss_val / 0.05))

            # 構造損失が十分下がっていない場合は構造学習を強化
            if avg_structure_loss > 0.3:
                structure_multiplier = 1.8
            elif avg_structure_loss > 0.1:
                structure_multiplier = 1.3
            else:
                structure_multiplier = 1.0

        else:
            # 再構成最適化期間
            recon_weight = 1.0
            structure_multiplier = 0.7  # 構造損失の重みを減らす

        # 重みを調整
        adapted_weights = base_weights.copy()
        adapted_weights['reconstruction'] = recon_weight

        # 構造関連損失の重み調整
        for key in ['orthogonal_loss', 'contrastive_loss', 'style_classification_loss',
                    'skill_regression_loss', 'manifold_loss']:
            if key in adapted_weights:
                adapted_weights[key] *= structure_multiplier

        return adapted_weights