"""
Author: Lynn Ye
Created on: 2025/4/14
Brief: 
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections import defaultdict
from typing import Mapping, Optional

from src.util.definitions import NOTE_TYPE, IGNORE_LABEL_INDEX, VEL_TYPE, TS_TYPE


class NoteCrossEntropy(nn.Module):
    def __init__(self, criterion: nn.Module = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL_INDEX)):
        super().__init__()
        self.criterion = criterion

    def forward(self, logits: torch.Tensor, types: torch.Tensor, all_labels):
        note_mask = (types == NOTE_TYPE)
        if not note_mask.any():
            return torch.tensor(0.0, device=logits.device)
        note_labels = all_labels[note_mask]
        note_logits = logits[note_mask]
        return self.criterion(note_logits, note_labels)


class VelocityLoss(nn.Module):
    def __init__(self, criterion: nn.Module = nn.SmoothL1Loss()):
        super().__init__()
        self.criterion = criterion

    def forward(self, preds: torch.Tensor, types: torch.Tensor, all_labels: torch.Tensor):
        """
        :param preds: complete prediction
        :param types:
        :param all_labels: may contain IGNORE_LABEL_INDEX
        :return:
        """
        velocity_mask = (types == VEL_TYPE)
        if not velocity_mask.any():
            return torch.tensor(0.0, device=preds.device)
        velocity_preds = preds[velocity_mask]
        velocity_targets = all_labels[velocity_mask].float()

        valid_pred_mask = (velocity_targets != IGNORE_LABEL_INDEX)
        velocity_preds = velocity_preds[valid_pred_mask]
        velocity_targets = velocity_targets[valid_pred_mask]  # Divide by 127 to avoid big loss
        return self.criterion(velocity_preds, velocity_targets / 127)  # pred in range [0, 1] after sigmoid


class TimeshiftLoss(nn.Module):
    def __init__(self, criterion: nn.Module = nn.SmoothL1Loss()):
        super().__init__()
        self.criterion = criterion

    def forward(self, preds: torch.Tensor, types: torch.Tensor, all_labels: torch.Tensor):
        """
        :param preds: complete prediction
        :param types:
        :param all_labels: may contain IGNORE_LABEL_INDEX, timeshift in milliseconds
        :return:
        """
        timeshift_mask = (types == TS_TYPE)
        if not timeshift_mask.any():
            return torch.tensor(0.0, device=preds.device)
        ts_preds = preds[timeshift_mask]
        ts_targets = all_labels[timeshift_mask].float()

        valid_pred_mask = (ts_targets != IGNORE_LABEL_INDEX)
        ts_preds = ts_preds[valid_pred_mask]
        ts_targets = ts_targets[valid_pred_mask]
        return self.criterion(ts_preds, torch.log(ts_targets + 1e-7))  # Predict log-transformed timeshift values


# reference: PESTO https://github.com/SonyCSLParis/pesto
class LossWeighting(pl.Callback):
    def __init__(self, weights: Mapping[str, float] or None = None) -> None:
        self.weights = weights if weights is not None else defaultdict(lambda: 1.)

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs,
                           batch,
                           batch_idx: int) -> None:
        pl_module.log_dict({f"hparams/{k}_weight": v for k, v in self.weights.items()}, prog_bar=False, logger=True)

    def combine_losses(self, **losses):
        self.update_weights(losses)
        return sum([self.weights[key] * losses[key] for key in self.weights.keys()])

    def update_weights(self, losses):
        pass

    def __str__(self):
        params = '\n'.join(f"\t{k}: {v}" for k, v in vars(self).items())
        return self.__class__.__name__ + "(\n" + params + "\n)"


class GradientsLossWeighting(LossWeighting):
    def __init__(self,
                 weights: Mapping[str, float] or None = None,
                 last_layer: Optional[torch.Tensor] = None,
                 ema_rate: float = 0.):
        super(GradientsLossWeighting, self).__init__(weights)
        self.last_layer = last_layer
        self.ema_rate = ema_rate
        self.grads = {k: 1 - v for k, v in weights.items()}
        self.weights_tensor = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.weights_tensor = torch.zeros(len(self.weights.keys()), device=pl_module.device)

    def update_weights(self, losses):
        # compute gradient w.r.t last layer for each loss term
        for i, (k, loss) in enumerate(losses.items()):
            if not loss.requires_grad:
                return

            grads = torch.autograd.grad(loss, self.get_last_layer(k), retain_graph=True)[0].norm().detach()
            old_grads = self.grads[k]
            if old_grads is not None:
                grads = self.ema_rate * old_grads + (1 - self.ema_rate) * grads
            self.grads[k] = grads
            self.weights_tensor[i] = grads

        # compute the weight of this loss based on these gradients
        self.weights_tensor = 1 - self.weights_tensor / self.weights_tensor.sum().clip(min=1e-7)

        # associate each weight with the right loss
        for i, k in enumerate(losses.keys()):
            self.weights[k] = self.weights_tensor[i]

    def get_last_layer(self, key: str) -> torch.Tensor:
        return self.last_layer


def main():
    print("Hello, world!")


if __name__ == "__main__":
    main()
