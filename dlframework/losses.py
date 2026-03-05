"""
Loss functions for time-series prediction.

How to add a new loss:
1. Define your nn.Module subclass in this file
2. Add it to LOSS_REGISTRY dict below
3. Reference it by name in your training config: cfg["loss"] = "my_loss"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberDirectionalLoss(nn.Module):
    """
    Combined loss:
      - Huber loss for magnitude accuracy (robust to outliers)
      - Sign penalty for directional accuracy (matters for trading)
    """
    def __init__(self, delta: float = 0.01, alpha: float = 0.3):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        magnitude = self.huber(pred, target)
        directional = torch.mean(F.relu(-pred * target))
        return magnitude + self.alpha * directional


class MapeLoss(nn.Module):
    """Mean Absolute Percentage Error."""
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs((target - pred) / (torch.abs(target) + self.epsilon)))


class QuantileLoss(nn.Module):
    def __init__(self, quantile: float = 0.5):
        super().__init__()
        self.q = quantile

    def forward(self, pred, target):
        e = target - pred
        return torch.mean(torch.max(self.q * e, (self.q - 1) * e))


class LogCoshLoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean(torch.log(torch.cosh(pred - target)))


class ScaledMSELoss(nn.Module):
    def __init__(self, scale: float = 100.0):
        super().__init__()
        self.scale = scale
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(pred * self.scale, target * self.scale)


LOSS_REGISTRY = {
    "mse":               nn.MSELoss(),
    "mae":               nn.L1Loss(),
    "huber":             nn.HuberLoss(delta=0.1),
    "huber_directional": HuberDirectionalLoss(delta=1, alpha=0.3),
    "mape":              MapeLoss(epsilon=1e-8),
    "quantile_0.5":      QuantileLoss(quantile=0.5),
    "log_cosh":          LogCoshLoss(),
    "scaled_mse":        ScaledMSELoss(scale=100.0),
}
