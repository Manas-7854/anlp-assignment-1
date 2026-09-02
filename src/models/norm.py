"""Normalization layers implemented from basic PyTorch operations."""

from __future__ import annotations

import torch
from torch import nn


class LayerNorm(nn.Module):
    """Layer normalization over the final feature dimension."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize each token across its final feature dimension.
        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        return self.gamma * normalized + self.beta


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the final feature dimension."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS(x) = sqrt(mean(x^2)); RMSNorm does not center or add bias.
        inverse_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * x * inverse_rms
