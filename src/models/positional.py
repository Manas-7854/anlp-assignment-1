"""Positional encodings implemented with basic PyTorch operations."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sinusoidal positions to token embeddings."""

    def __init__(self, d_model: int, max_seq_length: int = 10_000) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive.")

        position = torch.arange(max_seq_length, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        angles = position * frequency

        encoding = torch.zeros(max_seq_length, d_model)
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])

        # [1, max_seq_length, d_model] broadcasts across the batch.
        self.register_buffer("encoding", encoding.unsqueeze(0))
        self.d_model = d_model
        self.max_seq_length = max_seq_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positions to x with shape [batch, seq_len, d_model]."""

        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError(
                f"Expected x with shape [batch, seq_len, {self.d_model}]."
            )
        seq_len = x.size(1)
        if seq_len > self.max_seq_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum "
                f"{self.max_seq_length}."
            )
        return x + self.encoding[:, :seq_len].to(dtype=x.dtype)


class RotaryPositionalEmbedding(nn.Module):
    """Apply rotary positional embeddings to one attention tensor."""

    def __init__(
        self,
        head_dim: int,
        max_seq_length: int = 10_000,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even number.")
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive.")
        if base <= 0:
            raise ValueError("base must be positive.")

        inverse_frequency = base ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        positions = torch.arange(max_seq_length, dtype=torch.float32).unsqueeze(1)
        angles = positions * inverse_frequency.unsqueeze(0)

        # [1, 1, max_seq_length, head_dim / 2] broadcasts over batch and heads.
        self.register_buffer("cosine", torch.cos(angles).unsqueeze(0).unsqueeze(0))
        self.register_buffer("sine", torch.sin(angles).unsqueeze(0).unsqueeze(0))
        self.head_dim = head_dim
        self.max_seq_length = max_seq_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate x using its own sequence length, preserving its shape."""

        if x.ndim != 4 or x.size(-1) != self.head_dim:
            raise ValueError(
                "Expected an attention tensor with shape "
                f"[batch, heads, seq_len, {self.head_dim}]."
            )
        seq_len = x.size(-2)
        if seq_len > self.max_seq_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum "
                f"{self.max_seq_length}."
            )

        cosine = self.cosine[:, :, :seq_len].to(dtype=x.dtype)
        sine = self.sine[:, :, :seq_len].to(dtype=x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]

        # Rotate each adjacent pair: (x, y) -> (x cos - y sin, x sin + y cos).
        rotated_even = even * cosine - odd * sine
        rotated_odd = even * sine + odd * cosine
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
