"""Attention layers implemented from basic PyTorch operations."""

from __future__ import annotations

import math

import torch
from torch import nn

from .positional import RotaryPositionalEmbedding


class ScaledDotProductAttention(nn.Module):
    """Compute softmax(QK^T / sqrt(d_k))V."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _prepare_attention_mask(
        mask: torch.Tensor, score_dimensions: int
    ) -> torch.Tensor:
        # Add singleton head/group axes until the mask broadcasts over scores.
        if mask.dim() == 2:
            while mask.dim() < score_dimensions:
                mask = mask.unsqueeze(0)
            return mask
        if 3 <= mask.dim() <= score_dimensions:
            while mask.dim() < score_dimensions:
                mask = mask.unsqueeze(1)
            return mask
        raise ValueError("attention_mask has an incompatible number of dimensions.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [..., query_length, head_dim]
            key: Broadcastable [..., key_length, head_dim]
            value: Broadcastable [..., key_length, value_dim]
            attention_mask: Boolean True blocks attention; float masks are added.
            key_padding_mask: [batch, key_length], where True marks padding.
            causal: Block future positions for decoder self-attention.
        """

        head_dim = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        # scores: [..., query_length, key_length]

        if attention_mask is not None:
            mask = self._prepare_attention_mask(
                attention_mask, scores.dim()
            ).to(scores.device)
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, float("-inf"))
            else:
                scores = scores + mask.to(scores.dtype)

        if key_padding_mask is not None:
            if key_padding_mask.dim() != 2:
                raise ValueError("key_padding_mask must have shape [batch, key_length].")
            padding = key_padding_mask.to(device=scores.device, dtype=torch.bool)
            padding_shape = (
                padding.size(0),
                *([1] * (scores.dim() - 2)),
                padding.size(1),
            )
            scores = scores.masked_fill(
                padding.view(padding_shape), float("-inf")
            )

        if causal:
            query_length, key_length = query.size(-2), key.size(-2)
            if query_length != key_length:
                raise ValueError("Causal self-attention requires equal Q and K lengths.")
            future = torch.triu(
                torch.ones(
                    query_length,
                    key_length,
                    dtype=torch.bool,
                    device=scores.device,
                ),
                diagonal=1,
            )
            scores = scores.masked_fill(future, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        # A completely masked query should produce zero output rather than NaNs.
        weights = torch.nan_to_num(weights, nan=0.0)
        output = torch.matmul(self.dropout(weights), value)
        return output, weights


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with separate Q, K, and V projections."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        rope: RotaryPositionalEmbedding | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = rope

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attention = ScaledDotProductAttention(dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        # [B, L, d_model] -> [B, heads, L, head_dim]
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply self-attention, or cross-attention when key/value are supplied."""

        key = query if key is None else key
        value = key if value is None else value

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        attended, weights = self.attention(
            q,
            k,
            v,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            causal=causal,
        )

        batch, _, query_length, _ = attended.shape
        # [B, heads, Q, head_dim] -> [B, Q, d_model]
        attended = attended.transpose(1, 2).contiguous().view(
            batch, query_length, self.d_model
        )
        return self.out_proj(attended), weights


class GroupedQueryAttention(nn.Module):
    """GQA with many query heads and fewer shared key/value heads."""

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_query_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("Model and head dimensions must be positive.")
        if d_model % num_query_heads != 0:
            raise ValueError("d_model must be divisible by num_query_heads.")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads.")

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.queries_per_kv = num_query_heads // num_kv_heads
        kv_dimension = num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, kv_dimension, bias=bias)
        self.v_proj = nn.Linear(d_model, kv_dimension, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attention = ScaledDotProductAttention(dropout)

    @staticmethod
    def _split_heads(
        tensor: torch.Tensor, num_heads: int, head_dim: int
    ) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, num_heads, head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply grouped-query self-attention or cross-attention."""

        key = query if key is None else key
        value = key if value is None else value

        # Q: [B, query_heads, Q, head_dim]
        q = self._split_heads(
            self.q_proj(query), self.num_query_heads, self.head_dim
        )
        # K/V: [B, kv_heads, K, head_dim]
        k = self._split_heads(self.k_proj(key), self.num_kv_heads, self.head_dim)
        v = self._split_heads(self.v_proj(value), self.num_kv_heads, self.head_dim)

        batch, _, query_length, _ = q.shape
        key_length = k.size(-2)
        # [B, query_heads, Q, D] -> [B, kv_heads, queries_per_kv, Q, D]
        q = q.reshape(
            batch,
            self.num_kv_heads,
            self.queries_per_kv,
            query_length,
            self.head_dim,
        )
        # Singleton group axes broadcast shared K/V without duplicating storage.
        k = k.unsqueeze(2)  # [B, kv_heads, 1, K, D]
        v = v.unsqueeze(2)  # [B, kv_heads, 1, K, D]

        if attention_mask is not None and attention_mask.dim() == 4:
            mask_heads = attention_mask.size(1)
            if mask_heads == self.num_query_heads:
                attention_mask = attention_mask.reshape(
                    attention_mask.size(0),
                    self.num_kv_heads,
                    self.queries_per_kv,
                    attention_mask.size(-2),
                    attention_mask.size(-1),
                )
            elif mask_heads != 1:
                raise ValueError(
                    "GQA attention_mask head dimension must be 1 or "
                    "num_query_heads."
                )

        attended, weights = self.attention(
            q,
            k,
            v,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            causal=causal,
        )

        # Restore the public [B, query_heads, ...] representation.
        attended = attended.reshape(
            batch, self.num_query_heads, query_length, self.head_dim
        )
        weights = weights.reshape(
            batch, self.num_query_heads, query_length, key_length
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch, query_length, self.d_model
        )
        return self.out_proj(attended), weights
