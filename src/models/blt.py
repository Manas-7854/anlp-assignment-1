"""Lightweight byte-to-patch modules for the C5 BLT configuration."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .attention import MultiHeadAttention
from .norm import LayerNorm
from .positional import SinusoidalPositionalEncoding


class _LocalBlock(nn.Module):
    """Small Pre-LN Transformer block used inside each byte patch."""

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float
    ) -> None:
        super().__init__()
        self.attention_norm = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn_norm = LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor, causal: bool
    ) -> torch.Tensor:
        attended, _ = self.attention(
            self.attention_norm(x),
            key_padding_mask=padding_mask,
            causal=causal,
        )
        x = x + self.dropout(attended)
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class BLTLocalEncoder(nn.Module):
    """Encode raw bytes and pool every strided patch into one latent vector."""

    def __init__(
        self,
        byte_vocab_size: int,
        padding_idx: int,
        local_d_model: int,
        global_d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        patch_size: int,
        max_seq_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embedding = nn.Embedding(
            byte_vocab_size, local_d_model, padding_idx=padding_idx
        )
        self.position = SinusoidalPositionalEncoding(
            local_d_model, max_seq_length
        )
        self.blocks = nn.ModuleList(
            [
                _LocalBlock(local_d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.projection = nn.Linear(local_d_model, global_d_model)
        self.dropout = nn.Dropout(dropout)

    def patch_padding_mask(self, byte_mask: torch.Tensor) -> torch.Tensor:
        padding = (-byte_mask.size(1)) % self.patch_size
        byte_mask = F.pad(byte_mask, (0, padding), value=True)
        return byte_mask.view(
            byte_mask.size(0), -1, self.patch_size
        ).all(dim=-1)

    def forward(
        self, byte_ids: torch.Tensor, byte_padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length = byte_ids.shape
        x = self.dropout(self.position(self.embedding(byte_ids)))
        padding = (-length) % self.patch_size
        x = F.pad(x, (0, 0, 0, padding))
        mask = F.pad(byte_padding_mask, (0, padding), value=True)

        patch_count = x.size(1) // self.patch_size
        x = x.view(batch * patch_count, self.patch_size, -1)
        local_mask = mask.view(batch * patch_count, self.patch_size)
        for block in self.blocks:
            x = block(x, local_mask, causal=False)

        valid = (~local_mask).unsqueeze(-1)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        patches = self.projection(pooled).view(batch, patch_count, -1)
        patch_mask = local_mask.view(batch, patch_count, -1).all(dim=-1)
        return patches.masked_fill(patch_mask.unsqueeze(-1), 0.0), patch_mask


class BLTLocalDecoder(nn.Module):
    """Expand global patch states and predict raw bytes autoregressively."""

    def __init__(
        self,
        byte_vocab_size: int,
        padding_idx: int,
        local_d_model: int,
        global_d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        patch_size: int,
        max_seq_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embedding = nn.Embedding(
            byte_vocab_size, local_d_model, padding_idx=padding_idx
        )
        self.context_projection = nn.Linear(global_d_model, local_d_model)
        self.position = SinusoidalPositionalEncoding(
            local_d_model, max_seq_length
        )
        self.blocks = nn.ModuleList(
            [
                _LocalBlock(local_d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_projection = nn.Linear(local_d_model, byte_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        decoder_input: torch.Tensor,
        patch_states: torch.Tensor,
        byte_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length = decoder_input.shape
        context = self.context_projection(patch_states)
        context = context.unsqueeze(2).expand(-1, -1, self.patch_size, -1)
        context = context.reshape(batch, -1, context.size(-1))[:, :length]
        x = self.dropout(
            self.position(self.embedding(decoder_input) + context)
        )

        padding = (-length) % self.patch_size
        x = F.pad(x, (0, 0, 0, padding))
        mask = F.pad(byte_padding_mask, (0, padding), value=True)
        patch_count = x.size(1) // self.patch_size
        x = x.view(batch * patch_count, self.patch_size, -1)
        local_mask = mask.view(batch * patch_count, self.patch_size)
        for block in self.blocks:
            x = block(x, local_mask, causal=True)

        x = x.view(batch, patch_count * self.patch_size, -1)[:, :length]
        return self.output_projection(x)
