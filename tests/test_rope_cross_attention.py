"""Focused tests for unequal-length RoPE cross-attention."""

import unittest

import torch

from src.models.attention import MultiHeadAttention
from src.models.positional import RotaryPositionalEmbedding


class RoPECrossAttentionTest(unittest.TestCase):
    def test_unequal_query_and_key_lengths(self) -> None:
        attention = MultiHeadAttention(
            d_model=32,
            num_heads=4,
            dropout=0.0,
            rope=RotaryPositionalEmbedding(head_dim=8, max_seq_length=32),
        )
        query = torch.randn(2, 5, 32)
        key_value = torch.randn(2, 11, 32)

        output, weights = attention(query, key_value, key_value)

        self.assertEqual(output.shape, (2, 5, 32))
        self.assertEqual(weights.shape, (2, 4, 5, 11))


if __name__ == "__main__":
    unittest.main()
