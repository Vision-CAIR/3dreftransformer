"""
Multi-Head Attention implementation
Originally copy-pasted from https://github.com/CyberZHG/torch-multi-head-attention
We have our own implementation for 2 reasons:
    - pytorch's version is full of bugs
    - we are going to add relative positional encoding someday
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MultiHeadAttention', 'MultiHeadAttentionRelPosEmb']


class MultiHeadAttentionRelPosEmb(nn.Module):
    def __init__(self, dim, num_heads, bias=True, dropout_rate: float = 0.0):
        """Multi-head attention.

        :param dim: Size of each input sample.
        :param num_heads: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttentionRelPosEmb, self).__init__()

        if dim % num_heads != 0:
            raise ValueError('`dim`({}) should be divisible by `num_heads`({})'.format(dim, num_heads))

        self.dim = dim
        self.num_heads = num_heads
        self.bias = bias

        self.linear_q = nn.Linear(dim, dim, bias=bias)
        self.linear_k = nn.Linear(dim, dim, bias=bias)
        self.linear_v = nn.Linear(dim, dim, bias=bias)
        self.linear_o = nn.Linear(dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, rel_pos, mask=None, need_weights=False):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)

        rel_pos = rel_pos.repeat(self.num_heads, 1, 1)
        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)

        attn_weights = self.compute_attention_weights(q, k, rel_pos, mask=mask)
        attn_weights = self.dropout(attn_weights)

        y = attn_weights.matmul(v)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)

        if need_weights:
            return y, attn_weights
        else:
            return y

    @staticmethod
    def compute_attention_weights(query, key, rel_pos, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / np.sqrt(dk)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            rel_pos = rel_pos.masked_fill(mask == 0, -1e9)  # Mask the relative positional emb for padded tokens

        # Add the relative positional emb
        scores = scores + rel_pos
        weights = F.softmax(scores, dim=-1)

        return weights

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        return x.reshape(batch_size, seq_len, self.num_heads, sub_dim) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size * self.num_heads, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads

        return x.reshape(batch_size, self.num_heads, seq_len, in_feature) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, bias=True, dropout_rate: float = 0.0):
        """Multi-head attention.

        :param dim: Size of each input sample.
        :param num_heads: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()

        if dim % num_heads != 0:
            raise ValueError('`dim`({}) should be divisible by `num_heads`({})'.format(dim, num_heads))

        self.dim = dim
        self.num_heads = num_heads
        self.bias = bias

        self.linear_q = nn.Linear(dim, dim, bias=bias)
        self.linear_k = nn.Linear(dim, dim, bias=bias)
        self.linear_v = nn.Linear(dim, dim, bias=bias)
        self.linear_o = nn.Linear(dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, mask=None, need_weights=False):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)

        attn_weights = self.compute_attention_weights(q, k, mask)
        attn_weights = self.dropout(attn_weights)

        y = attn_weights.matmul(v)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)

        if need_weights:
            return y, attn_weights
        else:
            return y

    @staticmethod
    def compute_attention_weights(query, key, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / np.sqrt(dk)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)

        return weights

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        return x.reshape(batch_size, seq_len, self.num_heads, sub_dim) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size * self.num_heads, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads

        return x.reshape(batch_size, self.num_heads, seq_len, in_feature) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)
