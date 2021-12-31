import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from reftransformer.models.backbone.attention import MultiHeadAttention


class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, bias=True):
        super().__init__()

        self.embed_size = embed_size
        self.heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (self.head_dim * self.heads == embed_size), "Embed size needs to be multiple of num heads"

        self.values = nn.Linear(self.head_dim, self.heads, bias=bias)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=bias)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=bias)

        self.fc_out = nn.Linear(self.heads * self.head_dim, self.heads * self.head_dim)

    def forward(self, values: torch.Tensor, keys: torch.Tensor, queries: torch.Tensor, mask: torch.Tensor = None):
        batch_size = keys.size(0)

        values_len, keys_len, queries_len = values.size(1), keys.size(1), queries.size(1)

        # Split the values, keys, and queries into self.heads pieces
        values = values.reshape(batch_size, values_len, self.heads, self.head_dim)
        keys = keys.reshape(batch_size, keys_len, self.heads, self.head_dim)
        queries = queries.reshape(batch_size, queries_len, self.heads, self.head_dim)

        energy = torch.einsum("bqhd,bkhd->bhqk", [queries, keys])

        # Mask the energy matrix before taking the softmax
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(energy / (self.embed_size ** 0.5), dim=3)

        out = torch.einsum("bhql,blhd->bqhd", [attention, values]).reshape(batch_size,
                                                                           queries_len,
                                                                           self.heads * self.head_dim)
        out = self.fc_out(out)

        return out


