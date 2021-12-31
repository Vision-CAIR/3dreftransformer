"""
Visual Transformer to model the interaction between the pairs of (language, object features)
"""
import math

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel
# from transformers.modeling_bert import BertEmbeddings, BertEncoder

from reftransformer.models.backbone.attention import MultiHeadAttention
from reftransformer.models.backbone.memory_meshed_transformer.containers import Module


class WordPositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(WordPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

