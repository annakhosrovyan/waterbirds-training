from math import sin, cos, sqrt, log
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_seq_len, dropout):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)

        positional_encoding = torch.zeros(max_seq_len, self.embed_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * -(log(10000.0) / embed_dim)
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.pe = positional_encoding.to("cuda")

    def pe_sin(self, position, i):
        return sin(position / (10000 ** (2 * i) / self.embed_dim))

    def pe_cos(self, position, i):
        return cos(position / (10000 ** (2 * i) / self.embed_dim))

    def forward(self, x):
        x = x.to("cuda") + self.pe[: x.size(0), : x.size(1)].requires_grad_(False)
        return self.dropout(x)
