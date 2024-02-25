import torch.nn as nn

from src.model.transformer_architecture.utils import replicate
from src.model.transformer_architecture.attention import MultiHeadAttention
from src.model.transformer_architecture.embed import PositionalEncoding


class TransformerBlock(nn.Module):

    def __init__(self,
                 embed_dim,
                 expansion_factor,
                 dropout
                 ):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim)  
        self.norm = nn.LayerNorm(embed_dim)  

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim * expansion_factor, embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, key, query, value, mask=None):
        attention_out = self.attention(key, query, value, mask)
        attention_out = attention_out + value
        attention_norm = self.dropout(self.norm(attention_out))
        fc_out = self.feed_forward(attention_norm)
        fc_out = fc_out + attention_norm 
        fc_norm = self.dropout(self.norm(fc_out))
        return fc_norm


class Encoder(nn.Module):

    def __init__(self,
                 seq_len,
                 embed_dim,
                 num_blocks,
                 expansion_factor,
                 dropout
                 ):

        super(Encoder, self).__init__()

        self.positional_encoder = PositionalEncoding(embed_dim, seq_len, dropout)

        self.blocks = replicate(TransformerBlock(embed_dim, expansion_factor, dropout), num_blocks)

    def forward(self, x):
        x = self.positional_encoder(x) 
        for block in self.blocks:
            x = block(x, x, x)

        return x
