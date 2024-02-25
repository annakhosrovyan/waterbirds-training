import torch.nn as nn
import torch.nn.functional as F
from src.model.transformer_architecture.utils import replicate

from src.model.transformer_architecture.attention import MultiHeadAttention
from src.model.transformer_architecture.embed import PositionalEncoding
from src.model.transformer_architecture.encoder import TransformerBlock


class DecoderBlock(nn.Module):

    def __init__(self,
                 embed_dim,
                 expansion_factor,
                 dropout
                 ):

        super(DecoderBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.transformerBlock = TransformerBlock(embed_dim, expansion_factor, dropout)

    def forward(self, key, query, x, mask):
        decoder_attention = self.attention(x, x, x, mask)
        value = self.dropout(self.norm(decoder_attention + x))
        decoder_attention_output = self.transformerBlock(key, query, value)

        return decoder_attention_output


class Decoder(nn.Module):

    def __init__(self,
                 seq_len,
                 embed_dim,
                 num_blocks,  
                 expansion_factor,
                 dropout
                 ):

        super(Decoder, self).__init__()

        self.positional_encoder = PositionalEncoding(embed_dim, seq_len, dropout)
        self.blocks = replicate(DecoderBlock(embed_dim, expansion_factor, dropout), num_blocks)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, mask):
        x = self.dropout(self.positional_encoder(x)) 
        for block in self.blocks:
            x = block(encoder_output, x, encoder_output, mask)

        return x
