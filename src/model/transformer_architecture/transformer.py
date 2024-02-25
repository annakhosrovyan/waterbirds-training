import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.transformer_architecture.encoder import Encoder
from src.model.transformer_architecture.decoder import Decoder


class Transformer(nn.Module):

    def __init__(self,
                 embed_dim,
                 prototype_dim,
                 seq_len,
                 num_blocks,
                 expansion_factor,
                 dropout):
        super(Transformer, self).__init__()

        self.encoder = Encoder(seq_len=seq_len,
                               embed_dim=embed_dim,
                               num_blocks=num_blocks,
                               expansion_factor=expansion_factor,
                               dropout=dropout)

        self.decoder = Decoder(seq_len=seq_len,
                               embed_dim=embed_dim,
                               num_blocks=num_blocks,
                               expansion_factor=expansion_factor,
                               dropout=dropout)

        self.fc_out = nn.Linear(embed_dim, prototype_dim)


    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask

    def forward(self, source, target):
        trg_mask = self.make_trg_mask(target)
        enc_out = self.encoder(source)
        outputs = self.decoder(target, enc_out, trg_mask)
        output = F.softmax(self.fc_out(outputs), dim=-1)

        return output