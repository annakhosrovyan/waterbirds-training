from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim):
        """
        Multi-Head Attention class
        :param embed_dim: the embedding dimension
        :param heads: the number of heads, default equals 8
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim  

        # query, value, key:
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)  # the Query metrix
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)  # the Value metrix
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)  # the Key metrix

        # fully connected layer:
        self.fc_out = nn.Linear(embed_dim, embed_dim)
                

    def forward(self, key, query, value, mask=None):
        batch_size = key.size(0)
        k_len, q_len, v_len = key.size(1), query.size(1), value.size(1)

        key = key.reshape(batch_size, k_len)
        query = key.reshape(batch_size, q_len)
        value = key.reshape(batch_size, v_len)

        key = self.key(key)
        query = self.key(query)
        value = self.key(value)

        ############### query x key ###############
        product = query @ key.t()
        product = product / sqrt(self.embed_dim)
        scores = F.softmax(product, dim=-1)

        ############### scores x value ###############
        output = scores @ value
        output = self.fc_out(output)  

        return output
