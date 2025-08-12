import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalAttention(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads

        self.head_dim = self.embedding_dim // self.num_heads

        # query, key, and value projections
        # why do we need 3 different projections?
        self.K = nn.Linear(self.embedding_dim, self.embedding_dim, bias = config.qkv_bias)
        self.Q = nn.Linear(self.embedding_dim, self.embedding_dim, bias = config.qkv_bias)
        self.V = nn.Linear(self.embedding_dim, self.embedding_dim, bias = config.qkv_bias)

        self.dropout = nn.Dropout(config.dropout)
        self.project = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, x):

        batch_size, num_tokens = x.shape[0], x.shape[1]

        k = self.K(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.Q(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.V(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product
        product = torch.matmul(k, q.transpose(2, 3))
        product /= self.head_dim ** 0.5

        # mask for autoregressive
        mask = torch.triu(torch.ones_like(product), diagonal=1)
        product = torch.masked_fill(product, mask.bool(), -torch.inf)

        attention = F.softmax(product, dim = -1)
        attention = self.dropout(attention)

        embeddings = torch.matmul(attention, v)
        embeddings = embeddings.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.embedding_dim)

        # finally project it across the linear layer
        embeddings = self.project(embeddings)
        return embeddings