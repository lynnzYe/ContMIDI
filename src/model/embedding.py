"""
Author: Lynn Ye
Created on: 2025/4/6
Brief: embed both discrete and continuous tokens
"""
import torch
import torch.nn as nn


class HybridEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, continuous_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.continuous_projection = nn.Linear(continuous_dim, embed_dim)

    def forward(self, discrete_tokens, continuous_features, token_types):
        """
        token_types: tensor of shape (batch, seq_len), values are 'discrete' or 'continuous'
        """
        B, T = token_types.shape
        out = torch.zeros(B, T, self.token_embedding.embedding_dim, device=discrete_tokens.device)

        discrete_mask = token_types == 'discrete'
        continuous_mask = token_types == 'continuous'

        if discrete_mask.any():
            out[discrete_mask] = self.token_embedding(discrete_tokens[discrete_mask])
        if continuous_mask.any():
            out[continuous_mask] = self.continuous_projection(continuous_features[continuous_mask])

        return out
