"""
Author: Lynn Ye
Created on: 2025/4/6
Brief: embed both discrete and continuous tokens
"""
import torch
import torch.nn as nn


class HybridEmbedding(nn.Module):
    def __init__(self, discrete_vocab_size, continuous_dim, embed_dim, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(discrete_vocab_size, embed_dim)
        self.continuous_projection = nn.Linear(continuous_dim, embed_dim)
        self.position_embedding = torch.nn.Embedding(max_len, embed_dim)

    def forward(self, discrete_tokens, continuous_features, token_types):
        """
        token_types: tensor of shape (batch, seq_len), values are 'discrete' or 'continuous'
        """
        B, T = token_types.shape
        out = torch.zeros(B, T, self.token_embedding.embedding_dim, device=discrete_tokens.device)

        out[token_types == 1] = self.token_embedding(discrete_tokens[token_types == 1])
        out[token_types == 0] = self.continuous_projection(continuous_features[token_types == 0])

        return out


if __name__ == '__main__':
    pass
