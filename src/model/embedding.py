"""
Author: Lynn Ye
Created on: 2025/4/6
Brief: embed both discrete and continuous tokens
"""
import torch
import torch.nn as nn


class HybridEmbedding(nn.Module):
    def __init__(self, discrete_vocab_size, embed_dim, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(discrete_vocab_size, embed_dim)
        self.continuous_projection = nn.Linear(1, embed_dim)  # Place holder TODO @Bmois
        self.position_embedding = torch.nn.Embedding(max_len, embed_dim)

    def forward(self, input_ids, token_types):
        """
        token_types: tensor of shape (batch, seq_len), values are 'discrete' or 'continuous'
        """
        B, T = input_ids.shape
        device = input_ids.device

        embeddings = torch.zeros(B, T, self.token_embedding.embedding_dim, device=device)
        # --- Discrete tokens ---
        discrete_mask = (token_types == 0)  # (B, T)
        if discrete_mask.any():
            discrete_emb = self.token_embedding(input_ids)  # (B, T, D)
            embeddings[discrete_mask] = discrete_emb[discrete_mask]

        # --- Continuous values ---
        continuous_mask = (token_types == 1)
        if continuous_mask.any():
            cont_vals = input_ids.float().unsqueeze(-1)  # shape: (B, T, 1)
            continuous_emb = self.continuous_projection(cont_vals)
            embeddings[continuous_mask] = continuous_emb[continuous_mask]

        # --- Add position embedding ---
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        embeddings += self.position_embedding(pos_ids)

        return embeddings


if __name__ == '__main__':
    batch_size = 10
    seq_len = 128
    vocab_size = 356

    token_ids = torch.randint(0, 356, [batch_size, seq_len])
    types = torch.randint(0, 2, [batch_size, seq_len])

    emb = HybridEmbedding(vocab_size, 4, seq_len)
    emb.forward(token_ids, types)
    pass
