"""
Author: Lynn Ye
Created on: 2025/4/6
Brief: embed both discrete and continuous tokens
"""
import torch
import torch.nn as nn

from src.util.definitions import TS_TYPE, VEL_TYPE, NOTE_TYPE, TIMESHIFT_MASK, VELOCITY_MASK


def timeshift_fe(t):
    """
    Feature extraction for timeshift value
    :param t:
    :return:
    """
    t = torch.clamp(t, min=1e-3)
    return torch.stack([t, torch.log(t), t ** 2, t ** 0.5], dim=-1)


def velocity_fe(v):
    """
    Feature extraction for velocity value
    :param v:
    :return:
    """
    # Normalize to avoid log(0)
    v = torch.clamp(v, min=1e-3)
    return torch.stack([v, torch.log(v), v ** 2, v ** 0.5], dim=-1)


class HybridEmbedding(nn.Module):
    def __init__(self, discrete_vocab_size, embed_dim, max_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(discrete_vocab_size, embed_dim)
        self.timeshift_embedding = nn.Sequential(
            nn.Linear(4, embed_dim),  # four extracted features
            nn.ReLU()
        )
        self.velocity_embedding = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.ReLU())
        self.position_embedding = torch.nn.Embedding(max_len, embed_dim)

        self.discrete_layer_norm = nn.LayerNorm(embed_dim)
        self.timeshift_layer_norm = nn.LayerNorm(embed_dim)
        self.velocity_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor, token_types: torch.Tensor):
        """
        token_types: tensor of shape (batch, seq_len), values are 'discrete' or 'continuous'
        """
        B, T = input_ids.shape
        D = self.token_embedding.embedding_dim
        device = input_ids.device

        def embed_cont_tokens(masked_vals: torch.Tensor, feat_extr_func, embed_func, layer_norm, mask_id):
            N = masked_vals.size(0)
            real_input_pos = masked_vals >= 0
            cont_embedding = torch.zeros(N, D, dtype=torch.float,
                                         device=device)  # placeholder
            if real_input_pos.any():
                real_vals = masked_vals[real_input_pos].float()
                real_feats = feat_extr_func(real_vals)  # → (N_real, feat_dim)
                real_emb = embed_func(real_feats)
                real_emb = layer_norm(real_emb)
                cont_embedding[real_input_pos] = real_emb
            if (~real_input_pos).any():
                mask_pos = (~real_input_pos).nonzero(as_tuple=False).squeeze(-1)
                # Look up mask vector
                mask_emb = self.token_embedding(torch.full(
                    (mask_pos.size(0),), mask_id, dtype=torch.long, device=device
                ))
                cont_embedding[mask_pos] = mask_emb
            return cont_embedding

        embeddings = torch.zeros(B, T, self.token_embedding.embedding_dim, device=device)
        # --- Discrete tokens ---
        discrete_mask = (token_types == NOTE_TYPE)  # (B, T)
        if discrete_mask.any():
            # Only select the discrete token IDs
            discrete_ids = input_ids[discrete_mask]
            discrete_emb = self.token_embedding(discrete_ids)
            discrete_emb = self.discrete_layer_norm(discrete_emb)
            embeddings[discrete_mask] = discrete_emb

        # --- Continuous values ---
        timeshift_mask = (token_types == TS_TYPE)
        if timeshift_mask.any():
            embeddings[timeshift_mask] = embed_cont_tokens(masked_vals=input_ids[timeshift_mask],
                                                           feat_extr_func=timeshift_fe,
                                                           embed_func=self.timeshift_embedding,
                                                           layer_norm=self.timeshift_layer_norm,
                                                           mask_id=-TIMESHIFT_MASK)

        velocity_mask = (token_types == VEL_TYPE)
        if velocity_mask.any():
            embeddings[velocity_mask] = embed_cont_tokens(masked_vals=input_ids[velocity_mask],
                                                          feat_extr_func=velocity_fe,
                                                          embed_func=self.velocity_embedding,
                                                          layer_norm=self.velocity_layer_norm,
                                                          mask_id=-VELOCITY_MASK
                                                          )

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

    emb = HybridEmbedding(vocab_size, 16, seq_len)
    emb.forward(token_ids, types)
    pass
