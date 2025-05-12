"""
Author: Lynn Ye
Created on: 2025/5/12
Brief: reference - https://github.com/StepanTita/nano-BERT
"""

import torch
import math

from src.util.definitions import NOTE_MASK, TIMESHIFT_MASK, VELOCITY_MASK, IGNORE_LABEL_INDEX


class BertAttentionHead(torch.nn.Module):
    """
    A single attention head in MultiHeaded Self Attention layer.
    The idea is identical to the original paper ("Attention is all you need"),
    however instead of implementing multiple heads to be evaluated in parallel we matrix multiplication,
    separated in a distinct class for easier and clearer interpretability
    """

    def __init__(self, head_size, dropout=0.1, n_embed=3):
        super().__init__()

        self.query = torch.nn.Linear(in_features=n_embed, out_features=head_size)
        self.key = torch.nn.Linear(in_features=n_embed, out_features=head_size)
        self.values = torch.nn.Linear(in_features=n_embed, out_features=head_size)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        # B, Seq_len, N_embed
        B, seq_len, n_embed = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.values(x)

        if mask.dim() == 2:
            mask = mask.unsqueeze(1)  # (B, 1, seq_len)

        weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.query.out_features)  # (B, Seq_len, Seq_len)
        weights = weights.masked_fill(mask == 0, -1e9)  # mask out not attended tokens

        scores = torch.softmax(weights, dim=-1)
        scores = self.dropout(scores)

        context = scores @ v

        return context


class BertSelfAttention(torch.nn.Module):
    """
    MultiHeaded Self-Attention mechanism as described in "Attention is all you need"
    """

    def __init__(self, n_heads=1, dropout=0.1, n_embed=3):
        super().__init__()

        head_size = n_embed // n_heads
        n_heads = n_heads
        self.heads = torch.nn.ModuleList([BertAttentionHead(head_size, dropout, n_embed) for _ in range(n_heads)])
        self.proj = torch.nn.Linear(head_size * n_heads, n_embed)  # project from multiple heads to the single space
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        context = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        proj = self.proj(context)
        out = self.dropout(proj)

        return out


class FeedForward(torch.nn.Module):
    def __init__(self, dropout=0.1, n_embed=3):
        super().__init__()

        self.ffwd = torch.nn.Sequential(
            torch.nn.Linear(n_embed, 4 * n_embed),
            torch.nn.GELU(),
            torch.nn.Linear(4 * n_embed, n_embed),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.ffwd(x)

        return out

    def get_last_layer_weights(self):
        return self.ffwd[2].weight


class BertLayer(torch.nn.Module):
    """
    Single layer of BERT transformer model
    """

    def __init__(self, n_heads=1, dropout=0.1, n_embed=3):
        super().__init__()

        # unlike in the original paper, today in transformers it is more common to apply layer norm before other layers
        # this idea is borrowed from Andrej Karpathy's series on transformers implementation
        self.layer_norm1 = torch.nn.LayerNorm(n_embed)
        self.self_attention = BertSelfAttention(n_heads, dropout, n_embed)

        self.layer_norm2 = torch.nn.LayerNorm(n_embed)
        self.feed_forward = FeedForward(dropout, n_embed)

    def forward(self, x, mask):
        x = self.layer_norm1(x)
        x = x + self.self_attention(x, mask)

        x = self.layer_norm2(x)
        out = x + self.feed_forward(x)

        return out


class BertEncoder(torch.nn.Module):
    def __init__(self, n_layers=2, n_heads=1, dropout=0.1, n_embed=3):
        super().__init__()

        self.layers = torch.nn.ModuleList([BertLayer(n_heads, dropout, n_embed) for _ in range(n_layers)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return x

    def get_last_layer_weights(self):
        return self.layers[-1].feed_forward.get_last_layer_weights()


class BertPooler(torch.nn.Module):
    def __init__(self, dropout=0.1, n_embed=3):
        super().__init__()

        self.dense = torch.nn.Linear(in_features=n_embed, out_features=n_embed)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        pooled = self.dense(x)
        out = self.activation(pooled)

        return out


def mask_input(input_ids: torch.Tensor, token_types: torch.Tensor, mask_config: dict or None = None,
               mask_token_ids: dict or None = None):
    """
    :param input_ids:
    :param token_types:
    :param mask_config: assign mask probabilities
    :param mask_token_ids:  assign token id for masking
    :return:
    """
    if mask_config is None:
        mask_config = {
            'note': 0.15,
            'velocity': 0.15,
            'timeshift': 0.15
        }
    if mask_token_ids is None:
        mask_token_ids = {
            'note': NOTE_MASK,
            'timeshift': TIMESHIFT_MASK,  # Regressor cannot distinguish a positive mask value and a timeshift value
            'velocity': VELOCITY_MASK,  # Therefore, we use negative values - reserved for masks
        }

    device = input_ids.device
    labels = input_ids.clone()
    masked_input = input_ids.clone()

    # TODO @Bmois you did not add 10% keep 10% random walk....
    for type_id, type_name in zip([0, 1, 2], ['note', 'timeshift', 'velocity']):
        type_mask = (token_types == type_id)
        prob = mask_config.get(type_name, 0.0)
        mask_decision = torch.bernoulli(torch.full_like(input_ids, prob, dtype=torch.float, device=device)).bool()
        mask_mask = type_mask & mask_decision

        masked_input[mask_mask] = mask_token_ids[type_name]
        labels[~mask_mask & type_mask] = IGNORE_LABEL_INDEX  # Only mask label for unmasked tokens of this type

    labels[~((token_types == 0) | (token_types == 1) | (token_types == 2))] = IGNORE_LABEL_INDEX
    return masked_input, labels


def mask_discrete_input(input_ids: torch.Tensor, token_types: torch.Tensor, mask_config: dict or None = None,
                        mask_token_ids: dict or None = None):
    """
    :param input_ids:
    :param token_types:
    :param mask_config: assign mask probabilities
    :param mask_token_ids:  assign token id for masking
    :return:
    """
    if mask_config is None:
        mask_config = {
            'note': 0.15,
            'velocity': 0.15,
            'timeshift': 0.15
        }
    if mask_token_ids is None:
        mask_token_ids = {
            'note': NOTE_MASK,
            'timeshift': -1 * TIMESHIFT_MASK,  # Discrete model can take input of positive values (token map)
            'velocity': -1 * VELOCITY_MASK,
        }

    device = input_ids.device
    labels = input_ids.clone()
    masked_input = input_ids.clone()

    # TODO @Bmois you did not add 10% keep 10% random walk....
    for type_id, type_name in zip([0, 1, 2], ['note', 'timeshift', 'velocity']):
        type_mask = (token_types == type_id)
        prob = mask_config.get(type_name, 0.0)
        mask_decision = torch.bernoulli(torch.full_like(input_ids, prob, dtype=torch.float, device=device)).bool()
        mask_mask = type_mask & mask_decision

        masked_input[mask_mask] = mask_token_ids[type_name]
        labels[~mask_mask & type_mask] = IGNORE_LABEL_INDEX  # Only mask label for unmasked tokens of this type

    labels[~((token_types == 0) | (token_types == 1) | (token_types == 2))] = IGNORE_LABEL_INDEX
    return masked_input, labels
