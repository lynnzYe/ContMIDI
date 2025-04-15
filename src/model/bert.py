"""
Author: Lynn Ye
Created on: 2025/4/11
Brief: 
"""
import torch
import torch.functional as F
import math
import pytorch_lightning as pl

from src.model.embedding import HybridEmbedding
from src.model.loss import LossWeighting, GradientsLossWeighting, NoteCrossEntropy, VelocityLoss, TimeshiftLoss
from src.util.definitions import IGNORE_LABEL_INDEX, NOTE_TYPE, TS_TYPE, VEL_TYPE, NOTE_MASK, TIMESHIFT_MASK, \
    VELOCITY_MASK


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


class NanoBERT(torch.nn.Module):
    """
    NanoBERT is a almost an exact copy of a transformer decoder part described in the paper "Attention is all you need"
    This is a base model that can be used for various purposes such as Masked Language Modelling, Classification,
    Or any other kind of NLP tasks.
    This implementation does not cover the Seq2Seq problem, but can be easily extended to that.
    """

    def __init__(self, vocab_size, n_layers=2, n_heads=1, dropout=0.1, n_embed=3, max_seq_len=16):
        super().__init__()

        self.embedding = HybridEmbedding(discrete_vocab_size=vocab_size, embed_dim=n_embed, max_len=max_seq_len)

        self.encoder = BertEncoder(n_layers, n_heads, dropout, n_embed)

        self.pooler = BertPooler(dropout, n_embed)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        emb_output = self.embedding(input_ids, token_type_ids)
        encoded = self.encoder(emb_output, attention_mask)
        pooled = self.pooler(encoded)
        return pooled


class NanoBertMLM(torch.nn.Module):
    def __init__(self, vocab_size, n_layers=2, n_heads=1, dropout=0.1, n_embed=3, max_seq_len=16):
        super().__init__()
        self.bert = NanoBERT(vocab_size=vocab_size, n_layers=n_layers, n_heads=n_heads, dropout=dropout,
                             n_embed=n_embed, max_seq_len=max_seq_len)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_embed, out_features=n_embed),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_features=n_embed, out_features=vocab_size)
        )
        # Predicts value [0, 1]
        self.vel_regressor = torch.nn.Sequential(
            torch.nn.Linear(n_embed, n_embed),
            torch.nn.ReLU(),
            torch.nn.Linear(n_embed, 1),  # output a scalar
            torch.nn.Sigmoid()
        )
        # Timeshift regressor predicts log-transformed time shift values
        self.ts_regressor = torch.nn.Sequential(
            torch.nn.Linear(n_embed, n_embed),
            torch.nn.ReLU(),
            torch.nn.Linear(n_embed, 1),  # output a scalar
        )

    def forward(self, input_ids: torch.Tensor, token_types: torch.Tensor, attention_mask: torch.Tensor):
        hidden = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_types)
        cls_logits = self.classifier(hidden)
        ts_reg_preds = self.ts_regressor(hidden).squeeze(-1)  # learn to output log-transformed timeshift values
        vel_reg_preds = self.vel_regressor(hidden).squeeze(-1)  # output [0, 1]
        return {
            'note': cls_logits,
            'timeshift': ts_reg_preds,
            'velocity': vel_reg_preds
        }


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
            'timeshift': TIMESHIFT_MASK,
            'velocity': VELOCITY_MASK,
        }

    device = input_ids.device
    labels = input_ids.clone()
    masked_input = input_ids.clone()

    for type_id, type_name in zip([0, 1, 2], ['note', 'timeshift', 'velocity']):
        type_mask = (token_types == type_id)
        prob = mask_config.get(type_name, 0.0)
        mask_decision = torch.bernoulli(torch.full_like(input_ids, prob, dtype=torch.float, device=device)).bool()
        mask_mask = type_mask & mask_decision

        masked_input[mask_mask] = mask_token_ids[type_name]
        labels[~mask_mask & type_mask] = IGNORE_LABEL_INDEX  # Only mask label for unmasked tokens of this type

    labels[~((token_types == 0) | (token_types == 1) | (token_types == 2))] = IGNORE_LABEL_INDEX
    return masked_input, labels


class LitBertMLM(pl.LightningModule):
    def __init__(self, vocab_size, n_layers=2, n_heads=1, dropout=0.1, n_embed=3, max_seq_len=16, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = NanoBertMLM(vocab_size, n_layers, n_heads, dropout, n_embed, max_seq_len)
        self.lr = lr
        self.loss_weighting = None
        self.note_loss = NoteCrossEntropy()
        self.velocity_loss = VelocityLoss()
        self.timeshift_loss = TimeshiftLoss()
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def on_fit_start(self) -> None:
        for callback in self.trainer.callbacks:
            if isinstance(callback, LossWeighting):
                self.loss_weighting = callback
                break
        if self.loss_weighting is None:
            self.loss_weighting = LossWeighting(weights={'note': 1.0, 'vel': 1.0, 'ts': 1.0})
        self.loss_weighting.last_layer = self.model.bert.encoder.get_last_layer_weights()

    def forward(self, input_ids, token_types=None, attention_mask=None):
        return self.model(input_ids=input_ids, token_types=token_types, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        # TODO @Bmois write masking mechanism
        input_ids, attention_mask, token_types = batch
        labels = input_ids.clone()
        output_dict = self(input_ids, token_types, attention_mask)
        cls_logits = output_dict['note']
        vel_preds = output_dict['velocity']
        ts_preds = output_dict['timeshift']

        note_loss = self.note_loss(cls_logits, token_types, labels)
        velocity_loss = self.velocity_loss(vel_preds, token_types, labels)
        timeshift_loss = self.timeshift_loss(ts_preds, token_types, labels)
        total_loss = self.loss_weighting.combine_losses(note=note_loss, vel=velocity_loss, ts=timeshift_loss)
        loss_dict = dict(note=note_loss,
                         vel=velocity_loss,
                         ts=timeshift_loss,
                         loss=total_loss)

        self.log_dict({f"loss/{k}/train": v for k, v in loss_dict.items()}, sync_dist=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_types = batch
        masked_input, labels = mask_input(input_ids, token_types)
        output_dict = self(input_ids, token_types, attention_mask)
        cls_logits = output_dict['note']
        vel_preds = output_dict['velocity']
        ts_preds = output_dict['timeshift']

        note_loss = self.note_loss(cls_logits, token_types, labels)
        velocity_loss = self.velocity_loss(vel_preds, token_types, labels)
        timeshift_loss = self.timeshift_loss(ts_preds, token_types, labels)
        total_loss = self.loss_weighting.combine_losses(note=note_loss, vel=velocity_loss, ts=timeshift_loss)
        loss_dict = dict(note=note_loss,
                         vel=velocity_loss,
                         ts=timeshift_loss,
                         loss=total_loss)

        self.log_dict({f"loss/{k}/train": v for k, v in loss_dict.items()}, sync_dist=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def main():
    # Test configuration
    vocab_size = 356
    max_seq_len = 128
    batch_size = 8

    # Initialize model
    model = NanoBertMLM(vocab_size=vocab_size, max_seq_len=max_seq_len)

    # Generate random input data
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, max_seq_len))
    token_types = torch.randint(0, 3, [batch_size, max_seq_len])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(low=0, high=vocab_size, size=(batch_size, max_seq_len))

    # Set some labels to ignore index to simulate masked labels
    labels[torch.rand_like(labels, dtype=torch.float) < 0.2] = IGNORE_LABEL_INDEX
    # TODO @Bmois: assign special mask for different token type: noteon-off, velocity, timeshift

    # Perform a forward pass
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        logits, vel_preds, ts_preds = model(input_ids, token_type_ids=token_types, attention_mask=attention_mask)

    # Compute loss
    note_loss_fn = NoteCrossEntropy()
    velocity_loss_fn = VelocityLoss()
    timeshift_loss_fn = TimeshiftLoss()

    note_loss = note_loss_fn(logits, token_types, labels)
    timeshift_loss = timeshift_loss_fn(ts_preds, token_types, labels)
    velocity_loss = velocity_loss_fn(vel_preds, token_types, labels)
    loss_weighting = LossWeighting(weights={'note': 1.0, 'vel': 1.0, 'ts': 1.0})
    loss = loss_weighting.combine_losses(note=note_loss, vel=velocity_loss, ts=timeshift_loss)
    # Check output
    print(f"Loss: {loss.item() if loss is not None else 'N/A'}")

    assert logits.shape == (batch_size, max_seq_len, vocab_size), "Incorrect logits shape"
    assert vel_preds.shape == (batch_size, max_seq_len), "Incorrect preds shape"

    # Assertions to verify correct output
    if loss is not None:
        assert isinstance(loss.item(), float), "Loss is not a float value"


if __name__ == "__main__":
    main()
