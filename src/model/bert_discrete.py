"""
Author: Lynn Ye
Created on: 2025/4/11
Brief: 
"""
import torch
import math
import pytorch_lightning as pl

from src.model.bert import BertEncoder, BertPooler, mask_discrete_input
from src.model.embedding import HybridEmbedding, DiscreteEmbedding
from src.model.loss import LossWeighting, GradientsLossWeighting, NoteCrossEntropy, VelocityLoss, TimeshiftLoss, \
    TokenCrossEntropy
from src.util.definitions import IGNORE_LABEL_INDEX, NOTE_TYPE, TS_TYPE, VEL_TYPE, NOTE_MASK, TIMESHIFT_MASK, \
    VELOCITY_MASK


class NanoBERT(torch.nn.Module):
    """
    NanoBERT is a almost an exact copy of a transformer decoder part described in the paper "Attention is all you need"
    This is a base model that can be used for various purposes such as Masked Language Modelling, Classification,
    Or any other kind of NLP tasks.
    This implementation does not cover the Seq2Seq problem, but can be easily extended to that.
    """

    def __init__(self, vocab_size, n_layers=2, n_heads=1, dropout=0.1, n_embed=3, max_seq_len=16):
        super().__init__()

        self.embedding = DiscreteEmbedding(discrete_vocab_size=vocab_size, embed_dim=n_embed, max_len=max_seq_len)

        self.encoder = BertEncoder(n_layers, n_heads, dropout, n_embed)

        self.pooler = BertPooler(dropout, n_embed)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        emb_output = self.embedding(input_ids)
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

    def forward(self, input_ids: torch.Tensor, token_types: torch.Tensor, attention_mask: torch.Tensor):
        hidden = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_types)
        cls_logits = self.classifier(hidden)
        return {
            'note': cls_logits,
            'timeshift': cls_logits,
            'velocity': cls_logits
        }


def train_step(model, input_ids, attention_mask, token_types, note_loss, velocity_loss, timeshift_loss, loss_weighting):
    masked_input, labels = mask_discrete_input(input_ids, token_types)
    output_dict = model(masked_input, token_types, attention_mask)
    cls_logits = output_dict['note']
    vel_preds = output_dict['velocity']
    ts_preds = output_dict['timeshift']

    note_loss = note_loss(cls_logits, token_types, labels)
    velocity_loss = velocity_loss(vel_preds, token_types, labels)
    timeshift_loss = timeshift_loss(ts_preds, token_types, labels)
    combined_loss = loss_weighting.combine_losses(note=note_loss, vel=velocity_loss, ts=timeshift_loss)
    loss_dict = dict(note=note_loss,
                     vel=velocity_loss,
                     ts=timeshift_loss,
                     loss=combined_loss)
    return loss_dict, combined_loss


class LitBertMLM(pl.LightningModule):
    def __init__(self, vocab_size, n_layers=2, n_heads=1, dropout=0.1, n_embed=3, max_seq_len=16, lr=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = NanoBertMLM(vocab_size, n_layers, n_heads, dropout, n_embed, max_seq_len)
        self.lr = lr
        self.loss_weighting = None
        self.note_loss = NoteCrossEntropy()
        self.timeshift_loss = TokenCrossEntropy(types_considered=[TS_TYPE])
        self.velocity_loss = TokenCrossEntropy(types_considered=[VEL_TYPE])
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
        input_ids, attention_mask, token_types = batch
        loss_dict, total_loss = train_step(self.model, input_ids, attention_mask, token_types, self.note_loss,
                                           self.velocity_loss, self.timeshift_loss, self.loss_weighting)
        self.log_dict({f"loss/{k}/train": v for k, v in loss_dict.items()}, sync_dist=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_types = batch
        masked_input, labels = mask_discrete_input(input_ids, token_types)
        output_dict = self(masked_input, token_types, attention_mask)
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
        self.log("val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        ts_params = list(self.model.ts_regressor.parameters())
        ts_param_ids = set(id(p) for p in ts_params)
        base_params = [p for p in self.parameters() if id(p) not in ts_param_ids]

        return torch.optim.AdamW([{'params': base_params, 'lr': self.lr},
                                  {'params': ts_params, 'lr': self.lr * 5}])


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

    note_loss_fn = NoteCrossEntropy()
    timeshift_loss_fn = TokenCrossEntropy(types_considered=[TS_TYPE])
    velocity_loss_fn = TokenCrossEntropy(types_considered=[VEL_TYPE])
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        loss_dict, total_loss = train_step(model, input_ids, attention_mask, token_types, note_loss_fn,
                                           velocity_loss_fn, timeshift_loss_fn,
                                           LossWeighting(weights={'note': 1.0, 'vel': 1.0, 'ts': 1.0}))

    # Check output
    print(f"Loss dict: {loss_dict}")
    print(f"Loss: {total_loss}")


if __name__ == "__main__":
    main()
