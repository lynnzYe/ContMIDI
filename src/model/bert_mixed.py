"""
Author: Lynn Ye
Created on: 2025/4/11
Brief: 
"""
import torch
import math
import pytorch_lightning as pl

from src.model.bert import BertEncoder, BertPooler, mask_input
from src.model.embedding import HybridEmbedding
from src.model.loss import LossWeighting, GradientsLossWeighting, NoteCrossEntropy, VelocityLoss, TimeshiftLoss
from src.util.definitions import IGNORE_LABEL_INDEX, NOTE_TYPE, TS_TYPE, VEL_TYPE, NOTE_MASK, TIMESHIFT_MASK, \
    VELOCITY_MASK


class NanoMixBERT(torch.nn.Module):
    """
    NanoBERT with mixed types of token (continuous ones and discrete tokens)
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


class NanoMixBertMLM(torch.nn.Module):
    def __init__(self, vocab_size, n_layers=2, n_heads=1, dropout=0.1, n_embed=3, max_seq_len=16):
        super().__init__()
        self.bert = NanoMixBERT(vocab_size=vocab_size, n_layers=n_layers, n_heads=n_heads, dropout=dropout,
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
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(n_embed, n_embed),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(n_embed, 1),  # output a scalar
            torch.nn.Sigmoid()
        )
        # Timeshift regressor predicts log-transformed time shift values
        self.ts_regressor = torch.nn.Sequential(
            torch.nn.Linear(n_embed, n_embed),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(n_embed, n_embed),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
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


def train_step(model, input_ids, attention_mask, token_types, note_loss, velocity_loss, timeshift_loss, loss_weighting):
    masked_input, labels = mask_input(input_ids, token_types)
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


class LitMixBertMLM(pl.LightningModule):
    def __init__(self, vocab_size, n_layers=2, n_heads=1, dropout=0.1, n_embed=3, max_seq_len=16, lr=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = NanoMixBertMLM(vocab_size, n_layers, n_heads, dropout, n_embed, max_seq_len)
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
        input_ids, attention_mask, token_types = batch
        loss_dict, total_loss = train_step(self.model, input_ids, attention_mask, token_types, self.note_loss,
                                           self.velocity_loss, self.timeshift_loss, self.loss_weighting)
        self.log_dict({f"loss/{k}/train": v for k, v in loss_dict.items()}, sync_dist=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_types = batch
        masked_input, labels = mask_input(input_ids, token_types)
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
    model = NanoMixBertMLM(vocab_size=vocab_size, max_seq_len=max_seq_len)

    # Generate random input data
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, max_seq_len))
    token_types = torch.randint(0, 3, [batch_size, max_seq_len])
    attention_mask = torch.ones_like(input_ids)

    note_loss_fn = NoteCrossEntropy()
    velocity_loss_fn = VelocityLoss()
    timeshift_loss_fn = TimeshiftLoss()
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        loss_dict, total_loss = train_step(model, input_ids, attention_mask, token_types, note_loss_fn,
                                           velocity_loss_fn, timeshift_loss_fn,
                                           LossWeighting(weights={'note': 1.0, 'vel': 1.0, 'ts': 1.0}))

    # Check output
    print(f"Loss: {total_loss}")


if __name__ == "__main__":
    main()
