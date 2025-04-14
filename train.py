"""
Author: Lynn Ye
Created on: 2025/4/13
Brief: 
"""
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from src.data.create_dataset import load_dataset
from src.model.bert import LitBertMLM
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# --- Init WandB logger ---
wandb_logger = WandbLogger(
    project="nano-bert-mlm",  # give your project a name
    name="v0.1",  # optional run name
    log_model="all"  # optionally log model checkpoints as artifacts
)

# --- Model Checkpointing ---
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=1,
    mode="min",
    filename="contmidi-bert-{epoch:02d}-{val_loss:.2f}",
    save_weights_only=True  # or False if you want full model + optimizer
)


def train(data_dir):
    data_info = torch.load(data_dir + '/data_info.pt')
    vocab_size = data_info['vocab_size']
    max_seq_len = data_info['seq_len']
    config_name = data_info['tokenizer_type']

    train_data, val_data, test_data = load_dataset(data_dir)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    model = LitBertMLM(vocab_size=vocab_size, n_embed=128, max_seq_len=128)
    trainer = Trainer(max_epochs=30, accelerator='mps',
                      logger=wandb_logger,
                      callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    data_dir = ''
    train(data_dir)
