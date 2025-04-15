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
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import importlib.util


def get_logger():
    # if importlib.util.find_spec("wandb") is not None:
    #     try:
    #         import wandb
    #         return WandbLogger(
    #             project="mix-token-bert",
    #             name="v0.1",
    #             log_model="all"
    #         )
    #     except Exception as e:
    #         print(f"wandb import failed, falling back to CSV logger. Error: {e}")
    # print("Using CSV Logger instead of WandB.")
    return CSVLogger("logs", name="nano-bert-mlm")


# --- Model Checkpointing ---
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=1,
    mode="min",
    filename="mix-token-bert-{epoch:02d}-{val_loss:.2f}",
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
                      logger=get_logger(),
                      callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    data_dir = '/Users/kurono/Documents/github/ContinuousMIDI/tmp/maestro_data'
    train(data_dir)
