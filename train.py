"""
Author: Lynn Ye
Created on: 2025/4/13
Brief: 
"""
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from src.data.create_dataset import load_dataset
from src.model.bert_mixed import LitMixBertMLM
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model.loss import GradientsLossWeighting


def get_logger(cfg):
    if cfg.logger == 'wandb':
        try:
            import wandb
            return WandbLogger(
                project=cfg.project_name,
                name=cfg.run_name,
                log_model=cfg.log_model
            )
        except Exception as e:
            print(f"wandb import failed, falling back to CSV logger. Error: {e}")
    print("Using CSV Logger instead of WandB.")
    return CSVLogger("logs", name=cfg.project_name)


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    data_info = torch.load(cfg.data_dir + '/data_info.pt')
    vocab_size = data_info['vocab_size']
    max_seq_len = data_info['seq_len']

    train_data, val_data, test_data = load_dataset(cfg.data_dir)

    train_loader = DataLoader(train_data, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=cfg.training.num_workers_train,
                              persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers_val)
    # test_loader = DataLoader(test_data, batch_size=32, num_workers=2)

    model = LitMixBertMLM(vocab_size=vocab_size, n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
                          n_embed=cfg.model.n_embed, max_seq_len=max_seq_len, dropout=cfg.model.dropout,
                          lr=cfg.training.lr)

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        save_top_k=cfg.checkpoint.save_top_k,
        mode=cfg.checkpoint.mode,
        filename=cfg.checkpoint.filename,
        save_weights_only=cfg.checkpoint.save_weights_only
    )
    loss_weighting = GradientsLossWeighting(
        weights={'note': cfg.loss_weighting.note,
                 'vel': cfg.loss_weighting.vel,
                 'ts': cfg.loss_weighting.ts},
        ema_rate=cfg.loss_weighting.ema_rate
    )
    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        logger=get_logger(cfg),
        callbacks=[checkpoint_callback, loss_weighting]
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.training.ckpt_path
    )


if __name__ == "__main__":
    train()
