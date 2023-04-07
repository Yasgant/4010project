import numpy as np
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from torchmetrics import Accuracy
import wandb
from lightning.pytorch.loggers import WandbLogger
import time


def train(model, name, dataset, valid_dataset=None, batch_size=64, epochs=100, args=None):
    if args is not None:
        batch_size = args.batch_size
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    else:
        valid_loader = None
    logger = WandbLogger(project='4010project', name = name+time.strftime("-%m%d-%H%M"))
    wandb.config.update(args)
    trainer = pl.Trainer(max_epochs=epochs, logger=logger, default_root_dir=f'data/{name}')
    trainer.fit(model, train_loader, valid_loader)
    trainer.save_checkpoint(f'data/{name}.ckpt')
    wandb.finish()
    return model
