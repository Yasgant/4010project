import numpy as np
import torch
import lightning.pytorch as pl
from torch import nn, optim
from torchmetrics import Accuracy
from torch.nn.functional import one_hot

class AttackModel(pl.LightningModule):
    def __init__(self, type='mlp', lr=1e-3):
        super().__init__()
        self.lr = lr
        self.type = type
        if type == 'random':
            self.model = lambda yhat, y: torch.rand(y.shape[0], 1).squeeze()
        elif type == 'predict':
            self.model = lambda yhat, y: torch.tensor(yhat.argmax(dim=1) == y, dtype=torch.float)
        elif type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(20, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        self.loss = nn.BCELoss()
        self.Accuracy = Accuracy('binary')

    def forward(self, yhat, y):
        if self.type == 'mlp':
            y_ = torch.cat([yhat, one_hot(y, num_classes=10)], dim=1)
            return self.model(y_).squeeze()
        else:
            return self.model(yhat, y)
    
    def training_step(self, batch, batch_idx):
        (yhat, y), is_m = batch
        is_hat = self(yhat, y)
        loss = self.loss(is_hat, is_m)
        metrics = {'train_loss': loss, 'train_acc': self.Accuracy(is_hat, is_m)}
        self.log_dict(metrics)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (yhat, y), is_m = batch
        is_hat = self(yhat, y)
        loss = self.loss(is_hat, is_m)
        metrics = {'val_loss': loss, 'val_acc': self.Accuracy(is_hat, is_m)}
        self.log_dict(metrics)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        return [optimizer], [scheduler]