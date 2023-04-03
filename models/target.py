import numpy as np
import torch
import lightning.pytorch as pl
from torch import nn, optim
from torchmetrics import Accuracy

class TargetBaseModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.lr = args.model_lr
        self.loss = nn.CrossEntropyLoss()
        self.Accuracy = Accuracy('multiclass', num_classes=10)
        self.softmax = nn.Softmax(dim=1)
        self.train_acc, self.val_acc = None, None
        self.save_hyperparameters()
    
    def forward(self, x):
        return self.model(x)
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        metrics = {'train_loss': loss, 'train_acc': self.Accuracy(y_hat, y)}
        self.log_dict(metrics)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        metrics = {'val_loss': loss, 'val_acc': self.Accuracy(y_hat, y)}
        self.log_dict(metrics)
        return loss
    
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        return [optimizer], [scheduler]
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        y_hat = self.softmax(y_hat)
        return y_hat, y
    
class TargetMLPModel(TargetBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

class TargetCNNModel(TargetBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*4*4, 10)
        )