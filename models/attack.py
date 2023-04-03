import numpy as np
import torch
import lightning.pytorch as pl
from torch import nn, optim
from torchmetrics import Accuracy
from torch.nn.functional import one_hot

class AttackBaseModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.lr = args.attack_lr
        self.loss = nn.BCELoss()
        self.Accuracy = Accuracy('binary')
        self.save_hyperparameters()

    def forward(self, yhat, y):
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

class AttackMLPModel(AttackBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = nn.Sequential(
                nn.Linear(20, args.attack_hidden_size),
                nn.ReLU(),
                nn.Linear(args.attack_hidden_size, 1),
                nn.Sigmoid()
            )
        
    def forward(self, yhat, y):
        y_ = torch.cat([yhat, one_hot(y, num_classes=10)], dim=1)
        return self.model(y_).squeeze()
    
class AttackRandomModel(AttackBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = lambda yhat, y: torch.rand(y.shape[0], 1).squeeze()

class AttackPredictModel(AttackBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = lambda yhat, y: (yhat.argmax(dim=1) == y).float()