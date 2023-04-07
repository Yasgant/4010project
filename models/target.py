import numpy as np
import torch
import lightning.pytorch as pl
from torch import nn, optim
from torchmetrics import Accuracy
from .resnet import resnet18
from .effnet import Effnetv2_s
from .alexnet import alexnet
from opacus import PrivacyEngine

def mixup_data(x, y, alpha=1.0):
    '''From https://github.com/facebookresearch/mixup-cifar10
       Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class TargetBaseModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.lr = args.model_lr
        self.weight_decay = args.model_weight_decay
        self.topk = args.topk
        self.loss = nn.CrossEntropyLoss()
        self.Accuracy = Accuracy('multiclass', num_classes=10)
        self.softmax = nn.Softmax(dim=1)
        self.train_acc, self.val_acc = None, None
        self.mixup = args.mixup
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.mixup:
            x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)
            y_hat = self(x)
            loss = mixup_criterion(self.loss, y_hat, y_a, y_b, lam)
            acc = self.Accuracy(y_hat, y_a) * lam + self.Accuracy(y_hat, y_b) * (1 - lam)
            metrics = {'train_loss': loss, 'train_acc': acc}
            self.log_dict(metrics)
        else:
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
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        return [optimizer], [scheduler]
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        y_hat = self.softmax(y_hat)
        if self.topk != 0:
            topk, indices = torch.topk(y_hat, self.topk, dim=1)
            y_hat = torch.zeros_like(y_hat).scatter(1, indices, topk)
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

class TargetResnetModel(TargetBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = resnet18(num_classes=10)

class TargetAlexnetModel(TargetBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = alexnet(classes=10)

class TargetMLPModel_MNIST(TargetBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

class TargetCNNModel_MNIST(TargetBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(576, 10)
        )

class TargetMLPModel_CIFAR100(TargetBaseModel):
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
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100)
        )
        self.Accuracy = Accuracy('multiclass', num_classes=100)

class TargetCNNModel_CIFAR100(TargetBaseModel):
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
            nn.Linear(128*4*4, 100)
        )
        self.Accuracy = Accuracy('multiclass', num_classes=100)

class TargetResnetModel_CIFAR100(TargetBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = resnet18(num_classes=100)
        self.Accuracy = Accuracy('multiclass', num_classes=100)

class TargetAlexnetModel_CIFAR100(TargetBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = alexnet(classes=100)
        self.Accuracy = Accuracy('multiclass', num_classes=100)

class TargetResnetModel_A1K(TargetBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = resnet18(num_classes=1000)
        self.Accuracy = Accuracy('multiclass', num_classes=1000)

class TargetAlexnetModel_A1K(TargetBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = alexnet(classes=1000)
        self.Accuracy = Accuracy('multiclass', num_classes=1000)

class TargetCNNModel_A1K(TargetBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = Effnetv2_s()
        self.Accuracy = Accuracy('multiclass', num_classes=1000)

class TargetMLPModel_A1K(TargetBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*64*3, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1000)
        )
        self.Accuracy = Accuracy('multiclass', num_classes=1000)

class TargetPretrainModel(TargetBaseModel):
    def __init__(self, model, num_classes, args):
        super().__init__(args)
        self.model = Effnetv2_s()
        self.model.load_state_dict(torch.load('pretrained_model/effnetv2_s.pth'))
        self.model.fc = nn.Linear(1280, 1000)
        self.Accuracy = Accuracy('multiclass', num_classes=1000)