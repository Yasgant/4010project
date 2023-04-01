import numpy as np
import torch
import lightning.pytorch as pl
from torchvision import transforms
from torch import nn, optim
import tqdm
import matplotlib.pyplot as plt
import os
from torchmetrics import Accuracy
import wandb
from lightning.pytorch.loggers import WandbLogger
import time
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from train import train
import utils
from models import AttackModel, TargetModel

batch_size = 128
model_lr = 1e-3
attack_lr = 1e-3
model_epochs = 100
attack_epochs = 100

if __name__ == '__main__':
    if os.path.exists('API_KEY'):
        with open('API_KEY', 'r') as f:
            os.environ["WANDB_API_KEY"] = f.read()
    else:
        os.environ["WANDB_API_KEY"] = "SAMPLE" # replace with your own API key
    
    target_model = TargetModel(lr = model_lr)
    shadow_model = TargetModel(lr = model_lr)
    CIFAR10 = utils.get_CIFAR10()
    train_data, test_data, member_data, nonmember_data = utils.split_dataset(CIFAR10)
    if os.path.exists('data/target_model.ckpt'):
        target_model = target_model.load_from_checkpoint('data/target_model.ckpt')
    else:
        target_model = train(target_model, 'target_model', train_data, test_data, batch_size=batch_size, epochs=model_epochs)
    if os.path.exists('data/shadow_model.ckpt'):
        shadow_model = shadow_model.load_from_checkpoint('data/shadow_model.ckpt')
    else:
        shadow_model = train(shadow_model, 'shadow_model', member_data, nonmember_data)
    
    membership_dataset = utils.prepare_membership(shadow_model, member_data, nonmember_data)
    membership_test_dataset = utils.prepare_membership(target_model, train_data, test_data)
    attack_model = AttackModel(lr = attack_lr)
    if attack_model.type == 'mlp':
        if os.path.exists('data/attack_model.ckpt'):
            attack_model = attack_model.load_from_checkpoint('data/attack_model.ckpt')
        else:
            attack_model = train(attack_model, 'attack_model', membership_dataset, membership_test_dataset, batch_size=batch_size, epochs=attack_epochs)
    
    valid_loader = DataLoader(membership_test_dataset, batch_size=batch_size, shuffle=False)
    trainer = pl.Trainer(accelerator='cpu')
    trainer.validate(attack_model, valid_loader)

    predict_attack = AttackModel(type='predict')
    random_attack = AttackModel(type='random')
    # trainer.validate(predict_attack, valid_loader)
    # trainer.validate(random_attack, valid_loader)
    from torchmetrics import ROC
    # metrics, labels = utils.get_metrics(membership_test_dataset, type='entropy')
    # roc = ROC(task='binary')
    # fpr, tpr, thresholds = roc(metrics, labels)
    # plt.plot(fpr, tpr)
    # plt.title('entropy')
    # plt.xlabel('fpr')
    # plt.ylabel('tpr')
    # plt.show()
    metrics, labels = utils.get_metrics(membership_test_dataset, type='confidence')
    roc = ROC(task='binary')
    fpr, tpr, thresholds = roc(metrics, labels)
    plt.plot(fpr, tpr)
    plt.title('confidence')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.show()
    metrics, labels = utils.get_metrics(membership_test_dataset, type='loss')
    roc = ROC(task='binary')
    fpr, tpr, thresholds = roc(metrics, labels)
    plt.plot(fpr, tpr)
    plt.title('loss')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.show()
    metrics, labels = utils.get_metrics(membership_test_dataset, type='loss')
    optimal_threshold, acc = utils.get_optimal_threshold(metrics, labels)
    print(f'Optimal loss threshold: {optimal_threshold}, accuracy: {acc}')
    metrics, labels = utils.get_metrics(membership_test_dataset, type='confidence')
    optimal_threshold, acc = utils.get_optimal_threshold(metrics, labels)
    print(f'Optimal confidence threshold: {optimal_threshold}, accuracy: {acc}')
    # metrics, labels = utils.get_metrics(membership_test_dataset, type='entropy')
    # optimal_threshold, acc = utils.get_optimal_threshold(metrics, labels)
    # print(f'Optimal entropy threshold: {optimal_threshold}, accuracy: {acc}')



