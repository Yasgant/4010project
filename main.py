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
from models import get_target_model, get_attack_model
from torchmetrics import ROC
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--model_lr', type=float, default=1e-3)
parser.add_argument('--attack_lr', type=float, default=1e-3)
parser.add_argument('--model_epochs', type=int, default=50)
parser.add_argument('--attack_epochs', type=int, default=50)
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--target_model', type=str, default='cnn')
parser.add_argument('--attack_model', type=str, default='mlp')
parser.add_argument('--attack_hidden_size', type=int, default=20)
args = parser.parse_args()

if __name__ == '__main__':
    if os.path.exists('API_KEY'):
        with open('API_KEY', 'r') as f:
            os.environ["WANDB_API_KEY"] = f.read()
    else:
        os.environ["WANDB_API_KEY"] = "SAMPLE" # replace with your own API key

    TargetModel = get_target_model(args.target_model)
    target_model = TargetModel(args)
    shadow_model = TargetModel(args)

    DATASET = utils.get_dataset(args.dataset)
    train_data, test_data, member_data, nonmember_data = utils.split_dataset(DATASET)

    if os.path.exists(f'data/target_{args.target_model}model.ckpt'):
        target_model = target_model.load_from_checkpoint(f'data/target_{args.target_model}model.ckpt')
    else:
        target_model = train(target_model, f'target_{args.target_model}model', train_data, test_data, args=args)
    if os.path.exists(f'data/shadow_{args.target_model}model.ckpt'):
        shadow_model = shadow_model.load_from_checkpoint(f'data/shadow_{args.target_model}model.ckpt')
    else:
        shadow_model = train(shadow_model, f'shadow_{args.target_model}model', member_data, nonmember_data, args=args)
    
    membership_dataset = utils.prepare_membership(shadow_model, member_data, nonmember_data)
    membership_test_dataset = utils.prepare_membership(target_model, train_data, test_data)

    AttackModel = get_attack_model(args.attack_model)
    attack_model = AttackModel(args)

    if len(list(attack_model.parameters())):
        if os.path.exists('data/attack_model.ckpt'):
            attack_model = attack_model.load_from_checkpoint('data/attack_model.ckpt')
        else:
            attack_model = train(attack_model, 'attack_model', membership_dataset, membership_test_dataset, args=args)
    
    valid_loader = DataLoader(membership_test_dataset, batch_size=args.batch_size, shuffle=False)
    trainer = pl.Trainer(accelerator='cpu')
    attack_acc = trainer.validate(attack_model, valid_loader, verbose=False)[0]['val_acc']
    print(f'{args.attack_model} Attack Accuracy: {attack_acc}')
    PredictModel = get_attack_model('predict')
    predict_attacker = PredictModel(args)
    baseline_attack_acc = trainer.validate(predict_attacker, valid_loader, verbose=False)[0]['val_acc']
    print(f'Baseline Attack Accuracy: {baseline_attack_acc}')

    metrics, labels = utils.get_metrics(membership_dataset, type='loss')
    utils.plot_roc_curve(metrics, labels, 'loss')
    optimal_threshold = utils.get_optimal_threshold(metrics, labels)
    metrics, labels = utils.get_metrics(membership_test_dataset, type='loss')
    acc = utils.get_accuracy(metrics, labels, optimal_threshold)
    print(f'Optimal loss threshold: {optimal_threshold}, accuracy: {acc}')

    metrics, labels = utils.get_metrics(membership_dataset, type='confidence')
    utils.plot_roc_curve(metrics, labels, 'confidence')
    optimal_threshold = utils.get_optimal_threshold(metrics, labels)
    metrics, labels = utils.get_metrics(membership_test_dataset, type='confidence')
    acc = utils.get_accuracy(metrics, labels, optimal_threshold)
    print(f'Optimal confidence threshold: {optimal_threshold}, accuracy: {acc}')

    metrics, labels = utils.get_metrics(membership_dataset, type='entropy')
    utils.plot_roc_curve(metrics, labels, 'entropy')
    optimal_threshold = utils.get_optimal_threshold(metrics, labels)
    metrics, labels = utils.get_metrics(membership_test_dataset, type='entropy')
    acc = utils.get_accuracy(metrics, labels, optimal_threshold)
    print(f'Optimal entropy threshold: {optimal_threshold}, accuracy: {acc}')
