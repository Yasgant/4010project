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
from models import get_target_model, get_attack_model, get_shadow_model
from torchmetrics import ROC
from argparse import ArgumentParser
from torchvision import transforms

parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--model_lr', type=float, default=1e-3)
parser.add_argument('--attack_lr', type=float, default=1e-3)
parser.add_argument('--model_epochs', type=int, default=50)
parser.add_argument('--attack_epochs', type=int, default=20)
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--target_model', type=str, default='cnn')
parser.add_argument('--attack_model', type=str, default='mlp')
parser.add_argument('--attack_hidden_size', type=int, default=20)
parser.add_argument('--model_weight_decay', type=float, default=0.0)
parser.add_argument('--topk', type=int, default=0)
args = parser.parse_args()

if __name__ == '__main__':
    if os.path.exists('API_KEY'):
        with open('API_KEY', 'r') as f:
            os.environ["WANDB_API_KEY"] = f.read()
    else:
        os.environ["WANDB_API_KEY"] = "SAMPLE" # replace with your own API key
    
    target_model, target_trained = get_target_model(args.target_model, args.dataset, args)
    shadow_model, shadow_trained = get_shadow_model(args.target_model, args.dataset, args)

    if args.dataset != 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.ToTensor()
    dataset = utils.get_dataset(args.dataset, transform=transform)
    train_data, test_data, member_data, nonmember_data = utils.split_dataset(dataset)
    arghash = hash(str(args))
    if not target_trained:
        target_model = train(target_model, f'target_{args.target_model}model_{args.dataset}_{arghash}', train_data, test_data, args=args, epochs=args.model_epochs)
    if not shadow_trained:
        shadow_model = train(shadow_model, f'shadow_{args.target_model}model_{args.dataset}_{arghash}', member_data, nonmember_data, args=args, epochs=args.model_epochs)
    membership_dataset = utils.prepare_membership(shadow_model, member_data, nonmember_data)
    membership_test_dataset = utils.prepare_membership(target_model, train_data, test_data)

    attack_model, attack_trained = get_attack_model(args.attack_model, args.dataset, args)

    if not attack_trained:
        attack_model = train(attack_model, f'attack_{args.attack_model}model_{args.dataset}_{arghash}', membership_dataset, membership_test_dataset, args=args, epochs=args.attack_epochs)
    
    valid_loader = DataLoader(membership_test_dataset, batch_size=args.batch_size, shuffle=False)
    trainer = pl.Trainer(accelerator='cpu')
    attack_acc = trainer.validate(attack_model, valid_loader, verbose=False)[0]['val_acc']
    print(f'{args.attack_model} Attack Accuracy: {attack_acc}')
    
    predict_attacker, _ = get_attack_model('predict', args.dataset, args)
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
