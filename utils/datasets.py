import torch
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageFolder
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset, random_split
from torchvision import transforms
import lightning.pytorch as pl
import os
import numpy as np

def get_dataset(name, transform = transforms.ToTensor()):
    if name == 'MNIST':
        return get_MNIST(transform)
    elif name == 'CIFAR10':
        return get_CIFAR10(transform)
    elif name == 'CIFAR100':
        return get_CIAR100(transform)
    elif name == 'A1K':
        return get_A1K(transform)
    else:
        raise ValueError('Dataset not supported')
    
def get_MNIST(transform = transforms.ToTensor()):
    train = MNIST(root='./data', train=True, download=True, transform=transform)
    test = MNIST(root='./data', train=False, download=True, transform=transform)
    return ConcatDataset([train, test])

def get_CIFAR10(transform = transforms.ToTensor()):
    train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test = CIFAR10(root='./data', train=False, download=True, transform=transform)
    return ConcatDataset([train, test])

def get_CIAR100(transform = transforms.ToTensor()):
    train = CIFAR100(root='./data', train=True, download=True, transform=transform)
    test = CIFAR100(root='./data', train=False, download=True, transform=transform)
    return ConcatDataset([train, test])

def get_A1K(transform = transforms.ToTensor()):
    if not os.path.exists('./data/train'):
        raise ValueError('Please download the A1-Kaggle dataset and place it in the data folder')
    train = ImageFolder('./data/train', transform=transform)
    test = ImageFolder('./data/val', transform=transform)
    return ConcatDataset([train, test])

def split_dataset(dataset):
    # dataset_len = len(dataset)
    train_data, test_data, member_data, nonmember_data = random_split(dataset, [0.25, 0.25, 0.25, 0.25])
    return train_data, test_data, member_data, nonmember_data

class MembershipDataset(Dataset):
    def __init__(self, member_preds, member_labels, nonmember_preds, nonmember_labels):
        self.membership = list(zip(member_preds, member_labels))
        self.nonmembership = list(zip(nonmember_preds, nonmember_labels))
        self.memlen = len(self.membership)
        self.nonmemlen = len(self.nonmembership)
        self.length = self.memlen + self.nonmemlen
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx < self.memlen:
            return self.membership[idx], torch.tensor(1, dtype=torch.float)
        else:
            return self.nonmembership[idx - self.memlen], torch.tensor(0, dtype=torch.float)

def prepare_membership(model, member_data, nonmember_data):
    trainer = pl.Trainer()
    member_loader = DataLoader(member_data, batch_size=64, shuffle=False)
    nonmember_loader = DataLoader(nonmember_data, batch_size=64, shuffle=False)
    member_preds = trainer.predict(model, member_loader)
    nonmember_preds = trainer.predict(model, nonmember_loader)
    member_preds, member_labels = zip(*member_preds)
    nonmember_preds, nonmember_labels = zip(*nonmember_preds)
    member_preds = torch.cat(member_preds)
    nonmember_preds = torch.cat(nonmember_preds)
    member_labels = torch.cat(member_labels)
    nonmember_labels = torch.cat(nonmember_labels)
    return MembershipDataset(member_preds, member_labels, nonmember_preds, nonmember_labels)

