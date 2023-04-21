import torch
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageFolder
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset, random_split
from torchvision import transforms
import lightning.pytorch as pl
import os
import numpy as np
import utils

def get_dataset(name, transform = transforms.ToTensor(), limit=None):
    if name == 'MNIST':
        return get_MNIST(transform, limit)
    elif name == 'CIFAR10':
        return get_CIFAR10(transform, limit)
    elif name == 'CIFAR100':
        return get_CIAR100(transform, limit)
    elif name == 'A1K':
        return get_A1K(transform, limit)
    else:
        raise ValueError('Dataset not supported')
    
def get_MNIST(transform = transforms.ToTensor(), limit=None):
    train = MNIST(root='./data', train=True, download=True, transform=transform)
    test = MNIST(root='./data', train=False, download=True, transform=transform)
    if limit is not None:
        train = utils.limit_label(train, limit)
        test = utils.limit_label(test, limit)
    return ConcatDataset([train, test])

def get_CIFAR10(transform = transforms.ToTensor(), limit=None):
    train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test = CIFAR10(root='./data', train=False, download=True, transform=transform)
    if limit is not None:
        train = utils.limit_label(train, limit)
        test = utils.limit_label(test, limit)
    return ConcatDataset([train, test])

def get_CIAR100(transform = transforms.ToTensor(), limit=None):
    train = CIFAR100(root='./data', train=True, download=True, transform=transform)
    test = CIFAR100(root='./data', train=False, download=True, transform=transform)
    if limit is not None:
        train = utils.limit_label(train, limit)
        test = utils.limit_label(test, limit)
    return ConcatDataset([train, test])

def get_A1K(transform = transforms.ToTensor(), limit=None):
    print(limit)
    if not os.path.exists('./data/train'):
        raise ValueError('Please download the A1-Kaggle dataset and place it in the data folder')
    train = ImageFolder('./data/train', transform=transform)
    test = ImageFolder('./data/val', transform=transform)
    if limit is not None:
        train = utils.limit_label(train, limit)
        test = utils.limit_label(test, limit)
    return ConcatDataset([train, test])

def split_dataset(dataset):
    dataset_len = len(dataset)
    train_data, test_data, member_data, nonmember_data = Subset(dataset, range(int(0.25*dataset_len))), Subset(dataset, range(int(0.25*dataset_len), int(0.5*dataset_len))), Subset(dataset, range(int(0.5*dataset_len), int(0.75*dataset_len))), Subset(dataset, range(int(0.75*dataset_len), dataset_len))
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

def limit_label(dataset, limit):
    return Subset(dataset, np.where(np.array(dataset.targets) < limit)[0])