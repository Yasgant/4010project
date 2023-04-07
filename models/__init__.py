from .attack import *
from .target import *
from .resnet import *
import os

def _get_target_type(name, dataset):
    if dataset == 'CIFAR10':
        if name == 'mlp':
            return TargetMLPModel
        elif name == 'cnn':
            return TargetCNNModel
        elif name == 'resnet':
            return TargetResnetModel
        elif name == 'alexnet':
            return TargetAlexnetModel
        else:
            raise ValueError('Invalid model name')
    elif dataset == 'MNIST':
        if name == 'mlp':
            return TargetMLPModel_MNIST
        elif name == 'cnn':
            return TargetCNNModel_MNIST
        else:
            raise ValueError('Invalid model name')
    elif dataset == 'CIFAR100':
        if name == 'mlp':
            return TargetMLPModel_CIFAR100
        elif name == 'cnn':
            return TargetCNNModel_CIFAR100
        elif name == 'resnet':
            return TargetResnetModel_CIFAR100
        elif name == 'alexnet':
            return TargetAlexnetModel_CIFAR100
        else:
            raise ValueError('Invalid model name')
    elif dataset == 'A1K':
        if name == 'resnet':
            return TargetResnetModel_A1K
        elif name == 'alexnet':
            return TargetAlexnetModel_A1K
        else:
            raise ValueError('Invalid model name')
    else:
        raise ValueError('Invalid dataset name')

def _get_attack_type(name, dataset=None):
    if name == 'mlp':
        if dataset == 'CIFAR10' or dataset == 'MNIST':
            return AttackMLPModel
        elif dataset == 'CIFAR100':
            return AttackMLPModel_CIFAR100
        elif dataset == 'A1K':
            return AttackMLPModel_A1K
        else:
            raise ValueError('Invalid dataset name')
    elif name == 'random':
        return AttackRandomModel
    elif name == 'predict':
        return AttackPredictModel
    elif name == 'entropy':
        return AttackEntropyModel
    elif name == 'confidence':
        return AttackConfidenceModel
    elif name == 'loss':
        return AttackLossModel
    else:
        raise ValueError('Invalid model name')

def get_target_model(name, dataset, args):
    TargetModel = _get_target_type(name, dataset)
    trained = False
    arghash = hash(str(args))
    if os.path.exists(f'data/target_{name}model_{dataset}_{arghash}.ckpt'):
        model = TargetModel.load_from_checkpoint(f'data/target_{name}model_{dataset}_{arghash}.ckpt')
        trained = True
    else:
        model = TargetModel(args)
    return model, trained

def get_shadow_model(name, dataset, args):
    ShadowModel = _get_target_type(name, dataset)
    trained = False
    arghash = hash(str(args))
    if os.path.exists(f'data/shadow_{name}model_{dataset}_{arghash}.ckpt'):
        model = ShadowModel.load_from_checkpoint(f'data/shadow_{name}model_{dataset}_{arghash}.ckpt')
        trained = True
    else:
        model = ShadowModel(args)
    return model, trained

def get_attack_model(name, dataset, args, threshold=None):
    AttackModel = _get_attack_type(name, dataset)
    trained = False
    if not AttackModel.trainable:
        if threshold is None:
            return AttackModel(args), True
        else:
            return AttackModel(threshold, args), True
    arghash = hash(str(args))
    if os.path.exists(f'data/attack_{name}model_{dataset}_{arghash}.ckpt'):
        model = AttackModel.load_from_checkpoint(f'data/attack_{name}model_{dataset}_{arghash}.ckpt')
        trained = True
    else:
        model = AttackModel(args)
    return model, trained