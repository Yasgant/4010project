from .attack import AttackMLPModel, AttackPredictModel, AttackRandomModel
from .target import TargetMLPModel, TargetCNNModel

def get_target_model(name):
    if name == 'mlp':
        return TargetMLPModel
    elif name == 'cnn':
        return TargetCNNModel
    else:
        raise ValueError('Invalid model name')
    
def get_attack_model(name):
    if name == 'mlp':
        return AttackMLPModel
    elif name == 'random':
        return AttackRandomModel
    elif name == 'predict':
        return AttackPredictModel