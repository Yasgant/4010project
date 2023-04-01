import torch
import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader

def get_metrics(dataset, type='confidence', batch_size=64):
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    metrics = []
    labels = []
    criterion = None
    if type == 'confidence':
        criterion = lambda x, y: torch.max(x, dim=-1)[0]
    elif type == 'entropy':
        raise NotImplementedError
        criterion = lambda x, y: -torch.nansum(x * torch.log(x), dim=-1)
    elif type == 'loss':
        CE = torch.nn.CrossEntropyLoss(reduction='none')
        criterion = lambda x, y: -CE(x, y)
    else:
        raise ValueError('Invalid metric type')
    for pred, label in data_loader:
        metrics.append(criterion(pred[0], pred[1]))
        labels.append(label.long())
    metrics = torch.cat(metrics, dim=0)
    labels = torch.cat(labels, dim=0)
    return metrics, labels

def get_optimal_threshold(metrics, labels):
    metrics = metrics.numpy()
    labels = labels.numpy()
    thresholds = np.linspace(np.min(metrics), np.max(metrics), 1000)
    accs = []
    # import pdb; pdb.set_trace()
    for threshold in thresholds:
        preds = (metrics > threshold).astype(int)
        accs.append(np.mean(preds == labels))
    
    return thresholds[np.argmax(accs)], np.max(accs)
