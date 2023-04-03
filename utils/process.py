import torch
import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchmetrics import ROC
import matplotlib.pyplot as plt
from scipy.stats import entropy

def get_metrics(dataset, type='confidence', batch_size=64):
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    metrics = []
    labels = []
    criterion = None
    if type == 'confidence':
        criterion = lambda x, y: torch.max(x, dim=-1)[0]
    elif type == 'entropy':
        criterion = lambda x, y: -torch.tensor(entropy(x.T))
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
    for threshold in thresholds:
        preds = (metrics > threshold).astype(int)
        accs.append(np.mean(preds == labels))
    
    return thresholds[np.argmax(accs)]

def get_accuracy(metrics, labels, threshold, rule='greater'):
    metrics, labels = metrics.numpy(), labels.numpy()
    if rule == 'greater':
        preds = (metrics >= threshold).astype(int)
    elif rule == 'lower':
        preds = (metrics < threshold).astype(int)
    return np.mean(preds == labels)

def plot_roc_curve(metrics, labels, name):
    roc = ROC(task='binary')
    fpr, tpr, thresholds = roc(metrics, labels)
    plt.plot(fpr, tpr)
    plt.title(name)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.show()