import mlflow
import torch
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from settings.config import (
    EPOCHS,
    K_FOLDS,
)

# Dict to use in combination with prepare for plotting function in training and testing phase
metric_arrays = {
    "Train Accuracy": np.zeros((EPOCHS - 1)),
    "Train Precision": np.zeros((EPOCHS - 1)),
    "Train Recall": np.zeros((EPOCHS - 1)),
    "Train F1": np.zeros((EPOCHS - 1)),
    "Train Roc": np.zeros((EPOCHS - 1)),
    "Test Accuracy": np.zeros((EPOCHS - 1)),
    "Test Precision": np.zeros((EPOCHS - 1)),
    "Test Recall": np.zeros((EPOCHS - 1)),
    "Test F1": np.zeros((EPOCHS - 1)),
    "Test Roc": np.zeros((EPOCHS - 1)),
}

def calculate_metrics(y_pred, y_true):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred)
    return precision, recall, f1, accuracy, roc

def prepare_for_plotting(metric_storage: dict, metric_name: str, metric_value: float, type: str, epoch: int): 
    metric_storage[metric_name][(epoch - 1)] = metric_value
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def reset_weights(m: torch.nn.Module):
    """Try resetting model weights to avoid weight leakage.

    Args:
        m (torch.nn.Module): The model to reset.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
