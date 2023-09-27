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


def calculate_metrics(y_pred, y_true):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred)
    return precision, recall, f1, accuracy, roc

def prepare_for_plotting(y_pred, y_true, precision_arr, recall_arr, accuracy_arr, f1_arr, roc_arr, type, fold, epoch):
    precision, recall, f1, accuracy, roc = calculate_metrics(y_pred, y_true)
    
    if type == "tuning":
        precision_arr[fold, (epoch - 1)] = precision
        recall_arr[fold, (epoch - 1)] = recall
        f1_arr[fold, (epoch - 1)] = f1
        accuracy_arr[fold, (epoch - 1)] = accuracy
        roc_arr[fold, (epoch - 1)] = roc
        plot_results = [np.mean(precision_arr, axis=0), np.mean(recall_arr, axis=0), np.mean(f1_arr, axis=0), np.mean(accuracy_arr, axis=0), np.mean(roc_arr, axis=0)]
        plot_labels = ["precision", "recall", "f1", "accuracy", "roc"]
        
    elif type == "training":
        precision_arr[epoch - 1] = precision
        recall_arr[epoch - 1] = recall
        f1_arr[epoch - 1] = f1
        accuracy_arr[epoch - 1] = accuracy
        roc_arr[epoch - 1] = roc
        plot_results = precision_arr, recall_arr, f1_arr, accuracy_arr, roc_arr
        plot_labels = ["precision", "recall", "f1", "accuracy", "roc"]
    
    return plot_results, plot_labels

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
