import os
import pickle
import torch
from datetime import datetime
from settings.config import ROOT_PATH, CURRENT_DATE, DEVICE
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef
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
    matthews = matthews_corrcoef(y_true, y_pred)
    return precision, recall, f1, accuracy, roc, matthews

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


def model_saver(
    epoch: int,
    model: torch.nn.Module,
    filename: str,
):
    """Saves model parameters to a file.

    Arguments:
        epoch: The current epoch.
        model: The model to save.
        optimizer: The optimizer to save.
        filename: The name of the file to save to.
    """

    checkpoint = {
        "model_name": filename,
        "model_state": model.state_dict(),
    }
    path = f"{ROOT_PATH}/models/{CURRENT_DATE}/"
    file_name = f"{filename}_epoch_{epoch}.pth.tar"
    try:
        if os.path.exists(path):
            torch.save(checkpoint, path + file_name)

        if not os.path.exists(path):
            os.makedirs(path)
            torch.save(checkpoint, path + file_name)
        print(f"Model '{file_name}' is saved")
    except Exception as e:
        print("Error saving model: ", e)


def model_loader(
    filename: str,
    epoch: int,
    date: str,
):
    """Loads model parameters from a file.

    Args:
        filename (str): The name of the file to load from.
        epoch (int): The epoch to load.
        date (str): The date folder to load from.

    Returns:
        checkpoint (dict) : Returns a dictionary containing the model and optimizer states.
    """
    path = f"{ROOT_PATH}/models/{date}/"
    file_name = f"{filename}_epoch_{epoch}.pth.tar"
    try:
        # Load model parameters
        checkpoint = torch.load(path + file_name, map_location=DEVICE)
        print(f"Model '{file_name}' is loaded")
        return checkpoint
    except Exception as e:
        print("Error loading model: ", e)
        return None


def hyperparameter_saver(filename: str, best_trial_params: dict):
    path = f"{ROOT_PATH}/models/parameters/{CURRENT_DATE}/"
    file_name = f"{filename}_{datetime.now().strftime('%H%M%S')}.pkl"
    try:
        if os.path.exists(path):
            with open(path + file_name, "wb") as f:
                pickle.dump(best_trial_params, f)

        if not os.path.exists(path):
            os.makedirs(path)
            with open(path + file_name, "wb") as f:
                pickle.dump(best_trial_params, f)
        print(f"Hyperparameters '{file_name}' is saved")
    except Exception as e:
        print("Error saving hyperparameters: ", e)


def hyperparameter_loader(filename: str, date: str):
    path = f"{ROOT_PATH}/models/parameters/{date}/"
    file_name = f"{filename}.pkl"
    try:
        with open(path + file_name, "rb") as f:
            hyperparameters = pickle.load(f)
        print(f"Hyperparameters '{file_name}' is loaded")
        return hyperparameters
    except Exception as e:
        print("Error loading hyperparameters: ", e)
        return None
