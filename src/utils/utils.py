import os
import pickle
import torch
import warnings

warnings.filterwarnings("ignore")
from datetime import datetime
from settings.config import ROOT_PATH, CURRENT_DATE, DEVICE
import numpy as np
from visualization.visualize import Plot, LinePlot
import optuna.visualization.matplotlib as ovm
from sklearn.model_selection import train_test_split
from torch_geometric.explain import GNNExplainer, Explainer
from torchmetrics import F1Score, Accuracy, Precision, Recall, AUROC, MatthewsCorrCoef


def generate_optuna_plots(study):
    Plot.export_figure(
        ovm.plot_optimization_history(study),
        "optuna_optimization_history",
        overwrite=True,
    )
    Plot.export_figure(
        ovm.plot_param_importances(study), "hyperparameter_importance", overwrite=True
    )
    Plot.export_figure(
        ovm.plot_parallel_coordinate(study), "parallel_coordinate", overwrite=True
    )
    Plot.export_figure(ovm.plot_contour(study), "contour", overwrite=True)
    Plot.export_figure(ovm.plot_edf(study), "edf", overwrite=True)
    Plot.export_figure(ovm.plot_rank(study), "rank", overwrite=True)


def calculate_metrics(y_pred, y_true):
    accuracy_fn = Accuracy(task="binary").to(DEVICE)
    precision_fn = Precision(task="binary").to(DEVICE)
    recall_fn = Recall(task="binary").to(DEVICE)
    f1_fn = F1Score(task="binary").to(DEVICE)
    auroc_fn = AUROC(task="binary").to(DEVICE)
    matthews_fn = MatthewsCorrCoef(task="binary").to(DEVICE)

    precision = precision_fn(y_pred, y_true)
    recall = recall_fn(y_pred, y_true)
    f1 = f1_fn(y_pred, y_true)
    accuracy = accuracy_fn(y_pred, y_true)
    roc = auroc_fn(y_pred, y_true)
    matthews = matthews_fn(y_pred, y_true)
    return precision, recall, f1, accuracy, roc, matthews


def generate_storage_dict(epochs):
    metric_results = {
        "Train Accuracy": np.zeros((epochs - 1)),
        "Train Precision": np.zeros((epochs - 1)),
        "Train Recall": np.zeros((epochs - 1)),
        "Train F1": np.zeros((epochs - 1)),
        "Train Roc": np.zeros((epochs - 1)),
        "Train Matthews": np.zeros((epochs - 1)),
        "Test Accuracy": np.zeros((epochs - 1)),
        "Test Precision": np.zeros((epochs - 1)),
        "Test Recall": np.zeros((epochs - 1)),
        "Test F1": np.zeros((epochs - 1)),
        "Test Roc": np.zeros((epochs - 1)),
        "Test Matthews": np.zeros((epochs - 1)),
    }
    return metric_results


def store_metric_results(
    metric_results, train_y_true, train_y_pred, test_y_true, test_y_pred, e
):
    # Store metric results from training
    (
        train_precision,
        train_recall,
        train_f1,
        train_accuracy,
        train_roc,
        train_matthews,
    ) = calculate_metrics(train_y_pred, train_y_true)
    metric_results["Train Accuracy"][e - 1] = train_accuracy
    metric_results["Train Precision"][e - 1] = train_precision
    metric_results["Train Recall"][e - 1] = train_recall
    metric_results["Train F1"][e - 1] = train_f1
    metric_results["Train Roc"][e - 1] = train_roc
    metric_results["Train Matthews"][e - 1] = train_matthews

    # Store metric results from testing
    (
        test_precision,
        test_recall,
        test_f1,
        test_accuracy,
        test_roc,
        test_matthews,
    ) = calculate_metrics(test_y_pred, test_y_true)
    metric_results["Test Accuracy"][e - 1] = test_accuracy
    metric_results["Test Precision"][e - 1] = test_precision
    metric_results["Test Recall"][e - 1] = test_recall
    metric_results["Test F1"][e - 1] = test_f1
    metric_results["Test Roc"][e - 1] = test_roc
    metric_results["Test Matthews"][e - 1] = test_matthews


def generate_plots(metric_results):
    # Export plots
    accuracy_lineplot = LinePlot(
        x_label="Epoch", y_label="Accuracy", title="Train/Test Accuracy"
    ).multi_lineplot(
        [metric_results["Train Accuracy"], metric_results["Test Accuracy"]],
        labels=["Train Accuracy", "Test Accuracy"],
    )
    Plot.export_figure(accuracy_lineplot, "Accuracy", overwrite=True)

    precision_lineplot = LinePlot(
        x_label="Epoch", y_label="Precision", title="Train/Test Precision"
    ).multi_lineplot(
        [metric_results["Train Precision"], metric_results["Test Precision"]],
        labels=["Train Precision", "Test Precision"],
    )
    Plot.export_figure(precision_lineplot, "Precision", overwrite=True)

    recall_lineplot = LinePlot(
        x_label="Epoch", y_label="Recall", title="Train/Test Recall"
    ).multi_lineplot(
        [metric_results["Train Recall"], metric_results["Test Recall"]],
        labels=["Train Recall", "Test Recall"],
    )
    Plot.export_figure(recall_lineplot, "Recall", overwrite=True)

    f1_lineplot = LinePlot(
        x_label="Epoch", y_label="F1", title="Train/Test F1"
    ).multi_lineplot(
        [metric_results["Train F1"], metric_results["Test F1"]],
        labels=["Train F1", "Test F1"],
    )
    Plot.export_figure(f1_lineplot, "F1", overwrite=True)

    matthews_lineplot = LinePlot(
        x_label="Epoch",
        y_label="Matthews Correlation Coefficient",
        title="Train/Test Matthew Correlation",
    ).multi_lineplot(
        [metric_results["Train Matthews"], metric_results["Test Matthews"]],
        labels=["Train Matthews", "Test Matthews"],
    )
    Plot.export_figure(matthews_lineplot, "Matthews_corr", overwrite=True)

    roc_lineplot = LinePlot(
        x_label="Epoch", 
        y_label="ROC", 
        title="Train/Test ROC"
    ).multi_lineplot(
        [metric_results["Train Roc"], metric_results["Test Roc"]],
        labels=["Train ROC", "Test ROC"],
    )
    Plot.export_figure(roc_lineplot, "ROC", overwrite=True)


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


def train_test_splitter(dataset, train_size_percentage):
    train_idx, test_idx = train_test_split(
        range(len(dataset)), train_size=train_size_percentage, stratify=dataset.y
    )
    train_data = [dataset[i] for i in train_idx]
    test_data = [dataset[i] for i in test_idx]

    return train_data, test_data


def generate_explainer_plots(model, epochs, dataset):
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="binary_classification",
            task_level="graph",
            return_type="probs",
        ),
        # Include only the top 10 most important edges:
        threshold_config=dict(threshold_type="topk", value=10),
    )
    
    # Generate explanations
    for batch in dataset:
        explanation = explainer(batch.x.float(), batch.edge_index, target=batch.y.float(), batch_index=batch.batch)
        explanation.visualize_feature_importance()
        break # Only generate one explanation for now
    
    #Plot.export_figure(explanation.visualize_graph(), "explainer_graph", overwrite=True)