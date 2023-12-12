import os
import pickle
import torch
import warnings

from graphxai.explanation_metrics import fidelity, contrastivity, sparsity

warnings.filterwarnings("ignore")
from datetime import datetime
from settings.config import ROOT_PATH, CURRENT_DATE, DEVICE
import numpy as np
from visualization.visualize import Plot, LinePlot, CAMPlot
import optuna.visualization.matplotlib as ovm
from sklearn.model_selection import train_test_split
from torchmetrics import F1Score, Accuracy, Precision, Recall, AUROC, MatthewsCorrCoef
from graphxai.explanation import CAM


def generate_explanation_plots(
    mutag_dataset,
    model,
    filename,
    threshold=0.1,
    overwrite=True,
):
    important_masked_graphs = []
    unimportant_masked_graphs = []
    exp = None
    for i, graph in enumerate(mutag_dataset):
        model.eval()
        with torch.no_grad():
            prediction = model(
                graph.x,
                graph.edge_index,
                batch_index=torch.zeros(
                    1,
                    dtype=torch.int64,
                    device=DEVICE,
                ),
            )
        # predicted = torch.round(torch.sigmoid(prediction)).item()
        predicted = prediction.max(dim=1)[1]
        cam = CAM(model)
        exp = cam.get_explanation_graph(
            graph.x,
            edge_index=graph.edge_index,
            prediction=predicted.item(),
            label=graph.y,
        )
        masked_graphs = exp.generate_masked_graph(predicted, threshold=threshold)
        important_masked_graphs.append(masked_graphs[0])
        unimportant_masked_graphs.append(masked_graphs[1])
        # Export plots
        cam_plot = CAMPlot(
            x_label=None, y_label=None, title="Class Attention Map (CAM)"
        ).single_graph(
            exp=exp,
            y_pred=predicted,
            y_true=graph.y.item(),
            y_original_pred=None,
            use_node_importance=True,
        )
        Plot.export_figure(cam_plot, f"CAM/Graph{i}", overwrite=overwrite)
    exp.save_masked_graph(important_masked_graphs, filename + "_important")
    exp.save_masked_graph(unimportant_masked_graphs, filename + "_unimportant")

    combined_masked_graphs = [important_masked_graphs, unimportant_masked_graphs]
    updated_masked_graphs = [[], []]
    for i, mg in enumerate(combined_masked_graphs):
        for j, graph in enumerate(mg):
            model.eval()
            with torch.no_grad():
                prediction = model(
                    graph.x,
                    graph.edge_index,
                    batch_index=torch.zeros(
                        1,
                        dtype=torch.int64,
                        device=DEVICE,
                    ),
                )
                predicted = prediction.max(dim=1)[1]
                cam = CAM(model)
                exp = cam.get_explanation_graph(
                    graph.x,
                    edge_index=graph.edge_index,
                    prediction=predicted.item(),
                    label=graph.y,
                )
                graph.y_masked_pred = predicted
                updated_masked_graphs[i].append(graph)

                # Export plots
                cam_plot = CAMPlot(
                    x_label=None, y_label=None, title="Masked Graph"
                ).single_graph(
                    exp=exp,
                    y_pred=predicted,
                    y_true=graph.y.item(),
                    y_original_pred=graph.y_pred,
                )
                Plot.export_figure(
                    cam_plot, f"CAM/Masked_Graph_{i}-{j}", overwrite=overwrite
                )

    return updated_masked_graphs


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


def calculate_evaluation_metrics(model, masked_graphs, test_dataset):
    fidelity_plus, fidelity_minus = fidelity(masked_graphs)
    sparsity_score = sparsity(masked_graphs[0])
    contrastivity_score = contrastivity(model, test_dataset)

    return fidelity_plus, fidelity_minus, sparsity_score, contrastivity_score


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
        "Train Accuracy": torch.zeros((epochs - 1)),
        "Train Precision": torch.zeros((epochs - 1)),
        "Train Recall": torch.zeros((epochs - 1)),
        "Train F1": torch.zeros((epochs - 1)),
        "Train Roc": torch.zeros((epochs - 1)),
        "Train Matthews": torch.zeros((epochs - 1)),
        "Test Accuracy": torch.zeros((epochs - 1)),
        "Test Precision": torch.zeros((epochs - 1)),
        "Test Recall": torch.zeros((epochs - 1)),
        "Test F1": torch.zeros((epochs - 1)),
        "Test Roc": torch.zeros((epochs - 1)),
        "Test Matthews": torch.zeros((epochs - 1)),
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


def generate_plots(metric_results, overwrite=False):
    # Export plots
    accuracy_lineplot = LinePlot(
        x_label="Epoch", y_label="Accuracy", title="Train/Test Accuracy"
    ).multi_lineplot(
        [metric_results["Train Accuracy"], metric_results["Test Accuracy"]],
        labels=["Train Accuracy", "Test Accuracy"],
    )
    Plot.export_figure(accuracy_lineplot, "Accuracy", overwrite=overwrite)

    precision_lineplot = LinePlot(
        x_label="Epoch", y_label="Precision", title="Train/Test Precision"
    ).multi_lineplot(
        [metric_results["Train Precision"], metric_results["Test Precision"]],
        labels=["Train Precision", "Test Precision"],
    )
    Plot.export_figure(precision_lineplot, "Precision", overwrite=overwrite)

    recall_lineplot = LinePlot(
        x_label="Epoch", y_label="Recall", title="Train/Test Recall"
    ).multi_lineplot(
        [metric_results["Train Recall"], metric_results["Test Recall"]],
        labels=["Train Recall", "Test Recall"],
    )
    Plot.export_figure(recall_lineplot, "Recall", overwrite=overwrite)

    f1_lineplot = LinePlot(
        x_label="Epoch", y_label="F1", title="Train/Test F1"
    ).multi_lineplot(
        [metric_results["Train F1"], metric_results["Test F1"]],
        labels=["Train F1", "Test F1"],
    )
    Plot.export_figure(f1_lineplot, "F1", overwrite=overwrite)

    matthews_lineplot = LinePlot(
        x_label="Epoch",
        y_label="Matthews Correlation Coefficient",
        title="Train/Test Matthew Correlation",
    ).multi_lineplot(
        [metric_results["Train Matthews"], metric_results["Test Matthews"]],
        labels=["Train Matthews", "Test Matthews"],
    )
    Plot.export_figure(matthews_lineplot, "Matthews_corr", overwrite=overwrite)

    roc_lineplot = LinePlot(
        x_label="Epoch", y_label="ROC", title="Train/Test ROC"
    ).multi_lineplot(
        [metric_results["Train Roc"], metric_results["Test Roc"]],
        labels=["Train ROC", "Test ROC"],
    )
    Plot.export_figure(roc_lineplot, "ROC", overwrite=overwrite)


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
        "final_conv_acts": model.final_conv_acts,
        "final_conv_grads": model.final_conv_grads,
    }
    path = f"{ROOT_PATH}/models/{CURRENT_DATE}/{filename}/"
    file_name = f"{filename}_epoch_{epoch}.pth.tar"
    try:
        if os.path.exists(path):
            torch.save(checkpoint, path + file_name)

        if not os.path.exists(path):
            os.makedirs(path)
            torch.save(checkpoint, path + file_name)
        # print(f"Model '{file_name}' is saved")
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
    path = f"{ROOT_PATH}/models/{date}/{filename}/"
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


def masked_graphs_loader(filename: str, date: str):
    path = f"{ROOT_PATH}/data/MUTAG/masked_graphs/{date}/"
    file_name = f"{filename}.pkl"
    try:
        with open(path + file_name, "rb") as f:
            masked_graphs = pickle.load(f)
        print(f"Masked graphs '{file_name}' is loaded")
        return masked_graphs
    except Exception as e:
        print("Error loading masked graphs: ", e)
        return None


def train_test_splitter(dataset, train_size_percentage, seed=None):
    train_idx, test_idx = train_test_split(
        range(len(dataset)),
        train_size=train_size_percentage,
        stratify=dataset.y,
        shuffle=True,
        random_state=seed,
    )
    train_data = [dataset[i] for i in train_idx]
    test_data = [dataset[i] for i in test_idx]

    return train_data, test_data
