import os
import torch
import optuna
import numpy as np
import optuna.visualization.matplotlib as ovm
from visualization.visualize import Plot, LinePlot
from data.get_dataloader import MUTAGLoader
import torch.optim as optim
from networks.gnn_loader import GAT, GCN
from dotenv import load_dotenv
from utils.utils import (
    calculate_metrics,
    count_parameters,
    model_saver,
    model_loader,
    hyperparameter_loader,
    hyperparameter_saver,
)
from sklearn.metrics import confusion_matrix
from models.hyperparameter_tuning import objective_cv
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from models.train_model import ModelTrainer
from models.test_model import ModelTester
from settings.config import (
    TRAIN_SIZE,
    DOTENV_PATH,
    SEED,
    EPOCHS,
    GRAPH_BATCH_SIZE,
    DO_HYPERPARAMETER_TUNING,
    FILE_NAME,
    CURRENT_DATE,
    DO_TRAIN_MODEL,
)


def run_kfold_cv(model, train_dataset, n_trials=2):
    study = optuna.create_study(direction="maximize")
    
    study.optimize(
        lambda trial: objective_cv(
            trial=trial, model=model, train_dataset=train_dataset
        ),
        n_trials=n_trials,
    )

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print(f"Best trial: {trial.number}")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    # Store Optuna plots 
    Plot.export_figure(ovm.plot_optimization_history(study), "optuna_optimization_history", overwrite=True)
    Plot.export_figure(ovm.plot_param_importances(study), "hyperparameter_importance", overwrite=True)
    Plot.export_figure(ovm.plot_parallel_coordinate(study), "parallel_coordinate", overwrite=True)
    Plot.export_figure(ovm.plot_contour(study), "contour", overwrite=True)
    Plot.export_figure(ovm.plot_edf(study), "edf", overwrite=True)
    Plot.export_figure(ovm.plot_rank(study), "rank", overwrite=True)

    return trial.params


if __name__ == "__main__":
    load_dotenv(DOTENV_PATH)
    torch.manual_seed(SEED)
    mutag_dataset = MUTAGLoader().get_dataset().shuffle()
    train_dataset = mutag_dataset[: int(len(mutag_dataset) * TRAIN_SIZE)]
    test_dataset = mutag_dataset[int(len(mutag_dataset) * TRAIN_SIZE) :]

    model = GCN(mutag_dataset.num_features)
    # print(model)
    # print(f"Number of parameters: {count_parameters(model)}")

    # Perform k-fold cross validation to tune hyperparameters
    if DO_HYPERPARAMETER_TUNING:
        hyperparameters = run_kfold_cv(model, train_dataset)
        epoch = 1
        DO_TRAIN_MODEL = True
        hyperparameter_saver(FILE_NAME, hyperparameters)
    else:
        epoch = EPOCHS - 1
        checkpoint = model_loader(FILE_NAME, epoch, CURRENT_DATE)
        model.load_state_dict(checkpoint["model_state"])
        hyperparameters = hyperparameter_loader(f"{FILE_NAME}_172515", CURRENT_DATE)

    # Train model with best hyperparameters and evaluate on test set
    train_loader = DataLoader(dataset=train_dataset, batch_size=GRAPH_BATCH_SIZE)
    test_loader = DataLoader(dataset=test_dataset, batch_size=GRAPH_BATCH_SIZE)

    optimizer = getattr(optim, hyperparameters["optimizer"])(
        model.parameters(), lr=hyperparameters["lr"]
    )
    scheduler = getattr(optim.lr_scheduler, hyperparameters["scheduler"])(
        optimizer, gamma=hyperparameters["scheduler_gamma"]
    )

    model_trainer = ModelTrainer(
        model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    model_tester = ModelTester(model)

    # Dict for storing metric results
    metric_results = {
        "Train Accuracy": np.zeros((EPOCHS - 1)),
        "Train Precision": np.zeros((EPOCHS - 1)),
        "Train Recall": np.zeros((EPOCHS - 1)),
        "Train F1": np.zeros((EPOCHS - 1)),
        "Train Roc": np.zeros((EPOCHS - 1)),
        "Train Matthews": np.zeros((EPOCHS - 1)),
        "Test Accuracy": np.zeros((EPOCHS - 1)),
        "Test Precision": np.zeros((EPOCHS - 1)),
        "Test Recall": np.zeros((EPOCHS - 1)),
        "Test F1": np.zeros((EPOCHS - 1)),
        "Test Roc": np.zeros((EPOCHS - 1)),
        "Test Matthews": np.zeros((EPOCHS - 1)),
    }
    
    if DO_TRAIN_MODEL and epoch < (EPOCHS - 1):
        for e in range(epoch, EPOCHS):
            # Training phase
            model.train()
            train_loss, train_y_pred, train_y_true = model_trainer.train_model(
                train_loader
            )

            # Store metric results from training
            precision, recall, f1, accuracy, roc, matthews = calculate_metrics(train_y_pred, train_y_true)
            metric_results["Train Accuracy"][e - 1] = accuracy
            metric_results["Train Precision"][e - 1] = precision
            metric_results["Train Recall"][e - 1] = recall
            metric_results["Train F1"][e - 1] = f1
            metric_results["Train Roc"][e - 1] = roc
            #metric_results["Train Matthews"][e - 1] = matthews

            # Testing phase
            model_saver(e, model, FILE_NAME)
            model.eval()
            test_loss, test_y_pred, test_y_true = model_tester.test_model(test_loader)
            
            # Store metric results from testing
            precision, recall, f1, accuracy, roc, matthews = calculate_metrics(test_y_pred, test_y_true)
            metric_results["Test Accuracy"][e - 1] = accuracy
            metric_results["Test Precision"][e - 1] = precision
            metric_results["Test Recall"][e - 1] = recall
            metric_results["Test F1"][e - 1] = f1
            metric_results["Test Roc"][e - 1] = roc
            metric_results["Test Matthews"][e - 1] = matthews
            
            # Print intermediate results
            if e % 10 == 0 or e == 1:
                print(
                    f"Epoch {e} | Train Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f}"
                )
                print(confusion_matrix(test_y_true, test_y_pred, labels=[0, 1]))
        
        # Export plots
        accuracy_lineplot = LinePlot(x_label="Epoch", y_label="Accuracy", title="Train/Test Accuracy").multi_lineplot(
            [metric_results["Train Accuracy"], metric_results["Test Accuracy"]],
            labels=["Train Accuracy", "Test Accuracy"]
        )
        precision_lineplot = LinePlot(x_label="Epoch", y_label="Precision", title="Train/Test Precision").multi_lineplot(
            [metric_results["Train Precision"], metric_results["Test Precision"]],
            labels=["Train Precision", "Test Precision"]
        )
        recall_lineplot = LinePlot(x_label="Epoch", y_label="Recall", title="Train/Test Recall").multi_lineplot(
            [metric_results["Train Recall"], metric_results["Test Recall"]],
            labels=["Train Recall", "Test Recall"]
        )
        f1_lineplot = LinePlot(x_label="Epoch", y_label="F1", title="Train/Test F1").multi_lineplot(
            [metric_results["Train F1"], metric_results["Test F1"]],
            labels=["Train F1", "Test F1"]
        )
        roc_lineplot = LinePlot(x_label="Epoch", y_label="ROC", title="Train/Test ROC").multi_lineplot(
            [metric_results["Train Roc"], metric_results["Test Roc"]],
            labels=["Train ROC", "Test ROC"]
        )
        matthews_lineplot = LinePlot(x_label="Epoch", y_label="Matthews Correlation Coefficient", title="Train/Test Matthew Correlation").multi_lineplot(
            [metric_results["Train Matthews"] ,metric_results["Test Matthews"]],
            labels = ["Train Matthews", "Test Matthews"]
        )
        
        Plot.export_figure(accuracy_lineplot, "accuracy", overwrite=True)
        Plot.export_figure(precision_lineplot, "precision", overwrite=True)
        Plot.export_figure(recall_lineplot, "recall", overwrite=True)
        Plot.export_figure(f1_lineplot, "f1", overwrite=True)
        Plot.export_figure(roc_lineplot, "roc", overwrite=True)
        Plot.export_figure(roc_lineplot, "matthews", overwrite=True)
    else:
        model.eval()
        test_loss, test_y_pred, test_y_true = model_tester.test_model(test_loader)
        print(confusion_matrix(test_y_true, test_y_pred, labels=[0, 1]))
