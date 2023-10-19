import os
import torch
import optuna
import numpy as np
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
    store_metric_results,
    generate_plots,
    generate_storage_dict,
    generate_optuna_plots,
    reset_weights
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
    N_TRIALS,
)


def run_kfold_cv(model, train_dataset, n_trials):
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

    # Generate optuna plots
    generate_optuna_plots(study)

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
        hyperparameters = run_kfold_cv(model, train_dataset, N_TRIALS)
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
    metric_results_dict = generate_storage_dict(EPOCHS)

    if DO_TRAIN_MODEL and epoch < (EPOCHS - 1):
        model.apply(reset_weights)
        for e in range(epoch, EPOCHS):
            # Training phase
            model.train()
            train_loss, train_y_pred, train_y_true = model_trainer.train_model(
                train_loader
            )

            # Testing phase
            # model_saver(e, model, FILE_NAME)
            model.eval()
            test_loss, test_y_pred, test_y_true = model_tester.test_model(test_loader)

            # Save and print intermediate results
            store_metric_results(
                metric_results_dict,
                train_y_true,
                train_y_pred,
                test_y_true,
                test_y_pred,
                e,
            )
            if e % 10 == 0 or e == 1:
                print(
                    f"Epoch {e} | Train Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f}"
                )
                # Virker ikke på CUDA
                # print(confusion_matrix(test_y_true, test_y_pred, labels=[0, 1]))
        generate_plots(metric_results_dict)
    else:
        model.eval()
        test_loss, test_y_pred, test_y_true = model_tester.test_model(test_loader)
        # Virker ikke på CUDA
        # print(confusion_matrix(test_y_true, test_y_pred, labels=[0, 1]))
