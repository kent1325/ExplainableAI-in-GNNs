import os
import torch
import optuna
from optuna.visualization import plot_optimization_history
from visualization.visualize import Plot, LinePlot
from data.get_dataloader import MUTAGLoader
import torch.optim as optim
from networks.gnn_loader import GAT, GCN
from dotenv import load_dotenv
from utils.utils import (
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


def run_kfold_cv(model, train_dataset, n_trials=1):
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
    print("Best trial:", trial.number)
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

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

    if DO_TRAIN_MODEL and epoch < (EPOCHS - 1):
        for e in range(epoch, EPOCHS):
            model.train()
            train_loss, train_y_pred, train_y_true = model_trainer.train_model(
                train_loader
            )
            model_saver(e, model, FILE_NAME)

            model.eval()
            test_loss, test_y_pred, test_y_true = model_tester.test_model(test_loader)
            if e % 10 == 0 or e == 1:
                print(
                    f"Epoch {e} | Train Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f}"
                )
                print(confusion_matrix(test_y_true, test_y_pred, labels=[0, 1]))
    else:
        model.eval()
        test_loss, test_y_pred, test_y_true = model_tester.test_model(test_loader)
        print(confusion_matrix(test_y_true, test_y_pred, labels=[0, 1]))
