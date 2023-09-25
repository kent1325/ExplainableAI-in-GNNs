import os
import torch
import optuna
from data.get_dataloader import MUTAGLoader
from networks.gnn_loader import GAT, GCN
from dotenv import load_dotenv
from utils.utils import count_parameters
from models.hyperparameter_tuning import objective_cv
from settings.config import (
    TRAIN_SIZE,
    DOTENV_PATH,
    SEED,
)


if __name__ == "__main__":
    load_dotenv(DOTENV_PATH)
    torch.manual_seed(SEED)
    mutag_dataset = MUTAGLoader().get_dataset().shuffle()
    train_dataset = mutag_dataset[: int(len(mutag_dataset) * TRAIN_SIZE)]
    test_dataset = mutag_dataset[int(len(mutag_dataset) * TRAIN_SIZE) :]

    model = GCN(mutag_dataset.num_features)
    # print(model)
    # print(f"Number of parameters: {count_parameters(model)}")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_cv(
            trial=trial, model=model, train_dataset=train_dataset
        ),
        n_trials=20,
        timeout=600,
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

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
