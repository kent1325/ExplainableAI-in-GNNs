import torch
import optuna
from data.get_dataloader import MUTAGLoader
import torch.optim as optim
from networks.gnn_loader import GAT, GCN
from networks.top_k_pool_GCN import GCN_pool_layers
from dotenv import load_dotenv
from torchmetrics.classification import BinaryConfusionMatrix
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
    generate_explanation_plots,
    reset_weights,
    train_test_splitter,
)
from models.hyperparameter_tuning import objective_cv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from models.train_model import ModelTrainer
from models.test_model import ModelTester
from settings.config import (
    TRAIN_SIZE,
    DOTENV_PATH,
    SEED,
    EPOCHS,
    DO_HYPERPARAMETER_TUNING,
    FILE_NAME,
    DO_TRAIN_MODEL,
    N_TRIALS,
    SAMPLER,
    DEVICE,
    PARAMETER_TIMESTAMP,
    PARAMETER_DATE,
)


def run_kfold_cv(model, train_dataset, n_trials):
    study = optuna.create_study(
        direction=optuna.study.StudyDirection.MAXIMIZE,
        sampler=SAMPLER,
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
    )
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
    mutag_dataset = MUTAGLoader().get_dataset()
    train_dataset, test_dataset = train_test_splitter(
        mutag_dataset, TRAIN_SIZE, seed=SEED
    )

    model = GCN(mutag_dataset.num_features).to(device=DEVICE)
    # print(model)
    # print(f"Number of parameters: {count_parameters(model)}")

    # Perform k-fold cross validation to tune hyperparameters
    if DO_HYPERPARAMETER_TUNING:
        hyperparameters = run_kfold_cv(model, train_dataset, N_TRIALS)
        DO_TRAIN_MODEL = True
        hyperparameter_saver(FILE_NAME, hyperparameters)
    else:
        hyperparameters = hyperparameter_loader(
            f"{FILE_NAME}_{PARAMETER_TIMESTAMP}", PARAMETER_DATE
        )

    # Train model with best hyperparameters and evaluate on test set
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=hyperparameters["graph_batch_size"]
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=hyperparameters["graph_batch_size"]
    )

    optimizer = getattr(optim, hyperparameters["optimizer"])(
        model.parameters(),
        lr=hyperparameters["lr"],
        weight_decay=hyperparameters["weight_decay"],
    )
    print(f"Best params: {optimizer}")
    model_trainer = ModelTrainer(model, optimizer=optimizer)
    model_tester = ModelTester(model)

    # Dict for storing metric results
    metric_results_dict = generate_storage_dict(EPOCHS)

    if DO_TRAIN_MODEL:
        model.apply(reset_weights)
        for e in range(1, EPOCHS):
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
                _, _, _, train_accuracy, _, _ = calculate_metrics(
                    train_y_pred, train_y_true
                )
                _, _, _, test_accuracy, _, _ = calculate_metrics(
                    test_y_pred, test_y_true
                )
                print(
                    f"Epoch {e} | Train Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f} | Train Acc: {train_accuracy:.3f} | Test Acc: {test_accuracy:.3f}"
                )
                print(
                    torch.transpose(
                        BinaryConfusionMatrix()(test_y_true, test_y_pred), 0, 1
                    )
                )
        generate_plots(metric_results_dict, overwrite=False)
        generate_explanation_plots(test_dataset, model, overwrite=True)
    else:
        model.eval()
        test_loss, test_y_pred, test_y_true = model_tester.test_model(test_loader)
        print(torch.transpose(BinaryConfusionMatrix()(test_y_true, test_y_pred), 0, 1))
