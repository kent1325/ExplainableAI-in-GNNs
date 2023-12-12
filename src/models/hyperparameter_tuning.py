import optuna
import torch
from torchmetrics.classification import BinaryConfusionMatrix
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
import torch.optim as optim
from utils.utils import calculate_metrics, reset_weights
from models.train_model import ModelTrainer
from models.test_model import ModelTester
from settings.config import (
    EPOCHS,
    SEED,
)


def objective_cv(trial, model, train_dataset):
    # Create arrays for storing results.
    scores_list = []
    labels = [y_vals.y for y_vals in train_dataset]
    k_folds = trial.suggest_categorical("k_folds", [5])
    sk_fold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)

    for fold, (cv_train_idx, cv_validation_idx) in enumerate(
        sk_fold.split(train_dataset, labels)
    ):
        print(f"\nFold: {fold}")
        print("====================================")
        # Generate the optimizers.
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-6, log=True)
        optimizer = getattr(optim, optimizer_name)(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # graph_batch_size = trial.suggest_int("graph_batch_size", 4, 32)
        graph_batch_size = trial.suggest_categorical(
            "graph_batch_size", [4, 8, 12, 16, 24, 32]
        )

        model_trainer = ModelTrainer(model, optimizer)
        model_tester = ModelTester(model)
        model.apply(reset_weights)

        cv_train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=graph_batch_size,
            sampler=SubsetRandomSampler(cv_train_idx),
        )
        cv_validation_loader = DataLoader(
            dataset=train_dataset,
            batch_size=graph_batch_size,
            sampler=SubsetRandomSampler(cv_validation_idx),
        )

        for epoch in range(1, EPOCHS):
            model.train()
            train_loss, train_y_pred, train_y_true = model_trainer.train_model(
                cv_train_loader
            )
            _, _, _, train_accuracy, _, _ = calculate_metrics(
                train_y_pred, train_y_true
            )

            model.eval()
            test_loss, test_y_pred, test_y_true = model_tester.test_model(
                cv_validation_loader
            )
            _, _, _, test_accuracy, _, _ = calculate_metrics(test_y_pred, test_y_true)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch} | Train Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f} | Train Acc: {train_accuracy:.3f} | Test Acc: {test_accuracy:.3f}"
                )
                print(
                    torch.transpose(
                        BinaryConfusionMatrix()(test_y_true, test_y_pred), 0, 1
                    )
                )
            # Handle pruning based on the intermediate value.
            trial.report(test_accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            scores_list.append(test_accuracy)
    scores = torch.tensor(scores_list)
    return torch.mean(scores)
