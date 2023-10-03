import numpy as np
import optuna
from sklearn.metrics import confusion_matrix
from torch.utils.data import SubsetRandomSampler
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
import torch.optim as optim
from utils.utils import calculate_metrics, reset_weights, prepare_for_plotting
from visualization.visualize import LinePlot
from models.train_model import ModelTrainer
from models.test_model import ModelTester
from settings.config import (
    GRAPH_BATCH_SIZE,
    EPOCHS,
    SEED,
    K_FOLDS,
)

def objective_cv(trial, model, train_dataset):
    # Create arrays for storing results.
    scores = []
    sk_fold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    for fold, (cv_train_idx, cv_validation_idx) in enumerate(
        sk_fold.split(train_dataset, train_dataset.y)
    ):
        print(f"\nFold: {fold}")
        print("====================================")
        # Generate the optimizers.
        optimizer_name = trial.suggest_categorical(
            "optimizer", ["Adam", "SGD", "RMSprop"]
        )
        lr = trial.suggest_float("lr", 1e-5, 1, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        scheduler_name = trial.suggest_categorical("scheduler", ["ExponentialLR"])
        scheduler_gamme = trial.suggest_float("scheduler_gamma", 0, 1)
        scheduler = getattr(optim.lr_scheduler, scheduler_name)(
            optimizer, gamma=scheduler_gamme
        )

        model_trainer = ModelTrainer(model, optimizer, scheduler)
        model_tester = ModelTester(model)
        model.apply(reset_weights)

        cv_train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=GRAPH_BATCH_SIZE,
            sampler=SubsetRandomSampler(cv_train_idx),
        )
        cv_validation_loader = DataLoader(
            dataset=train_dataset,
            batch_size=GRAPH_BATCH_SIZE,
            sampler=SubsetRandomSampler(cv_validation_idx),
        )

        for epoch in range(1, EPOCHS):
            model.train()
            train_loss, train_y_pred, train_y_true = model_trainer.train_model(
                cv_train_loader
            )
            _,_,_, train_accuracy,_,_ = calculate_metrics(
                train_y_pred, train_y_true
            )

            model.eval()
            test_loss, test_y_pred, test_y_true = model_tester.test_model(
                cv_validation_loader
            )
            _,_,_, test_accuracy,_,_ = calculate_metrics(
                test_y_pred, test_y_true
            )
            
            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch} | Train Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f} | Train Acc: {train_accuracy:.3f} | Test Acc: {test_accuracy:.3f}"
                )
                print(confusion_matrix(test_y_true, test_y_pred, labels=[0, 1]))
            
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            scores.append(test_accuracy)
            
    return np.mean(scores)
