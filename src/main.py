import os
import mlflow
import torch
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from data.get_dataloader import MUTAGLoader
from networks.gnn_loader import GAT, GCN
from models.train_model import ModelTrainer
from models.test_model import ModelTester
from torch_geometric.loader import DataLoader
from dotenv import load_dotenv
from utils.utils import calculate_metrics, count_parameters
from settings.config import (
    GRAPH_BATCH_SIZE,
    TRAIN_SIZE,
    TEST_SIZE,
    DOTENV_PATH,
    EPOCHS,
    SEED,
    K_FOLDS,
)


if __name__ == "__main__":
    load_dotenv(DOTENV_PATH)
    torch.manual_seed(SEED)
    # Specify tracking server
    # mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    mutag_dataset = MUTAGLoader().get_dataset().shuffle()
    model = GAT(mutag_dataset.num_features)
    # print(model)
    # print(f"Number of parameters: {count_parameters(model)}")

    model_trainer = ModelTrainer(model)
    model_tester = ModelTester(model)

    train_dataset = mutag_dataset[: int(len(mutag_dataset) * TRAIN_SIZE)]
    test_dataset = mutag_dataset[
        int(len(mutag_dataset) * TRAIN_SIZE) : int(len(mutag_dataset) * TRAIN_SIZE)
        + int(len(mutag_dataset) * TEST_SIZE)
    ]
    validation_dataset = mutag_dataset[
        int(len(mutag_dataset) * TRAIN_SIZE) + int(len(mutag_dataset) * TEST_SIZE) :
    ]

    sk_fold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    for fold, (cv_train_idx, cv_validation_idx) in enumerate(
        sk_fold.split(train_dataset, train_dataset.y)
    ):
        print(f"\nFold: {fold}")
        print("====================================")
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
            # calculate_metrics(train_y_pred, train_y_true, epoch, "train")
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch} | Train Loss: {train_loss:.3f}")

                model.eval()
                test_loss, test_y_pred, test_y_true = model_tester.test_model(
                    cv_validation_loader
                )
                # calculate_metrics(test_y_pred, test_y_true, epoch, "test")
                print(f"Epoch {epoch} | Test Loss: {test_loss:.3f}")
        # mlflow.pytorch.log_model(model, "model")
