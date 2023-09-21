import os
import mlflow
from data.get_dataloader import MUTAGLoader
from networks.gnn_loader import GAT, GCN
from models.train_model import ModelTrainer
from models.test_model import ModelTester
from torch_geometric.loader import DataLoader
from dotenv import load_dotenv
from settings.config import GRAPH_BATCH_SIZE, TRAIN_TEST_SIZE, DOTENV_PATH, EPOCHS
from utils.utils import calculate_metrics


if __name__ == "__main__":
    load_dotenv(DOTENV_PATH)
    # Specify tracking server
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    mutag_dataset = MUTAGLoader().get_dataset()
    model = GAT(mutag_dataset.num_features)
    print(model)
    print(f"Number of parameters: {model.count_parameters()}")

    model_trainer = ModelTrainer(model)
    model_tester = ModelTester(model)

    train_loader = DataLoader(
        mutag_dataset[: int(len(mutag_dataset) * TRAIN_TEST_SIZE)],
        batch_size=GRAPH_BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        mutag_dataset[int(len(mutag_dataset) * TRAIN_TEST_SIZE) :],
        batch_size=GRAPH_BATCH_SIZE,
        shuffle=True,
    )

    with mlflow.start_run() as run:
        for epoch in range(1, EPOCHS):
            model.train()
            train_loss, train_y_pred, train_y_true = model_trainer.train_model(
                train_loader
            )
            calculate_metrics(train_y_pred, train_y_true, epoch, "train")
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch} | Train Loss: {train_loss:.3f}")

                model.eval()
                test_loss, test_y_pred, test_y_true = model_tester.test_model(
                    test_loader
                )
                calculate_metrics(test_y_pred, test_y_true, epoch, "test")
                print(f"Epoch {epoch} | Test Loss: {test_loss:.3f}")
        mlflow.pytorch.log_model(model, "model")
