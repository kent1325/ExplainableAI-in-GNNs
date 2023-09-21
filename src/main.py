import os
from data.get_dataloader import MUTAGLoader
from networks.gnn_loader import GAT, GCN
from models.train_model import ModelTrainer
from models.test_model import ModelTester
from torch_geometric.loader import DataLoader
from dotenv import load_dotenv
from settings.config import GRAPH_BATCH_SIZE, TRAIN_TEST_SIZE, DOTENV_PATH
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
import mlflow


def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_true, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    mlflow.log_metric(key=f"Precision-{type}", value=float(prec), step=epoch)
    mlflow.log_metric(key=f"Recall-{type}", value=float(rec), step=epoch)
    try:
        roc = roc_auc_score(y_true, y_pred)
        print(f"ROC AUC: {roc}")
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)
    except:
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)
        print(f"ROC AUC: notdefined")


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

    for epoch in range(1, 101):
        model.train()
        train_loss, train_y_pred, train_y_true = model_trainer.train_model(train_loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.3f}")

            model.eval()
            test_loss, test_y_pred, test_y_true = model_tester.test_model(test_loader)
            print(f"Epoch {epoch} | Test Loss: {test_loss:.3f}")
