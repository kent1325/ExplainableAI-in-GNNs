import mlflow
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


def calculate_metrics(y_pred, y_true, epoch, type):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    # mlflow.log_metric(key=f"F1-Score-{type}", value=float(f1), step=epoch)
    # mlflow.log_metric(key=f"Accuracy-{type}", value=float(accuracy), step=epoch)
    # mlflow.log_metric(key=f"Precision-{type}", value=float(precision), step=epoch)
    # mlflow.log_metric(key=f"Recall-{type}", value=float(recall), step=epoch)
    # try:
    #     roc = roc_auc_score(y_true, y_pred)
    #     # print(f"ROC AUC: {roc}")
    #     # mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)
    # except:
    #     # mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)
    #     # print(f"ROC AUC: notdefined")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
