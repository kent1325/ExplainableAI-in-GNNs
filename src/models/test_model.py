import torch
from torch.nn import BCEWithLogitsLoss
import numpy as np
import os
import sys

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir)))
from settings.config import DEVICE


class ModelTester:
    def __init__(self, model):
        super(ModelTester, self).__init__()
        self.model = model.to(DEVICE)
        self.loss_fn = BCEWithLogitsLoss()

    def test_model(self, test_dataset):
        step = 0
        running_loss = 0.0
        y_pred, y_true = [], []
        with torch.no_grad():
            for batch in test_dataset:
                batch.to(DEVICE)
                predictions = self.model(batch.x.float(), batch.edge_index, batch.batch)
                loss = self.loss_fn(torch.squeeze(predictions), batch.y.float())
                running_loss += loss.item()
                step += 1
                predictions.squeeze_()
                y_pred.append(torch.round(torch.sigmoid(predictions.detach())))
                y_true.append(batch.y.detach())

            y_pred = torch.cat(y_pred)
            y_true = torch.cat(y_true)

        return running_loss / step, y_pred, y_true
