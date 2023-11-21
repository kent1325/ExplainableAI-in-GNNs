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
        step = torch.tensor(0, dtype=torch.int32, device=DEVICE)
        running_loss = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
        y_pred, y_true = torch.tensor([], device=DEVICE), torch.tensor(
            [], device=DEVICE
        )
        with torch.no_grad():
            for batch in test_dataset:
                batch.to(DEVICE)
                predictions = torch.squeeze(
                    self.model(batch.x.float(), batch.edge_index, batch.batch)
                )
                loss = self.loss_fn(predictions, batch.y.float())
                running_loss += loss.item()
                step += 1
                y_pred = torch.cat((y_pred, torch.round(torch.sigmoid(predictions))))
                y_true = torch.cat((y_true, batch.y))

        return running_loss / step, y_pred, y_true
