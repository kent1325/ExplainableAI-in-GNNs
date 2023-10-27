import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import numpy as np
import os
import sys

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir)))
from settings.config import DEVICE


class ModelTrainer:
    def __init__(self, model, optimizer):
        super(ModelTrainer, self).__init__()
        self.model = model.to(DEVICE)
        # self.weights = torch.tensor([0.36], dtype=torch.float32).to(DEVICE)
        self.loss_fn = BCEWithLogitsLoss()
        self.optimizer = optimizer

    def train_model(self, train_dataset):
        step = 0
        running_loss = 0.0
        y_pred, y_true = [], []
        for batch in train_dataset:
            batch.to(DEVICE)
            self.optimizer.zero_grad()
            predictions = self.model(batch.x.float(), batch.edge_index, batch.batch)
            loss = self.loss_fn(torch.squeeze(predictions), batch.y.float())
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            step += 1
            predictions.squeeze_()
            y_pred.append(torch.round(torch.sigmoid(predictions.detach())))
            y_true.append(batch.y.detach())

        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)

        return running_loss / step, y_pred, y_true
