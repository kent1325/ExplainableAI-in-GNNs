import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import sys

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir)))
from settings.config import DEVICE


class ModelTrainer:
    def __init__(self, model):
        super(ModelTrainer, self).__init__()
        self.model = model.to(DEVICE)
        self.weights = torch.tensor([1.0], dtype=torch.float32).to(DEVICE)
        self.loss_fn = BCEWithLogitsLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        self.train_dataset = None

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

            y_pred.append(
                np.rint(torch.round(torch.sigmoid(predictions)).detach().cpu().numpy())
            )
            y_true.append(batch.y.detach().cpu().numpy())

        y_pred = np.concatenate(y_pred).ravel()
        y_true = np.concatenate(y_true).ravel()

        return running_loss / step, y_pred, y_true
