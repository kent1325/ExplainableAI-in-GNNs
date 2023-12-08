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
        self.loss_fn = CrossEntropyLoss()  # BCEWithLogitsLoss()
        self.optimizer = optimizer

    def train_model(self, train_dataset):
        step = torch.tensor(0, dtype=torch.int32, device=DEVICE)
        running_loss = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
        y_pred, y_true = torch.tensor([], device=DEVICE), torch.tensor(
            [], device=DEVICE
        )
        for batch in train_dataset:
            batch.to(DEVICE)
            self.optimizer.zero_grad()
            predictions = torch.squeeze(
                self.model(batch.x.float(), batch.edge_index, batch.batch)
            )
            classes = torch.tensor([], dtype=torch.int64, device=DEVICE)
            for label in batch.y.float():
                if label == 1.0:
                    classes = torch.cat(
                        (
                            classes,
                            torch.tensor(
                                [0.0, 1.0], dtype=torch.float32, device=DEVICE
                            ).unsqueeze(0),
                        )
                    )
                else:
                    classes = torch.cat(
                        (
                            classes,
                            torch.tensor(
                                [1.0, 0.0], dtype=torch.float32, device=DEVICE
                            ).unsqueeze(0),
                        )
                    )
            loss = self.loss_fn(predictions.float(), classes)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            step += 1
            argmax_pred = predictions.max(dim=1)
            y_pred = torch.cat((y_pred, argmax_pred[1]))
            # y_pred = torch.cat((y_pred, torch.round(torch.sigmoid(predictions))))
            y_true = torch.cat((y_true, batch.y))

        return running_loss / step, y_pred, y_true
