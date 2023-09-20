import os
import torch

ROOT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRAPH_BATCH_SIZE = 16
