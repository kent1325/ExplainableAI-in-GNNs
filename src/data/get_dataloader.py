from torch_geometric.datasets import TUDataset
import os
import sys

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir)))
from settings.config import ROOT_PATH


class MUTAGLoader:
    def __init__(self):
        self.data_path = os.path.join(ROOT_PATH, "data")

    def get_dataset(self):
        return TUDataset(root=self.data_path, name="MUTAG")
