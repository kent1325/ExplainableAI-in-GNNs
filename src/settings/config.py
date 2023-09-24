import os
import torch

ROOT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRAPH_BATCH_SIZE = 16
TRAIN_SIZE = 0.7
TEST_SIZE = 0.15
DOTENV_PATH = os.path.normpath(os.path.join(ROOT_PATH, ".env"))
EPOCHS = 101
SEED = 12345
SEEDS = [12345, 54321, 13245]
