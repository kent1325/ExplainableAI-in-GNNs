import datetime
import os
import torch

ROOT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILE_NAME = "test_run"
GRAPH_BATCH_SIZE = 10
TRAIN_SIZE = 0.8
DO_HYPERPARAMETER_TUNING = True
DO_TRAIN_MODEL = True
DOTENV_PATH = os.path.normpath(os.path.join(ROOT_PATH, ".env"))
EPOCHS = 101
SEED = 12345
K_FOLDS = 5
CURRENT_DATE = datetime.date.today().strftime("%Y%m%d")
