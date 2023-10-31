import datetime
import os
import torch
import optuna

# Path settings
ROOT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
DOTENV_PATH = os.path.normpath(os.path.join(ROOT_PATH, ".env"))
FILE_NAME = "TestAcc"
CURRENT_DATE = datetime.date.today().strftime("%Y%m%d")

# Device settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train, test, validation settings
TRAIN_SIZE = 0.8
DO_HYPERPARAMETER_TUNING = True
DO_TRAIN_MODEL = True
EPOCHS = 101
SEED = 12345

# Sampler options
N_TRIALS = 3
# SEARCH_SPACE = {
#     "lr": [1e-4, 1e-3, 1e-2, 1e-1],
#     "optimizer": ["Adam", "SGD", "RMSprop"],
#     "k_folds": [3, 5, 10],
#     "weight_decay": [1e-8, 1e-7, 1e-6, 1e-5],
#     "graph_batch_size": [4, 8, 16, 32],
# }
# SAMPLER = optuna.samplers.GridSampler(search_space=SEARCH_SPACE)
SAMPLER = optuna.samplers.TPESampler()
