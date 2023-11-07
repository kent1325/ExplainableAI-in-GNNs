import datetime
import os
import torch
import optuna

# Path settings
ROOT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
DOTENV_PATH = os.path.normpath(os.path.join(ROOT_PATH, ".env"))
FILE_NAME = "model_parameters"
CURRENT_DATE = "20231103"

# Device settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train, test, validation settings
TRAIN_SIZE = 0.8
DO_HYPERPARAMETER_TUNING = False
DO_TRAIN_MODEL = True
EPOCHS = 101
SEED = 12345

# Sampler options
N_TRIALS = 10
SAMPLER = optuna.samplers.TPESampler()
