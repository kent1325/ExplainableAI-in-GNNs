import datetime
import os
import torch
import optuna

# Path settings
ROOT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
DOTENV_PATH = os.path.normpath(os.path.join(ROOT_PATH, ".env"))

# Device settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train, test, validation settings
SEED = 1233
EPOCHS = 101
TRAIN_SIZE = 0.8
DO_HYPERPARAMETER_TUNING = False
DO_TRAIN_MODEL = False

# Parameter and Model path settings
FILE_NAME = "new_parameters"
CURRENT_DATE = datetime.date.today().strftime("%Y%m%d")
PARAMETER_TIMESTAMP = "223248"
PARAMETER_DATE = "20231208"
MODEL_DATE = "20231208"
MODEL_EPOCH = 50

# Sampler options
N_TRIALS = 1000
SAMPLER = optuna.samplers.TPESampler()
