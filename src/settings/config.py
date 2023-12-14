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
FILE_NAME = "report_model"
CURRENT_DATE = datetime.date.today().strftime("%Y%m%d")
PARAMETER_TIMESTAMP = "132845"
PARAMETER_DATE = "20231214"
MODEL_DATE = "20231214"
MODEL_EPOCH = 80

# Sampler options
N_TRIALS = 200
SAMPLER = optuna.samplers.TPESampler()
