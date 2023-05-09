# %%
from os.path import join as joinp
from time import perf_counter_ns
import pickle
import json

import torch
from torch import nn
from torchmetrics import R2Score
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import mlflow
import matplotlib.pyplot as plt
import numpy as np

from src.datasets import (
    load_dataset, WeibullMixtureSampler)
from src.plotting import (
    hist_plot, pdf_plot)
from src.utils import (
    set_commit_tag, del_folder_content)
from src.models import WM, Optimized_WM, Manual_GD_WM
from src.trainers import (
    EM_Trainer, ManualGD_Trainer, OptimizedGD_Trainer
)
from src.losses import nll
# from src.initializers import KMeansInitializer
from config import (
    DATASETS_ARTIFACTS_PATH,
    TRAINED_MODELS_PATH,
    TRAIN_PLOTS_PATH,
    MODELS_TAG_KEY,
    TRAINING_TAG_VALUE,
    TRAIN_DATA_PATH)


# ALGORITHM = "gd"
ALGORITHM = "opt_gd"
# ALGORITHM = "manual_gd"
# ALGORITHM = "em"

START_TRAIN_SEED = 107
LR = 10**(-1)
N_EPOCHS = 500
WEIGHT_DECAY = 0.0
LOSS_PREFIX = "NLL"
METRIC_PREFIX = "R2"
PLOT_EVERY = 500000
BATCH_SIZE = 1.0
K_INIT = "random"
LMD_INIT = "random"
Q_INIT = "1/m"


if __name__ == '__main__':
    # clearing folders
    del_folder_content(TRAIN_PLOTS_PATH)

    # starting mlflow
    train_run = mlflow.start_run()
    
    # commit
    set_commit_tag()
    # tag
    mlflow.set_tag(MODELS_TAG_KEY, TRAINING_TAG_VALUE)
    
    # mlflow logging
    # params
    mlflow.log_params({
        "START_TRAIN_SEED": START_TRAIN_SEED,
        "LR": LR,
        "N_EPOCHS": N_EPOCHS,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "LOSS": LOSS_PREFIX,
        "BATCH_SIZE": BATCH_SIZE,
        "K_INIT": K_INIT,
        "LMD_INIT": LMD_INIT,
        "Q_INIT": Q_INIT,
        "ALGORITHM": ALGORITHM
    })

    with open(joinp(DATASETS_ARTIFACTS_PATH, "num_of_datasets.json"), "r") as f:
        num_of_datasets = int(json.load(f))
    with open(joinp(DATASETS_ARTIFACTS_PATH, "run_id.json"), "r") as f:
        datasets_run_id = json.load(f)
        mlflow.log_param("DATASETS_RUN_ID", datasets_run_id)

    # metrics and losses
    loss_fn = nll
    metric_fn = R2Score()

    results = []
    for i in tqdm(range(num_of_datasets), position=0):
        # setting seed of torch and numpy
        seed = START_TRAIN_SEED + i
        torch.manual_seed(seed)
        rnd = np.random.RandomState(seed)

        # loading dataset
        # TODO: add dataset-generation-run logging
        with open(joinp(DATASETS_ARTIFACTS_PATH, f"{i}", "wms.pkl"), 'rb') as f:
            wm_sampler = WeibullMixtureSampler.load(f)
        with open(joinp(DATASETS_ARTIFACTS_PATH, f"{i}", "Xy.pkl"), 'rb') as f:
            X, y = load_dataset(f)
        m = wm_sampler.m

        # trainer creating
        if ALGORITHM == "gd":
            pass
        elif ALGORITHM == "opt_gd":
            trainer = OptimizedGD_Trainer(
                m, opt_name="adam", lr=LR,
                k_init=K_INIT, lmd_init=LMD_INIT, q_init=Q_INIT,
                loss_fn=loss_fn)
        elif ALGORITHM == "manual_gd":
            pass
        elif ALGORITHM == "em":
            trainer = EM_Trainer(
                m, K_INIT, LMD_INIT, Q_INIT,
                max_newtone_iter=5
            )
        else:
            raise ValueError(f"Unknown algorithm = {ALGORITHM}")

        # training
        res = trainer.train(
            X=torch.from_numpy(X), y_true=torch.from_numpy(wm_sampler.pdf(X)),
            n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, loss_fn=loss_fn, metric_fn=metric_fn,
            loss_prefix=LOSS_PREFIX, metric_prefix=METRIC_PREFIX
        )
        results.append({
            "i": i,
            "res": res
        })
        
    with open(joinp(TRAIN_DATA_PATH, "train_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    mlflow.log_artifact(TRAIN_PLOTS_PATH)
    # mlflow.pytorch.log_model(gm, TRMP)
    mlflow.log_artifact(TRAIN_DATA_PATH)