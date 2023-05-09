# %%
from os.path import join as joinp
from time import perf_counter_ns
import pickle

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


TRAIN_SEED = 107
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
    # starting mlflow
    train_run = mlflow.start_run()
    
    # mlflow logging
    # params
    mlflow.log_params({
        "TRAIN_SEED": TRAIN_SEED,
        "LR": LR,
        "N_EPOCHS": N_EPOCHS,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "LOSS": LOSS_PREFIX,
        "BATCH_SIZE": BATCH_SIZE})
    # commit
    set_commit_tag()
    # tag
    mlflow.set_tag(MODELS_TAG_KEY, TRAINING_TAG_VALUE)

    # clearing folders
    del_folder_content(TRAIN_PLOTS_PATH)

    # setting seed of torch and numpy
    torch.manual_seed(TRAIN_SEED)
    rnd = np.random.RandomState(TRAIN_SEED)

    # loading dataset
    # TODO: add dataset-generation-run logging
    with open(joinp(DATASETS_ARTIFACTS_PATH, "wms.pkl"), 'rb') as f:
        wm_sampler = WeibullMixtureSampler.load(f)
    with open(joinp(DATASETS_ARTIFACTS_PATH, "Xy.pkl"), 'rb') as f:
        X, y = load_dataset(f)
    m = wm_sampler.m

    # metrics and losses
    loss_fn = nll
    metric_fn = R2Score()

    # trainer creating
    trainer = OptimizedGD_Trainer(
        m, opt_name="adam", lr=LR,
        k_init=K_INIT, lmd_init=LMD_INIT, q_init=Q_INIT,
        loss_fn=loss_fn)
    
    # trainer = EM_Trainer(
    #     m, K_INIT, LMD_INIT, Q_INIT,
    #     max_newtone_iter=5
    # )
    
    mlflow.log_param("ALGORITHM", trainer.__class__)

    # training
    res = trainer.train(
        X=torch.from_numpy(X), y_true=torch.from_numpy(wm_sampler.pdf(X)),
        n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, loss_fn=loss_fn, metric_fn=metric_fn,
        loss_prefix=LOSS_PREFIX, metric_prefix=METRIC_PREFIX
    )

    mlflow.log_artifact(TRAIN_PLOTS_PATH)
    # mlflow.pytorch.log_model(gm, TRMP)
    mlflow.log_artifact(TRAIN_DATA_PATH)