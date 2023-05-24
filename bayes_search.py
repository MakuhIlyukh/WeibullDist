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
from skopt import Optimizer, gp_minimize

from src.datasets import (
    load_dataset, WeibullMixtureSampler)
from src.plotting import (
    hist_plot, pdf_plot)
from src.utils import (
    set_commit_tag, del_folder_content)
from src.models import WM, Optimized_WM, Manual_GD_WM
from src.trainers import (
    EM_Trainer, ManualGD_Trainer, Moments_EM_Trainer, Moments_GD_Trainer, Moments_Trainer, OptimizedGD_Trainer, GD_Trainer, EM_GD_Trainer, LMoments_Trainer
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
# ALGORITHM = "opt_gd"
# ALGORITHM = "manual_gd"
# ALGORITHM = "em"
# ALGORITHM = "emgd"
# ALGORITHM = "lmoments"
# ALGORITHM = "moments"
# ALGORITHM = "moments_gd"
ALGORITHM = "moments_em"

START_TRAIN_SEED = 107
LR = 10**(-1)
N_EPOCHS = 200
WEIGHT_DECAY = 0.0
LOSS_PREFIX = "NLL"
METRIC_PREFIX = "R2"
PLOT_EVERY = 500000
BATCH_SIZE = 1.0
K_INIT = "random"
LMD_INIT = "random"
Q_INIT = "1/m"
MAX_NEWTON_ITER = 5
NEWTON_TOL = 0.01
OPT_NAME = "adam"
SWITCH_ITER = 2


def area(iter_times, y, max_time=0.05):
    seconds = iter_times / 10**9
    mask = np.cumsum(seconds, dtype=np.float64) < max_time
    return (seconds[mask] * y[mask]).sum(axis=0)


def transform_r2(r2_vals):
    return 1 - r2_vals


def eval_hyperparams(hyperparams):
    with open(joinp(DATASETS_ARTIFACTS_PATH, "num_of_datasets.json"), "r") as f:
        num_of_datasets = int(json.load(f))
    with open(joinp(DATASETS_ARTIFACTS_PATH, "run_id.json"), "r") as f:
        datasets_run_id = json.load(f)

    # metrics and losses
    loss_fn = nll
    metric_fn = R2Score()

    iter_times = []
    r2_scores = []
    for i in tqdm(range(num_of_datasets), position=1, leave=False):
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

        trainer = OptimizedGD_Trainer(m, "adam", hyperparams[0], K_INIT, LMD_INIT, Q_INIT, loss_fn)

        res = trainer.train(
            X=torch.from_numpy(X), y_true=torch.from_numpy(wm_sampler.pdf(X)),
            n_epochs=N_EPOCHS, batch_size=hyperparams[1], loss_fn=loss_fn, metric_fn=metric_fn,
            loss_prefix=LOSS_PREFIX, metric_prefix=METRIC_PREFIX
        )

        r2_scores.append([elem.metric for elem in res])
        iter_times.append([elem.iter_time for elem in res])
    
    areas = []
    for times, scores in zip(iter_times, r2_scores):
        times = np.array(times)
        transformed_score = transform_r2(np.array(scores))
        areas.append(area(times, transformed_score))
    return np.mean([ar for ar in areas if not np.isnan(ar)])
    

def bayes_search(n_search_iter):
    search_opt = Optimizer(
        [(0.001, 0.5, "uniform"),
         (0.1, 1.0, "uniform")]
    )
    for i in tqdm(range(n_search_iter), position=0, leave=False):
        suggested = search_opt.ask()
        area_score = eval_hyperparams(suggested)
        res = search_opt.tell(suggested, area_score)
    return res


if __name__ == '__main__':
    np.random.seed(26)
    res = bayes_search(30)
