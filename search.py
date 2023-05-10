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
    EM_Trainer, ManualGD_Trainer, OptimizedGD_Trainer, GD_Trainer
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
ALGORITHM = "em"

START_TRAIN_SEED = 107
N_EPOCHS = 500
LOSS_PREFIX = "NLL"
METRIC_PREFIX = "R2"
K_INIT = "random"
LMD_INIT = "random"
Q_INIT = "1/m"


if __name__ == '__main__':
    # clearing folders
    del_folder_content(TRAIN_PLOTS_PATH)
    del_folder_content(TRAIN_DATA_PATH)

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
        "N_EPOCHS": N_EPOCHS,
        "LOSS": LOSS_PREFIX,
        "K_INIT": K_INIT,
        "LMD_INIT": LMD_INIT,
        "Q_INIT": Q_INIT,
        "ALGORITHM": ALGORITHM,
    })

    with open(joinp(DATASETS_ARTIFACTS_PATH, "num_of_datasets.json"), "r") as f:
        num_of_datasets = int(json.load(f))
    with open(joinp(DATASETS_ARTIFACTS_PATH, "run_id.json"), "r") as f:
        datasets_run_id = json.load(f)
        mlflow.log_param("DATASETS_RUN_ID", datasets_run_id)

    # metrics and losses
    loss_fn = nll
    metric_fn = R2Score()

    if ALGORITHM == "em":
        grid = [
            {
                "MAX_NEWTON_ITER": max_newtone_iter,
                "NEWTON_TOL": newton_tol,
                "BATCH_SIZE": batch_size
            }
            for max_newtone_iter in [3, 5, 10]
            for newton_tol in [0.01]
            for batch_size in [1.0]
        ]
    elif ALGORITHM == "opt_gd":
        grid = [
            {
                "LR": lr,
                "OPT_NAME": opt_name,
                "BATCH_SIZE": batch_size
            }
            for lr in [0.1, 0.05, 0.01]
            for batch_size in [0.2, 0.5, 0.7, 1.0]
            for opt_name in ["sgd", "adam"]
        ]
    else:
        raise ValueError(f"Unknown algorithm = {ALGORITHM}")

    search_results = []
    bar = tqdm(grid, position=0, leave=False)
    for hyperparams in bar:
        bar.set_postfix({"hp": str(hyperparams)})
        results = []
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

            # trainer creating
            if ALGORITHM == "gd":
                trainer = GD_Trainer(
                    m, opt_name=hyperparams["OPT_NAME"], lr=hyperparams["LR"],
                    k_init=K_INIT, lmd_init=LMD_INIT, q_init=Q_INIT,
                    loss_fn=loss_fn)
            elif ALGORITHM == "opt_gd":
                trainer = OptimizedGD_Trainer(
                    m, opt_name=hyperparams["OPT_NAME"], lr=hyperparams["LR"],
                    k_init=K_INIT, lmd_init=LMD_INIT, q_init=Q_INIT,
                    loss_fn=loss_fn)
            elif ALGORITHM == "manual_gd":
                trainer = ManualGD_Trainer(
                    m, opt_name=hyperparams["OPT_NAME"], lr=hyperparams["LR"],
                    k_init=K_INIT, lmd_init=LMD_INIT, q_init=Q_INIT,
                    c=0, eps=1e-6)
            elif ALGORITHM == "em":
                trainer = EM_Trainer(
                    m, K_INIT, LMD_INIT, Q_INIT,
                    max_newton_iter=hyperparams["MAX_NEWTON_ITER"],
                    newton_tol=hyperparams["NEWTON_TOL"])
            else:
                raise ValueError(f"Unknown algorithm = {ALGORITHM}")

            # training
            res = trainer.train(
                X=torch.from_numpy(X), y_true=torch.from_numpy(wm_sampler.pdf(X)),
                n_epochs=N_EPOCHS, batch_size=hyperparams["BATCH_SIZE"], loss_fn=loss_fn, metric_fn=metric_fn,
                loss_prefix=LOSS_PREFIX, metric_prefix=METRIC_PREFIX
            )
            results.append({
                "i": i,
                "res": res
            })
        search_results.append(
            {
                "hyperparams": hyperparams,
                "results": results
            }
        )
        
    with open(joinp(TRAIN_DATA_PATH, "search_results.pkl"), "wb") as f:
        pickle.dump(search_results, f)

    mlflow.log_artifact(TRAIN_PLOTS_PATH)
    # mlflow.pytorch.log_model(gm, TRMP)
    mlflow.log_artifact(TRAIN_DATA_PATH)