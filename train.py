# %%
from os.path import join as joinp

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
from src.models import WM, Optimized_WM
from src.losses import nll
# from src.initializers import KMeansInitializer
from config import (
    DATASETS_ARTIFACTS_PATH,
    TRAINED_MODELS_PATH,
    TRAIN_PLOTS_PATH,
    MODELS_TAG_KEY,
    TRAINING_TAG_VALUE)


TRAIN_SEED = 107
LR = 10**(-2)
N_EPOCHS = 5000
WEIGHT_DECAY = 0.0
LOSS_PREFIX = "NLL"
METRIC_PREFIX = "R2"
PLOT_EVERY = 500000
BATCH_SIZE = 0.2


def train(X, wm_sampler, wm, n_epochs, optimizer, loss_fn, metric_fn,
          batch_size=1.0, loss_prefix="", metric_prefix="", plot_every=100):
    # loss_name is used for logging
    loss_name = loss_prefix + "_loss"
    metric_name = metric_prefix + "_score"
    X_numpy = X
    y_true = torch.from_numpy(wm_sampler.pdf(X_numpy))
    X = torch.from_numpy(X)
    # TODO: add batch splitting
    # TODO: add stop on plateu
    if isinstance(batch_size, float):
        batch_size = int(X_numpy.shape[0] * batch_size)
    elif isinstance(batch_size, int):
        pass
    else:
        raise ValueError("batch_size must be int or float <= 1.0")
    tqdm_bar = tqdm(range(n_epochs))
    for epoch in tqdm_bar:
        wm.train()
        for batch_start in range(0, X_numpy.shape[0], batch_size):
            optimizer.zero_grad()
            dens = wm(X[batch_start : batch_start + batch_size])
            loss = loss_fn(dens)
            loss.backward()
            optimizer.step()

        wm.eval()
        with torch.no_grad():
            dens = wm(X)
            loss = loss_fn(dens)
            metric_score = metric_fn(dens, y_true).item()

        tqdm_bar.set_postfix({
            loss_name: loss.item(),
            metric_name: metric_score})

        if epoch % plot_every == 0:
            fig, ax = plt.subplots()
            pdf_plot(
                X_numpy,
                wm_sampler.pdf(X_numpy),
                axis=ax
            )
            pdf_plot(
                X_numpy,
                dens.detach().numpy(),
                axis=ax)
            fig.savefig(joinp(TRAIN_PLOTS_PATH, f"wm_plot_{epoch}.png"))
            plt.close(fig)  # ???: может лучше очищать фигуру и создавать не в цикле?
        
        mlflow.log_metrics({
            loss_name: loss.item(),
            metric_name: metric_score
        }, step=epoch)


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

    # creating a model
    wm = Optimized_WM(
        m,
        k_init="random",
        lmd_init="random",
        q_init="1/m")
    mlflow.log_param("ALGORITHM", wm.__class__)

    # choosing an optimizer
    # TODO: add lr sheduler
    optimizer = torch.optim.Adam(
        params=wm.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY)
    mlflow.log_param("OPTIMIZER", optimizer.__class__)
    
    # preprocessing steps
    X_proc = X

    # training
    loss_fn = nll
    metric_fn = R2Score()
    train(X_proc, wm_sampler, wm, N_EPOCHS, optimizer, loss_fn, metric_fn,
          loss_prefix=LOSS_PREFIX, metric_prefix=METRIC_PREFIX, plot_every=PLOT_EVERY,
          batch_size=BATCH_SIZE)

    # TODO: artifacts logging
    # TODO: Обрати внимание на то, чтобы папка plots очищалась
    #       перед запуском, ибо иначе будут логироваться данные с прошлых
    #       запусков. А может должна очищаться не только папка plots?
    #       А может вообще нужно, чтобы файлы не сохранялись в папки,
    #       а сразу заносились в mlflow runs?
    mlflow.log_artifact(TRAIN_PLOTS_PATH)
    # mlflow.pytorch.log_model(gm, TRMP)