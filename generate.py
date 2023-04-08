# %%
from os.path import join as joinp
import json

import numpy as np
import matplotlib.pyplot as plt
import mlflow

from src.datasets import WeibullMixtureSampler, save_dataset
from src.plotting import pdf_plot, hist_plot
from src.utils import set_commit_tag
from config import (
    DATASETS_ARTIFACTS_PATH as DAP,
    DATASETS_TAG_KEY,
    GENERATION_TAG_VALUE)


M = 3
N = 200_000
SEED = 20


if __name__ == '__main__':
    with mlflow.start_run() as data_gen_run:
        # log params
        mlflow.log_params({
            "M": M,
            "N": N,
            "SEED": SEED})
        # adding tags
        mlflow.set_tag(DATASETS_TAG_KEY, GENERATION_TAG_VALUE)
        set_commit_tag()
        
        # dataset generation
        rnd = np.random.RandomState(SEED)
        wms = WeibullMixtureSampler(M, rnd,
                                    lambda m, rnd: rnd.uniform(1, 2, size=m),
                                    lambda m, rnd: rnd.uniform(1, 2, size=m),
                                    lambda m, rnd: rnd.dirichlet(alpha=[1]*m))
        X, y = wms.sample(N)
        fig, axis = plt.subplots()
        hist_plot(X, int(np.sqrt(N)), axis=axis)
        pdf_plot(X, wms.pdf(X), axis=axis)
        plt.show()

        # saving data and log
        # dataset
        with open(joinp(DAP, 'Xy.pkl'), 'wb') as f:
            save_dataset(X, y, f)
        # sampler
        with open(joinp(DAP, 'gms.pkl'), 'wb') as f:
            wms.save(f)
        # run_id
        run_id = data_gen_run.info.run_id
        with open(joinp(DAP, 'run_id.json'), 'w') as f:
            json.dump(run_id, f)
        # figure
        fig.savefig(joinp(DAP, "figure.png"))
        # log data
        mlflow.log_artifact(DAP)
