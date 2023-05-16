# %%
from os.path import join as joinp
from os import makedirs
import json
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import mlflow

from src.datasets import WeibullMixtureSampler, save_dataset
from src.plotting import pdf_plot, hist_plot
from src.utils import set_commit_tag, del_folder_content
from config import (
    DATASETS_ARTIFACTS_PATH as DAP,
    DATASETS_TAG_KEY,
    GENERATION_TAG_VALUE)


N = 250
START_SEED = 505
NUM_DATASETS = 300
MIN_M = 1
MAX_M = 4


if __name__ == '__main__':
    with mlflow.start_run() as data_gen_run:
        seed = START_SEED
        mlflow.log_params({
            "START_SEED": START_SEED,
            "N": N,
            "NUM_DATASETS": NUM_DATASETS,
            "MIN_M": MIN_M,
            "MAX_M": MAX_M})
        # adding tags
        mlflow.set_tag(DATASETS_TAG_KEY, GENERATION_TAG_VALUE)
        set_commit_tag()
        # clearing folders
        del_folder_content(DAP)
        for i in tqdm(range(NUM_DATASETS)):
            m = np.random.RandomState(seed).randint(MIN_M, MAX_M + 1)
            # dataset generation
            rnd = np.random.RandomState(seed)
            wms = WeibullMixtureSampler(m, rnd,
                                        lambda m, rnd: rnd.uniform(1, 10, size=m),
                                        lambda m, rnd: rnd.uniform(1, 10, size=m),
                                        lambda m, rnd: rnd.dirichlet(alpha=[1]*m))
            X, y = wms.sample(N)
            fig, axis = plt.subplots()
            hist_plot(X, int(np.sqrt(N)), axis=axis)
            pdf_plot(X, wms.pdf(X), axis=axis)

            # saving data and log
            makedirs(joinp(DAP, f'{i}'), exist_ok=True)
            # dataset
            with open(joinp(DAP, f'{i}', 'Xy.pkl'), 'wb') as f:
                save_dataset(X, y, f)
            # sampler
            with open(joinp(DAP, f'{i}', 'wms.pkl'), 'wb') as f:
                wms.save(f)
            # figure
            fig.savefig(joinp(DAP, f'{i}', "figure.png"))
            plt.close(fig)
            with open(joinp(DAP, f'{i}', 'seed.json'), 'w') as f:
                json.dump(seed, f)
            seed += 1
        # run_id
        run_id = data_gen_run.info.run_id
        with open(joinp(DAP, 'run_id.json'), 'w') as f:
            json.dump(run_id, f)
        with open(joinp(DAP, "num_of_datasets.json"), "w") as f:
            json.dump(NUM_DATASETS, f)            
        mlflow.log_artifact(DAP)
    
