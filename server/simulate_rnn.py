import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from banditpy.models import BanditTrainer2Arm
from banditpy.utils import generate_probs_2arm
from pathlib import Path
from joblib import Parallel, delayed
from datetime import datetime

n_sessions_train = 70000
n_sessions_test = 1000
probs = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
frac_impurity = 0.10
lstm_kwargs = dict(
    lr=1e-4, lr_min=1e-6, gamma=0.9, beta_entropy=0.045, beta_value=0.025, device="cpu"
)
train_kwargs = dict(hidden_reset_every="window", update_every=("window", 1))
common_kwargs = dict(
    min_block_trials=100,
    max_block_trials=500,
    p_switch=0.02,
    n_block_min=4,
    n_block_max=8,
    progress_bar=False,
)

basedir = Path("/mnt/pve/Homes/bapun/Data/RNNdataset/Paradigm_9010/")
sim_folder = f"Train{n_sessions_train}_Test{n_sessions_test}_LR{lstm_kwargs['lr']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
basepath = basedir / sim_folder
basepath.mkdir(parents=True, exist_ok=True)

params_lines = [
    f"Simulation parameters",
    f"=====================",
    f"Date/Time        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"",
    f"[Probabilities]",
    f"probs            : {probs}",
    f"frac_impurity    : {frac_impurity}",
    f"",
    f"[Sessions]",
    f"n_sessions_train : {n_sessions_train}",
    f"n_sessions_test  : {n_sessions_test}",
    f"",
    f"[LSTM / optimizer]",
    f"lr               : {lstm_kwargs['lr']}",
    f"lr_min           : {lstm_kwargs['lr_min']}",
    f"gamma            : {lstm_kwargs['gamma']}",
    f"beta_entropy     : {lstm_kwargs['beta_entropy']}",
    f"beta_value       : {lstm_kwargs['beta_value']}",
    f"device           : {lstm_kwargs['device']}",
    f"",
    f"[Training]",
    f"hidden_reset_every : {train_kwargs['hidden_reset_every']}",
    f"update_every       : {train_kwargs['update_every']}",
    f"",
    f"[Episode structure]",
    f"min_block_trials : {common_kwargs['min_block_trials']}",
    f"max_block_trials : {common_kwargs['max_block_trials']}",
    f"p_switch         : {common_kwargs['p_switch']}",
    f"n_block_min      : {common_kwargs['n_block_min']}",
    f"n_block_max      : {common_kwargs['n_block_max']}",
]
(basepath / "params.txt").write_text("\n".join(params_lines) + "\n")


def train_structured(i):
    struc_folder = basepath / f"BGModelS{i}"
    struc_folder.mkdir(parents=True, exist_ok=True)
    struc_model_path = struc_folder / f"BGModelS{i}.pt"

    _, struc_probs_train = generate_probs_2arm(
        probs, N=n_sessions_train, frac_impurity=frac_impurity
    )
    _, struc_probs_test = generate_probs_2arm(
        probs, N=n_sessions_test, frac_impurity=frac_impurity
    )

    b2a_s = BanditTrainer2Arm(model_path=struc_model_path, **lstm_kwargs)
    b2a_s.train(
        reward_probs=struc_probs_train, return_df=False, **train_kwargs, **common_kwargs
    )
    b2a_s.save_model()

    dfs = b2a_s.evaluate(reward_probs=struc_probs_test, **common_kwargs)
    dfs.to_csv(struc_folder / f"BGModelS{i}.csv", index=False)


def train_unstructured(i):
    unstruc_folder = basepath / f"BGModelU{i}"
    unstruc_folder.mkdir(parents=True, exist_ok=True)
    unstruc_model_path = unstruc_folder / f"BGModelU{i}.pt"

    unstruc_probs_train, _ = generate_probs_2arm(
        probs, N=n_sessions_train, frac_impurity=frac_impurity
    )
    unstruc_probs_test, _ = generate_probs_2arm(
        probs, N=n_sessions_test, frac_impurity=frac_impurity
    )

    b2a_u = BanditTrainer2Arm(model_path=unstruc_model_path, **lstm_kwargs)
    b2a_u.train(
        reward_probs=unstruc_probs_train,
        return_df=False,
        **train_kwargs,
        **common_kwargs,
    )
    b2a_u.save_model()

    dfu = b2a_u.evaluate(reward_probs=unstruc_probs_test, **common_kwargs)
    dfu.to_csv(unstruc_folder / f"BGModelU{i}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=40,
        help="Total number of models to train (split evenly between structured and unstructured)",
    )
    parser.add_argument(
        "--n-jobs-subject",
        type=int,
        default=40,
        help="Number of parallel workers (passed to joblib)",
    )
    parser.add_argument(
        "--n-jobs-inner",
        type=int,
        default=1,
        help="Reserved for inner parallelism (unused)",
    )
    args = parser.parse_args()

    n_models = args.max_subjects // 2  # models per network type
    n_jobs = args.n_jobs_subject

    jobs = [(train_structured, i) for i in range(n_models)] + [
        (train_unstructured, i) for i in range(n_models)
    ]

    Parallel(n_jobs=n_jobs)(delayed(fn)(i) for fn, i in jobs)
