import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from banditpy.models import BanditTrainer2Arm
from banditpy.utils import generate_probs_2arm
from pathlib import Path
from joblib import Parallel, delayed
from banditpy.core import Bandit2Arm
import mab_subjects

n_sessions_train = 20000
probs = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
unstruc_probs_train, struc_probs_train = generate_probs_2arm(
    probs, N=n_sessions_train, frac_impurity=0.2
)

n_sessions_test = 1000
unstruc_probs_test, struc_probs_test = generate_probs_2arm(
    probs, N=n_sessions_test, frac_impurity=0.2
)

train_test_kwargs = dict(
    min_block_trials=100,
    max_block_trials=500,
    p_switch=0.02,
    n_block_min=4,
    n_block_max=8,
    progress_bar=False,
)

basepath = Path("/mnt/pve/Homes/bapun/Data/RNNdataset/Paradigm_8020/")


def train_structured(i, lr):
    struc_folder = basepath / f"BGModelS{i}"
    struc_folder.mkdir(parents=True, exist_ok=True)
    struc_model_path = struc_folder / f"BGModelS{i}.pt"

    model = BanditTrainer2Arm(lr=lr, model_path=struc_model_path, device="cpu")
    model.train(reward_probs=struc_probs_train, return_df=False, **train_test_kwargs)
    # model.save_model()

    dfs = model.evaluate(reward_probs=struc_probs_test, **train_test_kwargs)
    task = Bandit2Arm(
        probs=dfs[["arm1_reward_prob", "arm2_reward_prob"]].values,
        choices=dfs["chosen_action"].values,
        rewards=dfs["reward"].values,
        session_ids=dfs["session_id"].values,
    )

    perf = task.filter_by_trials(100, 100).get_optimal_choice_probability()[-10:].mean()

    return dict(model=f"S{i}", lr=lr, max_perf=perf, group="struc")


def train_unstructured(i, lr):
    unstruc_folder = basepath / f"BGModelU{i}"
    unstruc_folder.mkdir(parents=True, exist_ok=True)
    unstruc_model_path = unstruc_folder / f"BGModelU{i}.pt"

    model = BanditTrainer2Arm(lr=lr, model_path=unstruc_model_path, device="cpu")
    model.train(reward_probs=unstruc_probs_train, return_df=False, **train_test_kwargs)
    # model.save_model()

    dfu = model.evaluate(reward_probs=unstruc_probs_test, **train_test_kwargs)
    task = Bandit2Arm(
        probs=dfu[["arm1_reward_prob", "arm2_reward_prob"]].values,
        choices=dfu["chosen_action"].values,
        rewards=dfu["reward"].values,
        session_ids=dfu["session_id"].values,
    )
    perf = task.filter_by_trials(100, 100).get_optimal_choice_probability()[-10:].mean()
    return dict(model=f"U{i}", lr=lr, max_perf=perf, group="unstruc")


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
    lrs = np.geomspace(1e-6, 0.1, num=n_models)

    jobs = [(train_structured, i, lr) for i, lr in zip(range(n_models), lrs)] + [
        (train_unstructured, i, lr) for i, lr in zip(range(n_models), lrs)
    ]

    results = Parallel(n_jobs=n_jobs)(delayed(fn)(i, lr) for fn, i, lr in jobs)
    results_df = pd.DataFrame(results)

    mab_subjects.GroupData().save(results_df, "rnn_lr_search_perf", write_stub=False)
