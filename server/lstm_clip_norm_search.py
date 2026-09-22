import numpy as np
import pandas as pd
from banditpy.models import BanditTrainer2Arm
from banditpy.core import Bandit2Arm
from banditpy.utils import generate_probs_2arm
from pathlib import Path
from joblib import Parallel, delayed
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

rng = np.random.default_rng()

n_train_sessions = 50000
n_test_sessions = 500
frac_impurity = 0.16
probs = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
unstruc_probs_train, struc_probs_train = generate_probs_2arm(
    probs, N=n_train_sessions, frac_impurity=frac_impurity
)

unstruc_probs_test, struc_probs_test = generate_probs_2arm(
    probs, N=n_test_sessions, frac_impurity=frac_impurity
)

model_kwargs = {
    "lr": 0.00004,
    "gamma": 0.92,
    "beta_entropy": 0.045,
    "beta_value": 0.025,
    "device": "cpu",
}

folder_name = f"Train{n_train_sessions}Impure{frac_impurity}_{timestamp}"
model_folder = Path("/mnt/pve/Homes/bapun/Data/rnn_models/") / folder_name
model_folder.mkdir(parents=True, exist_ok=True)

data_folder_name = (
    f"Train{n_train_sessions}Test{n_test_sessions}Impure{frac_impurity}_{timestamp}"
)
datapath = Path("/mnt/pve/Homes/bapun/Data/rnn_data/") / data_folder_name
datapath.mkdir(parents=True, exist_ok=True)


def train_rnn_models(i, clip_norm, train_type):

    if train_type == "structured":
        probs_train = rng.permutation(struc_probs_train, axis=0)
        probs_test = rng.permutation(struc_probs_test, axis=0)

    if train_type == "unstructured":
        probs_train = rng.permutation(unstruc_probs_train, axis=0)
        probs_test = rng.permutation(unstruc_probs_test, axis=0)

    # ------ Structured network ----------
    model_path = model_folder / f"{train_type}_2arm_model{i}.pt"
    model = BanditTrainer2Arm(model_path=model_path, **model_kwargs)
    model.train(
        n_sessions=n_train_sessions,
        mode=probs_train,
        clip_norm=clip_norm,
        progress_bar=False,
        save_model=False,
        return_df=False,
    )
    task_df = model.evaluate(
        n_sessions=n_test_sessions,
        mode=probs_test,
        progress_bar=False,
    )

    task = Bandit2Arm.from_df(
        df=task_df,
        probs=["arm1_reward_prob", "arm2_reward_prob"],
        choices="chosen_action",
        rewards="reward",
        session_ids="session_id",
    )
    perf = task.get_optimal_choice_probability()[-5:].mean()
    print(f"{train_type} model {i} performance: {perf:.4f}")

    model.save_model()
    model_name = model.model_path.stem
    model_data_folder = datapath / model_name
    model_data_folder.mkdir(exist_ok=True)

    # ------ Congruent environment testing ------
    exp_folder = model_data_folder / f"{model_name}_{train_type}"
    exp_folder.mkdir(exist_ok=True)

    task_df.to_csv(exp_folder / f"{model_name}_{train_type}.csv", index=False)

    # ------ Incongruent environment testing ------

    other_train_type = "unstructured" if train_type == "structured" else "structured"
    other_exp_folder = model_data_folder / f"{model_name}_{other_train_type}"
    other_exp_folder.mkdir(exist_ok=True)

    other_probs_test = (
        rng.permutation(unstruc_probs_test, axis=0)
        if train_type == "structured"
        else rng.permutation(struc_probs_test, axis=0)
    )

    other_task_df = model.evaluate(mode=other_probs_test, n_sessions=n_test_sessions)
    other_task_df.to_csv(
        other_exp_folder / f"{model_name}_{other_train_type}.csv", index=False
    )


clip_norm_values = np.linspace(0.5, 5, 10)
print("Training structured models...")
Parallel(n_jobs=10)(
    delayed(train_rnn_models)(i, clip_norm=val, train_type="structured")
    for i, val in enumerate(clip_norm_values)
)

print("Training unstructured models...")
Parallel(n_jobs=10)(
    delayed(train_rnn_models)(i, train_type="unstructured")
    for i in enumerate(clip_norm_values)
)

print("All models trained and saved.")
