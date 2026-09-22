import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from banditpy.models import BanditTrainer2Arm
from banditpy.utils import generate_probs_2arm
from pathlib import Path
from banditpy.core import Bandit2Arm
from joblib import Parallel, delayed

n_train_sessions = 30000
n_test_sessions = 300
grid_size = 25

probs = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
unstruc_probs_train, struc_probs_train = generate_probs_2arm(
    probs, N=n_train_sessions, frac_impurity=0.16
)
unstruc_probs_test, struc_probs_test = generate_probs_2arm(
    probs, N=n_test_sessions, frac_impurity=0.16
)

basepath = Path("/mnt/pve/Homes/bapun/Data/beta_search")
beta_entropys = np.linspace(0.02, 0.1, grid_size)
beta_values = np.linspace(0.02, 0.1, grid_size)
lr = 0.00004
gamma = 0.9


def run_models(be, bv):
    # Structured
    b2a_s = BanditTrainer2Arm(
        lr=lr,
        gamma=gamma,
        beta_entropy=be,
        beta_value=bv,
        device="cpu",
        model_path=basepath / "beta_search_structured.pt",
    )
    b2a_s.train(
        n_sessions=n_train_sessions,
        mode=struc_probs_train,
        save_model=False,
        progress_bar=False,
        return_df=False,
    )
    dfs = b2a_s.evaluate(
        n_sessions=n_test_sessions, mode=struc_probs_test, progress_bar=False
    )
    task_s = Bandit2Arm.from_df(
        df=dfs,
        probs=["arm1_reward_prob", "arm2_reward_prob"],
        choices="chosen_action",
        rewards="reward",
        session_ids="session_id",
    )
    final_perf_s = task_s.get_optimal_choice_probability()[-5:].mean()

    # Unstructured
    b2a_u = BanditTrainer2Arm(
        lr=lr,
        gamma=gamma,
        beta_entropy=be,
        beta_value=bv,
        device="cpu",
        model_path=basepath / "beta_search_unstructured.pt",
    )
    b2a_u.train(
        n_sessions=n_train_sessions,
        mode=unstruc_probs_train,
        save_model=False,
        progress_bar=False,
        return_df=False,
    )
    dfu = b2a_u.evaluate(
        n_sessions=n_test_sessions, mode=unstruc_probs_test, progress_bar=False
    )
    task_u = Bandit2Arm.from_df(
        df=dfu,
        probs=["arm1_reward_prob", "arm2_reward_prob"],
        choices="chosen_action",
        rewards="reward",
        session_ids="session_id",
    )
    final_perf_u = task_u.get_optimal_choice_probability()[-5:].mean()

    return pd.DataFrame(
        dict(
            beta_entropy=be,
            beta_value=bv,
            final_perf_s=final_perf_s,
            final_perf_u=final_perf_u,
        ),
        index=[0],
    )


# Create list of parameter combinations
param_grid = [(be, bv) for be in beta_entropys for bv in beta_values]

# Run in parallel
with Parallel(n_jobs=24) as parallel:
    #helps to avoid no child processes error/warning
    results = parallel(delayed(run_models)(be, bv) for be, bv in param_grid)

# Combine results and save
search_df = pd.concat(results, ignore_index=True)
search_df.to_csv(basepath / "beta_search_results.csv", index=False)
