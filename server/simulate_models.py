import argparse
import numpy as np
import mab_subjects
from banditpy.models.policy import Qlearn2Arm, StaticBeta
from banditpy.utils import generate_probs_2arm
from banditpy.models import DecisionModel
from joblib import Parallel, delayed
from pathlib import Path
import pandas as pd
from datetime import datetime

probs_unstruc, probs_struc = generate_probs_2arm(N=1000, frac_impurity=0.2)
corr_struc = np.corrcoef(probs_struc[:, 0], probs_struc[:, 1])[0, 1]
corr_unstruc = np.corrcoef(probs_unstruc[:, 0], probs_unstruc[:, 1])[0, 1]
print(corr_unstruc, corr_struc)


task_common_kwargs = dict(min_trials_per_block=100, prob_switch=0.02)

# Define parameter grid
betas = np.geomspace(0.1, 40.0, 40)
alpha_c = np.linspace(0, 0.9, 40)
alpha_u = np.linspace(-0.9, 0.9, 40)
param_combos = np.array(np.meshgrid(alpha_c, alpha_u, betas)).T.reshape(-1, 3)


def train_structured(i, ac, au, beta):
    beta_schedule = StaticBeta()
    beta_schedule.params.beta.set_value(beta)
    policy = Qlearn2Arm(beta_schedule=beta_schedule)
    policy_name = policy.__class__.__name__
    policy.params.alpha_c.set_value(ac)
    policy.params.alpha_u.set_value(au)

    task_struc = DecisionModel.simulate_policy(
        policy=policy, reward_schedule=probs_struc, **task_common_kwargs
    )
    max_perf = (
        task_struc.filter_by_trials(100, 100)
        .get_optimal_choice_probability()[-10:]
        .mean()
    )
    return dict(
        model=f"{policy_name}_S{i}",
        alpha_c=ac,
        alpha_u=au,
        beta=beta,
        max_perf=max_perf,
        group="struc",
    )


def train_unstructured(i, ac, au, beta):
    beta_schedule = StaticBeta()
    beta_schedule.params.beta.set_value(beta)
    policy = Qlearn2Arm(beta_schedule=beta_schedule)
    policy_name = policy.__class__.__name__
    policy.params.alpha_c.set_value(ac)
    policy.params.alpha_u.set_value(au)

    task_unstruc = DecisionModel.simulate_policy(
        policy=policy, reward_schedule=probs_unstruc, **task_common_kwargs
    )
    max_perf = (
        task_unstruc.filter_by_trials(100, 100)
        .get_optimal_choice_probability()[-10:]
        .mean()
    )
    return dict(
        model=f"{policy_name}_U{i}",
        alpha_c=ac,
        alpha_u=au,
        beta=beta,
        max_perf=max_perf,
        group="unstruc",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-sims",
        type=int,
        default=80,
        help="Number of parallel workers (passed to joblib)",
    )
    args = parser.parse_args()

    n_jobs = args.n_sims

    jobs = [
        (train_structured, i, ac, au, beta)
        for i, (ac, au, beta) in enumerate(param_combos)
    ] + [
        (train_unstructured, i, ac, au, beta)
        for i, (ac, au, beta) in enumerate(param_combos)
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(fn)(i, ac, au, beta) for fn, i, ac, au, beta in jobs
    )
    results_df = pd.DataFrame(results)

    mab_subjects.GroupData().save(
        results_df, "simulated_policies_perf", write_stub=False
    )
