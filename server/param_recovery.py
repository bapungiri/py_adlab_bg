import argparse
import numpy as np
import mab_subjects
import pandas as pd
from banditpy.models import DecisionModel
from banditpy.models.policy import StateInference, Qlearn, ThompsonShared
from banditpy.utils.probs import generate_probs_2arm
from banditpy.models.optim import OptunaOptimizer
from scipy.stats import pearsonr
from numpy.random import default_rng
from joblib import Parallel, delayed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parameter recovery for Qlearn"
    )
    parser.add_argument(
        "--max-subjects", type=int, default=1000, help="Number of simulations"
    )
    parser.add_argument(
        "--n-jobs-subject",
        type=int,
        default=20,
        help="Number of outer parallel jobs",
    )
    parser.add_argument(
        "--n-jobs-inner",
        type=int,
        default=5,
        help="Number of inner optimizer jobs per simulation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible simulations",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    main_policy = Qlearn

    n_simulations = args.max_subjects

    n_sessions = 200
    min_trials_per_block = 100
    prob_switch = 0.02
    fit_kwargs = {
        "optimizer": OptunaOptimizer(n_trials=80),
        "n_starts": 5,
        "n_jobs": args.n_jobs_inner,
        "early_stop": True,
        "es_warmup_trials": 3000,
        "es_check_every": 250,
        "es_slack": 0.01,
    }

    probs_unstruc, probs_struc = generate_probs_2arm(N=n_sessions, frac_impurity=0.2)

    print(f"Prob correlation: {pearsonr(probs_unstruc[:, 0], probs_unstruc[:, 1])[0]}")
    print(f"Prob correlation: {pearsonr(probs_struc[:, 0], probs_struc[:, 1])[0]}")

    policy1_bounds = main_policy().get_bounds()
    child_seed_seqs = np.random.SeedSequence(args.seed).spawn(n_simulations)

    # Rate/scale-like params span orders of magnitude, so a linear-uniform
    # draw over-samples large values relative to small ones; log-uniform
    # gives equal weight to equal ratios instead.
    LOG_SCALE_PARAMS = {"beta"}

    def sample_true_param(rng, name, lower, upper):
        if name in LOG_SCALE_PARAMS:
            return float(np.exp(rng.uniform(np.log(lower), np.log(upper))))
        return float(rng.uniform(lower, upper))

    def run_simulation(seed_seq):
        rng = default_rng(seed_seq)
        policy1 = main_policy()
        param_dict = {}
        for param, (lower, upper) in policy1_bounds.items():
            param_dict[param] = sample_true_param(rng, param, lower, upper)

        policy1.set_params(param_dict)

        task_unstruc = DecisionModel.simulate_policy(
            policy=policy1,
            reward_schedule=probs_unstruc,
            min_trials_per_block=min_trials_per_block,
            prob_switch=prob_switch,
        )

        task_struc = DecisionModel.simulate_policy(
            policy=policy1,
            reward_schedule=probs_struc,
            min_trials_per_block=min_trials_per_block,
            prob_switch=prob_switch,
        )

        policy2 = main_policy()
        model_unstruc = DecisionModel(
            task=task_unstruc, policy=policy2, reset_mode="session"
        )
        model_unstruc.fit(**fit_kwargs)

        policy3 = main_policy()
        model_struc = DecisionModel(
            task=task_struc, policy=policy3, reset_mode="session"
        )
        model_struc.fit(**fit_kwargs)

        def _true_value(policy, name):
            # Some params (e.g. 'beta') live on policy.beta_schedule.params
            # rather than policy.params -- policy1_bounds combines both.
            try:
                return policy.params[name]
            except KeyError:
                return policy.beta_schedule.params[name]

        df = pd.DataFrame()
        df["param"] = list(policy1_bounds.keys())
        df["true_value"] = [_true_value(policy1, param) for param in policy1_bounds.keys()]
        df["estimated_value_unstruc"] = [
            model_unstruc.params[param] for param in policy1_bounds.keys()
        ]
        df["estimated_value_struc"] = [
            model_struc.params[param] for param in policy1_bounds.keys()
        ]
        return df

    results = Parallel(n_jobs=args.n_jobs_subject)(
        delayed(run_simulation)(child_seed_seqs[i]) for i in range(n_simulations)
    )

    recovery_df = pd.concat(results, ignore_index=True)
    mab_subjects.GroupData().save(
        recovery_df, "param_recovery_qlearn", write_stub=False
    )


if __name__ == "__main__":
    main()
