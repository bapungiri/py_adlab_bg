import argparse
import numpy as np
import mab_subjects
import pandas as pd
from banditpy.models import DecisionModel
from banditpy.models.policy import StateInference2Arm, Qlearn2Arm, ThompsonShared2Arm
from banditpy.utils.probs import generate_probs_2arm
from banditpy.models.optim import OptunaOptimizer
from scipy.stats import pearsonr
from numpy.random import default_rng
from joblib import Parallel, delayed


CANDIDATE_POLICIES = [Qlearn2Arm, StateInference2Arm, ThompsonShared2Arm]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Model recovery analysis for 2-arm bandit policies"
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=100,
        help="Number of simulations per generating model",
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
        help="Number of inner optimizer jobs per fit",
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

    print(
        f"Prob correlation (unstruc): {pearsonr(probs_unstruc[:, 0], probs_unstruc[:, 1])[0]}"
    )
    print(
        f"Prob correlation (struc): {pearsonr(probs_struc[:, 0], probs_struc[:, 1])[0]}"
    )

    policy_names = [cls.__name__ for cls in CANDIDATE_POLICIES]
    print(f"Candidate policies: {policy_names}")

    total_sims = len(CANDIDATE_POLICIES) * n_simulations
    child_seed_seqs = np.random.SeedSequence(args.seed).spawn(total_sims)

    def run_simulation(gen_policy_cls, seed_seq):
        rng = default_rng(seed_seq)

        # Sample random parameters for the generating policy
        gen_policy = gen_policy_cls()
        gen_bounds = gen_policy.get_bounds()
        param_dict = {}
        for param, (lower, upper) in gen_bounds.items():
            param_dict[param] = rng.uniform(lower, upper)
        gen_policy.set_params(param_dict)

        # Simulate data from the generating model (unstructured)
        task_unstruc = DecisionModel.simulate_policy(
            policy=gen_policy,
            reward_schedule=probs_unstruc,
            min_trials_per_block=min_trials_per_block,
            prob_switch=prob_switch,
        )

        # Simulate data from the generating model (structured)
        gen_policy2 = gen_policy_cls()
        gen_policy2.set_params(param_dict)
        task_struc = DecisionModel.simulate_policy(
            policy=gen_policy2,
            reward_schedule=probs_struc,
            min_trials_per_block=min_trials_per_block,
            prob_switch=prob_switch,
        )

        n_trials_unstruc = task_unstruc.n_trials
        n_trials_struc = task_struc.n_trials

        # Fit all candidate models and record NLL + n_params for BIC
        rows = []
        for fit_policy_cls in CANDIDATE_POLICIES:
            # Unstructured
            fit_policy_u = fit_policy_cls()
            model_u = DecisionModel(
                task=task_unstruc, policy=fit_policy_u, reset_mode="session"
            )
            model_u.fit(**fit_kwargs)
            n_params_u = len(fit_policy_u.param_names())
            bic_u = n_params_u * np.log(n_trials_unstruc) + 2 * model_u.nll

            # Structured
            fit_policy_s = fit_policy_cls()
            model_s = DecisionModel(
                task=task_struc, policy=fit_policy_s, reset_mode="session"
            )
            model_s.fit(**fit_kwargs)
            n_params_s = len(fit_policy_s.param_names())
            bic_s = n_params_s * np.log(n_trials_struc) + 2 * model_s.nll

            rows.append(
                {
                    "generating_model": gen_policy_cls.__name__,
                    "fit_model": fit_policy_cls.__name__,
                    "nll_unstruc": model_u.nll,
                    "bic_unstruc": bic_u,
                    "nll_struc": model_s.nll,
                    "bic_struc": bic_s,
                    "n_params": n_params_u,
                }
            )

        return pd.DataFrame(rows)

    results = Parallel(n_jobs=args.n_jobs_subject)(
        delayed(run_simulation)(
            CANDIDATE_POLICIES[i // n_simulations],
            child_seed_seqs[i],
        )
        for i in range(total_sims)
    )

    recovery_df = pd.concat(results, ignore_index=True)
    mab_subjects.GroupData().save(recovery_df, "model_recovery", write_stub=False)


if __name__ == "__main__":
    main()
