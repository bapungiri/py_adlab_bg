import argparse
import numpy as np
import mab_subjects
import pandas as pd
from banditpy.core import Bandit2Arm
from banditpy.models import DecisionModel
from banditpy.models.policy import (
    StateInference,
    Qlearn,
    ThompsonShared,
    Qlearn2Regime,
    Qlearn3Regime,
)
from banditpy.utils.probs import generate_probs_2arm
from banditpy.models.optim import OptunaOptimizer
from scipy.stats import pearsonr
from numpy.random import default_rng
from joblib import Parallel, delayed


POLICY_REGISTRY = {
    "Qlearn": Qlearn,
    "Qlearn2Regime": Qlearn2Regime,
    "Qlearn3Regime": Qlearn3Regime,
    "StateInference": StateInference,
    "ThompsonShared": ThompsonShared,
}


# Same arm probabilities as generate_probs_2arm; tiers follow fit_policy_core's
# n_high rule (number of arms with p >= 0.5).
ARM_PROBS = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
TIER_N_HIGH = {"low_low": 0, "high_low": 1, "high_high": 2}


def n_high_arms(probs):
    return (np.asarray(probs) >= 0.5).sum(axis=1)


def generate_probs_tiers(n_blocks_per_tier, rng):
    """Mixed schedule with 'n_blocks_per_tier' blocks of each tier, shuffled.

    Pairs are drawn uniformly from the ordered, unequal pairs of 'ARM_PROBS'
    within each tier.
    """
    pairs = np.array([(a, b) for a in ARM_PROBS for b in ARM_PROBS if a != b])
    pairs_n_high = n_high_arms(pairs)
    blocks = []
    for n_high in TIER_N_HIGH.values():
        tier_pairs = pairs[pairs_n_high == n_high]
        idx = rng.integers(len(tier_pairs), size=n_blocks_per_tier)
        blocks.append(tier_pairs[idx])
    probs = np.vstack(blocks)
    rng.shuffle(probs)
    return probs


def concat_tasks(tasks):
    """Stack simulated 'Bandit2Arm' tasks, offsetting session/block ids so
    they stay unique across tasks."""
    probs, choices, rewards, session_ids, block_ids = [], [], [], [], []
    offset = 0
    for task in tasks:
        probs.append(task.probs)
        choices.append(task.choices)
        rewards.append(task.rewards)
        session_ids.append(task.session_ids + offset)
        block_ids.append(task.block_ids + offset)
        offset = max(session_ids[-1].max(), block_ids[-1].max())
    return Bandit2Arm(
        probs=np.vstack(probs),
        choices=np.concatenate(choices),
        rewards=np.concatenate(rewards),
        session_ids=np.concatenate(session_ids),
        block_ids=np.concatenate(block_ids),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Parameter recovery")
    parser.add_argument(
        "--policy",
        choices=list(POLICY_REGISTRY),
        default="Qlearn",
        help="Policy class to simulate/recover",
    )
    parser.add_argument(
        "--design",
        choices=["struc_unstruc", "tier", "tier_params"],
        default="struc_unstruc",
        help=(
            "struc_unstruc: fit structured and unstructured schedules separately; "
            "tier: simulate one mixed low-low/high-low/high-high schedule and "
            "fit all blocks plus each tier separately; "
            "tier_params: like tier, but each tier is simulated with its own "
            "true params"
        ),
    )
    parser.add_argument(
        "--n-blocks-per-tier",
        type=int,
        default=200,
        help="Blocks per tier in the mixed schedule (--design tier only)",
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
    main_policy = POLICY_REGISTRY[args.policy]

    n_simulations = args.max_subjects

    n_sessions = 200
    min_trials_per_block = 100
    prob_switch = 0.02
    # Rate/scale-like params span orders of magnitude, so a linear-uniform
    # draw over-samples large values relative to small ones; log-uniform
    # gives equal weight to equal ratios instead. Used both for the true
    # values and for the Optuna search.
    LOG_SCALE_PARAMS = {"beta", "beta_0", "beta_1", "beta_2"}

    fit_kwargs = {
        "optimizer": OptunaOptimizer(n_trials=80, log_params=LOG_SCALE_PARAMS),
        "n_starts": 5,
        "n_jobs": args.n_jobs_inner,
        "early_stop": True,
        "es_warmup_trials": 3000,
        "es_check_every": 250,
        "es_slack": 0.01,
    }

    if args.design == "struc_unstruc":
        probs_unstruc, probs_struc = generate_probs_2arm(
            N=n_sessions, frac_impurity=0.2
        )
        print(
            f"Prob correlation: {pearsonr(probs_unstruc[:, 0], probs_unstruc[:, 1])[0]}"
        )
        print(f"Prob correlation: {pearsonr(probs_struc[:, 0], probs_struc[:, 1])[0]}")

    # Only sample/set ground truth for params the fit actually optimizes.
    # get_bounds() also returns inactive params (e.g. StaticBeta's epsilon,
    # which defaults to 0 and isn't touched by fit()), so using it directly
    # here would inject real lapse-rate noise into the simulated data while
    # the fit still assumes epsilon=0 -- silently corrupting other estimates.
    _probe_policy = main_policy()
    active_names = _probe_policy.active_parameter_names()
    all_bounds = _probe_policy.get_bounds()
    policy1_bounds = {name: all_bounds[name] for name in active_names}
    child_seed_seqs = np.random.SeedSequence(args.seed).spawn(n_simulations)

    def sample_true_param(rng, name, lower, upper):
        if name in LOG_SCALE_PARAMS:
            return float(np.exp(rng.uniform(np.log(lower), np.log(upper))))
        return float(rng.uniform(lower, upper))

    def _true_value(policy, name):
        # Some params (e.g. 'beta') live on policy.beta_schedule.params
        # rather than policy.params -- policy1_bounds combines both.
        try:
            return policy.params[name]
        except KeyError:
            return policy.beta_schedule.params[name]

    def sample_true_policy(rng):
        policy = main_policy()
        param_dict = {}
        for param, (lower, upper) in policy1_bounds.items():
            param_dict[param] = sample_true_param(rng, param, lower, upper)
        policy.set_params(param_dict)
        return policy

    def run_simulation_struc_unstruc(seed_seq):
        rng = default_rng(seed_seq)
        policy1 = sample_true_policy(rng)

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

        df = pd.DataFrame()
        df["param"] = list(policy1_bounds.keys())
        df["true_value"] = [
            _true_value(policy1, param) for param in policy1_bounds.keys()
        ]
        df["estimated_value_unstruc"] = [
            model_unstruc.params[param] for param in policy1_bounds.keys()
        ]
        df["estimated_value_struc"] = [
            model_struc.params[param] for param in policy1_bounds.keys()
        ]
        return df

    def run_simulation_tier(sim_id, seed_seq):
        rng = default_rng(seed_seq)
        policy1 = sample_true_policy(rng)
        probs = generate_probs_tiers(args.n_blocks_per_tier, rng)

        # Every simulated block is its own session (policy resets between
        # blocks), so tier subsets can be fit with reset_mode="session".
        task = DecisionModel.simulate_policy(
            policy=policy1,
            reward_schedule=probs,
            min_trials_per_block=min_trials_per_block,
            prob_switch=prob_switch,
            seed=rng.integers(2**32),
        )
        trial_n_high = n_high_arms(task.probs)
        scope_tasks = {"all": task}
        for tier, n_high in TIER_N_HIGH.items():
            scope_tasks[tier] = task._filtered(trial_n_high == n_high)

        param_names = list(policy1_bounds.keys())
        true_values = [_true_value(policy1, param) for param in param_names]
        dfs = []
        for scope, scope_task in scope_tasks.items():
            model = DecisionModel(
                task=scope_task, policy=main_policy(), reset_mode="session"
            )
            model.fit(**fit_kwargs)
            dfs.append(
                pd.DataFrame(
                    {
                        "sim_id": sim_id,
                        "scope": scope,
                        "n_trials": len(scope_task.choices),
                        "param": param_names,
                        "true_value": true_values,
                        "estimated_value": [model.params[p] for p in param_names],
                    }
                )
            )
        return pd.concat(dfs, ignore_index=True)

    def run_simulation_tier_params(sim_id, seed_seq):
        rng = default_rng(seed_seq)
        probs = generate_probs_tiers(args.n_blocks_per_tier, rng)
        probs_n_high = n_high_arms(probs)

        # One independent true param set per tier. Each simulated block is its
        # own session (policy resets between blocks), so stacking the per-tier
        # tasks gives a valid mixed task for reset_mode="session" fits.
        param_names = list(policy1_bounds.keys())
        true_values = {}
        scope_tasks = {}
        for tier, n_high in TIER_N_HIGH.items():
            tier_policy = sample_true_policy(rng)
            true_values[tier] = [_true_value(tier_policy, p) for p in param_names]
            scope_tasks[tier] = DecisionModel.simulate_policy(
                policy=tier_policy,
                reward_schedule=probs[probs_n_high == n_high],
                min_trials_per_block=min_trials_per_block,
                prob_switch=prob_switch,
                seed=rng.integers(2**32),
            )
        scope_tasks = {"all": concat_tasks(scope_tasks.values()), **scope_tasks}

        dfs = []
        for scope, scope_task in scope_tasks.items():
            model = DecisionModel(
                task=scope_task, policy=main_policy(), reset_mode="session"
            )
            model.fit(**fit_kwargs)
            df = pd.DataFrame(
                {
                    "sim_id": sim_id,
                    "scope": scope,
                    "n_trials": len(scope_task.choices),
                    "param": param_names,
                    # 'all' has no single ground truth; compare it against
                    # the true_<tier> columns instead.
                    "true_value": true_values.get(scope, np.nan),
                    "estimated_value": [model.params[p] for p in param_names],
                }
            )
            for tier in TIER_N_HIGH:
                df[f"true_{tier}"] = true_values[tier]
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    if args.design == "tier_params":
        results = Parallel(n_jobs=args.n_jobs_subject)(
            delayed(run_simulation_tier_params)(i, child_seed_seqs[i])
            for i in range(n_simulations)
        )
        save_name = f"param_recovery_tier_params_{args.policy.lower()}"
    elif args.design == "tier":
        results = Parallel(n_jobs=args.n_jobs_subject)(
            delayed(run_simulation_tier)(i, child_seed_seqs[i])
            for i in range(n_simulations)
        )
        save_name = f"param_recovery_tier_{args.policy.lower()}"
    else:
        results = Parallel(n_jobs=args.n_jobs_subject)(
            delayed(run_simulation_struc_unstruc)(child_seed_seqs[i])
            for i in range(n_simulations)
        )
        save_name = f"param_recovery_{args.policy.lower()}"

    recovery_df = pd.concat(results, ignore_index=True)
    mab_subjects.GroupData().save(recovery_df, save_name, write_stub=False)


if __name__ == "__main__":
    main()
