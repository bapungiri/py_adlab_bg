import argparse
import fnmatch
import os
from pathlib import Path
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
from banditpy.models.optim import LBFGSOptimizer, OptunaOptimizer
from scipy.stats import pearsonr
from numpy.random import default_rng
from joblib import Parallel, delayed

# Per-task results of SLURM array runs (--task-index), combined by --merge.
PARTIALS_ROOT = Path(
    os.environ.get(
        "RECOVERY_PARTIALS_DIR", "/mnt/pve/Homes/bapun/Data/results/partials"
    )
)

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


# Tier proportions that generate_probs_2arm(frac_impurity=0.2) produces, used
# by --design tier_mix to build structured/unstructured tasks from tiers.
TIER_MIX = {
    "unstruc": {"low_low": 0.20, "high_low": 0.60, "high_high": 0.20},
    "struc": {"low_low": 0.05, "high_low": 0.90, "high_high": 0.05},
}


def generate_probs_tier_mix(n_blocks, proportions, rng):
    """'n_blocks' blocks split across tiers by 'proportions', shuffled.

    Like generate_probs_tiers, pairs are drawn uniformly within each tier,
    so (unlike generate_probs_2arm) high_low blocks are not specifically
    the pairs summing to 1.
    """
    pairs = np.array([(a, b) for a in ARM_PROBS for b in ARM_PROBS if a != b])
    pairs_n_high = n_high_arms(pairs)
    counts = {t: int(round(n_blocks * p)) for t, p in proportions.items()}
    counts["high_low"] += n_blocks - sum(counts.values())  # rounding remainder
    blocks = []
    for tier, n in counts.items():
        tier_pairs = pairs[pairs_n_high == TIER_N_HIGH[tier]]
        blocks.append(tier_pairs[rng.integers(len(tier_pairs), size=n)])
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


# Qlearn variants: params to enable/disable on top of the class defaults.
# Each variant saves to its own GroupData basename ('_<variant>' suffix),
# so versions/cleanup of one variant never touch another.
QLEARN_VARIANTS = {
    "default": {"enable": [], "disable": []},  # alpha_c, alpha_u, bias, beta
    "nobias": {"enable": [], "disable": ["bias"]},  # bias fixed at 0
    "sticky": {"enable": ["alpha_h", "sticky"], "disable": []},  # + perseverance
}


def make_policy_factory(policy_cls, variant):
    if variant != "default" and policy_cls is not Qlearn:
        raise ValueError(f"--variant {variant} is only defined for Qlearn")
    spec = QLEARN_VARIANTS[variant]

    def make_policy():
        policy = policy_cls()
        for name in spec["enable"]:
            getattr(policy.params, name).enable()
        for name in spec["disable"]:
            getattr(policy.params, name).disable()
        return policy

    return make_policy


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
        choices=[
            "struc_unstruc",
            "tier",
            "tier_params",
            "struc_unstruc_tier",
            "tier_mix",
        ],
        default="struc_unstruc",
        help=(
            "struc_unstruc: fit structured and unstructured schedules separately; "
            "tier: simulate one mixed low-low/high-low/high-high schedule and "
            "fit all blocks plus each tier separately; "
            "tier_params: like tier, but each tier is simulated with its own "
            "true params; "
            "struc_unstruc_tier: structured and unstructured tasks from "
            "generate_probs_2arm, each fit on all blocks and per tier; "
            "tier_mix: same, but each task is built by mixing tier blocks in "
            "TIER_MIX proportions (high_low pairs not restricted to sum-to-1)"
        ),
    )
    parser.add_argument(
        "--n-blocks-per-task",
        type=int,
        default=200,
        help="Blocks per task (--design struc_unstruc_tier / tier_mix)",
    )
    parser.add_argument(
        "--variant",
        choices=list(QLEARN_VARIANTS),
        default="default",
        help="Qlearn variant (which params are fitted); non-default adds a suffix to the save name",
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
    parser.add_argument(
        "--true-range",
        action="append",
        default=[],
        metavar="NAME=LO,HI",
        help="draw true values of NAME (glob allowed, e.g. 'beta_q*') from [LO, HI] "
        "instead of the fit bounds; repeatable",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="suffix for the save name (letters, digits, underscores), e.g. truesticky0to5",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="SLURM array mode: simulate and fit only subject i, write a partial result",
    )
    parser.add_argument(
        "--merge",
        metavar="RUN_ID",
        default=None,
        help="merge the partial results of array run RUN_ID into one GroupData save",
    )
    parser.add_argument(
        "--optimizer",
        choices=["optuna", "lbfgs"],
        default="optuna",
        help="optuna: TPE, 80 trials per start with early stopping; "
        "lbfgs: L-BFGS-B without early stopping (pruned NLLs would corrupt its gradients)",
    )
    parser.add_argument(
        "--n-starts", type=int, default=5, help="Optimizer restarts per fit"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    main_policy = make_policy_factory(POLICY_REGISTRY[args.policy], args.variant)

    n_simulations = args.max_subjects

    n_sessions = 200
    min_trials_per_block = 100
    prob_switch = 0.02
    # Rate/scale-like params span orders of magnitude, so a linear-uniform
    # draw over-samples large values relative to small ones; log-uniform
    # gives equal weight to equal ratios instead. Used both for the true
    # values and for the Optuna search.
    LOG_SCALE_PARAMS = {"beta", "beta_0", "beta_1", "beta_2"}

    if args.optimizer == "optuna":
        fit_kwargs = {
            "optimizer": OptunaOptimizer(n_trials=80, log_params=LOG_SCALE_PARAMS),
            "early_stop": True,
            "es_warmup_trials": 3000,
            "es_check_every": 250,
            "es_slack": 0.01,
        }
    else:
        fit_kwargs = {"optimizer": LBFGSOptimizer(), "early_stop": False}
    fit_kwargs.update(n_starts=args.n_starts, n_jobs=args.n_jobs_inner)

    def fit_columns(model, true_theta=None):
        """How the fit was run (model.fit_info) plus its NLL, and the NLL at
        the true params when there is a single ground truth."""
        cols = dict(model.fit_info, nll=float(model.nll))
        if true_theta is not None:
            cols["nll_true"] = float(model._nll(np.asarray(true_theta)))
        return cols

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
    # True values can be drawn from narrower ranges than the fit's bounds
    # (--true-range NAME=LO,HI, NAME may be a glob such as 'beta_q*').
    true_ranges = {}
    for spec in args.true_range:
        pattern, _, rng_txt = spec.partition("=")
        lo, hi = (float(x) for x in rng_txt.split(","))
        matched = fnmatch.filter(active_names, pattern)
        if not matched:
            raise SystemExit(
                f"--true-range {spec}: no fitted parameter matches '{pattern}'"
            )
        for name in matched:
            b_lo, b_hi = policy1_bounds[name]
            if lo < b_lo or hi > b_hi or lo >= hi:
                raise SystemExit(
                    f"--true-range {spec}: must lie within fit bounds [{b_lo}, {b_hi}]"
                )
            true_ranges[name] = (lo, hi)
    sample_ranges = {**policy1_bounds, **true_ranges}
    true_ranges_txt = ", ".join(
        f"{n}: [{lo:g}, {hi:g}]" for n, (lo, hi) in sample_ranges.items()
    )
    print("true values drawn from:", true_ranges_txt)
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
        for param, (lower, upper) in sample_ranges.items():
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
        true_theta = df["true_value"].to_numpy()
        for grp, model in (("unstruc", model_unstruc), ("struc", model_struc)):
            cols = fit_columns(model, true_theta)
            df[f"nll_{grp}"] = cols.pop("nll")
            df[f"nll_true_{grp}"] = cols.pop("nll_true")
        for k, v in cols.items():  # fit settings, identical for both fits
            df[k] = v
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
                        **fit_columns(model, true_values),
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
                    **fit_columns(model, true_values.get(scope)),
                }
            )
            for tier in TIER_N_HIGH:
                df[f"true_{tier}"] = true_values[tier]
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def run_simulation_task_tiers(sim_id, seed_seq):
        """Structured and unstructured tasks, each fit on all blocks and per tier.

        struc_unstruc_tier: one true param set; tasks from generate_probs_2arm
        (its own unseeded RNG, so schedules aren't reproducible with --seed).
        tier_mix: one true param set *per tier*, shared by both tasks; each
        task mixes tier blocks in TIER_MIX proportions. The 'all' fit then has
        no single truth; compare it with the true_<tier> columns.
        """
        rng = default_rng(seed_seq)
        param_names = list(policy1_bounds.keys())
        per_tier = args.design == "tier_mix"
        if per_tier:
            tier_policies = {tier: sample_true_policy(rng) for tier in TIER_N_HIGH}
            true_values = {
                tier: [_true_value(pol, p) for p in param_names]
                for tier, pol in tier_policies.items()
            }
        else:
            policy1 = sample_true_policy(rng)
            truth = [_true_value(policy1, p) for p in param_names]
            probs_unstruc, probs_struc = generate_probs_2arm(
                N=args.n_blocks_per_task, frac_impurity=0.2
            )
            schedules = {"unstruc": probs_unstruc, "struc": probs_struc}

        def simulate(policy, probs):
            return DecisionModel.simulate_policy(
                policy=policy,
                reward_schedule=probs,
                min_trials_per_block=min_trials_per_block,
                prob_switch=prob_switch,
                seed=rng.integers(2**32),
            )

        dfs = []
        for task_name in ("unstruc", "struc"):
            if per_tier:
                # Every block is its own session (policy resets between blocks),
                # so stacking per-tier simulations is a valid mixed task.
                probs = generate_probs_tier_mix(
                    args.n_blocks_per_task, TIER_MIX[task_name], rng
                )
                probs_n_high = n_high_arms(probs)
                task = concat_tasks(
                    simulate(tier_policies[tier], probs[probs_n_high == n_high])
                    for tier, n_high in TIER_N_HIGH.items()
                    if (probs_n_high == n_high).any()
                )
            else:
                task = simulate(policy1, schedules[task_name])

            trial_n_high = n_high_arms(task.probs)
            scope_tasks = {"all": task}
            for tier, n_high in TIER_N_HIGH.items():
                if (trial_n_high == n_high).any():
                    scope_tasks[tier] = task._filtered(trial_n_high == n_high)

            for scope, scope_task in scope_tasks.items():
                model = DecisionModel(
                    task=scope_task, policy=main_policy(), reset_mode="session"
                )
                model.fit(**fit_kwargs)
                scope_truth = true_values.get(scope) if per_tier else truth
                df = pd.DataFrame(
                    {
                        "sim_id": sim_id,
                        "task": task_name,
                        "scope": scope,
                        "n_blocks": len(np.unique(scope_task.session_ids)),
                        "n_trials": len(scope_task.choices),
                        "param": param_names,
                        "true_value": (
                            scope_truth if scope_truth is not None else np.nan
                        ),
                        "estimated_value": [model.params[p] for p in param_names],
                        **fit_columns(model, scope_truth),
                    }
                )
                if per_tier:
                    for tier in TIER_N_HIGH:
                        df[f"true_{tier}"] = true_values[tier]
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    if args.design in ("struc_unstruc_tier", "tier_mix"):
        simulate_one = run_simulation_task_tiers
        save_name = f"param_recovery_{args.design}_{args.policy.lower()}"
    elif args.design == "tier_params":
        simulate_one = run_simulation_tier_params
        save_name = f"param_recovery_tier_params_{args.policy.lower()}"
    elif args.design == "tier":
        simulate_one = run_simulation_tier
        save_name = f"param_recovery_tier_{args.policy.lower()}"
    else:

        def simulate_one(i, seed_seq):
            return run_simulation_struc_unstruc(seed_seq).assign(sim_id=i)

        save_name = f"param_recovery_{args.policy.lower()}"
    if args.variant != "default":
        save_name = f"{save_name}_{args.variant}"
    if args.tag:
        save_name = f"{save_name}_{args.tag}"

    def run_one(i):
        return simulate_one(i, child_seed_seqs[i]).assign(true_ranges=true_ranges_txt)

    # SLURM array runs write one file per simulated subject, then --merge
    # combines them into a single GroupData save.
    partials_dir = PARTIALS_ROOT / save_name

    if args.merge is not None:
        run_dir = partials_dir / args.merge
        files = sorted(run_dir.glob("*.pkl"))
        missing = sorted(set(range(n_simulations)) - {int(f.stem) for f in files})
        if missing:
            raise SystemExit(
                f"Not merging {run_dir}: {len(missing)} missing tasks, e.g. {missing[:10]}"
            )
        recovery_df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
        print(f"Merging {len(files)} partials from {run_dir} -> {save_name}")
        mab_subjects.GroupData().save(recovery_df, save_name, write_stub=False)
        return

    if args.task_index is not None:
        run_dir = partials_dir / os.environ.get("SLURM_ARRAY_JOB_ID", "local")
        run_dir.mkdir(parents=True, exist_ok=True)
        df = run_one(args.task_index)
        out = run_dir / f"{args.task_index:04d}.pkl"
        df.to_pickle(out)
        print(f"Wrote {out}")
        return

    results = Parallel(n_jobs=args.n_jobs_subject)(
        delayed(run_one)(i) for i in range(n_simulations)
    )
    recovery_df = pd.concat(results, ignore_index=True)
    mab_subjects.GroupData().save(recovery_df, save_name, write_stub=False)


if __name__ == "__main__":
    main()
