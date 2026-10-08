import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import mab_subjects

from banditpy.models.policy import (
    Qlearn,
    # QlearnDiff,
    # QlearnRegimeDiffStays,
    Qlearn2Regime,
    # MoARegime,
    # QlearnHierarchical,
    # QlearnAdaptiveLR,
    StateInference,
    # ThompsonSplit,
    # ThompsonShared,
    # BayesianUCB,
    StaticBeta,
)
from banditpy.models.optim import LBFGSOptimizer, OptunaOptimizer
from fit_policy_core import fit_experiments


def QlearnSticky():
    """Qlearn with all six params (alpha_c, alpha_u, bias, alpha_h, sticky, beta)."""
    policy = Qlearn()
    policy.params.alpha_h.enable()
    policy.params.sticky.enable()
    return policy


def QlearnNoSticky():
    """Qlearn without perseverance (alpha_c, alpha_u, bias, beta), as in the Optuna presets."""
    policy = Qlearn()
    policy.params.alpha_h.disable()
    policy.params.sticky.disable()
    return policy


def Qlearn2RegimeSticky():
    """Qlearn2Regime + shared perseverance (alpha_h, sticky) + learned b_init."""
    policy = Qlearn2Regime()
    policy.params.alpha_h.enable()
    policy.params.sticky.enable()
    policy.params.b_init.enable()
    return policy


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------
# EXPS = (
#     mab_subjects.unstruc.p8020_good_intact_sess
#     + mab_subjects.struc.p8020_good_intact_sess
#     + mab_subjects.unstruc_rnn.p8020_good_rnn_sess
#     + mab_subjects.struc_rnn.p8020_good_rnn_sess
# )

# Fit settings shared by presets unless a preset overrides them.
OPTUNA_DEFAULTS = dict(
    policies=[QlearnNoSticky],
    optimizer=OptunaOptimizer(n_trials=80),
    fit_kwargs={"n_starts": 5, "n_jobs": 5, "early_stop": False, "progress": False},
    require_expert=None,  # legacy: intact sessions skip their first 30 days
)
# 6-param Qlearn with L-BFGS-B: parameter recovery showed 10 starts are
# needed for this model (5 left ~9% of fits unconverged); 12 keeps it a
# multiple of the 6 CPUs per array task, so no worker idles in the last round.
LBFGS_STICKY = dict(
    policies=[QlearnSticky],
    optimizer=LBFGSOptimizer(),
    fit_kwargs={"n_starts": 12, "n_jobs": 6, "early_stop": False, "progress": False},
)
LBFGS_Q2R_STICKY = dict(
    policies=[Qlearn2RegimeSticky],
    optimizer=LBFGSOptimizer(),
    fit_kwargs={"n_starts": 24, "n_jobs": 6, "early_stop": False, "progress": False},
)
# Two-state Bayesian inference of which port is good (c, y, b0, beta).
LBFGS_SI = dict(
    policies=[StateInference],
    optimizer=LBFGSOptimizer(),
    fit_kwargs={"n_starts": 12, "n_jobs": 6, "early_stop": False, "progress": False},
)
# 11-param 2-regime mixture of agents: recovery with 10 starts left 37-49%
# of fits short of the truth, so use 24 (4 rounds on 6 CPUs).
LBFGS_Q2R = dict(
    policies=[Qlearn2Regime],
    optimizer=LBFGSOptimizer(),
    fit_kwargs={"n_starts": 24, "n_jobs": 6, "early_stop": False, "progress": False},
)

# Named configs, picked with --preset so several can run as separate SLURM
# jobs: sbatch job_fit_policy.slurm <preset>. Each gives the sessions, the
# GroupData save name and any overrides of OPTUNA_DEFAULTS.
PRESETS = {
    "lesion_mPFC": dict(
        OPTUNA_DEFAULTS,
        exps=lambda: mab_subjects.unstruc.p8020_lesion_mPFC_intact_post_sess
        + mab_subjects.struc.p8020_lesion_mPFC_intact_post_sess,
        save_name="fit_qlearn_high_low_lesion_mPFC",
    ),
    "p9505": dict(
        OPTUNA_DEFAULTS,
        exps=lambda: mab_subjects.unstruc.p9505_good_intact_sess
        + mab_subjects.struc.p9505_good_intact_sess,
        save_name="fit_qlearn_high_low_p9505",
    ),
    # Intact sessions trimmed from the expertise day (as perf_tier).
    "sticky_p8020_intact": dict(
        LBFGS_STICKY,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_qlearn_sticky_p8020_intact",
        require_expert=True,
    ),
    # Lesion sessions only, kept whole.
    "sticky_p8020_lesion_mPFC": dict(
        LBFGS_STICKY,
        exps=lambda: mab_subjects.unstruc.p8020_lesion_mPFC_post_sess
        + mab_subjects.struc.p8020_lesion_mPFC_post_sess,
        save_name="fit_qlearn_sticky_p8020_lesion_mPFC",
        require_expert=False,
    ),
    # Kept whole: 9505 data is still short.
    "sticky_p9505": dict(
        LBFGS_STICKY,
        exps=lambda: mab_subjects.unstruc.p9505_good_intact_sess
        + mab_subjects.struc.p9505_good_intact_sess,
        save_name="fit_qlearn_sticky_p9505",
        require_expert=False,
    ),
    # Qlearn2Regime on the same sessions and trimming as the sticky presets.
    "q2r_p8020_intact": dict(
        LBFGS_Q2R,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_qlearn2regime_p8020_intact",
        require_expert=True,
    ),
    "q2r_p8020_lesion_mPFC": dict(
        LBFGS_Q2R,
        exps=lambda: mab_subjects.unstruc.p8020_lesion_mPFC_post_sess
        + mab_subjects.struc.p8020_lesion_mPFC_post_sess,
        save_name="fit_qlearn2regime_p8020_lesion_mPFC",
        require_expert=False,
    ),
    "q2r_p9505": dict(
        LBFGS_Q2R,
        exps=lambda: mab_subjects.unstruc.p9505_good_intact_sess
        + mab_subjects.struc.p9505_good_intact_sess,
        save_name="fit_qlearn2regime_p9505",
        require_expert=False,
    ),
    # + perseverance and learned b_init (14 params).
    "q2r_sticky_p8020_intact": dict(
        LBFGS_Q2R_STICKY,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_qlearn2regime_sticky_p8020_intact",
        require_expert=True,
    ),
    "q2r_sticky_p8020_lesion_mPFC": dict(
        LBFGS_Q2R_STICKY,
        exps=lambda: mab_subjects.unstruc.p8020_lesion_mPFC_post_sess
        + mab_subjects.struc.p8020_lesion_mPFC_post_sess,
        save_name="fit_qlearn2regime_sticky_p8020_lesion_mPFC",
        require_expert=False,
    ),
    "q2r_sticky_p9505": dict(
        LBFGS_Q2R_STICKY,
        exps=lambda: mab_subjects.unstruc.p9505_good_intact_sess
        + mab_subjects.struc.p9505_good_intact_sess,
        save_name="fit_qlearn2regime_sticky_p9505",
        require_expert=False,
    ),
    # State inference on the same sessions and trimming as the sticky presets.
    "si_p8020_intact": dict(
        LBFGS_SI,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_state_inference_p8020_intact",
        require_expert=True,
    ),
    "si_p8020_lesion_mPFC": dict(
        LBFGS_SI,
        exps=lambda: mab_subjects.unstruc.p8020_lesion_mPFC_post_sess
        + mab_subjects.struc.p8020_lesion_mPFC_post_sess,
        save_name="fit_state_inference_p8020_lesion_mPFC",
        require_expert=False,
    ),
    "si_p9505": dict(
        LBFGS_SI,
        exps=lambda: mab_subjects.unstruc.p9505_good_intact_sess
        + mab_subjects.struc.p9505_good_intact_sess,
        save_name="fit_state_inference_p9505",
        require_expert=False,
    ),
    # mab_subjects.unstruc.p8020_good_intact_sess
    # + mab_subjects.struc.p8020_good_intact_sess
    # + mab_subjects.unstruc.p8020_lesion_OFC_post_sess
    # + mab_subjects.struc.p8020_lesion_OFC_post_sess
    # + mab_subjects.unstruc_rnn.p8020_good_rnn_sess
    # + mab_subjects.struc_rnn.p8020_good_rnn_sess
}

parser = argparse.ArgumentParser()
parser.add_argument("--preset", choices=PRESETS, default="lesion_mPFC")
parser.add_argument(
    "--task-index",
    type=int,
    default=None,
    help="SLURM array mode: fit only session i and write a partial result",
)
parser.add_argument(
    "--merge",
    metavar="RUN_ID",
    default=None,
    help="merge the partial results of array run RUN_ID into one GroupData save",
)
parser.add_argument(
    "--count", action="store_true", help="print the number of sessions and exit"
)
args = parser.parse_args()

PRESET = PRESETS[args.preset]
EXPS = PRESET["exps"]()
SAVE_NAME = PRESET["save_name"]
FIT_KWARGS = dict(PRESET["fit_kwargs"])
if args.count:
    print(len(EXPS))
    sys.exit(0)

FILTER_BY_DATETIME = True  # legacy 30-day rule, only when require_expert is None
FALLBACK_DIR = Path("/mnt/pve/Homes/bapun/Data/results")
# Array runs write one file per session here, then a merge job combines them.
PARTIALS_DIR = FALLBACK_DIR / "partials" / SAVE_NAME
_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", FIT_KWARGS["n_jobs"] * len(EXPS)))


def save_groupdata(df):
    try:
        mab_subjects.GroupData().save(df, SAVE_NAME, write_stub=False)
    except Exception:
        df.to_csv(FALLBACK_DIR / f"{SAVE_NAME}.csv", index=False)


# ----------------------------------------------------------
# Merge mode: combine an array run's partials
# ----------------------------------------------------------
if args.merge is not None:
    run_dir = PARTIALS_DIR / args.merge
    files = sorted(run_dir.glob("*.pkl"))
    done = {int(f.name.split("_", 1)[0]) for f in files}
    missing = sorted(set(range(len(EXPS))) - done)
    if missing:
        names = [EXPS[i].sub_name for i in missing]
        sys.exit(f"Not merging {run_dir}: missing tasks {missing} ({names})")
    params_df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    print(f"Merging {len(files)} partials from {run_dir} -> {SAVE_NAME}")
    save_groupdata(params_df)
    sys.exit(0)

# ----------------------------------------------------------
# Array mode: one session per task
# ----------------------------------------------------------
if args.task_index is not None:
    exp = EXPS[args.task_index]
    # Optimizer starts use the task's own CPUs.
    FIT_KWARGS["n_jobs"] = max(1, min(FIT_KWARGS["n_starts"], _cpus))
    print(
        f"Preset {args.preset}: task {args.task_index}/{len(EXPS) - 1} = {exp.sub_name} "
        f"({exp.data_tag}, {exp.lesion_tag}); {FIT_KWARGS['n_jobs']} cores"
    )
    params_df = fit_experiments(
        exps=[exp],
        policies=PRESET["policies"],
        fit_kwargs=FIT_KWARGS,
        optimizer=PRESET["optimizer"],
        n_jobs=1,
        filter_by_datetime=FILTER_BY_DATETIME,
        require_expert=PRESET["require_expert"],
    )
    run_dir = PARTIALS_DIR / os.environ.get("SLURM_ARRAY_JOB_ID", "local")
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / f"{args.task_index:03d}_{exp.sub_name}_{exp.lesion_tag}.pkl"
    params_df.to_pickle(out)
    print(f"Wrote {out}")
    sys.exit(0)

# ----------------------------------------------------------
# Single-job mode: all sessions in one allocation
# ----------------------------------------------------------
# Subjects fit in parallel: as many as the SLURM allocation fits, given each
# subject uses fit_kwargs["n_jobs"] cores for its optimizer starts.
PARALLEL_JOBS = max(1, min(len(EXPS), _cpus // FIT_KWARGS["n_jobs"]))
print(
    f"Preset {args.preset}: {len(EXPS)} sessions -> {SAVE_NAME}; "
    f"{PARALLEL_JOBS} in parallel x {FIT_KWARGS['n_jobs']} cores"
)
params_df = fit_experiments(
    exps=EXPS,
    policies=PRESET["policies"],
    fit_kwargs=FIT_KWARGS,
    optimizer=PRESET["optimizer"],
    n_jobs=PARALLEL_JOBS,
    filter_by_datetime=FILTER_BY_DATETIME,
    require_expert=PRESET["require_expert"],
)
save_groupdata(params_df)
