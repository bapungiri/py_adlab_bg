import argparse
import os
from pathlib import Path
import mab_subjects

from banditpy.models.policy import (
    Qlearn,
    # QlearnDiff,
    # QlearnRegimeDiffStays,
    # Qlearn2Regime,
    # MoARegime,
    # QlearnHierarchical,
    # QlearnAdaptiveLR,
    # StateInference,
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
    policies=[Qlearn],
    optimizer=OptunaOptimizer(n_trials=80),
    fit_kwargs={"n_starts": 5, "n_jobs": 5, "early_stop": False, "progress": False},
    require_expert=None,  # legacy: intact sessions skip their first 30 days
)
# 6-param Qlearn with L-BFGS-B: parameter recovery showed 10 starts are
# needed for this model (5 left ~9% of fits unconverged).
LBFGS_STICKY = dict(
    policies=[QlearnSticky],
    optimizer=LBFGSOptimizer(),
    fit_kwargs={"n_starts": 10, "n_jobs": 10, "early_stop": False, "progress": False},
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
    # mab_subjects.unstruc.p8020_good_intact_sess
    # + mab_subjects.struc.p8020_good_intact_sess
    # + mab_subjects.unstruc.p8020_lesion_OFC_post_sess
    # + mab_subjects.struc.p8020_lesion_OFC_post_sess
    # + mab_subjects.unstruc_rnn.p8020_good_rnn_sess
    # + mab_subjects.struc_rnn.p8020_good_rnn_sess
}

parser = argparse.ArgumentParser()
parser.add_argument("--preset", choices=PRESETS, default="lesion_mPFC")
args = parser.parse_args()

PRESET = PRESETS[args.preset]
EXPS = PRESET["exps"]()
SAVE_NAME = PRESET["save_name"]
FIT_KWARGS = PRESET["fit_kwargs"]

# Subjects fit in parallel: as many as the SLURM allocation fits, given each
# subject uses fit_kwargs["n_jobs"] cores for its optimizer starts.
_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", FIT_KWARGS["n_jobs"] * len(EXPS)))
PARALLEL_JOBS = max(1, min(len(EXPS), _cpus // FIT_KWARGS["n_jobs"]))
print(
    f"Preset {args.preset}: {len(EXPS)} sessions -> {SAVE_NAME}; "
    f"{PARALLEL_JOBS} in parallel x {FIT_KWARGS['n_jobs']} cores"
)

FILTER_BY_DATETIME = True  # legacy 30-day rule, only when require_expert is None
FALLBACK_DIR = Path("/mnt/pve/Homes/bapun/Data/results")


# ----------------------------------------------------------
# Run
# ----------------------------------------------------------
params_df = fit_experiments(
    exps=EXPS,
    policies=PRESET["policies"],
    fit_kwargs=FIT_KWARGS,
    optimizer=PRESET["optimizer"],
    n_jobs=PARALLEL_JOBS,
    filter_by_datetime=FILTER_BY_DATETIME,
    require_expert=PRESET["require_expert"],
)

try:
    mab_subjects.GroupData().save(params_df, SAVE_NAME, write_stub=False)
except Exception:
    params_df.to_csv(FALLBACK_DIR / f"{SAVE_NAME}.csv", index=False)
