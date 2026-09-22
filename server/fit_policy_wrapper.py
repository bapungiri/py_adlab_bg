from pathlib import Path
import mab_subjects

from banditpy.models.policy import (
    # Qlearn,
    # QlearnDiff,
    # QlearnRegimeDiffStays,
    Qlearn2Regime,
    # MoARegime,
    # QlearnHierarchical,
    # QlearnAdaptiveLR,
    # StateInference,
    # ThompsonSplit,
    # ThompsonShared,
    # BayesianUCB,
    StaticBeta,
)
from banditpy.models.optim import OptunaOptimizer
from fit_policy_core import fit_experiments

# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------
# EXPS = (
#     mab_subjects.unstruc.p8020_good_intact_sess
#     + mab_subjects.struc.p8020_good_intact_sess
#     + mab_subjects.unstruc_rnn.p8020_good_rnn_sess
#     + mab_subjects.struc_rnn.p8020_good_rnn_sess
# )

EXPS = (
    mab_subjects.unstruc.p8020_good_intact_sess
    + mab_subjects.struc.p8020_good_intact_sess
    # + mab_subjects.unstruc.p8020_lesion_OFC_post_sess
    # + mab_subjects.struc.p8020_lesion_OFC_post_sess
    # + mab_subjects.unstruc.p8020_lesion_mPFC_post_sess
    # + mab_subjects.struc.p8020_lesion_mPFC_post_sess
    # + mab_subjects.unstruc_rnn.p8020_good_rnn_sess
    # + mab_subjects.struc_rnn.p8020_good_rnn_sess
)


FIT_KWARGS = {
    "n_starts": 5,
    "n_jobs": 5,
    "early_stop": False,
    # "es_warmup_trials": 3000,
    # "es_check_every": 250,
    # "es_slack": 0.01,
    "progress": False,
}

POLICIES = [
    # Qlearn,
    # QlearnAdaptiveLR,
    # QlearnDiff,
    # QlearnRegimeDiffStays,
    # MoARegime,
    Qlearn2Regime,
    # QlearnHierarchical,
    # BayesianUCB,
    # ThompsonSplit2Arm,
    # ThompsonShared,
    # StateInference,
]
OPTIMIZER = OptunaOptimizer(n_trials=80)

PARALLEL_JOBS = len(EXPS)
FILTER_BY_DATETIME = True  # False for lesion, True for intact-experts
SAVE_NAME = "fit_Qlearn2Regime_policy_combinations"
FALLBACK_DIR = Path("/mnt/pve/Homes/bapun/Data/results")


# ----------------------------------------------------------
# Run
# ----------------------------------------------------------
params_df = fit_experiments(
    exps=EXPS,
    policies=POLICIES,
    fit_kwargs=FIT_KWARGS,
    optimizer=OPTIMIZER,
    n_jobs=PARALLEL_JOBS,
    filter_by_datetime=FILTER_BY_DATETIME,
)

try:
    mab_subjects.GroupData().save(params_df, SAVE_NAME, write_stub=False)
except Exception:
    params_df.to_csv(FALLBACK_DIR / f"{SAVE_NAME}.csv", index=False)
