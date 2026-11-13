import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import mab_subjects

from banditpy.models.policy import (
    Qlearn,
    QlearnCost,
    BayesForager,
    MVTForager,
    MVTForagerKappa,
    MVTForagerSatiety,
    # QlearnDiff,
    # QlearnRegimeDiffStays,
    Qlearn2Regime,
    # MoARegime,
    # QlearnHierarchical,
    QlearnAdaptiveLR,
    QlearnAdaptiveBeta,
    QlearnDynamicSticky,
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


def QlearnStickyDecay():
    """Qlearn with sticky (outside beta) and decay of the unchosen value toward
    0.5: alpha_c, alpha_u, bias, alpha_h, sticky, decay, beta. beta is bounded
    to [1, 10]: below 1 the value gap (at most 1) barely moves choices."""
    policy = QlearnSticky()
    policy.params.decay.enable()
    policy.beta_schedule.set_bounds(beta=(1.0, 10.0))
    return policy


def QlearnStickyBeta1():
    """6-param Qlearn (sticky outside beta) with beta in [1, 10], as in
    QlearnStickyDecay, so the two fits nest and compare directly."""
    policy = QlearnSticky()
    policy.beta_schedule.set_bounds(beta=(1.0, 10.0))
    return policy


def QlearnCostSticky():
    """QlearnCost default variant: the 6 Qlearn params + alpha_rho (rho0 = 0, so
    alpha_rho = 0 is QlearnStickyBeta1), sticky outside beta, beta in [1, 10]."""
    policy = QlearnCost()
    policy.beta_schedule.set_bounds(beta=(1.0, 10.0))
    return policy


def QlearnAdaptiveLRSticky():
    """QlearnAdaptiveLR: the 6 Qlearn params (sticky outside beta) + reward-rate
    scaling of both learning rates (kappa_c, kappa_u, w_r); kappa_c = kappa_u = 0
    is QlearnStickyBeta1. beta in [1, 10]."""
    policy = QlearnAdaptiveLR()
    policy.beta_schedule.set_bounds(beta=(1.0, 10.0))
    return policy


def QlearnAdaptiveBetaSticky():
    """QlearnAdaptiveBeta: the 6 Qlearn params (sticky outside beta) + a reward-rate gain on
    the whole decision (kappa_beta, w_r); kappa_beta = 0 is QlearnStickyBeta1. beta in [1, 10]."""
    policy = QlearnAdaptiveBeta()
    policy.beta_schedule.set_bounds(beta=(1.0, 10.0))
    return policy


def MVTForagerSticky():
    """MVTForager + a choice (perseverance) trace on the switch logit (alpha_h, sticky);
    sticky = 0 is the six-parameter MVTForager."""
    policy = MVTForager()
    policy.params.alpha_h.enable()
    policy.params.sticky.enable()
    return policy


def MVTForagerR0Fixed():
    """MVTForager with r0 fixed at 0 (a newly chosen port starts from r_hat = 0): five
    fitted params. The animals' r0 fits sat at 0 and r0 recovered poorly."""
    policy = MVTForager()
    policy.params.r0.default = 0.0
    policy.params.r0.disable()
    return policy


def MVTForagerKappaR0Fixed():
    """MVTForagerKappa (switch logit kappa * rho - beta * r_hat + gamma) with r0 fixed
    at 0, as in MVTForagerR0Fixed: six fitted params (alpha_l, alpha_rho, beta, kappa,
    gamma, rho0). kappa = beta is MVTForagerR0Fixed."""
    policy = MVTForagerKappa()
    policy.params.r0.default = 0.0
    policy.params.r0.disable()
    return policy

def MVTForagerStickyR0Fixed():
    """MVTForagerR0Fixed + the choice (perseverance) trace (alpha_h, sticky): seven fitted
    params. sticky = 0 is MVTForagerR0Fixed."""
    policy = MVTForagerR0Fixed()
    policy.params.alpha_h.enable()
    policy.params.sticky.enable()
    return policy


def MVTForagerKappaSticky():
    """MVTForagerKappaR0Fixed + the choice (perseverance) trace (alpha_h, sticky): eight
    fitted params. sticky = 0 is MVTForagerKappaR0Fixed."""
    policy = MVTForagerKappaR0Fixed()
    policy.params.alpha_h.enable()
    policy.params.sticky.enable()
    return policy


def MVTForagerConstRho():
    """MVTForager with a constant environment rate: alpha_rho = 0, gamma = 0, r0 = 0, so
    three fitted params (alpha_l, beta, rho0). P(switch) = sigmoid(beta * (rho0 - r_hat)),
    so rho0 is the leave threshold (P(switch) = 0.5 where r_hat = rho0). With rho constant,
    a free gamma would only trade off with beta * rho0, so gamma is dropped. rho0 may be
    negative (bounds -1 to 1): a threshold below 0 means switching never reaches 0.5."""
    policy = MVTForagerR0Fixed()
    policy.params.alpha_rho.default = 0.0
    policy.params.alpha_rho.disable()
    policy.params.gamma.default = 0.0
    policy.params.gamma.disable()
    policy.params.rho0.set_bounds(-1.0, 1.0)
    return policy


def QlearnDynamicStickyFit():
    """QlearnDynamicSticky: the 6 Qlearn params (sticky outside beta) + a within-window
    ramp of the perseverance weight (k0, tau); k0 = 1 is QlearnStickyBeta1. beta in [1, 10]."""
    policy = QlearnDynamicSticky()
    policy.beta_schedule.set_bounds(beta=(1.0, 10.0))
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
# 24 starts: with 12 the fits split between two solutions (decay ~0, alpha_u < 0 vs
# decay ~0.7, alpha_u > 0) and often ended worse than the nested 6-param fit.
LBFGS_STICKY_DECAY = dict(
    LBFGS_STICKY,
    policies=[QlearnStickyDecay],
    fit_kwargs={"n_starts": 24, "n_jobs": 6, "early_stop": False, "progress": False},
)
# Same 24 starts for the models compared with sticky + decay.
LBFGS_STICKY_BETA1 = dict(LBFGS_STICKY_DECAY, policies=[QlearnStickyBeta1])
LBFGS_COST = dict(LBFGS_STICKY_DECAY, policies=[QlearnCostSticky])
# Stay/leave forager, all six params (alpha_l, alpha_rho, beta, gamma, r0, rho0).
LBFGS_MVT = dict(LBFGS_STICKY_DECAY, policies=[MVTForager])
LBFGS_MVT_STICKY = dict(LBFGS_STICKY_DECAY, policies=[MVTForagerSticky])
LBFGS_MVT_R0FIXED = dict(LBFGS_STICKY_DECAY, policies=[MVTForagerR0Fixed])
LBFGS_MVT_CONSTRHO = dict(LBFGS_STICKY_DECAY, policies=[MVTForagerConstRho])
LBFGS_MVT_KAPPA = dict(LBFGS_STICKY_DECAY, policies=[MVTForagerKappaR0Fixed])
LBFGS_MVT_STICKY_R0FIXED = dict(LBFGS_STICKY_DECAY, policies=[MVTForagerStickyR0Fixed])
LBFGS_MVT_KAPPA_STICKY = dict(LBFGS_STICKY_DECAY, policies=[MVTForagerKappaSticky])
# Stay/leave on the Beta belief that the current port pays less than rho (tau, n0, alpha_rho, beta, gamma, rho0).
LBFGS_BAYES_FORAGER = dict(LBFGS_STICKY_DECAY, policies=[BayesForager])
# rho = rho0 * exp(-n / tau_rho), n = rewards earned in the window (clock='rewards').
LBFGS_MVT_SATIETY = dict(LBFGS_STICKY_DECAY, policies=[MVTForagerSatiety])
LBFGS_ADAPTLR = dict(LBFGS_STICKY_DECAY, policies=[QlearnAdaptiveLRSticky])
LBFGS_ADAPTBETA = dict(LBFGS_STICKY_DECAY, policies=[QlearnAdaptiveBetaSticky])
LBFGS_DYNSTICKY = dict(LBFGS_STICKY_DECAY, policies=[QlearnDynamicStickyFit])
LBFGS_Q2R_STICKY = dict(
    policies=[Qlearn2RegimeSticky],
    optimizer=LBFGSOptimizer(),
    fit_kwargs={"n_starts": 24, "n_jobs": 6, "early_stop": False, "progress": False},
)
# Two-state Bayesian inference of which port is good (c, y, b0, beta).
# 6-param Qlearn fitted on dependent blocks only (probabilities sum to 1).
LBFGS_STICKY_DEP = dict(LBFGS_STICKY, scopes=("dependent",))
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
    # Sticky + decay, beta in [1, 10]; same sessions and trimming as above.
    "sticky_decay_p8020_intact": dict(
        LBFGS_STICKY_DECAY,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_qlearn_sticky_decay_p8020_intact",
        require_expert=True,
    ),
    # 6-param Qlearn and QlearnCost with sticky outside beta and beta in [1, 10]:
    # the clean comparison set for sticky_decay_p8020_intact (same sessions).
    "sticky_b1_p8020_intact": dict(
        LBFGS_STICKY_BETA1,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_qlearn_sticky_b1_p8020_intact",
        require_expert=True,
    ),
    "cost_p8020_intact": dict(
        LBFGS_COST,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_qlearncost_p8020_intact",
        require_expert=True,
    ),
    "mvt_p8020_intact": dict(
        LBFGS_MVT,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_mvtforager_p8020_intact",
        require_expert=True,
    ),
    "mvtr0fixed_p8020_intact": dict(
        LBFGS_MVT_R0FIXED,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_mvtforager_r0fixed_p8020_intact",
        require_expert=True,
    ),
    "mvtconstrho_p8020_intact": dict(
        LBFGS_MVT_CONSTRHO,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_mvtforager_constrho_p8020_intact",
        require_expert=True,
    ),
    "mvtkappa_p8020_intact": dict(
        LBFGS_MVT_KAPPA,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_mvtforagerkappa_p8020_intact",
        require_expert=True,
    ),
    "bayesforager_p8020_intact": dict(
        LBFGS_BAYES_FORAGER,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_bayesforager_p8020_intact",
        require_expert=True,
    ),
    "mvtstickyr0fixed_p8020_intact": dict(
        LBFGS_MVT_STICKY_R0FIXED,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_mvtforagersticky_r0fixed_p8020_intact",
        require_expert=True,
    ),
    "mvtkappasticky_p8020_intact": dict(
        LBFGS_MVT_KAPPA_STICKY,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_mvtforagerkappasticky_p8020_intact",
        require_expert=True,
    ),
    "mvtsticky_p8020_intact": dict(
        LBFGS_MVT_STICKY,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_mvtforagersticky_p8020_intact",
        require_expert=True,
    ),
    "mvtsatiety_p8020_intact": dict(
        LBFGS_MVT_SATIETY,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_mvtforagersatiety_p8020_intact",
        require_expert=True,
    ),
    "adaptlr_p8020_intact": dict(
        LBFGS_ADAPTLR,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_qlearnadaptivelr_p8020_intact",
        require_expert=True,
    ),
    "adaptbeta_p8020_intact": dict(
        LBFGS_ADAPTBETA,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_qlearnadaptivebeta_p8020_intact",
        require_expert=True,
    ),
    "dynsticky_p8020_intact": dict(
        LBFGS_DYNSTICKY,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_qlearndynamicsticky_p8020_intact",
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
    # OFC lesion (post) sessions only, kept whole; intact baselines are in
    # sticky_p8020_intact.
    "sticky_p8020_lesion_OFC": dict(
        LBFGS_STICKY,
        exps=lambda: mab_subjects.unstruc.p8020_lesion_OFC_post_sess
        + mab_subjects.struc.p8020_lesion_OFC_post_sess,
        save_name="fit_qlearn_sticky_p8020_lesion_OFC",
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
    # 6-param Qlearn, dependent blocks only, both groups.
    "sticky_dep_p8020_intact": dict(
        LBFGS_STICKY_DEP,
        exps=lambda: mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess,
        save_name="fit_qlearn_sticky_dependent_p8020_intact",
        require_expert=True,
    ),
    "sticky_dep_p8020_lesion_mPFC": dict(
        LBFGS_STICKY_DEP,
        exps=lambda: mab_subjects.unstruc.p8020_lesion_mPFC_post_sess
        + mab_subjects.struc.p8020_lesion_mPFC_post_sess,
        save_name="fit_qlearn_sticky_dependent_p8020_lesion_mPFC",
        require_expert=False,
    ),
    "sticky_dep_p8020_lesion_OFC": dict(
        LBFGS_STICKY_DEP,
        exps=lambda: mab_subjects.unstruc.p8020_lesion_OFC_post_sess
        + mab_subjects.struc.p8020_lesion_OFC_post_sess,
        save_name="fit_qlearn_sticky_dependent_p8020_lesion_OFC",
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
parser.add_argument(
    "--allow-missing",
    action="store_true",
    help="with --merge: save even if some sessions have no partial (they are listed)",
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
        if not args.allow_missing:
            sys.exit(f"Not merging {run_dir}: missing tasks {missing} ({names})")
        print(f"Merging without missing tasks {missing} ({names})")
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
        scopes=PRESET.get("scopes", ("all", "high_low")),
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
    scopes=PRESET.get("scopes", ("all", "high_low")),
)
save_groupdata(params_df)
