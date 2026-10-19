"""Build the commonly used GroupData products over every good session.

Each product runs once over all good animals (every paradigm and lesion
condition) plus the good RNN models; plot cells then narrow it down with
'mab_builders.select.select' instead of rebuilding per subset.

    python -m mab_builders.build_all                     # every product
    python -m mab_builders.build_all perf_tier --n-jobs 8
    python -m mab_builders.build_all --list

From a notebook: build("perf_tier", n_jobs=8).
"""

import argparse
import time

import mab_subjects

from .learning import tau_tier
from .performance import perf_probability_matrix, perf_tier
from .switching import (
    swp_around_flip,
    swp_by_block,
    swp_by_prev_best_arm,
    swp_probability_matrix,
    swp_tier,
)
from .task import prob_combinations

RNN_PARADIGMS = ("P100", "P9505", "P9010", "P8020")

# Cosyne 2026 abstract: 10-trial windows over the first 30 trials of each
# session, intact sessions trimmed from the expertise day
ABSTRACT_WINDOWED = dict(
    require_expert="auto",
    kwargs_trial_filter=dict(min_trials=100, clip_max=30),
)

# save name -> (builder, params). Params must be valid for every session type
# (intact, lesion, RNN); add a separate entry (another save name) for variants.
PRODUCTS = {
    "perf_tier": (perf_tier, dict(require_expert=False)),
    # intact sessions trimmed from the expertise day, lesion/RNN kept whole
    "perf_tier_expert": (perf_tier, dict(require_expert="auto")),
    "perf_probability_matrix": (perf_probability_matrix, dict(n_last_trials=90)),
    "swp_by_prev_best_arm": (swp_by_prev_best_arm, dict()),
    "swp_probability_matrix": (swp_probability_matrix, dict(trials=(2, 100))),
    "prob_combinations": (prob_combinations, dict()),
    "abstract_perf_tier": (
        perf_tier,
        dict(kwargs_perf=dict(by="choice", trial_window=10), **ABSTRACT_WINDOWED),
    ),
    "abstract_swp_tier": (swp_tier, dict(trial_window=10, **ABSTRACT_WINDOWED)),
    # per-trial (no windows) over the first 100 trials of each session
    "abstract_swp_by_trial": (swp_tier, dict(require_expert="auto")),
    # same per-trial P(High) / P(Switch) tiers, built over 8020 + 9505 good intact
    # animals for the paradigm comparison (keeps the shared products untouched)
    "perf_tier_8020_9505": (perf_tier, dict(require_expert="auto")),
    "swp_tier_8020_9505": (swp_tier, dict(require_expert="auto")),
    "perf_probability_matrix_8020_9505": (
        perf_probability_matrix,
        dict(n_last_trials=90, require_expert="auto"),
    ),
    "swp_probability_matrix_8020_9505": (
        swp_probability_matrix,
        dict(trials=(2, 100), require_expert="auto"),
    ),
    "abstract_swp_by_prev_best_arm": (swp_by_prev_best_arm, dict(require_expert="auto")),
    # per-trial, all blocks vs first / second block of each window
    "swp_by_block_expert": (swp_by_block, dict(require_expert="auto")),
    # previous block's last 10 trials -> new block's first 50, flipped vs same
    "abstract_swp_around_flip": (swp_around_flip, dict(require_expert="auto")),
    # same, built over 8020 animals + good 8020 RNN models
    "abstract_swp_around_flip_rnn": (swp_around_flip, dict(require_expert="auto")),
    # same, built over 8020 animals + good 8020 RNN models
    "abstract_swp_by_prev_best_arm_rnn": (
        swp_by_prev_best_arm,
        dict(require_expert="auto"),
    ),
    "abstract_perf_probability_matrix": (
        perf_probability_matrix,
        dict(n_last_trials=90, require_expert="auto"),
    ),
    # same grid, built over 8020 animals + good 8020 RNN models for animal - RNN
    "abstract_perf_probability_matrix_rnn": (
        perf_probability_matrix,
        dict(n_last_trials=90, require_expert="auto"),
    ),
    "abstract_swp_probability_matrix": (
        swp_probability_matrix,
        dict(trials=(2, 100), require_expert="auto"),
    ),
    # same grids, built over the 8020 mPFC-lesion animals' intact + lesion sessions
    "abstract_perf_probability_matrix_lesion_mPFC": (
        perf_probability_matrix,
        dict(n_last_trials=90, require_expert="auto"),
    ),
    "abstract_swp_probability_matrix_lesion_mPFC": (
        swp_probability_matrix,
        dict(trials=(2, 100), require_expert="auto"),
    ),
    # flip response and block-wise P(Switch), same animals and sessions
    "abstract_swp_around_flip_lesion_mPFC": (
        swp_around_flip,
        dict(require_expert="auto"),
    ),
    "swp_by_block_expert_lesion_mPFC": (swp_by_block, dict(require_expert="auto")),
    # same grids, over the 8020 OFC-lesion animals' intact + lesion sessions
    "abstract_perf_probability_matrix_lesion_OFC": (
        perf_probability_matrix,
        dict(n_last_trials=90, require_expert="auto"),
    ),
    "abstract_swp_probability_matrix_lesion_OFC": (
        swp_probability_matrix,
        dict(trials=(2, 100), require_expert="auto"),
    ),
    # same grid, built over 8020 animals + good 8020 RNN models
    "abstract_swp_probability_matrix_rnn": (
        swp_probability_matrix,
        dict(trials=(2, 100), require_expert="auto"),
    ),
    "abstract_tau_tier": (tau_tier, dict(model="single", require_expert="auto")),
    "abstract_tau_tier_double": (tau_tier, dict(model="double", require_expert="auto")),
}

# Same analyses on the fitted QlearnSticky simulated on each animal's own task, one
# product per (parameter source, analysis). Build with
# exps=mab_builders.simulate.simulated_exps(..., params_from=<source>). The
# simulated task is already expert-trimmed like the fit, so require_expert=False.
SIM_PARAM_SOURCES = {  # simulate.simulated_exps params_from -> save-name infix
    "own": "",
    "own_group_median": "groupmedian_",
    "other_group_median": "swapped_",
}
SIM_ANALYSES = {
    "perf_probability_matrix": (
        perf_probability_matrix,
        dict(n_last_trials=90, require_expert=False),
    ),
    "swp_probability_matrix": (
        swp_probability_matrix,
        dict(trials=(2, 100), require_expert=False),
    ),
    "swp_by_prev_best_arm": (swp_by_prev_best_arm, dict(require_expert=False)),
}
for _infix in SIM_PARAM_SOURCES.values():
    for _analysis, _entry in SIM_ANALYSES.items():
        PRODUCTS[f"abstract_sim_qlearn_sticky_{_infix}{_analysis}"] = _entry
    # the dependent-blocks fit (fit_qlearn_sticky_dependent_p8020_intact), simulated
    # on dependent blocks only (simulated_exps(..., fit_scope="dependent"))
    PRODUCTS[f"abstract_sim_qlearn_sticky_dep_{_infix}perf_probability_matrix"] = (
        SIM_ANALYSES["perf_probability_matrix"]
    )
    # P(High) per trial (the simulated task is dependent blocks only: tier "all")
    PRODUCTS[f"abstract_sim_qlearn_sticky_dep_{_infix}perf_tier"] = (
        perf_tier,
        dict(require_expert=False),
    )


# Learning curves, prob grids and flip response of each animal's fitted RNN
# simulated on the animal's own block schedule (compare with perf_tier_expert and
# the abstract_* animal products). Build with
# exps=mab_builders.simulate.rnn_simulated_exps(exps, "rnn_fit_n48_session") etc.;
# the schedule is already trimmed, so require_expert=False. "48win": the N48 fit with
# hidden state reset at window starts (rnn_fit_n48_window).
for _n in (48, 6, "48win"):
    PRODUCTS[f"sim_rnn_n{_n}_perf_tier"] = (perf_tier, dict(require_expert=False))
    PRODUCTS[f"sim_rnn_n{_n}_perf_probability_matrix"] = (
        perf_probability_matrix,
        dict(n_last_trials=90, require_expert=False),
    )
    PRODUCTS[f"sim_rnn_n{_n}_swp_probability_matrix"] = (
        swp_probability_matrix,
        dict(trials=(2, 100), require_expert=False),
    )
    PRODUCTS[f"sim_rnn_n{_n}_swp_around_flip"] = (
        swp_around_flip,
        dict(require_expert=False),
    )


def all_good_sessions(include_rnn=True):
    """Every good-quality animal session set, plus the curated good RNN models."""
    exps = mab_subjects.unstruc.sess(quality="good") + mab_subjects.struc.sess(
        quality="good"
    )
    if include_rnn:
        for group in (mab_subjects.unstruc_rnn, mab_subjects.struc_rnn):
            for paradigm in RNN_PARADIGMS:
                condition = getattr(mab_subjects.Datasets.RNN, paradigm)
                exps += group.rnn_good_sess(condition)
    return exps


def build(*names, n_jobs=1, include_rnn=True, exps=None):
    """Build the named products (all of them if none given) and save each to
    GroupData. Sessions are loaded once and shared across products."""
    names = names or tuple(PRODUCTS)
    unknown = set(names) - set(PRODUCTS)
    if unknown:
        raise ValueError(f"unknown products {sorted(unknown)}; have {list(PRODUCTS)}")

    if exps is None:
        t = time.time()
        exps = all_good_sessions(include_rnn=include_rnn)
        print(f"loaded {len(exps)} session sets in {time.time() - t:.0f}s")

    for name in names:
        builder, params = PRODUCTS[name]
        t = time.time()
        builder(exps, save_as=name, n_jobs=n_jobs, **params)
        print(f"built {name} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("products", nargs="*", help="products to build (default: all)")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--no-rnn", action="store_true", help="animals only")
    parser.add_argument("--list", action="store_true", help="list products and exit")
    args = parser.parse_args()

    if args.list:
        for name, (builder, params) in PRODUCTS.items():
            print(f"{name}: {builder.__name__}({params})")
    else:
        build(*args.products, n_jobs=args.n_jobs, include_rnn=not args.no_rnn)
