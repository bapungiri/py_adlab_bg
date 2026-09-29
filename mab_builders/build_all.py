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

from .performance import perf_probability_matrix, perf_tier
from .switching import swp_by_prev_best_arm, swp_probability_matrix

RNN_PARADIGMS = ("P100", "P9505", "P9010", "P8020")

# save name -> (builder, params). Params must be valid for every session type
# (intact, lesion, RNN); add a separate entry (another save name) for variants.
PRODUCTS = {
    "perf_tier": (perf_tier, dict(require_expert=False)),
    # intact sessions trimmed from the expertise day, lesion/RNN kept whole
    "perf_tier_expert": (perf_tier, dict(require_expert="auto")),
    "perf_probability_matrix": (perf_probability_matrix, dict(n_last_trials=90)),
    "swp_by_prev_best_arm": (swp_by_prev_best_arm, dict()),
    "swp_probability_matrix": (swp_probability_matrix, dict(trials=(2, 100))),
}


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
