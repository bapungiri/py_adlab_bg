"""Builders for mab_choice_performance1.ipynb."""

import numpy as np

from ._core import group_builder
from .preprocess import make_tiered_task
from mab_data_core import MABData
from banditpy.core import Bandit2Arm


@group_builder("perf_tier")
def perf_tier(
    exp: MABData,
    require_expert: bool | str = True,
    min_sessions: int = 3,
    kwargs_perf: dict = dict(by="choice", trial_window=None),
    kwargs_trial_filter: dict = dict(min_trials=100, clip_max=100),
    kwargs_expert: dict = dict(
        day_start_hour=19,
        threshold_pct=65,
        n_consecutive=3,
        min_trials_per_day=500,
    ),
):
    """Performance per trial for each probability tier, in long format.

    Tiers: all, equalized (equal weight per probability combo), low-low,
    high-low, high-high, dependent (probs sum to 1) and independent. Returns
    one row per (tier_label, trial_id), with 'n_sessions' giving the number of
    sessions behind each tier. Tiers with fewer than 'min_sessions' sessions
    are filled with NaN.

    For intact vs lesion comparisons use require_expert="auto" (expert
    filtering on intact sessions only) and 'save_as' to name the product, e.g.
    perf_tier(exps, require_expert="auto", save_as="perf_tier_mPFC_lesion").
    """

    tiers = make_tiered_task(
        exp,
        require_expert=require_expert,
        min_sessions=min_sessions,
        kwargs_trial_filter=kwargs_trial_filter,
        kwargs_expert=kwargs_expert,
    )

    task_all, n_sess_all = tiers["all"]
    if task_all is None:
        print(f"{exp.sub_name}: only {n_sess_all} session(s), skipping.")
        return dict(trial_id=[], tier_label=[], tier_perf=[], n_sessions=[])

    def get_perf(b2a: Bandit2Arm, **kwargs):
        return b2a.get_performance(**kwargs_perf, **kwargs)

    # equalized is a reweighting of 'all', not a separate trial subset
    curves = {"all": (get_perf(task_all), n_sess_all)}
    curves["equalized"] = (get_perf(task_all, equalize_by="combo"), n_sess_all)

    n = len(curves["all"][0])
    for label, (task, n_sessions) in tiers.items():
        if label == "all":
            continue
        perf = np.full(n, np.nan) if task is None else get_perf(task)
        curves[label] = (perf, n_sessions)

    labels = list(curves)
    return dict(
        trial_id=np.tile(np.arange(n) + 1, len(labels)),  # 1..n, 1..n, ...
        tier_label=np.repeat(labels, n),  # "all"*n, "equalized"*n, ...
        tier_perf=np.concatenate([perf for perf, _ in curves.values()]),
        n_sessions=np.repeat([n_sess for _, n_sess in curves.values()], n),
    )
