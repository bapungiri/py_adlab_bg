"""Shared task preprocessing used by multiple builders."""

import numpy as np

DEFAULT_KWARGS_EXPERT = dict(
    day_start_hour=19,
    threshold_pct=65,
    n_consecutive=3,
    min_trials_per_day=500,
)


def prep_task(
    exp,
    require_expert=True,
    kwargs_trial_filter: dict = dict(min_trials=100, clip_max=100),
    kwargs_expert: dict = DEFAULT_KWARGS_EXPERT,
):
    """Return one exp's Bandit2Arm task, optionally trimmed to post-expertise
    trials, with 'kwargs_trial_filter' applied.

    require_expert : bool or "auto"
        If True, trim to trials from the animal's expertise day onward
        (skipped for RNN datasets, which have no real datetime). If "auto",
        trim only when exp.lesion_tag == "intact", so lesion sessions are kept
        whole. False keeps all trials.
    """
    task = exp.b2a

    if require_expert == "auto":
        require_expert = exp.lesion_tag == "intact"

    if require_expert and exp.data_tag != "RNNdataset":
        task.auto_block_window_ids()
        _, expert_datetime, _, _ = task.get_expertise_day(
            by="datetime", **kwargs_expert
        )
        task = task.filter_by_datetime(start=expert_datetime)
    elif require_expert:
        print(
            f"{exp.sub_name}(data_tag={exp.data_tag}), Not an animal, skipping expertise filtering."
        )

    return task.filter_by_trials(**kwargs_trial_filter)


def make_tiered_task(
    exp,
    require_expert=True,
    min_sessions=3,
    kwargs_trial_filter: dict = dict(min_trials=100, clip_max=100),
    kwargs_expert: dict = DEFAULT_KWARGS_EXPERT,
):
    """Prep one exp's Bandit2Arm task for a tiered abstract GroupData product.

    Runs 'prep_task' (optional expertise trimming + trial filter), then splits
    into low-low/high-low/high-high probability tiers and dependent/independent
    combinations.

    Parameters
    ----------
    exp : experiment object with .b2a (Bandit2Arm) and .data_tag
    require_expert : bool or "auto", optional
        If True (default), trim to trials from the animal's expertise day
        onward (skipped for RNN datasets, which have no real datetime). If
        "auto", trim only when exp.lesion_tag == "intact", so lesion sessions
        are kept whole.
    min_sessions : int, optional
        Tiers with fewer sessions than this are returned as None instead of a
        Bandit2Arm, by default 3.
    kwargs_trial_filter : dict
        Passed to Bandit2Arm.filter_by_trials (e.g. min_trials, clip_max).
    kwargs_expert : dict
        Passed to Bandit2Arm.get_expertise_day.

    Returns
    -------
    dict
        Maps tier label ('all', 'low-low', 'high-low', 'high-high',
        'dependent', 'independent') to a (task, n_sessions) tuple, where task
        is a Bandit2Arm or None if n_sessions < min_sessions.
    """
    task = prep_task(exp, require_expert, kwargs_trial_filter, kwargs_expert)

    n_high = (task.probs >= 0.5).sum(axis=1)  # 0, 1, or 2 arms at/above 0.5
    is_corr = corr_mask(task)
    masks = {
        "all": np.ones(len(task.probs), dtype=bool),
        "low-low": n_high == 0,
        "high-low": n_high == 1,
        "high-high": n_high == 2,
        "dependent": is_corr,
        "independent": ~is_corr,
    }
    return {label: _tier(task, mask, min_sessions) for label, mask in masks.items()}


def _tier(task, mask, min_sessions):
    """Return (filtered task, n_sessions), with task None if too few sessions."""
    n_sessions = np.unique(task.session_ids[mask]).size
    if n_sessions < min_sessions:
        return None, n_sessions
    return task._filtered(mask), n_sessions


def corr_mask(task):
    """Trials whose arm probabilities sum to 1 (correlated combinations)."""
    return task.probs.sum(axis=1).round(2) == 1.0
