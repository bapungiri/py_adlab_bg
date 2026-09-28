"""Builders for mab_switching_probability1.ipynb."""

import numpy as np
from banditpy.analyses import SwitchProb2Arm

from ._core import group_builder
from mab_data_core import MABData


def best_arm_transitions(task):
    """Classify each session by whether its best arm moved from the previous one.

    Uses the task as given (call before trial filtering, so short sessions still
    count as the "previous" block). The previous session must be in the same
    window; each window's first session and sessions with tied (or tied
    previous) probabilities are left unclassified.

    Returns
    -------
    flipped_ids, same_ids : np.ndarray
        Session ids whose best arm switched port / stayed on the same port.
    """
    starts = task.is_session_start
    session_ids = task.session_ids[starts]
    window_ids = task.window_ids[starts]
    probs = task.probs[starts]

    best = np.where(probs[:, 0] > probs[:, 1], 1, 2)
    best[probs[:, 0] == probs[:, 1]] = 0  # tie: no best arm
    prev_best = np.r_[0, best[:-1]]
    same_window = np.r_[False, window_ids[1:] == window_ids[:-1]]

    valid = same_window & (best > 0) & (prev_best > 0)
    flipped_ids = session_ids[valid & (best != prev_best)]
    same_ids = session_ids[valid & (best == prev_best)]
    return flipped_ids, same_ids


@group_builder("swp_by_prev_best_arm")
def swp_by_prev_best_arm(
    exp: MABData,
    min_sessions: int = 3,
    kwargs_trial_filter: dict = dict(min_trials=100, clip_max=100),
):
    """Switch probability per trial, split by whether the better port moved to
    the other arm relative to the previous block ('flipped') or not ('same').

    Returns one row per (prev_best_arm, trial_id), with 'n_sessions' giving the
    sessions behind each curve. Conditions with fewer than 'min_sessions'
    sessions are filled with NaN.
    """
    task = exp.b2a
    task.auto_block_window_ids()
    flipped_ids, same_ids = best_arm_transitions(task)

    task = task.filter_by_trials(**kwargs_trial_filter)

    curves = {}
    for label, ids in (("flipped", flipped_ids), ("same", same_ids)):
        mask = np.isin(task.session_ids, ids)
        n_sessions = np.unique(task.session_ids[mask]).size
        swp = None
        if n_sessions >= min_sessions:
            swp = SwitchProb2Arm(task._filtered(mask)).by_trial()
        curves[label] = (swp, n_sessions)

    lengths = [len(swp) for swp, _ in curves.values() if swp is not None]
    if not lengths:
        print(f"{exp.sub_name}: too few sessions in both conditions, skipping.")
        return dict(trial_id=[], prev_best_arm=[], switch_prob=[], n_sessions=[])
    n = lengths[0]

    labels = list(curves)
    return dict(
        trial_id=np.tile(np.arange(n) + 1, len(labels)),
        prev_best_arm=np.repeat(labels, n),
        switch_prob=np.concatenate(
            [np.full(n, np.nan) if swp is None else swp for swp, _ in curves.values()]
        ),
        n_sessions=np.repeat([n_sess for _, n_sess in curves.values()], n),
    )
