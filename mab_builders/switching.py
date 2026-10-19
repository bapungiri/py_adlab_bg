"""Builders for mab_switching_probability1.ipynb."""

import numpy as np
import pandas as pd
from banditpy.analyses import SwitchProb2Arm

from ._core import group_builder
from .preprocess import DEFAULT_KWARGS_EXPERT, make_tiered_task, prep_task
from mab_data_core import MABData


@group_builder("swp_tier")
def swp_tier(
    exp: MABData,
    require_expert: bool | str = True,
    min_sessions: int = 3,
    trial_window: int | None = None,
    kwargs_trial_filter: dict = dict(min_trials=100, clip_max=100),
    kwargs_expert: dict = DEFAULT_KWARGS_EXPERT,
):
    """Switch probability per trial for each probability tier, in long format.

    Tiers as in 'make_tiered_task' (all, low-low, high-low, high-high,
    dependent, independent). 'outcome' is "all", or "win"/"lose" for switch
    probability conditioned on the previous trial being rewarded or not.
    Returns one row per (tier_label, outcome, trial_id); 'trial_id' indexes
    'trial_window'-trial windows when 'trial_window' is set. Tiers with fewer
    than 'min_sessions' sessions are left out. See 'prep_task' for
    'require_expert'.
    """
    tiers = make_tiered_task(
        exp,
        require_expert=require_expert,
        min_sessions=min_sessions,
        kwargs_trial_filter=kwargs_trial_filter,
        kwargs_expert=kwargs_expert,
    )

    rows = []
    for label, (task, n_sessions) in tiers.items():
        if task is None:
            continue
        swp = SwitchProb2Arm(task)
        curves = {"all": swp.by_trial(trial_window=trial_window)}
        curves["win"], curves["lose"] = swp.by_trial(
            trial_window=trial_window, split_by_reward=True
        )
        for outcome, curve in curves.items():
            rows.append(
                pd.DataFrame(
                    dict(
                        tier_label=label,
                        outcome=outcome,
                        trial_id=np.arange(len(curve)) + 1,
                        switch_prob=curve,
                        n_sessions=n_sessions,
                    )
                )
            )

    if not rows:
        print(f"{exp.sub_name}: too few sessions in every tier, skipping.")
        return dict(tier_label=[], outcome=[], trial_id=[], switch_prob=[], n_sessions=[])
    return pd.concat(rows, ignore_index=True)


@group_builder("swp_by_block")
def swp_by_block(
    exp: MABData,
    blocks: tuple = (1, 2),
    min_sessions: int = 3,
    require_expert: bool | str = False,
    kwargs_trial_filter: dict = dict(min_trials=100, clip_max=100),
    kwargs_expert: dict = DEFAULT_KWARGS_EXPERT,
):
    """Switch probability per trial over all blocks and for each block position
    in 'blocks' (block_id within its window, 1 = first block of the window).

    Returns one row per (block_label, trial_id), block_label "all" or
    "block<id>", with 'n_sessions' giving the sessions behind each curve.
    Curves with fewer than 'min_sessions' sessions are NaN. Block ids are
    assigned on the full session history, before 'require_expert' trimming and
    trial filtering (see 'prep_task'), so dropped sessions don't renumber them.
    """
    # RNN CSVs and simulated tasks (no datetime) carry their own block ids
    if exp.data_tag != "RNNdataset" and exp.b2a.datetime is not None:
        exp.b2a.auto_block_window_ids()
    task = prep_task(exp, require_expert, kwargs_trial_filter, kwargs_expert)

    subsets = {"all": task}
    for block in blocks:
        subsets[f"block{block}"] = task.filter_by_block_id(ids=[block])

    curves = {}
    for label, sub in subsets.items():
        n_sessions = np.unique(sub.session_ids).size
        swp = SwitchProb2Arm(sub).by_trial() if n_sessions >= min_sessions else None
        curves[label] = (swp, n_sessions)

    if curves["all"][0] is None:
        print(f"{exp.sub_name}: only {curves['all'][1]} session(s), skipping.")
        return dict(trial_id=[], block_label=[], switch_prob=[], n_sessions=[])
    n = len(curves["all"][0])

    labels = list(curves)
    return dict(
        trial_id=np.tile(np.arange(n) + 1, len(labels)),
        block_label=np.repeat(labels, n),
        switch_prob=np.concatenate(
            [np.full(n, np.nan) if swp is None else swp for swp, _ in curves.values()]
        ),
        n_sessions=np.repeat([n_sess for _, n_sess in curves.values()], n),
    )


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
    require_expert: bool | str = False,
    kwargs_trial_filter: dict = dict(min_trials=100, clip_max=100),
    kwargs_expert: dict = DEFAULT_KWARGS_EXPERT,
):
    """Switch probability per trial, split by whether the better port moved to
    the other arm relative to the previous block ('flipped') or not ('same').

    Returns one row per (prev_best_arm, trial_id), with 'n_sessions' giving the
    sessions behind each curve. Conditions with fewer than 'min_sessions'
    sessions are filled with NaN. Transitions are classified on the full
    session history, before 'require_expert' trimming (see 'prep_task'), so
    the first post-expertise block still has its previous block.
    """
    task = exp.b2a
    # RNN CSVs and simulated tasks (no datetime) carry their own window ids
    if exp.data_tag != "RNNdataset" and task.datetime is not None:
        task.auto_block_window_ids()
    flipped_ids, same_ids = best_arm_transitions(task)

    task = prep_task(exp, require_expert, kwargs_trial_filter, kwargs_expert)

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


@group_builder("swp_around_flip")
def swp_around_flip(
    exp: MABData,
    n_pre: int = 10,
    n_post: int = 50,
    min_blocks: int = 3,
    require_expert: bool | str = False,
    kwargs_trial_filter: dict = dict(min_trials=100, clip_max=100),
    kwargs_expert: dict = DEFAULT_KWARGS_EXPERT,
):
    """Switch probability around a block transition, from the previous block's
    last 'n_pre' trials into the new block's first 'n_post' trials.

    'trial_id' runs -n_pre..-1 (previous block) and 1..n_post (new block);
    trial 1 is a switch when the new block's first choice differs from the
    previous block's last one. Split by whether the better port moved
    ('flipped') or not ('same'), as in 'swp_by_prev_best_arm', and averaged
    over blocks. The new block must pass 'prep_task' (expertise trimming and
    trial filter); the previous block is taken untrimmed from the full session
    history and needs at least n_pre + 1 trials. Conditions with fewer than
    'min_blocks' blocks are NaN.
    """
    full = exp.b2a
    # RNN CSVs and simulated tasks (no datetime) carry their own window ids
    if exp.data_tag != "RNNdataset" and full.datetime is not None:
        full.auto_block_window_ids()
    flipped_ids, same_ids = best_arm_transitions(full)
    kept_ids = np.unique(
        prep_task(exp, require_expert, kwargs_trial_filter, kwargs_expert).session_ids
    )

    starts = np.flatnonzero(full.is_session_start)
    start_ids = full.session_ids[starts]
    ends = np.r_[starts[1:], len(full.session_ids)]
    trial_id = np.r_[-n_pre:0, 1 : n_post + 1]

    labels, curves, n_blocks = [], [], []
    for label, ids in (("flipped", flipped_ids), ("same", same_ids)):
        switches = []
        for k in np.flatnonzero(np.isin(start_ids, ids) & np.isin(start_ids, kept_ids)):
            s = starts[k]  # classified blocks always have a previous block (k > 0)
            if s - starts[k - 1] < n_pre + 1 or ends[k] - s < n_post:
                continue
            seg = full.choices[s - n_pre - 1 : s + n_post]
            switches.append((seg[1:] != seg[:-1]).astype(float))
        curve = np.full(len(trial_id), np.nan)
        if len(switches) >= min_blocks:
            curve = np.mean(switches, axis=0)
        labels.append(label)
        curves.append(curve)
        n_blocks.append(len(switches))

    return dict(
        trial_id=np.tile(trial_id, len(labels)),
        prev_best_arm=np.repeat(labels, len(trial_id)),
        switch_prob=np.concatenate(curves),
        n_blocks=np.repeat(n_blocks, len(trial_id)),
    )


@group_builder("swp_probability_matrix")
def swp_probability_matrix(
    exp: MABData,
    trials: tuple = (2, 100),
    min_sessions: int = 2,
    require_expert: bool | str = False,
    kwargs_trial_filter: dict = dict(min_trials=100, clip_max=100),
):
    """Switch probability for every (order-independent) reward probability pair,
    one row per exp.

    Each cell of 'swp_mat' is the session-averaged switch probability
    (SwitchProb2Arm.by_trial) averaged over trial positions 'trials'
    (inclusive, 1-based; trial 1 has no previous choice). Indexed by the sorted
    unique arm probabilities in 'probs'. Pairs with fewer than 'min_sessions'
    sessions are NaN. Unlike performance, the diagonal (equal probabilities) is
    kept, since switching is defined there. See 'prep_task' for
    'require_expert'.
    """
    task = prep_task(exp, require_expert, kwargs_trial_filter)
    probs = task.probs
    unique_probs = np.unique(probs.flatten())
    n = len(unique_probs)
    swp_mat = np.full((n, n), np.nan)

    for i1 in range(n):
        for i2 in range(i1, n):  # upper triangle incl. diagonal; mirrored below
            p1, p2 = unique_probs[i1], unique_probs[i2]
            mask = ((probs[:, 0] == p1) & (probs[:, 1] == p2)) | (
                (probs[:, 0] == p2) & (probs[:, 1] == p1)
            )
            if np.unique(task.session_ids[mask]).size < min_sessions:
                continue
            curve = SwitchProb2Arm(task._filtered(mask)).by_trial()
            swp = np.nanmean(curve[trials[0] - 1 : trials[1]])
            swp_mat[i1, i2] = swp_mat[i2, i1] = swp

    return dict(probs=[unique_probs], swp_mat=[swp_mat])
