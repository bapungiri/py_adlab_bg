"""Model-simulated stand-ins for subjects, so any builder can be run on what a
fitted policy would have done on the same task.

    exps = mab_subjects.unstruc.p8020_good_intact_sess + ...
    sim_exps = simulated_exps(exps, "fit_qlearn_sticky_p8020_intact")
    perf_probability_matrix(sim_exps, require_expert=False, save_as=...)

Build simulated products with require_expert=False: the simulated task is
already trimmed the way the fit was, and expertise computed from simulated
choices would trim it a second time, differently.
"""

import re

import pandas as pd
from banditpy.core import Bandit2Arm
from banditpy.models import DecisionModel
from banditpy.models.policy import Qlearn

from mab_group_data import GroupData
from .preprocess import DEFAULT_KWARGS_EXPERT, prep_task


def qlearn_sticky():
    """Qlearn with all six params, as fitted by server/fit_policy_wrapper.py
    ('QlearnSticky', presets sticky_*)."""
    policy = Qlearn()
    policy.params.alpha_h.enable()
    policy.params.sticky.enable()
    return policy


def fit_task(exp, require_expert="auto", fit_scope="all"):
    """The task a fit_policy_core fit saw: window ids, expertise trimming
    ('require_expert' as in fit_blocks), sessions >= 100 trials, unclipped, and
    for a 'high_low' / 'dependent' fit only those trials (as in fit_blocks)."""
    task = exp.b2a
    if exp.data_tag != "RNNdataset":
        task.auto_block_window_ids()
    if require_expert == "auto":
        require_expert = exp.lesion_tag == "intact"
    if require_expert and exp.data_tag != "RNNdataset":
        _, expert_datetime, _, _ = task.get_expertise_day(
            by="datetime", **DEFAULT_KWARGS_EXPERT
        )
        task = task.filter_by_datetime(start=expert_datetime)
    task = task.filter_by_trials(min_trials=100, clip_max=None)
    scope_masks = {
        "all": None,
        "high_low": lambda t: (t.probs >= 0.5).sum(axis=1) == 1,
        "dependent": lambda t: t.probs.sum(axis=1).round(2) == 1.0,
    }
    if scope_masks[fit_scope] is not None:
        task = task._filtered(scope_masks[fit_scope](task))
    return task


class SimulatedExp:
    """An exp whose 'b2a' is the fitted policy's simulated behaviour on the
    exp's own task; every other attribute (sub_name, group_tag, lesion_tag,
    common_kwargs, ...) is the real exp's, so builders treat it as that subject."""

    def __init__(self, exp, b2a):
        self._exp = exp
        self.b2a = b2a

    def __getattr__(self, name):
        # dunders (pickling for n_jobs > 1 probes __getstate__ etc.) and '_exp'
        # itself must not forward, or lookups recurse before '_exp' is set
        if name.startswith("__") or name == "_exp":
            raise AttributeError(name)
        return getattr(self._exp, name)


PARAM_SOURCES = ("own", "own_group_median", "other_group_median")
OTHER_GROUP = {"struc": "unstruc", "unstruc": "struc"}


def simulated_exps(
    exps,
    fit_name,
    policy_factory=qlearn_sticky,
    fit_scope="all",
    require_expert="auto",
    params_from="own",
    seed=0,
):
    """Simulate fitted 'fit_scope' parameters (GroupData 'fit_name') on each
    exp's own fit task, resetting at window starts as the fit did.

    params_from : {"own", "own_group_median", "other_group_median"}
        Whose parameters drive each exp's simulation: the exp's own fit, the
        median fit of its group, or the median fit of the other group
        (struc <-> unstruc). Medians are per parameter, over the fit's subjects
        in the same lesion condition. With "other_group_median" the task stays
        the exp's own, so a struc exp is the struc task played with unstruc
        parameters; comparing it with "own_group_median" isolates the effect of
        the parameters, and S - U under it isolates the effect of the task.

    Returns SimulatedExp objects; exps without parameters in the fit are skipped.
    'require_expert' must match the fit's preset ("auto" covers both the intact
    presets, trimmed, and the lesion presets, kept whole).
    """
    if params_from not in PARAM_SOURCES:
        raise ValueError(f"params_from must be one of {PARAM_SOURCES}")
    params_df = GroupData().__getattr__(fit_name).latest
    params_df = params_df[params_df["fit_scope"] == fit_scope].copy()
    names = list(policy_factory().parameter_specs())
    params_df = params_df[params_df["param_names"].isin(names)]
    params_df["param_values"] = params_df["param_values"].astype(float)
    medians = params_df.groupby(["group", "lesion", "param_names"])["param_values"].median()

    out = []
    for exp in exps:
        rows = params_df[
            (params_df["name"] == exp.sub_name)
            & (params_df["group"] == exp.group_tag)
            & (params_df["lesion"] == exp.lesion_tag)
        ]
        if rows.empty:
            print(f"{exp.sub_name}: no '{fit_scope}' params in {fit_name}, skipping.")
            continue
        if params_from == "own":
            params = dict(zip(rows["param_names"], rows["param_values"]))
        else:
            group = exp.group_tag
            if params_from == "other_group_median":
                group = OTHER_GROUP[group]
            params = medians.loc[(group, exp.lesion_tag)].to_dict()

        task = fit_task(exp, require_expert, fit_scope)  # the trials the fit used
        policy = policy_factory()
        model = DecisionModel(task, policy=policy, reset_mode=task.is_window_start)
        # parameter_specs covers the beta schedule (beta, epsilon) as well as the
        # policy's own params; param_names() alone leaves beta unset
        model.params = {k: params[k] for k in names if k in params}
        out.append(SimulatedExp(exp, model.simulate_posterior_predictive(seed=seed)))
    return out


RNN_FITS = (
    "rnn_fit_n48_session",
    "rnn_fit_n48_window",
    "rnn_fit_n6_session",
    "rnn_fit_n32_session",
)
# training epochs of fits not trained for the default 500
RNN_FIT_EPOCHS = {"rnn_fit_n48_window": 1000}


def rnn_sim_path(exp, rnn_attr, require_expert="auto", seed=0):
    """(fit file, cached simulation file) for one exp's RNN fit.

    Simulations are cached in the animal's folder as
    'rnn_simulated/<model>/seed<seed>_expert-<require_expert>.csv', where
    <model> is the fit's tag, e.g. 'N48_LR0.001_E500_Ssession'.
    """
    m = re.fullmatch(r"rnn_fit_n(\d+)_(\w+)", rnn_attr)
    if m is None:
        raise ValueError(f"rnn_attr must look like 'rnn_fit_n48_session', got {rnn_attr!r}")
    fit = exp.rnn_fit_path(
        int(m.group(1)), m.group(2), n_epochs=RNN_FIT_EPOCHS.get(rnn_attr, 500)
    )
    model = fit.stem.split("_RNNfit_", 1)[1]
    sim = fit.parent / "rnn_simulated" / model / f"seed{seed}_expert-{require_expert}.csv"
    return fit, sim


def _task_to_csv(task, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    task.to_df().to_csv(path, index=False)


def _task_from_csv(path):
    df = pd.read_csv(path)

    def col(name):  # columns that were None in the simulated task come back all-NaN
        return name if name in df and df[name].notna().any() else None

    return Bandit2Arm.from_df(
        df,
        probs=["probs_1", "probs_2"],
        choices="choices",
        rewards="rewards",
        session_ids="session_ids",
        block_ids=col("block_ids"),
        window_ids=col("window_ids"),
    )


def _simulate_rnn(exp, rnn_attr, require_expert, seed, overwrite=False):
    """One exp's RNN fit played on its own block schedule, from the cache when it
    is newer than the fit; otherwise simulated (single-threaded torch, so joblib
    workers don't oversubscribe) and cached."""
    fit, path = rnn_sim_path(exp, rnn_attr, require_expert, seed)
    if (
        not overwrite
        and path.is_file()
        and fit.is_file()
        and path.stat().st_mtime >= fit.stat().st_mtime
    ):
        return _task_from_csv(path)

    import torch

    torch.set_num_threads(1)
    task = prep_task(
        exp, require_expert, kwargs_trial_filter=dict(min_trials=100, clip_max=None)
    )
    block_probs = task.probs[task.is_session_start, :]
    sim = getattr(exp, rnn_attr).simulate(
        reward_schedule=block_probs, return_hidden=False, seed=seed
    )
    _task_to_csv(sim, path)
    return sim


def rnn_simulated_exps(
    exps,
    rnn_attr="rnn_fit_n48_session",
    require_expert="auto",
    seed=0,
    n_jobs=1,
    overwrite=False,
):
    """Each exp's RNN fit ('rnn_attr', a MABData property such as
    'rnn_fit_n48_session') simulated on the exp's own block schedule.

    The schedule is the exp's blocks after 'prep_task' trimming ("auto": intact
    sessions from the expertise day, lesion sessions whole; blocks >= 100
    trials), one simulated block per real block; block lengths and window
    resets are drawn by the RNN simulator. Simulations run in parallel over
    exps with 'n_jobs'. Build products from the result with
    require_expert=False (already trimmed). Exps whose fit fails to load or
    simulate are skipped with a message.

    Each simulation is cached as a CSV in the animal's folder (see
    'rnn_sim_path') and reused on later calls, so other analyses don't
    re-simulate; it is redone when the fit file is newer or 'overwrite' is True.
    """
    if n_jobs == 1:
        results = []
        for exp in exps:
            try:
                results.append(_simulate_rnn(exp, rnn_attr, require_expert, seed, overwrite))
            except Exception as e:
                results.append(e)
    else:
        from joblib import Parallel, delayed

        def safe(exp):
            try:
                return _simulate_rnn(exp, rnn_attr, require_expert, seed, overwrite)
            except Exception as e:
                return e

        results = Parallel(n_jobs=n_jobs)(delayed(safe)(exp) for exp in exps)

    out = []
    for exp, res in zip(exps, results):
        if isinstance(res, Exception):
            print(f"{exp.sub_name} ({exp.lesion_tag}): {rnn_attr} skipped: {res}")
        else:
            out.append(SimulatedExp(exp, res))
    return out
