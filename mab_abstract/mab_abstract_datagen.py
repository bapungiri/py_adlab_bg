"""GroupData product creators for the Cosyne 2026 abstract notebook.

Each build_* function computes one GroupData product from raw sessions and
saves it via mab_subjects.GroupData().save(...), mirroring what used to be
copy-pasted, hand-run notebook cells. Centralizing them here means the
notebook's data-prep cells are just a function call, and any future change
to the preprocessing pipeline (see make_tiered_task) automatically applies
to every product built from it.
"""

import numpy as np
import pandas as pd
import mab_subjects
from scipy.optimize import curve_fit
from banditpy.analyses import SwitchProb2Arm
from banditpy.utils.curve_fitting import fit_single_exp, fit_double_exp


def make_tiered_task(
    exp,
    trial_filter,
    require_expert=True,
    day_start_hour=19,
    threshold_pct=65,
    n_consecutive=3,
    min_trials_per_day=500,
):
    """Prep one exp's Bandit2Arm task for a tiered abstract GroupData product.

    Sets block/window IDs, optionally trims to post-expertise trials, applies
    trial_filter, then splits into low-low/high-low/high-high probability
    tiers.

    Parameters
    ----------
    exp : experiment object with .b2a (Bandit2Arm) and .data_tag
    trial_filter : dict
        Passed to Bandit2Arm.filter_by_trials (e.g. min_trials, clip_max).
    require_expert : bool, optional
        If True (default), trim to trials from the animal's expertise day
        onward (skipped for RNN datasets, which have no real datetime).

    Returns
    -------
    task, task_low_low, task_high_low, task_high_high : Bandit2Arm
    """
    task = exp.b2a

    if require_expert and exp.data_tag != "RNNdataset":
        task.auto_block_window_ids()
        _, expert_datetime, _, _ = task.get_expertise_day(
            by="datetime",
            day_start_hour=day_start_hour,
            threshold_pct=threshold_pct,
            fill_missing=True,
            n_consecutive=n_consecutive,
            min_trials_per_day=min_trials_per_day,
            baseline_frac=1,  # all data
        )
        task = task.filter_by_datetime(start=expert_datetime)

    task = task.filter_by_trials(**trial_filter)

    is_high = task.probs >= 0.5  # (n_trials, 2) bool
    n_high = is_high.sum(axis=1)  # 0, 1, or 2
    task_low_low = task._filtered(n_high == 0)
    task_high_low = task._filtered(n_high == 1)
    task_high_high = task._filtered(n_high == 2)

    return task, task_low_low, task_high_low, task_high_high


def build_abstract_perf_tier(trial_filter=None, by="choice", trial_window=10):
    """Compute and save the "abstract_perf_tier" GroupData product.

    P(High) per trial-window, tiered into low-low/high-low/high-high
    probability conditions, for every intact 80/20 session.

    Parameters
    ----------
    trial_filter : dict, optional
        Passed to Bandit2Arm.filter_by_trials. Default
        dict(min_trials=100, clip_max=30).
    by : str, optional
        Passed to get_performance. Default "choice".
    trial_window : int, optional
        Passed to get_performance. Default 10.

    Returns
    -------
    pd.DataFrame
    """
    if trial_filter is None:
        trial_filter = dict(min_trials=100, clip_max=30)

    exps = (
        mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess
    )

    get_perf = lambda task: task.get_performance(by=by, trial_window=trial_window)
    get_perf_equalize = lambda task: task.get_performance(
        by=by, equalize_by="combo", trial_window=trial_window
    )

    perf_df = []
    for exp in exps:
        print(exp.sub_name)

        task, task_low_low, task_high_low, task_high_high = make_tiered_task(
            exp, trial_filter
        )

        perf = get_perf_equalize(task)
        perf_low_low = get_perf(task_low_low)
        perf_high_low = get_perf(task_high_low)
        perf_high_high = get_perf(task_high_high)

        trial_starts = np.arange(0, perf.size * trial_window, trial_window) + 1
        trial_stops = trial_starts + trial_window - 1

        df = pd.DataFrame(
            dict(
                trial_id=[
                    f"{start}-{stop}"
                    for start, stop in zip(trial_starts, trial_stops)
                ],
                perf=perf,
                perf_low_low=perf_low_low,
                perf_high_low=perf_high_low,
                perf_high_high=perf_high_high,
                **exp.common_kwargs,
            )
        )
        perf_df.append(df)

    perf_df = pd.concat(perf_df, ignore_index=True)
    mab_subjects.GroupData().save(perf_df, "abstract_perf_tier")
    return perf_df


def build_abstract_swp_tier(trial_filter=None, trial_window=15):
    """Compute and save the "abstract_swp_tier" GroupData product.

    Switch probability per trial-window (overall, and split by whether the
    previous trial was rewarded), tiered into low-low/high-low/high-high
    probability conditions, for every intact 80/20 session.

    Parameters
    ----------
    trial_filter : dict, optional
        Passed to Bandit2Arm.filter_by_trials. Default
        dict(min_trials=100, clip_max=100).
    trial_window : int, optional
        Passed to SwitchProb2Arm.by_trial. Default 15.

    Returns
    -------
    pd.DataFrame
    """
    if trial_filter is None:
        trial_filter = dict(min_trials=100, clip_max=100)

    exps = (
        mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess
    )

    df_main = []
    for exp in exps:
        print(exp.sub_name)

        task, task_low_low, task_high_low, task_high_high = make_tiered_task(
            exp, trial_filter
        )

        swp_task = SwitchProb2Arm(task)
        swp_task_low_low = SwitchProb2Arm(task_low_low)
        swp_task_high_low = SwitchProb2Arm(task_high_low)
        swp_task_high_high = SwitchProb2Arm(task_high_high)

        swp = swp_task.by_trial(trial_window=trial_window)
        swp_win, swp_lose = swp_task.by_trial(
            trial_window=trial_window, split_by_reward=True
        )

        swp_low_low = swp_task_low_low.by_trial(trial_window=trial_window)
        swp_win_low_low, swp_lose_low_low = swp_task_low_low.by_trial(
            trial_window=trial_window, split_by_reward=True
        )

        swp_high_low = swp_task_high_low.by_trial(trial_window=trial_window)
        swp_win_high_low, swp_lose_high_low = swp_task_high_low.by_trial(
            trial_window=trial_window, split_by_reward=True
        )

        swp_high_high = swp_task_high_high.by_trial(trial_window=trial_window)
        swp_win_high_high, swp_lose_high_high = swp_task_high_high.by_trial(
            trial_window=trial_window, split_by_reward=True
        )

        df = pd.DataFrame(
            dict(
                trial_id=np.arange(swp.size) + 1,
                swp_all=swp,
                swp_all_low_low=swp_low_low,
                swp_all_high_low=swp_high_low,
                swp_all_high_high=swp_high_high,
                swp_win=swp_win,
                swp_lose=swp_lose,
                swp_win_low_low=swp_win_low_low,
                swp_lose_low_low=swp_lose_low_low,
                swp_win_high_low=swp_win_high_low,
                swp_lose_high_low=swp_lose_high_low,
                swp_win_high_high=swp_win_high_high,
                swp_lose_high_high=swp_lose_high_high,
                **exp.common_kwargs,
            )
        )
        df_main.append(df)

    df_main = pd.concat(df_main, ignore_index=True)
    mab_subjects.GroupData().save(df_main, "abstract_swp_tier")
    return df_main


def _fit_tau_single(perf):
    """Fit fit_single_exp to one animal's per-trial performance curve.

    Returns np.nan on too few points or a failed fit, rather than raising,
    so one bad animal doesn't stop the whole group's tau computation.
    """
    perf = np.asarray(perf, dtype=float)
    good = ~np.isnan(perf)
    t = np.arange(len(perf))

    if good.sum() < 4:  # need more points than fit_single_exp's 3 free params
        return np.nan

    try:
        popt, _ = curve_fit(
            fit_single_exp,
            t[good],
            perf[good],
            p0=[perf[good][0], perf[good][-1], len(t) / 3],
            maxfev=10000,
        )
    except RuntimeError:
        return np.nan

    return popt[2]  # tau


def _fit_tau_double(perf):
    """Fit fit_double_exp to one animal's per-trial performance curve.

    Returns (tau1, tau2, weight) — tau1 <= tau2 (fast, slow component) and
    weight is fit_double_exp's A1, the mixing weight for tau1 (tau2's
    weight is 1 - weight). Returns (nan, nan, nan) on too few points or a
    failed fit, rather than raising, so one bad animal doesn't stop the
    whole group's tau computation.
    """
    perf = np.asarray(perf, dtype=float)
    good = ~np.isnan(perf)
    t = np.arange(len(perf))

    if good.sum() < 8:  # need more points than fit_double_exp's 5 free params
        return np.nan, np.nan, np.nan

    try:
        popt, _ = curve_fit(
            fit_double_exp,
            t[good],
            perf[good],
            p0=[perf[good][0], perf[good][-1], 0.5, len(t) / 6, len(t) / 2],
            bounds=([0, 0, 0, 1e-3, 1e-3], [1, 1, 1, np.inf, np.inf]),
            maxfev=20000,
        )
    except RuntimeError:
        return np.nan, np.nan, np.nan

    _, _, A1, tau1, tau2 = popt
    if tau1 > tau2:
        tau1, tau2, A1 = tau2, tau1, 1 - A1

    return tau1, tau2, A1


def build_abstract_tau_tier(trial_filter=None, by="choice", model="single"):
    """Compute and save the "abstract_tau_tier" GroupData product.

    Fit a learning-speed time constant (tau) per animal, per probability tier.

    Fits a rise-to-asymptote curve to each animal's performance curve,
    giving a threshold-free measure of learning speed — tau is trials to
    reach a fixed fraction of the way from that animal's own baseline to
    its own asymptote — instead of comparing "trials to reach P(High)=0.6"
    against an arbitrary fixed value.

    Fits on the full per-trial resolution (get_performance's default, no
    trial_window), not the coarser trial-window curves used for the
    plotting cells, since curve_fit needs more points per animal than those
    curves have for a stable fit. trial_filter also defaults to more
    retained trials than the plotting cells use, for the same reason.

    Parameters
    ----------
    trial_filter : dict, optional
        Passed to Bandit2Arm.filter_by_trials. Default
        dict(min_trials=100, clip_max=100).
    by : str, optional
        Passed to get_performance. Default "choice".
    model : {"single", "double"}, optional
        Which rise-to-asymptote model to fit per animal's curve:

        - "single": fit_single_exp (P0, P_inf, tau) — one time constant.
        - "double": fit_double_exp (P0, P_inf, A1, tau1, tau2) — a fast and
          a slow learning component instead of one rate. Needs more points
          per animal for a stable fit (5 free params vs 3) — if fits come
          back mostly NaN, retain more trials via trial_filter.

        Default "single". Saved GroupData name is "abstract_tau_tier" for
        "single" and "abstract_tau_tier_double" for "double", so both can
        coexist without overwriting each other.

    Returns
    -------
    pd.DataFrame
        Long format: one row per animal per tau_type
        ({"all", "low_low", "high_low", "high_high"} — "all" is the overall
        curve equalized by combo, the rest are the three probability
        tiers). Columns beyond tau_type depend on model:

        - "single": tau_value — the fitted tau.
        - "double": tau1_value, tau2_value (tau1 <= tau2, fast/slow) and
          weight — fit_double_exp's A1, the mixing weight for tau1 (tau2's
          weight is 1 - weight).

        All NaN where the fit failed or had too few points. Plus the
        animal's identifying common_kwargs columns (name, group, dataset,
        lesion, paradigm, ...), repeated per tau_type row.
    """
    if model not in ("single", "double"):
        raise ValueError("model must be 'single' or 'double'")

    if trial_filter is None:
        trial_filter = dict(min_trials=100, clip_max=100)

    exps = (
        mab_subjects.unstruc.p8020_good_intact_sess
        + mab_subjects.struc.p8020_good_intact_sess
    )

    rows = []
    for exp in exps:
        print(exp.sub_name)

        task, task_low_low, task_high_low, task_high_high = make_tiered_task(
            exp, trial_filter
        )

        perf_by_type = dict(
            all=task.get_performance(by=by, equalize_by="combo"),
            low_low=task_low_low.get_performance(by=by),
            high_low=task_high_low.get_performance(by=by),
            high_high=task_high_high.get_performance(by=by),
        )

        for tau_type, perf in perf_by_type.items():
            if model == "single":
                fit_result = dict(tau_value=_fit_tau_single(perf))
            else:
                tau1, tau2, weight = _fit_tau_double(perf)
                fit_result = dict(tau1_value=tau1, tau2_value=tau2, weight=weight)

            rows.append(
                dict(tau_type=tau_type, **fit_result, **exp.common_kwargs)
            )

    tau_df = pd.DataFrame(rows)
    save_name = "abstract_tau_tier" if model == "single" else "abstract_tau_tier_double"
    mab_subjects.GroupData().save(tau_df, save_name)
    return tau_df
