"""Builders for learning-speed measures fit to performance curves."""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from banditpy.utils.curve_fitting import fit_single_exp, fit_double_exp

from ._core import group_builder
from .performance import perf_tier
from .preprocess import DEFAULT_KWARGS_EXPERT
from mab_data_core import MABData


def _fit_tau_single(perf):
    """Fit fit_single_exp to one per-trial performance curve.

    Returns np.nan on too few points or a failed fit, rather than raising,
    so one bad curve doesn't stop the whole group's tau computation.
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
    """Fit fit_double_exp to one per-trial performance curve.

    Returns (tau1, tau2, weight) -- tau1 <= tau2 (fast, slow component) and
    weight is fit_double_exp's A1, the mixing weight for tau1 (tau2's weight
    is 1 - weight). Returns (nan, nan, nan) on too few points or a failed fit.
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


@group_builder("tau_tier")
def tau_tier(
    exp: MABData,
    model: str = "single",
    require_expert: bool | str = True,
    min_sessions: int = 3,
    by: str = "choice",
    kwargs_trial_filter: dict = dict(min_trials=100, clip_max=100),
    kwargs_expert: dict = DEFAULT_KWARGS_EXPERT,
):
    """Learning-speed time constant (tau) per probability tier, one row per tier.

    Fits a rise-to-asymptote curve to each tier's per-trial performance curve
    from 'perf_tier', giving a threshold-free measure of learning speed: tau
    is trials to reach a fixed fraction of the way from the subject's own
    baseline to its own asymptote. Fits on full per-trial resolution (no
    trial_window), since curve_fit needs more points than windowed curves
    have for a stable fit.

    model : {"single", "double"}
        "single" fits fit_single_exp (P0, P_inf, tau) and returns 'tau_value'.
        "double" fits fit_double_exp (P0, P_inf, A1, tau1, tau2) -- a fast and
        a slow component -- and returns 'tau1_value', 'tau2_value' (tau1 <=
        tau2) and 'weight' (A1, the mixing weight for tau1). Needs more points
        per curve (5 free params vs 3); retain more trials via
        'kwargs_trial_filter' if fits come back mostly NaN.

    All NaN where the fit failed, had too few points, or the tier had fewer
    than 'min_sessions' sessions.
    """
    if model not in ("single", "double"):
        raise ValueError("model must be 'single' or 'double'")

    curves = pd.DataFrame(
        perf_tier.compute(
            exp,
            require_expert=require_expert,
            min_sessions=min_sessions,
            kwargs_perf=dict(by=by, trial_window=None),
            kwargs_trial_filter=kwargs_trial_filter,
            kwargs_expert=kwargs_expert,
        )
    )

    rows = []
    for label, curve in curves.groupby("tier_label", sort=False):
        perf = curve.sort_values("trial_id")["tier_perf"].to_numpy()
        if model == "single":
            fit_result = dict(tau_value=_fit_tau_single(perf))
        else:
            tau1, tau2, weight = _fit_tau_double(perf)
            fit_result = dict(tau1_value=tau1, tau2_value=tau2, weight=weight)
        rows.append(
            dict(tier_label=label, **fit_result, n_sessions=curve["n_sessions"].iat[0])
        )

    if not rows:
        return dict(tier_label=[], n_sessions=[])
    return pd.DataFrame(rows)
