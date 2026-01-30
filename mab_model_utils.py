"""Reusable plotting/model-fit helpers for MAB analyses."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from statplotannot.plots import SeabornPlotter
from statplotannot.plots.plot_utils import xtick_format
import matplotlib.gridspec as gridspec
from mab_colors import Palette2Arm
from neuropy import plotting
from banditpy.models import DecisionModel

MODEL_PARAM_GROUPS = {
    # Qlearn
    "qlearnH": {
        "params": [("alpha_c", "alpha_u", "alpha_h"), ("scaler"), ("beta"), ("nll")],
        "width_ratios": [2, 1, 1, 1],
        "scopes": ["block_all", "block1", "block2_plus"],
    },
    # Qlearn
    "ucb": {
        "params": [("alpha_c", "alpha_u", "alpha_h"), ("scaler"), ("beta"), ("nll")],
        "width_ratios": [2, 1, 1, 1],
        "scopes": ["block_all", "block1", "block2_plus"],
    },
    # Thompson Sampling
    "ts_shared": {
        "params": [("alpha_c", "alpha_u", "alpha_h"), ("scaler"), ("beta"), ("nll")],
        "width_ratios": [2, 1, 1, 1],
        "scopes": ["block_all", "block1", "block2_plus"],
    },
    # Thompson Sampling
    "ts_split": {
        "params": [("alpha_c", "alpha_u", "alpha_h"), ("scaler"), ("beta"), ("nll")],
        "width_ratios": [3, 1, 1, 1],
        "scopes": ["block_all", "block1", "block2_plus"],
    },
    # State inference 2-arm model
    "si": {
        "params": [("c"), ("y", "b0"), ("beta"), ("nll")],
        "width_ratios": [1, 2, 1, 1],
        "scopes": ["block_all", "block1", "block2_plus"],
    },
    # Add other models here, e.g. "qlearn": {"alpha": ("alpha",), "beta": "beta", ...}
}


def build_actual_vs_sim_perf_df(
    exps,
    est_params_df,
    n_trials: int = 100,
    min_trials: int = 100,
    clip_max: int = 100,
    policy_factory=None,
):
    """Construct dataframe comparing actual vs simulated performance per subject and scope."""

    policy_factory = policy_factory

    perf_df = []

    for exp in exps:
        name = exp.sub_name

        task = exp.b2a.filter_by_trials(min_trials=min_trials, clip_max=clip_max)
        task.auto_block_window_ids()

        task_specs = [
            ("block_all", task, task.get_block_start_mask(start=1)),
            (
                "block1",
                task_block1 := task.filter_by_block_id(start=1, stop=1),
                task_block1.get_block_start_mask(start=1, stop=1),
            ),
            (
                "block2_plus",
                task_block2 := task.filter_by_block_id(start=2),
                task_block2.get_block_start_mask(start=2, stop=2),
            ),
        ]

        sims_perf = {}
        for scope, scoped_task, reset_mask in task_specs:
            scope_rows = (est_params_df["name"] == name) & (
                est_params_df["fit_scope"] == scope
            )
            params = dict(
                zip(
                    est_params_df.loc[scope_rows, "param_names"],
                    est_params_df.loc[scope_rows, "param_values"],
                )
            )
            policy = policy_factory()
            model = DecisionModel(scoped_task, policy=policy, reset_mode=reset_mask)
            model.params = {k: params[k] for k in policy.param_names()}
            sims_perf[scope] = (
                model.simulate_posterior_predictive().get_optimal_choice_probability()
            )

        sub_df = pd.DataFrame(
            {
                "name": name,
                "trial_id": np.arange(n_trials) + 1,
                "perf_all": task.get_optimal_choice_probability(),
                "sim_perf_all": sims_perf["block_all"],
                "perf_block1": task_block1.get_optimal_choice_probability(),
                "sim_perf_block1": sims_perf["block1"],
                "perf_block2plus": task_block2.get_optimal_choice_probability(),
                "sim_perf_block2plus": sims_perf["block2_plus"],
                "grp": exp.group_tag,
            }
        )
        perf_df.append(sub_df)

    return pd.concat(perf_df, ignore_index=True)


def build_actual_vs_sim_swp_df(exps, params_df: pd.DataFrame, policy_factory):
    """Generate

    Parameters
    ----------
    exps : _type_
        _description_
    """
    from banditpy.analyses import SwitchProb2Arm

    swp_df = []

    for exp in exps:
        name = exp.sub_name
        print(name)

        task = exp.b2a.filter_by_trials(min_trials=100, clip_max=100)
        task.auto_block_window_ids()

        task_specs = [
            ("block_all", task, task.get_block_start_mask(start=1)),
            (
                "block1",
                task_block1 := task.filter_by_block_id(start=1, stop=1),
                task_block1.get_block_start_mask(start=1, stop=1),
            ),
            (
                "block2_plus",
                task_block2 := task.filter_by_block_id(start=2),
                task_block2.get_block_start_mask(start=2, stop=2),
            ),
        ]

        swp = {}
        for scope, scoped_task, reset_mask in task_specs:
            scope_rows = (params_df["name"] == name) & (params_df["fit_scope"] == scope)
            params = dict(
                zip(
                    params_df.loc[scope_rows, "param_names"],
                    params_df.loc[scope_rows, "param_values"],
                )
            )
            policy = policy_factory()
            model = DecisionModel(scoped_task, policy=policy, reset_mode=reset_mask)
            model.params = {k: params[k] for k in policy.param_names()}
            task_sim = model.simulate_posterior_predictive()

            swp[scope] = SwitchProb2Arm(task_sim).by_trial()

        sub_df = pd.DataFrame(
            {
                "name": name,
                "trial_id": np.arange(99) + 1,
                "switch_prob_all": SwitchProb2Arm(task).by_trial(),
                "sim_switch_prob_all": swp["block_all"],
                "switch_prob_block1": SwitchProb2Arm(task_block1).by_trial(),
                "sim_switch_prob_block1": swp["block1"],
                "switch_prob_block2plus": SwitchProb2Arm(task_block2).by_trial(),
                "sim_switch_prob_block2plus": swp["block2_plus"],
                "grp": exp.group_tag,
            }
        )
        swp_df.append(sub_df)

    return pd.concat(swp_df, ignore_index=True)


def plot_param_block_grid(data_df, model_key: str, fs=10):
    """Plot for each scope"""

    groups = MODEL_PARAM_GROUPS.get(model_key)
    params = groups["params"]
    scopes = groups["scopes"]
    n_cols = len(params)
    n_rows = len(scopes)

    fig = plotting.Fig(
        n_rows, n_cols, width_ratios=groups.get("width_ratios"), fontsize=fs
    )

    palette = Palette2Arm().as_dict()
    hue_order = ["unstruc", "struc"]
    strip_kw = dict(size=5, linewidth=0.3, alpha=0.7, palette=palette)
    bar_kw = dict(alpha=0.5, palette=palette)
    plot_kw = dict(
        x="param_names", y="param_values", hue="grp", hue_order=list(hue_order)
    )

    for i, block in enumerate(scopes):
        scope_df = data_df[data_df["fit_scope"] == block]

        for i1, param in enumerate(params):
            if isinstance(param, tuple):
                param_df = scope_df[scope_df["param_names"].isin(param)]
            else:
                param_df = scope_df[scope_df["param_names"] == param]

            if param == "nll":
                trial_vals = scope_df.loc[
                    scope_df["param_names"] == "n_trials", "param_values"
                ].to_numpy()

                param_df.loc[:, "param_values"] = (
                    param_df["param_values"].to_numpy() / trial_vals
                )

            ax = fig.subplot(fig.gs[i, i1])
            SeabornPlotter(data=param_df, ax=ax, **plot_kw).stripplot(
                **strip_kw
            ).barplot(**bar_kw).bootstrap_test()
            ax.set_xlabel("")
            if getattr(ax, "legend_", None):
                ax.legend_.remove()
            if i1 == 0:
                ax.set_title(f"fit scope={block}")
            # if p == 1 and rotate_beta_ticks:
            #     xtick_format(ax, rotation=45)


__all__ = [
    "plot_param_block_grid",
    "build_actual_vs_sim_perf_df",
    "build_actual_vs_sim_swp_df",
]
__all__.append("MODEL_PARAM_GROUPS")
