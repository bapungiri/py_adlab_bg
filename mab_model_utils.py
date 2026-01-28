"""Reusable plotting/model-fit helpers for MAB analyses."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from statplotannot.plots import SeabornPlotter
from statplotannot.plots.plot_utils import xtick_format
import matplotlib.gridspec as gridspec
from mab_colors import Palette2Arm
from neuropy import plotting

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
        "width_ratios": [2, 1, 1, 1],
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
            if i == 0:
                ax.set_title(f"fit scope={block}")
            # if p == 1 and rotate_beta_ticks:
            #     xtick_format(ax, rotation=45)


__all__ = ["plot_param_block_grid"]
__all__.append("MODEL_PARAM_GROUPS")
