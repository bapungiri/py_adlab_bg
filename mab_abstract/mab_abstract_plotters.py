"""Shared plotting helpers for the Cosyne 2026 abstract notebook.

Centralizing these keeps every tiered figure in the abstract using the same
plot style/statistics (boxplot_filled + BH-corrected bootstrap test), rather
than each figure cell hand-copying and silently drifting from the others.
Data-prep (make_tiered_task) lives in mab_abstract_datagen instead, since
it's used to build GroupData products, not to draw figures.
"""

import numpy as np
from mab_colors import Palette2Arm
from statplotannot.plots import SeabornPlotter, xtick_format, Fig

TIER_TITLES = ["All (normalized)", "Low-Low", "High-Low", "High-High"]


def get_fig(nrows, ncols):
    return Fig(size=(8.5, 11), nrows=nrows, ncols=ncols, fontsize=8)


def plot_tier_row(
    fig,
    row,
    df,
    y_cols,
    ylabel,
    titles=TIER_TITLES,
    hue_order=("unstruc", "struc"),
    palette=None,
    ylim=None,
    ytick_step=None,
    sep=0.7,
    fontsize=6,
    stat_method="bootstrap",
    comparisons_correction="BH",
    subject_col="name",
    stat_kwargs=None,
    rotation=45,
):
    """Draw one row of tiered boxplots (one panel per y column) into fig.

    Every panel is built the same way — boxplot_filled, plus one shared
    significance test — so any figure built from this helper shares
    identical statistics and style with every other one.

    Parameters
    ----------
    fig : neuropy.plotting.Fig
    row : int
        Row index into fig.gs to draw this panel row into.
    df : pd.DataFrame
        Must have a "trial_id" (str) and "group" column.
    y_cols : list of str
        One column name per panel, e.g. [y_all, y_low_low, y_high_low, y_high_high].
    ylabel : str
    titles : list of str, optional
        Panel titles, default TIER_TITLES.
    ylim : tuple, optional
        Applied to every panel in the row if given.
    ytick_step : float, optional
        If given (with ylim), sets explicit shared y-ticks across the row so
        tick density/spacing can't drift panel-to-panel.
    stat_method : {"bootstrap", "cluster_permutation"}, optional
        Which significance test to run per panel:

        - "bootstrap": independent per-trial-window bootstrap test, with
          `comparisons_correction` applied across all windows in the panel.
          Simple and fast, but treats windows as independent tests even
          though adjacent windows are temporally correlated.
        - "cluster_permutation": cluster-based permutation test across the
          whole row of windows (see SeabornPlotter.stat_cluster_permutation).
          Shuffles group labels across subjects — keeping each subject's
          whole trajectory intact — so it uses the actual temporal
          correlation instead of needing to correct for it. Reports
          contiguous significant clusters rather than one star per window.

        Default "bootstrap".
    comparisons_correction : str, optional
        Only used when stat_method="bootstrap".
    subject_col : str, optional
        Only used when stat_method="cluster_permutation" — column
        identifying the repeated-measures unit (e.g. animal name).
        Default "name".
    stat_kwargs : dict, optional
        Extra keyword arguments forwarded to the chosen stat method (e.g.
        n_permutations, seed for cluster_permutation; p_thresh, n_resamples
        for bootstrap).

    Returns
    -------
    list of Axes
    """
    if palette is None:
        palette = Palette2Arm().as_dict()
    if stat_kwargs is None:
        stat_kwargs = {}
    if stat_method not in ("bootstrap", "cluster_permutation"):
        raise ValueError("stat_method must be 'bootstrap' or 'cluster_permutation'")

    axes = []
    for i, y in enumerate(y_cols):
        ax = fig.subplot(fig.gs[row, i])
        plotter = SeabornPlotter(
            data=df, x="trial_id", y=y, hue="group", hue_order=list(hue_order), ax=ax
        ).boxplot_filled(palette=palette, sep=sep)

        if stat_method == "bootstrap":
            plotter.stat_bootstrap(
                comparisons_correction=comparisons_correction,
                fontsize=fontsize,
                **stat_kwargs,
            )
        else:
            plotter.stat_cluster_permutation(
                subject_col=subject_col, fontsize=fontsize, **stat_kwargs
            )

        if ylim is not None:
            ax.set_ylim(*ylim)
            if ytick_step is not None:
                ax.set_yticks(np.arange(ylim[0], ylim[1] + ytick_step / 2, ytick_step))

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Trials")
        ax.set_title(titles[i])
        xtick_format(ax, rotation=rotation)
        axes.append(ax)

    return axes
