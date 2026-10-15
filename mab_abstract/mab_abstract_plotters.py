"""Shared plotting helpers for the Cosyne 2026 abstract notebook.

Centralizing these keeps every tiered figure in the abstract using the same
plot style/statistics (boxplot_filled + BH-corrected bootstrap test), rather
than each figure cell hand-copying and silently drifting from the others.
The GroupData products themselves are built by mab_builders (see the
"abstract_*" entries in mab_builders.build_all.PRODUCTS).
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import offset_copy
from mab_colors import Palette2Arm
from statplotannot.plots import SeabornPlotter, xtick_format, Fig

TIER_TITLES = {
    "all": "All",
    "equalized": "All (normalized)",
    "low-low": "Low-Low",
    "high-low": "High-Low",
    "high-high": "High-High",
    "dependent": "Dependent",
    "independent": "Independent",
}


# Each gridspec cell is this size (inches), so panels come out at their printed
# size on the single-figure A4 layout and are placed in Affinity at 100%: a
# probability grid is ~0.7 in square, a trial curve ~0.65 x 0.8 in.
CELL_SIZE = (0.9, 0.85)
# smaller cells for probability-grid figures: grids ~0.62 in square (fonts unchanged)
GRID_CELL_SIZE = (0.83, 0.78)
# every probability grid is set to exactly this size (inches) by fix_grid_size
GRID_SIZE = 0.6


def three_yticks(ax, lo=0.0, hi=None):
    """Exactly three y-ticks -- bottom, middle, top -- with the axis spanning them.

    The step is the smallest round value (1, 1.5, 2, 2.5 or 5 x 10^k) whose two steps
    above the bottom tick cover 'hi' (default: the top of the plotted data). The
    bottom tick is 'lo', or with lo=None a multiple of the step at or below the
    bottom of the plotted data (for data that doesn't start near 0).
    """
    if hi is None:
        hi = ax.dataLim.y1
    data_lo = ax.dataLim.y0 if lo is None else lo
    raw = (hi - data_lo) / 2
    mag = 10 ** np.floor(np.log10(raw))
    for step in mag * np.array([1, 1.5, 2, 2.5, 5, 10, 15, 20, 25, 50]):
        bottom = lo if lo is not None else np.floor(data_lo / step + 1e-9) * step
        if bottom + 2 * step >= hi - 1e-12:
            break
    ticks = bottom + step * np.arange(3)
    ax.set_ylim(ticks[0], ticks[-1])
    ax.set_yticks(ticks)
    return ticks


def fix_axes_size(fig, sizes):
    """Set axes to exact sizes, 'sizes' = {ax: (width, height)} in inches. Call
    just before saving: it runs the layout, then freezes it and resizes each
    axes about its own centre (colorbars, titles and annotations move with it).
    Only shrinks reliably: a size larger than the layout gave can overlap
    neighbouring panels.
    """
    f = getattr(fig, "fig", fig)
    f.canvas.draw()  # let constrained layout place everything first
    f.set_layout_engine("none")
    fig_w, fig_h = f.get_size_inches()
    for ax, (width, height) in sizes.items():
        pos = ax.get_position()
        cx, cy = pos.x0 + pos.width / 2, pos.y0 + pos.height / 2
        w, h = width / fig_w, height / fig_h
        ax.set_position([cx - w / 2, cy - h / 2, w, h])
    return fig


def fix_grid_size(fig, size=GRID_SIZE):
    """Make every probability grid (equal-aspect axes) in 'fig' exactly 'size'
    inches square, so grids match across figures whatever their titles and
    colorbars. Call just before saving (see fix_axes_size)."""
    f = getattr(fig, "fig", fig)
    grids = [ax for ax in f.axes if ax.get_aspect() == 1.0]
    return fix_axes_size(fig, {ax: (size, size) for ax in grids})


GROUP_LETTER = {"unstruc": "U", "struc": "S"}


def group_token(group):
    """'U' / 'S' in the group's color, bold: a piece for rich_title."""
    color = Palette2Arm().as_dict()[group]
    return GROUP_LETTER[group], dict(color=color, fontweight="bold")


def s_minus_u():
    """'S - U' with S and U in their group colors, bold: pieces for rich_title."""
    return [group_token("struc"), (" - ", {}), group_token("unstruc")]


def rich_title(ax, lines, pad=4, linespacing=1.25):
    """Title built from differently styled pieces, centred over 'ax'.

    lines : list of lines, top first; each line a list of pieces, each piece a
        str or a (text, text_kwargs) pair, e.g.
        [["P(High): ", *s_minus_u()], ["(n=11 vs 8)"]].
    Pieces are placed side by side by their rendered widths, in points from the
    top centre of the axes, so they stay put when the layout resizes the axes.
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    size = FontProperties(size=mpl.rcParams["axes.titlesize"]).get_size_in_points()
    ax.set_title("")
    for i, parts in enumerate(lines):
        parts = [(p, {}) if isinstance(p, str) else p for p in parts]
        texts = [
            ax.text(0.5, 1, s, fontsize=size, ha="left", va="baseline", **kw)
            for s, kw in parts
        ]
        widths = [t.get_window_extent(renderer).width * 72 / fig.dpi for t in texts]
        x = -sum(widths) / 2
        y = pad + (len(lines) - 1 - i) * size * linespacing  # lines stack upwards
        for t, w in zip(texts, widths):
            t.set_transform(
                offset_copy(ax.transAxes, fig=fig, x=x, y=y, units="points")
            )
            x += w
    return ax


def get_fig(nrows, ncols, cell_size=CELL_SIZE):
    fig = Fig(
        size=(ncols * cell_size[0], nrows * cell_size[1]),
        nrows=nrows,
        ncols=ncols,
        fontsize=6,
        axis_lw=0.8,  # also the tick width
        tick_size=2,
    )
    mpl.rcParams["lines.linewidth"] = 0.8  # curves (lineplot) default
    return fig


def window_labels(trial_id, trial_window):
    """'start-stop' trial labels for 1-based 'trial_window'-trial window ids."""
    starts = (np.asarray(trial_id) - 1) * trial_window + 1
    return [f"{start}-{start + trial_window - 1}" for start in starts]


def prob_grid_group_means(df, mat_col, groups=("unstruc", "struc"), min_animals=1):
    """Per-group mean of a per-subject probability grid (e.g. mab_builders'
    perf_probability_matrix/swp_probability_matrix).

    Each row of 'df' holds one subject's grid in 'mat_col', indexed by its own
    sorted probabilities in "probs". Grids are placed on the union of
    probabilities (subjects can differ in which they saw), averaged ignoring
    NaN, and cut to the lower triangle incl. diagonal, indexed [higher, lower].
    Cells sampled by fewer than 'min_animals' subjects are NaN.

    Returns
    -------
    probs : 1D array
    means : dict
        group -> mean grid
    counts : dict
        group -> number of subjects
    """
    probs = np.unique(np.concatenate(df["probs"].to_list()))
    means, counts = {}, {}
    for group in groups:
        mats = np.stack(prob_grids_on(df[df["group"] == group], mat_col, probs))
        mean = np.nanmean(mats, axis=0)
        mean[(~np.isnan(mats)).sum(axis=0) < min_animals] = np.nan
        mean[np.triu_indices_from(mean, k=1)] = np.nan
        means[group], counts[group] = mean, len(mats)
    return probs, means, counts


def prob_grids_on(df, mat_col, probs):
    """Each row's grid in 'mat_col' (indexed by its own "probs") placed on the
    common sorted 'probs', NaN where the row didn't sample a probability."""
    mats = []
    for _, row in df.iterrows():
        idx = np.searchsorted(probs, row["probs"])
        mat = np.full((len(probs), len(probs)), np.nan)
        mat[np.ix_(idx, idx)] = row[mat_col]
        mats.append(mat)
    return mats


def draw_prob_grid(ax, mat, probs, has_diag, **pcolor_kw):
    """Lower-triangle probability grid, 'mat' indexed [higher, lower] (higher p
    on x). 'has_diag' shows the equal-probability diagonal. Returns the mesh,
    for a colorbar."""
    im = ax.pcolormesh(mat.T, shading="auto", **pcolor_kw)
    n = len(probs)
    ticks, labels = np.arange(n) + 0.5, [f"{p * 100:.0f}" for p in probs]
    ax.set_xticks(ticks, labels)  # tick label size from the figure style (get_fig)
    ax.set_yticks(ticks, labels)
    ax.set_xlim(0 if has_diag else 1, n)
    ax.set_ylim(0, n if has_diag else n - 1)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(True)
    ax.set_xlabel("Higher p")
    ax.set_ylabel("Lower p")
    return im


# color-block width per grid measure for diverging_bins, shared by every
# difference-grid figure so a measure reads the same everywhere
BIN_STEPS = {"perf_mat": 0.05, "swp_mat": 0.01}


def diverging_bins(halfrange, step=None, cmap="coolwarm", zero_bin=True):
    """Binned diverging colors for difference grids: equal-width blocks of
    'step' out to 'halfrange'.

    zero_bin : bool
        True: blocks centred on multiples of 'step', one centred on 0 (grey with
        'coolwarm', so near-zero cells stay distinct from empty white cells).
        False: blocks split at 0, every negative block a shade of the cmap's low
        end (blue) and every positive one of its high end (red), darker further
        from 0; no neutral block.

    'step' defaults to a round value (1, 2, 2.5 or 5 x 10^k) giving about three
    blocks per side. Values beyond the outer blocks take the end colors.
    Returns dict(cmap=..., norm=...) to pass on as **kwargs.
    """
    if step is None:
        raw = halfrange / 3
        mag = 10 ** np.floor(np.log10(raw))
        step = mag * min((1, 2, 2.5, 5, 10), key=lambda m: abs(m * mag - raw))
    n = int(np.ceil(halfrange / step - 1e-9))
    n += n % 2  # even blocks per side, so the colorbar can tick ends, 0 and halfway
    base = plt.get_cmap(cmap)
    if zero_bin:
        edges = (np.arange(-n, n + 2) - 0.5) * step
        colors = base(np.linspace(0, 1, 2 * n + 1))
    else:
        edges = np.arange(-n, n + 1) * step
        # each half from its end colour to just short of the neutral middle
        colors = np.vstack(
            [base(np.linspace(0, 0.4, n)), base(np.linspace(0.6, 1, n))]
        )
    return dict(
        cmap=ListedColormap(colors),
        norm=BoundaryNorm(edges, len(colors), clip=True),
    )


def add_side_colorbar(fig, ax, im, label, fontsize=4, fmt="%.2f", max_ticks=5):
    """Colorbar in an inset right of 'ax', so 'ax' keeps its size."""
    cax = ax.inset_axes([1.08, 0.15, 0.035, 0.7])  # x, y, width, height (axes fraction)
    cb = fig.fig.colorbar(im, cax=cax)
    cb.ax.minorticks_off()
    if isinstance(im.norm, BoundaryNorm):  # binned colors (see diverging_bins)
        edges = im.norm.boundaries
        # blocks split at 0: ticks on the block edges; a block centred on 0: on
        # the block centres only (no boundary ticks)
        ticks = edges if np.isclose(edges, 0).any() else (edges[:-1] + edges[1:]) / 2
    else:  # continuous: bottom, middle and top
        ticks = np.linspace(im.norm.vmin, im.norm.vmax, 3)
    # at most 'max_ticks', evenly spaced, always including both ends and the
    # middle (0 on a diverging bar): the smallest stride dividing the half-span
    # that leaves few enough ticks
    span = len(ticks) - 1
    half = span // 2 if span % 2 == 0 else span
    stride = next(
        (k for k in range(1, half + 1) if half % k == 0 and span // k + 1 <= max_ticks),
        span,
    )
    ticks = ticks[::stride]
    cb.set_ticks(ticks)
    cb.set_ticklabels([fmt % t for t in ticks])
    cb.set_label(label, fontsize=fontsize)
    cb.ax.tick_params(labelsize=fontsize, width=0.4, length=1.2)
    cb.outline.set_linewidth(0.3)
    return cb


def plot_tier_row(
    fig,
    row,
    df,
    value_col,
    ylabel,
    tiers=("equalized", "low-low", "high-low", "high-high"),
    tier_col="tier_label",
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
    """Draw one row of tiered boxplots (one panel per tier) into fig.

    Every panel is built the same way — boxplot_filled, plus one shared
    significance test — so any figure built from this helper shares
    identical statistics and style with every other one.

    Parameters
    ----------
    fig : neuropy.plotting.Fig
    row : int
        Row index into fig.gs to draw this panel row into.
    df : pd.DataFrame
        Long-format tier product (e.g. mab_builders' perf_tier/swp_tier). Must
        have "trial_id" (str), "group", 'tier_col' and 'value_col' columns.
    value_col : str
        Column plotted on the y axis, e.g. "tier_perf" or "switch_prob".
    ylabel : str
    tiers : sequence of str, optional
        'tier_col' values to plot, one panel each, in order.
    tier_col : str, optional
        Column holding the tier label, default "tier_label".
    titles : dict, optional
        Tier label -> panel title, default TIER_TITLES.
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
    for i, tier in enumerate(tiers):
        ax = fig.subplot(fig.gs[row, i])
        plotter = SeabornPlotter(
            data=df[df[tier_col] == tier],
            x="trial_id",
            y=value_col,
            hue="group",
            hue_order=list(hue_order),
            ax=ax,
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
        ax.set_title(titles[tier])
        xtick_format(ax, rotation=rotation)
        axes.append(ax)

    return axes
