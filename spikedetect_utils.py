# Spike detection and pairing helpers

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation as mad
from neuropy.plotting import Fig


def detect_spikes_1ch(signal, fs, k=5.0, min_isi_ms=0.7):
    """
    Detect negative-going spikes using MAD threshold.
    Returns indices and positive amplitudes (baseline-relative).
    """
    baseline = np.median(signal)
    mad_y = 1.4826 * mad(signal)
    thr = baseline - k * mad_y
    min_distance = max(1, int(min_isi_ms * 1e-3 * fs))
    trough_idx, props = find_peaks(-signal, height=-(thr), distance=min_distance)
    # positive amplitudes relative to baseline (peak-to-baseline)
    amps = (baseline - signal[trough_idx]).astype(float)
    return trough_idx, amps


def detect_spikes_all(signals, time, fs, k=5.0, min_isi_ms=0.7):
    """
    signals: array (n_channels, n_samples)
    time: array (n_samples,)
    Returns: lists per channel of spike times and amplitudes
    """
    n_ch = signals.shape[0]
    spike_times = []
    spike_amps = []
    spike_idxs = []
    for i in range(n_ch):
        idxs, amps = detect_spikes_1ch(signals[i], fs=fs, k=k, min_isi_ms=min_isi_ms)
        spike_idxs.append(idxs)
        spike_times.append(time[idxs])
        spike_amps.append(amps)
    return spike_idxs, spike_times, spike_amps


def pair_amplitudes(spike_times, spike_amps, window_ms=0.3):
    """
    Pair spikes across channel pairs if they occur within +/- window_ms.
    Returns dict: (i,j) -> (amps_i, amps_j) arrays for co-occurring spikes.
    """
    win = window_ms * 1e-3
    n_ch = len(spike_times)
    pairs = {}
    for i in range(n_ch):
        ti, ai = spike_times[i], spike_amps[i]
        if len(ti) == 0:
            continue
        for j in range(i + 1, n_ch):
            tj, aj = spike_times[j], spike_amps[j]
            if len(tj) == 0:
                continue
            x, y = [], []
            # use binary search windows for speed
            for t_i, a_i in zip(ti, ai):
                left = np.searchsorted(tj, t_i - win, side="left")
                right = np.searchsorted(tj, t_i + win, side="right")
                if right > left:
                    sub_t = tj[left:right]
                    sub_a = aj[left:right]
                    m = np.argmin(np.abs(sub_t - t_i))
                    x.append(a_i)
                    y.append(sub_a[m])
            if len(x) > 0:
                pairs[(i, j)] = (np.asarray(x), np.asarray(y))
    return pairs


def plot_pair_scatter(pairs, channel_names, title):
    if not pairs:
        print(f"{title}: no coincident spikes found.")
        return
    import math

    n = len(pairs)
    cols = min(2, n)
    rows = math.ceil(n / cols)
    # fig, axs = plt.subplots(rows, cols, figsize=(5, 4))
    fig = Fig(nrows=rows, ncols=cols, size=(5, 4))
    for k, ((ci, cj), (xi, yj)) in enumerate(pairs.items()):
        r, c = divmod(k, cols)
        # ax = axs[r][c]
        ax = fig.subplot(fig.gs[r, c])
        ax.scatter(xi, yj, s=8, c="k", alpha=0.6)
        ax.set_xlabel(f"{channel_names[ci]} (uV)")
        ax.set_ylabel(f"{channel_names[cj]} (uV)")
        ax.set_title(
            f"{channel_names[ci]} vs {channel_names[cj]} (N={len(xi)})", fontsize=8
        )
        # ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        # ax.axvline(0, color="k", lw=0.5, alpha=0.5)
    # remove unused axes
    # for k in range(n, rows * cols):
    #     fig.delaxes(axs[k // cols][k % cols])
    fig.fig.suptitle(title)
    # fig.tight_layout()
    plt.show()


def plot_spike_locations(lfp, time, spike_times, spike_amps, channel_names, k):
    """
    Plot spike locations for each channel.
    """
    n_ch = len(spike_times)
    fig, axs = plt.subplots(n_ch, 1, figsize=(10, 5), sharex=True)
    for i in range(n_ch):
        baseline = np.median(lfp[i])
        mad_y = 1.4826 * mad(lfp[i])
        thr = baseline - k * mad_y

        axs[i].plot(time, lfp[i], color="gray", linewidth=0.5)
        axs[i].axhline(thr, color="r", linestyle="--")
        axs[i].plot(spike_times[i], -spike_amps[i], "k.", markersize=2)
        axs[i].set_ylabel(channel_names[i])
    axs[-1].set_xlabel("Time (s)")
    fig.suptitle("Spike Locations")
    plt.show()


    
