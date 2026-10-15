"""Builders describing the task structure each subject experienced."""

import numpy as np

from ._core import group_builder
from mab_data_core import MABData


@group_builder("prob_combinations")
def prob_combinations(exp: MABData):
    """Proportion of the exp's sessions that started with each reward
    probability combination, in long format.

    Built from Bandit2Arm.get_prob_hist2d, with mirror-image combinations
    (port1/port2 swapped) folded together, so each row is an unordered
    (p_high, p_low) pair regardless of which port had the higher probability.
    One row per pair with p_high >= p_low, including pairs the exp never saw
    (prop = 0).
    """
    H, xedges, _ = exp.b2a.get_prob_hist2d(stat="prop")  # H[p2_bin, p1_bin]
    mat = np.tril(H + H.T, k=-1) + np.diag(np.diag(H))  # [higher, lower]
    probs = np.round((xedges[:-1] + xedges[1:]) / 2, 2)

    i_high, i_low = np.tril_indices_from(mat)
    return dict(p_high=probs[i_high], p_low=probs[i_low], prop=mat[i_high, i_low])
