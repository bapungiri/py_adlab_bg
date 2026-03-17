from typing import Any

class VersionedAccessor:
    def __init__(self, parent: 'GroupData', basename: str): ...
    @property
    def all(self) -> Any: ...
    @property
    def latest(self) -> Any: ...
    def __call__(self) -> Any: ...
    @property
    def versions(self) -> list[str]: ...

class GroupData:
    def __init__(self, keep_versions: int = 3): ...
    def save(self, data: Any, basename: str, clean: bool = True) -> str: ...
    def load(self, stem: str) -> dict: ...
    bias: VersionedAccessor
    cheeku: VersionedAccessor
    fit_qlearnH: VersionedAccessor
    fit_qlearnH_sim: VersionedAccessor
    fit_si: VersionedAccessor
    fit_si_sim: VersionedAccessor
    fit_thompson_split: VersionedAccessor
    fit_ucb: VersionedAccessor
    logreg: VersionedAccessor
    perf_AAdataset_Block1: VersionedAccessor
    perf_all_corr_uncorr: VersionedAccessor
    perf_easy_hard_transitions: VersionedAccessor
    perf_easy_to_hard: VersionedAccessor
    perf_flip_transitions: VersionedAccessor
    perf_learning: VersionedAccessor
    perf_logreg: VersionedAccessor
    perf_meta_learning: VersionedAccessor
    perf_probability_matrix: VersionedAccessor
    perf_sliding: VersionedAccessor
    perf_vs_lesion: VersionedAccessor
    qlearnH: VersionedAccessor
    reward_prob: VersionedAccessor
    reward_probability_matrix: VersionedAccessor
    switch_density: VersionedAccessor
    switch_prob: VersionedAccessor
    switch_prob_by_delta_prob: VersionedAccessor
    switch_prob_logreg: VersionedAccessor
    switch_prob_seq: VersionedAccessor
    switch_prob_seq_previous: VersionedAccessor
    switchprob_si: VersionedAccessor
    swp_AAdataset_Block1: VersionedAccessor
    swp_by_quartiles: VersionedAccessor
    swp_trial_history: VersionedAccessor