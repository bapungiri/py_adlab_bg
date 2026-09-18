"""Versioned-results storage (GroupData/VersionedAccessor), split out of
mab_subjects.py so its self-rewriting stub (see `GroupData._write_stub`)
only ever touches this small, dedicated file -- never the hand-edited
Struc/Unstruc/MABData classes in mab_subjects.py.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd


class VersionedAccessor:
    def __init__(self, parent: "GroupData", basename: str):
        self.parent = parent
        self.basename = basename

    # load all versioned files
    @property
    def all(self):
        return self.parent._all_versions(self.basename)

    # load the newest version
    @property
    def latest(self):
        fp = self.parent._latest_version(self.basename)
        return self.parent.load(fp)["data"]

    # calling the object returns latest
    def __call__(self):
        return self.latest

    # give list of version names (basename_yyyymmdd_hhmmss.npy)
    @property
    def versions(self):
        return [p.stem for p in self.parent._all_versions(self.basename)]


class GroupData:
    if TYPE_CHECKING:
        # === BEGIN AUTO-GENERATED BASENAME ANNOTATIONS (written by _write_stub) ===
        abstract_perf_combinations: VersionedAccessor
        abstract_perf_difficulty_level: VersionedAccessor
        abstract_perf_tier: VersionedAccessor
        abstract_swp_tier: VersionedAccessor
        abstract_tau_tier: VersionedAccessor
        abstract_tau_tier_double: VersionedAccessor
        bias: VersionedAccessor
        breaks_vs_swp: VersionedAccessor
        cheeku: VersionedAccessor
        compressibility_ratio: VersionedAccessor
        fit_MoARegime_policy_combinations: VersionedAccessor
        fit_Qlearn2Regime_policy_combinations: VersionedAccessor
        fit_multi_policy: VersionedAccessor
        fit_multi_policy_combinations: VersionedAccessor
        fit_multi_policy_lesion: VersionedAccessor
        fit_qlearnH: VersionedAccessor
        fit_qlearnH_sim: VersionedAccessor
        fit_qlearnRegimeDiffStays_policy_combinations: VersionedAccessor
        fit_qlearn_combinations_lesion: VersionedAccessor
        fit_qlearn_corr_uncorr: VersionedAccessor
        fit_qlearn_easy_hard: VersionedAccessor
        fit_qlearn_low_high_combinations: VersionedAccessor
        fit_qlearn_per_prob: VersionedAccessor
        fit_qlearn_policy: VersionedAccessor
        fit_si: VersionedAccessor
        fit_si_sim: VersionedAccessor
        fit_thompson_split: VersionedAccessor
        fit_ucb: VersionedAccessor
        gcca_meta_rnn: VersionedAccessor
        gcca_rnn_fit: VersionedAccessor
        gcca_rnn_fit_separate: VersionedAccessor
        logreg: VersionedAccessor
        logreg_AAdataset: VersionedAccessor
        model_recovery: VersionedAccessor
        nll_fit_multi_policy: VersionedAccessor
        nll_history_rnn_fit: VersionedAccessor
        param_recovery_qlearn: VersionedAccessor
        param_recovery_si: VersionedAccessor
        pca_mean_rnn_fit: VersionedAccessor
        pca_rnn_fit: VersionedAccessor
        perf_AAdataset: VersionedAccessor
        perf_AAdataset_Block1: VersionedAccessor
        perf_all_corr_uncorr: VersionedAccessor
        perf_animal_vs_rnn_fit: VersionedAccessor
        perf_difficulty_level: VersionedAccessor
        perf_easy_hard_transitions: VersionedAccessor
        perf_easy_to_hard: VersionedAccessor
        perf_fit_multi_policy: VersionedAccessor
        perf_flip_transitions: VersionedAccessor
        perf_learning: VersionedAccessor
        perf_logreg: VersionedAccessor
        perf_logreg_AAdataset: VersionedAccessor
        perf_mat_fit_multi_policy: VersionedAccessor
        perf_meta_learning: VersionedAccessor
        perf_old_vs_new: VersionedAccessor
        perf_per_day: VersionedAccessor
        perf_probability_matrix: VersionedAccessor
        perf_short_blocks: VersionedAccessor
        perf_sliding: VersionedAccessor
        perf_swp_fit_multi_policy: VersionedAccessor
        perf_vs_lesion: VersionedAccessor
        phase_portrait_lesion_model_vs_rnn: VersionedAccessor
        phase_portrait_model_vs_rnn: VersionedAccessor
        poster_perf_difficulty_level: VersionedAccessor
        poster_perf_probability_matrix: VersionedAccessor
        poster_perf_vs_mpfc_lesion: VersionedAccessor
        poster_perf_vs_ofc_lesion: VersionedAccessor
        qlearnH: VersionedAccessor
        reward_prob: VersionedAccessor
        reward_probability_matrix: VersionedAccessor
        rnn_fit_accuracy: VersionedAccessor
        rnn_lr_search_perf: VersionedAccessor
        rnn_models_fit_accuracy: VersionedAccessor
        simulated_policies_perf: VersionedAccessor
        state_traj_Qlearn2Regime: VersionedAccessor
        state_traj_Qlearn2Regime_avg: VersionedAccessor
        switch_density: VersionedAccessor
        switch_prob: VersionedAccessor
        switch_prob_by_delta_prob: VersionedAccessor
        switch_prob_by_trial_100trials: VersionedAccessor
        switch_prob_logreg: VersionedAccessor
        switch_prob_logreg_AAdataset: VersionedAccessor
        switch_prob_seq: VersionedAccessor
        switch_prob_seq_previous: VersionedAccessor
        switch_pure_prob_seq: VersionedAccessor
        switching_reward_rate: VersionedAccessor
        switchprob_si: VersionedAccessor
        swp_AAdataset_Block1: VersionedAccessor
        swp_after_reward: VersionedAccessor
        swp_by_quartiles: VersionedAccessor
        swp_trial_history: VersionedAccessor
        # === END AUTO-GENERATED BASENAME ANNOTATIONS ===
        pass

    def __init__(self, keep_versions: int = 3):
        self.keep_versions = keep_versions

        if os.name == "nt":
            self.path = Path(
                "C:/Users/asheshlab/OneDrive/academia/analyses/adlab/results"
            )
        else:
            self.path = Path("/mnt/pve/Homes/bapun/Data/results")

        self.path.mkdir(exist_ok=True, parents=True)

        # discover basenames from existing files
        self._basenames = self._discover_basenames()

        # Debug: print discovered basenames
        # print(f"Discovered basenames: {self._basenames}")

        # Dynamically create attributes for autocomplete
        for basename in self._basenames:
            setattr(self, basename, VersionedAccessor(self, basename))

        # write stub at init so VS Code sees current basenames
        if os.name == "nt":
            self._write_stub()

    def __dir__(self):
        """Enable autocomplete for discovered basenames"""
        base_attrs = list(super().__dir__())
        return base_attrs + list(self._basenames)

    def _discover_basenames(self):
        basenames = set()
        for f in self.path.glob("*.npy"):  # Uses self.path which is already defined
            stem = f.stem
            # assume format: basename_YYYYMMDD_HHMMSS
            parts = stem.rsplit("_", 2)  # basename, yyyymmdd, hhmmss
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                basenames.add(parts[0])
        return sorted(basenames)

    def filename_to_attr(self, stem: str) -> str | None:
        """Extract basename from a versioned filename stem (basename_YYYYMMDD_HHMMSS)."""
        parts = stem.rsplit("_", 2)
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            return parts[0]
        return None

    def __getattr__(self, name: str):
        if name in self._basenames:
            return VersionedAccessor(self, name)
        raise AttributeError(f"No attribute or data basename named '{name}'")

    def _versioned_name(self, basename: str):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{basename}_{now}.npy"

    def _all_versions(self, basename: str):
        files = sorted(self.path.glob(f"{basename}_*.npy"))
        return [f for f in files if self.filename_to_attr(f.stem) == basename]

    def _latest_version(self, basename: str):
        files = self._all_versions(basename)
        if not files:
            raise FileNotFoundError(f"No versions found for '{basename}'")
        return files[-1].stem  # no .npy

    def save(self, data, basename: str, clean: bool = True, write_stub: bool = True):
        # convert DataFrame to dict
        if isinstance(data, pd.DataFrame):
            data = data.to_dict()

        filename = self._versioned_name(basename)
        np.save(self.path / filename, {"data": data})
        print(f"[GroupData] Saved: {filename}")

        # register new basename if new
        if basename not in self._basenames:
            self._basenames.append(basename)
            setattr(self, basename, VersionedAccessor(self, basename))
            if write_stub:
                self._write_stub()  # update stub on new basename

        if clean:
            self._cleanup_versions(basename)

        return filename

    def _cleanup_versions(self, basename: str):
        files = self._all_versions(basename)
        if len(files) > self.keep_versions:
            old = files[: len(files) - self.keep_versions]
            for f in old:
                f.unlink()
                print(f"Deleted old version: {f.name}")

    def load(self, stem: str):
        data = np.load(self.path / f"{stem}.npy", allow_pickle=True).item()
        try:
            data["data"] = pd.DataFrame(data["data"])
        except Exception:
            pass

        # ensure basename from stem is registered for autocomplete
        base = self.filename_to_attr(stem)
        if base and base not in self._basenames:
            self._basenames.append(base)
            setattr(self, base, VersionedAccessor(self, base))
            self._write_stub()

        return data

    _STUB_BEGIN = "        # === BEGIN AUTO-GENERATED BASENAME ANNOTATIONS (written by _write_stub) ==="
    _STUB_END = "        # === END AUTO-GENERATED BASENAME ANNOTATIONS ==="

    def _write_stub(self):
        """Rewrite the `if TYPE_CHECKING:` basename-annotation block in this
        module's own source, so Pylance/VS Code autocompletes current
        basenames on GroupData.

        Deliberately NOT a sibling `mab_group_data.pyi` file: once such a
        stub exists, Pyright/Pylance uses it *exclusively* for this module's
        type info, hiding VersionedAccessor/GroupData's real methods from
        static completion. Splicing annotations into the real source
        instead means Pylance keeps reading the actual file for everything.
        """
        source_path = Path(__file__)
        text = source_path.read_text()

        try:
            start = text.index(self._STUB_BEGIN) + len(self._STUB_BEGIN)
            stop = text.index(self._STUB_END)
        except ValueError:
            raise RuntimeError(
                "_write_stub: basename-annotation markers not found in "
                f"{source_path.name} -- did someone edit/remove them?"
            )

        annotations = "".join(
            f"        {b}: VersionedAccessor\n" for b in sorted(self._basenames)
        )
        new_text = text[:start] + "\n" + annotations + text[stop:]

        if new_text != text:
            source_path.write_text(new_text)
