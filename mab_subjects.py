from pathlib import Path
import os
import numpy as np
import neuropy
from banditpy.core import TwoArmedBandit
from typing import List
import pandas as pd
from dataclasses import dataclass


class MABData:
    def __init__(self, basepath, tag=None):
        basepath = Path(basepath)
        try:
            csv_file = sorted(basepath.glob("*.csv"))
            fp = csv_file.with_suffix("")
        except:
            fp = basepath / basepath.name

        self.filePrefix = fp
        self.sub_name = fp.name

        self.tag = tag

        if (f := self.filePrefix.with_suffix(".animal.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.animal = neuropy.core.Animal.from_dict(d)
            self.name = self.animal.name + self.animal.day

        self.mab = TwoArmedBandit.from_csv(fp.with_suffix(".csv"))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.sub_name})\n"


# basedir = Path(r"D:\\Data\\mab")
# BewilderbeastExp1 = MABData(
#     basedir / r"anirudh_data\bewilderbeast\BewilderbeastExp1Structured"
# )
# BuffalordExp1 = MABData(basedir / r"anirudh_data\buffalord\BuffalordExp1Structured")

# exp1 = [BewilderbeastExp1, BuffalordExp1]


class Group:
    tag = None

    # @property
    # def basedir(self):
    if os.name == "nt":
        basedir = Path(r"D:\\Data\\mab")
    else:
        basedir = ath("/data/Clustering/sessions/")

    def _process(self, rel_path):
        return [MABData(self.basedir / rel_path, self.tag)]

    def data_exist(self):
        self.allsess


class Struc(Group):
    @property
    def BewilderbeastExp1(self):
        return self._process(r"anirudh_data\bewilderbeast\BewilderbeastExp1Structured")

    # @property
    # def bratexp2(self): # bad animal
    #     return self._process("anirudh_data\brat\bratexp2structured")

    @property
    def BuffalordExp1(self):
        return self._process(r"anirudh_data\buffalord\BuffalordExp1Structured")

    @property
    def GronckleExp1(self):
        return self._process(r"anirudh_data\gronckle\GronckleExp1Structured")

    @property
    def GrumpExp2(self):
        return self._process(r"anirudh_data\grump\GrumpExp2Structured")

    @property
    def ToothlessExp1(self):
        return self._process(r"anirudh_data\toothless\ToothlessExp1Structured")

    @property
    def allsess(self):
        pipelines: List[MABData]
        pipelines = (
            self.BewilderbeastExp1
            + self.BuffalordExp1
            + self.GronckleExp1
            + self.GrumpExp2
            + self.ToothlessExp1
        )
        return pipelines

    @property
    def first_exposure(self):
        "First exposure was structured env, had no prior experience with any type of env before this."
        pipelines: List[MABData]
        pipelines = (
            self.BewilderbeastExp1
            + self.BuffalordExp1
            + self.GronckleExp1
            + self.ToothlessExp1
        )
        return pipelines

    @property
    def second_exposure(self):
        """Animals whose second experience is structured env."""
        pipelines: List[MABData]
        pipelines = self.GrumpExp2
        return pipelines


class Unstruc(Group):
    @property
    def AggroExp1(self):
        return self._process(r"anirudh_data\aggro\AggroExp1UnStructured")

    @property
    def AuromaExp1(self):
        return self._process(r"anirudh_data\auroma\AuromaExp1Unstructured")

    @property
    def BratExp1(self):
        return self._process(r"anirudh_data\brat\BratExp1Unstructured")

    @property
    def GronckleExp2(self):
        return self._process(r"anirudh_data\gronckle\GronckleExp2Unstructured")

    @property
    def GrumpExp1(self):
        return self._process(r"anirudh_data\grump\GrumpExp1Unstructured")

    @property
    def ToothlessExp2(self):
        return self._process(r"anirudh_data\toothless\ToothlessExp2Unstructured")

    @property
    def allsess(self):
        pipelines: List[MABData]
        pipelines = (
            self.AggroExp1
            + self.AuromaExp1
            + self.BratExp1
            + self.GronckleExp2
            + self.GrumpExp1
            + self.ToothlessExp2
        )
        return pipelines

    @property
    def first_exposure(self):
        """Animals who were not exposed to any other environments before this."""
        pipelines: List[MABData]
        pipelines = self.AggroExp1 + self.AuromaExp1 + self.BratExp1 + self.GrumpExp1
        return pipelines

    @property
    def second_exposure(self):
        """Animals whose second experience is unstructured env."""
        pipelines: List[MABData]
        pipelines = self.GronckleExp2 + self.ToothlessExp2
        return pipelines


struc = Struc()
unstruc = Unstruc()


class GroupData:
    __slots__ = (
        "path",
        "history_coef_alltrials_anirudh",
        "history_coef_100trials_anirudh",
        "qlearning_2alpha_params_anirudh",
        "qlearning_2alpha_persev",
        "qlearning_2alpha_persev_correlated_within_unstructured_anirudh",
        "switch_prob_100trials",  # This mean across all sessions
        "switch_prob_by_trial_100trials",  # Trialwise switch prob
        "switch_prob_by_trial_100trials_first_exposure",  # Trialwise switch prob
        "perf_100min150max_10bin",
        "perf_100min150max_10bin_deltaprob_40",
        "perf_100min150max_10bin_deltaprob_0min35max",
        "perf_qlearning_assess_params",
        "perf_qlearning_switch_params",
        "switch_prob_seq",
        "switch_prob_seq_first_exposure",
        "switch_prob_seq_with_simulated_FE",
        "switch_prob_seq_with_simulated_switched_params_FE",
    )

    def __init__(self) -> None:
        self.path = Path(
            "C:\\Users\\asheshlab\\OneDrive\\academia\\analyses\\adlab\\processed_data"
        )

    def save(self, d, fp):
        if isinstance(d, pd.DataFrame):
            d = d.to_dict()
        data = {"data": d}
        np.save(self.path / fp, data)
        print(f"{fp} saved")

    def load(self, fp):
        data = np.load(self.path / f"{fp}.npy", allow_pickle=True).item()
        try:
            data["data"] = pd.DataFrame(data["data"])
        except:
            pass
        return data

    def __getattr__(self, name: str):
        return self.load(name)["data"]
