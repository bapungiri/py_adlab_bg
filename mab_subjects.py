from pathlib import Path
import os
import numpy as np
import neuropy
from banditpy.core import Bandit2Arm
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

        if "model" in self.sub_name:
            self.b2a = Bandit2Arm.from_csv(
                fp.with_suffix(".csv"),
                probs=["arm1_reward_prob", "arm2_reward_prob"],
                choices="chosen_action",
                rewards="reward",
                session_ids="session_id",
            )
        else:
            self.b2a = Bandit2Arm.from_csv(
                fp.with_suffix(".csv"),
                probs=["rewprobfull1", "rewprobfull2"],
                choices="port",
                rewards="reward",
                session_ids="session#",
                starts="trialstart",
                stops="trialend",
                datetime="datetime",
            )

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


def get_rnn_experiments(basedir):
    """
    Get RNN experiments for structured and unstructured environments.
    Returns:
        List of MABData objects for RNN experiments.
    """
    # rnn_basedir = Path(r"D:\\Data\\mab\\rnn_data\\trained_1decimals\\tested_1decimals")

    s_models = sorted(basedir.glob("structured_2arm*"))
    u_models = sorted(basedir.glob("unstructured_2arm*"))

    s_on_s = [MABData(_ / f"{_.stem}_structured", tag="s_on_s") for _ in s_models]
    s_on_u = [MABData(_ / f"{_.stem}_unstructured", tag="s_on_u") for _ in s_models]

    u_on_s = [MABData(_ / f"{_.stem}_structured", tag="u_on_s") for _ in u_models]
    u_on_u = [MABData(_ / f"{_.stem}_unstructured", tag="u_on_u") for _ in u_models]

    return u_on_u + u_on_s + s_on_s + s_on_u


# ----- RNN Data with 1 decimal training/testing --------
rnn_exps1 = get_rnn_experiments(
    Path(r"D:\\Data\\mab\\rnn_data\\trained_1decimals\\tested_1decimals")
)

# ----- RNN Data with 1 decimal training/testing with impure probabilities --------
rnn_exps2 = get_rnn_experiments(
    Path(r"D:\\Data\\mab\\rnn_data\\Train1Test1_0.16impure")
)

# ----- RNN Data with Train1, Tst1, impure probabilities --------
rnn_exps3 = get_rnn_experiments(
    Path(r"D:\\Data\\mab\\rnn_data\\Train1Test1_0.16impure_345reset")
)


struc = Struc()
unstruc = Unstruc()


class GroupData:
    __slots__ = (
        "path",
        "logreg_10past",
        "qlearning_2alpha_params_anirudh",
        "qlearning_2alpha_persev",
        "qlearning_2alpha_persev_correlated_within_unstructured_anirudh",
        "switch_prob_100trials",  # This mean across all sessions
        "switch_prob_by_trial_100trials",  # Trialwise switch prob
        "switch_prob_by_trial_100trials_first_exposure",  # Trialwise switch prob
        "switch_prob_logreg_10past",  # Trialwise switch prob
        "perf_logreg_10past",  # Trialwise switch prob
        "perf_difficulty_level",  # Performance across difficulty levels
        "perf_100min150max_10bin",
        "perf_100min150max_10bin_deltaprob_40",
        "perf_100min150max_10bin_deltaprob_0min35max",
        "perf_qlearning_assess_params",
        "perf_qlearning_switch_params",
        "switch_prob_seq",  # Switch prob like Beron2022 with difficulty level
        "switch_prob_consecutive",
        "switch_prob_seq_with_simulated_FE",
        "switch_prob_seq_with_simulated_switched_params_FE",
        "reward_rate_probability_matrix",  # Reward rate matrix as a function arm probs
        "perf_probability_matrix",  # Performance matrix as a function of arm probs
        "entropy_equal_probs",  # Entropy of equal probability arms in unstructured env
        # ------ Regular RNN data ------
        "rnn_perf",  # RNN performance s_on_s, s_on_u, u_on_s, u_on_u
        "rnn_switch_prob",  # RNN switch prob s_on_s, s_on_u, u_on_s, u_on_u
        "rnn_cond_switch_prob",  # RNN conditional switch prob s_on_s, s_on_u etc.
        "rnn_perf_probability_matrix",  # RNN perf. matrix as a function of arm probs
        "rnn_qlearn_2alphaH",  # Qlearn params for RNN
        # ------ Impure probabilities RNN data ------
        "rnn_perf_impure",  # RNN performance s_on_s, s_on_u, u_on_s, u_on_u
        "rnn_switch_prob_impure",  # RNN switch prob s_on_s, s_on_u, u_on_s, u_on_u
        "rnn_cond_switch_prob_impure",  # RNN conditional switch prob s_on_s, s_on_u etc.
        "rnn_perf_probability_matrix_impure",  # RNN perf. matrix as a function of arm probs
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
