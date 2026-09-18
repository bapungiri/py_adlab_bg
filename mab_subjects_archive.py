"""Old/stale subject-group definitions, kept working but no longer actively
maintained: short-blocks single-animal groups and the pre-table-driven LSTM
experiment loader. Split out of mab_subjects.py so they don't clutter the
classes that are actually maintained (Struc/Unstruc/StrucRNN/UnstrucRNN).

Re-exported from mab_subjects.py so existing `mab_subjects.rnn_exps1`-style
access (used in mab_rnn1.ipynb, mab_rnn_impure.ipynb,
mab_rnn_models_analysis.ipynb, mab_rnn_test.ipynb) keeps working unchanged.
"""

from pathlib import Path
from typing import List

import numpy as np
from banditpy.models import BanditTrainer2Arm

from mab_data_core import Group, MABData


class MostlyStrucShortBlocks(Group):
    group_tag = "struc"
    dirstr_BG = "BGdataset/short_blocks/"  # directory base string for BGdataset

    @property
    def BGF4(self):
        return self._process(
            self.dirstr_BG + "BGF4",
            data_tag="BGdataset",
            lesion_tag="naive",
        )

    @property
    def allsess(self):
        pipelines: List[MABData]
        pipelines = self.BGF4
        return pipelines

    @property
    def BGdataset(self):
        pipelines: List[MABData]
        pipelines = self.BGF4
        return pipelines


class MostlyUnstrucShortBlocks(Group):
    group_tag = "unstruc"
    dirstr_BG = "BGdataset/short_blocks/"  # directory base string for BGdataset

    @property
    def BGM5(self):
        return self._process(
            self.dirstr_BG + "BGM5",
            data_tag="BGdataset",
            lesion_tag="naive",
        )

    @property
    def allsess(self):
        pipelines: List[MABData]
        pipelines = self.BGM5
        return pipelines

    @property
    def BGdataset(self):
        pipelines: List[MABData]
        pipelines = self.BGM5
        return pipelines


class LSTMData:
    """
    Get RNN experiments for structured and unstructured environments.
    Returns:
        List of MABData objects for RNN experiments.
    """

    def __init__(self, basedir, modeldir=None, tag=None):
        self.basedir = Path(basedir)
        self.tag = tag

        self.s_fp = sorted(basedir.glob("structured_2arm*"))
        self.u_fp = sorted(basedir.glob("unstructured_2arm*"))
        self.modeldir = modeldir

    @property
    def s_models(self):
        if self.modeldir is None:
            models_fp = [sorted(_.glob("*.pt"))[0] for _ in self.s_fp]
            models: List[BanditTrainer2Arm]
            models = [
                BanditTrainer2Arm.load_model(model_path=_, verbose=False)
                for _ in models_fp
            ]
            return models

        if self.modeldir is not None:

            models_fp = sorted(self.modeldir.glob("structured_2arm*"))
            models: List[BanditTrainer2Arm]
            models = [
                BanditTrainer2Arm(model_path=self.modeldir / _) for _ in models_fp
            ]
            [_.load_model() for _ in models]
            return models

    @property
    def u_models(self):
        if self.modeldir is None:
            models_fp = [sorted(_.glob("*.pt"))[0] for _ in self.u_fp]
            models: List[BanditTrainer2Arm]
            models = [
                BanditTrainer2Arm.load_model(model_path=_, verbose=False)
                for _ in models_fp
            ]
            return models

        if self.modeldir is not None:

            models_fp = sorted(self.modeldir.glob("unstructured_2arm*"))
            models: List[BanditTrainer2Arm]
            models = [
                BanditTrainer2Arm(model_path=self.modeldir / _) for _ in models_fp
            ]
            [_.load_model() for _ in models]
            return models

    @property
    def all_models(self):
        return self.u_models + self.s_models

    @property
    def s_on_s(self):
        return [MABData(_ / f"{_.stem}_structured", tag="s_on_s") for _ in self.s_fp]

    @property
    def s_on_u(self):
        return [MABData(_ / f"{_.stem}_unstructured", tag="s_on_u") for _ in self.s_fp]

    @property
    def u_on_s(self):
        return [MABData(_ / f"{_.stem}_structured", tag="u_on_s") for _ in self.u_fp]

    @property
    def u_on_u(self):
        return [MABData(_ / f"{_.stem}_unstructured", tag="u_on_u") for _ in self.u_fp]

    @property
    def all_exps(self):
        return self.u_on_u + self.u_on_s + self.s_on_s + self.s_on_u

    def best_of(self, n=None):
        """
        Get the best n experiments based on the optimal choice probability.
        If n_best is None, return all experiments.
        """
        if n is None:
            return self.allexps

        # get_best = lambda x: sorted(
        #     x,
        #     key=lambda y: y.b2a.get_optimal_choice_probability()[-5:].mean(),
        #     reverse=True,
        # )[:n]

        s_indx = np.argsort(
            [_.b2a.get_optimal_choice_probability()[-5:].mean() for _ in self.s_on_s]
        )[-1 : -(n + 1) : -1]
        u_indx = np.argsort(
            [_.b2a.get_optimal_choice_probability()[-5:].mean() for _ in self.u_on_u]
        )[-1 : -(n + 1) : -1]

        return (
            [self.u_on_u[_] for _ in u_indx]
            + [self.u_on_s[_] for _ in u_indx]
            + [self.s_on_s[_] for _ in s_indx]
            + [self.s_on_u[_] for _ in s_indx]
        )

    # def get_data(self):
    #     files = sorted(self.basedir.glob("*.csv"))
    #     return [MABData(f, tag=self.tag) for f in files]


# ----- RNN Data with 1 decimal training/testing --------
rnn_exps1 = LSTMData(
    Path(r"D:\\Data\\mab\\rnn_data\\trained_1decimals\\tested_1decimals")
)

# ----- RNN Data with 1 decimal training/testing with impure probabilities --------
rnn_exps2 = LSTMData(Path(r"D:\\Data\\mab\\rnn_data\\Train1Test1_0.16impure"))

# ----- RNN Data with Train1, Tst1, impure probabilities --------
rnn_exps3 = LSTMData(Path(r"D:\\Data\\mab\\rnn_data\\Train1Test1_0.16impure_345reset"))

# ----- RNN Data with Train1, Tst1, impure probabilities and filtered performance --------
rnn_exps4 = LSTMData(
    Path(r"D:\\Data\\mab\\rnn_data\\Train1Test1custom_0.16impure_performance_filtered")
)

# ----- RNN Data with Train1, Tst1, impure probabilities and filtered performance --------
rnn_exps5 = LSTMData(Path(r"D:\\Data\\mab\\rnn_data\\Train40000Test1000_0.16impure"))

rnn_exps6 = LSTMData(Path(r"D:\\Data\\mab\\rnn_data\\Train50000Test500_0.16impure"))

rnn_exps7 = LSTMData(
    Path(r"D:\\Data\\mab\\rnn_data\\Train50000Test500Impure0.16_2025-07-05_09-41-18"),
    Path(r"D:\\Data\\mab\\rnn_models\\Train50000Impure0.16_2025-07-05_09-41-18"),
)
rnn_exps8 = LSTMData(
    Path(r"D:\\Data\\mab\\rnn_data\\Train50000Test500Impure0.16_2025-07-07_12-27-58"),
    Path(r"D:\\Data\\mab\\rnn_models\\Train50000Impure0.16_2025-07-07_12-27-58"),
)

rnn_exps9 = LSTMData(
    Path(r"D:\\Data\\mab\\rnn_data\\Train30000Test500Impure0.16_2025-07-07_18-09-58"),
    Path(r"D:\\Data\\mab\\rnn_models\\Train30000Impure0.16_2025-07-07_18-09-58"),
)
rnn_exps10 = LSTMData(
    Path("D:/Data/mab/rnn_data/Train30000Test500Impure0.16_2025-07-09_10-45-55"),
)


mostly_struc_short_blocks = MostlyStrucShortBlocks()
mostly_unstruc_short_blocks = MostlyUnstrucShortBlocks()
