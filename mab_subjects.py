from pathlib import Path
import os
import numpy as np
import neuropy
from banditpy.core import Bandit2Arm
from banditpy.models import BanditTrainer2Arm
from typing import List
import pandas as pd
from dataclasses import dataclass

figpath = Path("C:/Users/asheshlab/OneDrive/academia/analyses/adlab/figures")


class MABData:
    """_summary_
    Notes
    ------
    20-11-2025: Three new attributes added: group_tag, data_tag, lesion_tag to keep track of animal groups and lesion status.

    """

    def __init__(self, basepath, group_tag=None, data_tag=None, lesion_tag=None):
        basepath = Path(basepath)
        try:
            csv_file = sorted(basepath.glob("*.csv"))
            if len(csv_file) == 0:
                raise FileNotFoundError(f"No CSV files found in {basepath}")
            fp = csv_file[0].with_suffix("")
            # print(csv_file)
        except:
            fp = basepath / basepath.name
            # pass

        self.filePrefix = fp
        sub_name = basepath.name

        if sub_name in ["pre_lesion", "post_lesion"]:
            sub_name = basepath.parent.name
        self.sub_name = sub_name

        self.group_tag = group_tag
        self.data_tag = data_tag
        self.lesion_tag = lesion_tag

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
            csv_data = pd.read_csv(fp.with_suffix(".csv"))

            if "rewprobfull1" in csv_data.columns:
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
            if "probs_1" in csv_data.columns:
                self.b2a = Bandit2Arm.from_csv(
                    fp.with_suffix(".csv"),
                    probs=["probs_1", "probs_2"],
                    choices="choices",
                    rewards="rewards",
                    session_ids="session_ids",
                    datetime="datetime",
                )
            if "p1" in csv_data.columns:
                self.b2a = Bandit2Arm.from_csv(
                    fp.with_suffix(".csv"),
                    probs=["p1", "p2"],
                    choices=["port"],
                    rewards=["reward"],
                    session_ids=["session_id"],
                    starts=["start"],
                    stops=["stop"],
                    datetime=["stop_time"],
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
    group_tag = None

    # @property
    # def basedir(self):
    if os.name == "nt":
        basedir = Path(r"D:\\Data\\mab")
    else:
        basedir = Path("/mnt/pve/Homes/bapun/Data")

    def _process(self, rel_path, data_tag=None, lesion_tag=None):
        return [
            MABData(
                self.basedir / rel_path,
                group_tag=self.group_tag,
                data_tag=data_tag,
                lesion_tag=lesion_tag,
            )
        ]

    def data_exist(self):
        self.allsess


class Struc(Group):
    group_tag = "struc"
    """_summary_

    Notes
    ----------
    20-11-2025: Updated Bewilderbeast and Gronckle data paths to Aarushi's new data. Commented old paths and animals who were subjected to change of environment (Brat and Grump). Removed "Exp1" suffix from property names.
    """

    #!SECTION ======= Aarushi's dataset =======
    @property
    def Bewilderbeast1(self):
        # Aarushi's old data path
        # return self._process("AAdataset/bewilderbeast/BewilderbeastExp1Structured")
        return self._process(
            "ACdataset/Bewilderbeast/pre_lesion",
            data_tag="ACdataset",
            lesion_tag="pre_lesion",
        )

    # Bewilderbeast2 is bad data
    # @property
    # def Bewilderbeast2(self):
    #     # Aarushi's old data path
    #     # return self._process("AAdataset/bewilderbeast/BewilderbeastExp1Structured")
    #     return self._process(
    #         "ACdataset/Bewilderbeast/post_lesion",
    #         data_tag="ACdataset",
    #         lesion_tag="post_lesion_OFC",
    #     )

    @property
    def Aguero1(self):
        return self._process(
            "ACdataset/Aguero/pre_lesion", data_tag="ACdataset", lesion_tag="pre_lesion"
        )

    @property
    def Aguero2(self):
        return self._process(
            "ACdataset/Aguero/post_lesion",
            data_tag="ACdataset",
            lesion_tag="post_lesion_OFC",
        )

    @property
    def Sterling(self):
        return self._process(
            "ACdataset/Sterling", data_tag="ACdataset", lesion_tag="pre_lesion"
        )

    @property
    def Phil(self):
        return self._process(
            "ACdataset/Phil", data_tag="ACdataset", lesion_tag="naive_lesion_OFC"
        )

    @property
    def Rodri(self):
        return self._process(
            "ACdataset/Rodri", data_tag="ACdataset", lesion_tag="naive_lesion_OFC"
        )

    #!SECTION ======= Anirudh's dataset =======
    # @property
    # def GrumpExp2(self):
    #     return self._process("AAdataset/grump/GrumpExp2Structured")

    @property
    def Gronckle1(self):
        # Aarushi's old data path
        # return self._process("AAdataset/gronckle/GronckleExp1Structured")
        return self._process(
            "ASdataset/gronckle/pre_lesion",
            data_tag="ASdataset",
            lesion_tag="pre_lesion",
        )

    @property
    def Toothless(self):
        return self._process(
            "ASdataset/toothless/ToothlessExp1Structured",
            data_tag="ASdataset",
            lesion_tag="pre_lesion",
        )

    # @property
    # def bratexp2(self): # bad animal
    #     return self._process("AAdataset/brat/bratexp2structured")

    @property
    def Buffalord(self):
        return self._process(
            "ASdataset/buffalord/BuffalordExp1Structured",
            data_tag="ASdataset",
            lesion_tag="pre_lesion",
        )

    @property
    def allsess(self):
        pipelines: List[MABData]
        pipelines = (
            self.Bewilderbeast1
            + self.Aguero1
            + self.Aguero2
            + self.Sterling
            + self.Phil
            + self.Rodri
            + self.Gronckle1
            + self.Toothless
            + self.Buffalord
        )
        return pipelines

    # @property
    # def first_exposure(self):
    #     "First exposure was structured env, had no prior experience with any type of env before this."
    #     pipelines: List[MABData]
    #     pipelines = (
    #         self.BewilderbeastExp1
    #         + self.BuffalordExp1
    #         + self.GronckleExp1
    #         + self.ToothlessExp1
    #     )
    #     return pipelines

    # @property
    # def second_exposure(self):
    #     """Animals whose second experience is structured env."""
    #     pipelines: List[MABData]
    #     pipelines = self.GrumpExp2
    #     return pipelines


class Unstruc(Group):
    group_tag = "unstruc"
    """

    Notes
    ----------
    20-11-2025: Updated Bewilderbeast and Gronckle data paths to Aarushi's new data. Commented old paths and animals who were subjected to change of environment (Brat and Grump). Removed "Exp1" suffix from property names.
    """

    #!SECTION ======= Aarushi's dataset =======
    @property
    def Aggro1(self):
        return self._process(
            "ACdataset/Aggro/pre_lesion", data_tag="ACdataset", lesion_tag="pre_lesion"
        )

    @property
    def Aggro2(self):
        return self._process(
            "ACdataset/Aggro/post_lesion",
            data_tag="ACdataset",
            lesion_tag="post_lesion_OFC",
        )

    @property
    def Auroma(self):
        return self._process(
            "ACdataset/Auroma", data_tag="ACdataset", lesion_tag="pre_lesion"
        )

    @property
    def Torres(self):
        return self._process(
            "ACdataset/Torres", data_tag="ACdataset", lesion_tag="pre_lesion"
        )

    @property
    def Debruyne(self):
        return self._process(
            "ACdataset/Debruyne", data_tag="ACdataset", lesion_tag="naive_lesion_OFC"
        )

    @property
    def Kompany(self):
        return self._process(
            "ACdataset/Kompany", data_tag="ACdataset", lesion_tag="naive_lesion_OFC"
        )

    # @property
    # def Gronckle(self):
    #     return self._process("AAdataset/gronckle/GronckleExp2Unstructured")

    #!SECTION ======= Anirudh's dataset =======
    @property
    def Grump(self):
        return self._process(
            "ASdataset/grump/GrumpExp1Unstructured",
            data_tag="ASdataset",
            lesion_tag="pre_lesion",
        )

    # @property
    # def Toothless(self):
    #     return self._process("ASdataset/toothless/ToothlessExp2Unstructured")

    # @property
    # def Brat(self):
    #     return self._process(
    #         "AAdataset/brat/BratExp1Unstructured",
    #         data_tag="ASdataset",
    #         lesion_tag="pre_lesion",
    #     )

    # Gronckle2 is DLS lesion and environment was changed to unstructured post lesion
    # @property
    # def Gronckle2(self):
    #     # Aarushi's old data path
    #     # return self._process("AAdataset/gronckle/GronckleExp1Structured")
    #     return self._process(
    #         "ASdataset/Gronckle/post_lesion",
    #         data_tag="ASdataset",
    #         lesion_tag="post_lesion_DLS",
    #     )

    @property
    def allsess(self):
        pipelines: List[MABData]
        pipelines = (
            self.Aggro1
            + self.Aggro2
            + self.Auroma
            + self.Torres
            + self.Debruyne
            + self.Kompany
            + self.Grump
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


class MostlyStruc(Group):
    group_tag = "struc"

    @property
    def BGM1(self):
        return self._process("BGdataset/BGM1", data_tag="BGdataset", lesion_tag="naive")

    @property
    def BGF0(self):
        return self._process("BGdataset/BGF0", data_tag="BGdataset", lesion_tag="naive")

    @property
    def BGM3(self):
        return self._process("BGdataset/BGM3", data_tag="BGdataset", lesion_tag="naive")

    @property
    def BGM4(self):
        return self._process("BGdataset/BGM4", data_tag="BGdataset", lesion_tag="naive")

    @property
    def BGF4(self):
        return self._process("BGdataset/BGF4", data_tag="BGdataset", lesion_tag="naive")

    # -------- Aarushi dataset ----------
    @property
    def Gavi(self):
        return self._process(
            "ACdataset/impure_paradigm/Gavi", data_tag="ACdataset", lesion_tag="naive"
        )

    @property
    def Haaland(self):
        return self._process(
            "ACdataset/impure_paradigm/Haaland",
            data_tag="ACdataset",
            lesion_tag="naive",
        )

    @property
    def Pedri(self):
        return self._process(
            "ACdataset/impure_paradigm/Pedri", data_tag="ACdataset", lesion_tag="naive"
        )

    @property
    def Xavi(self):
        return self._process(
            "ACdataset/impure_paradigm/Xavi", data_tag="ACdataset", lesion_tag="naive"
        )

    @property
    def good_sess(self):
        pipelines: List[MABData]
        pipelines = self.BGM1 + self.BGF0 + self.BGM3 + self.BGF4
        return pipelines

    @property
    def allsess(self):
        pipelines: List[MABData]
        pipelines = (
            self.BGM1
            + self.BGF0
            + self.BGM3
            + self.BGM4
            + self.BGF4
            + self.Gavi
            + self.Haaland
            + self.Pedri
            + self.Xavi
        )
        return pipelines

    @property
    def BGdataset(self):
        pipelines: List[MABData]
        pipelines = self.BGM1 + self.BGF0 + self.BGM3 + self.BGM4 + self.BGF4
        return pipelines

    @property
    def ACdataset(self):
        pipelines: List[MABData]
        pipelines = self.Gavi + self.Haaland + self.Pedri + self.Xavi
        return pipelines


class MostlyUnstruc(Group):
    group_tag = "unstruc"

    @property
    def BGM0(self):
        return self._process("BGdataset/BGM0", data_tag="BGdataset", lesion_tag="naive")

    # @property
    # def BGM2(self): # BAD animal
    #     return self._process("BGdataset/BGM2")

    @property
    def BGF1(self):
        return self._process("BGdataset/BGF1", data_tag="BGdataset", lesion_tag="naive")

    @property
    def BGF2(self):
        return self._process("BGdataset/BGF2", data_tag="BGdataset", lesion_tag="naive")

    @property
    def BGF3(self):
        return self._process("BGdataset/BGF3", data_tag="BGdataset", lesion_tag="naive")

    @property
    def BGM5(self):
        return self._process("BGdataset/BGM5", data_tag="BGdataset", lesion_tag="naive")

    # -------- Aarushi dataset ----------

    @property
    def Messi(self):
        return self._process(
            "ACdataset/impure_paradigm/Messi", data_tag="ACdataset", lesion_tag="naive"
        )

    @property
    def Neymar(self):
        return self._process(
            "ACdataset/impure_paradigm/Neymar", data_tag="ACdataset", lesion_tag="naive"
        )

    @property
    def Son(self):
        return self._process(
            "ACdataset/impure_paradigm/Son", data_tag="ACdataset", lesion_tag="naive"
        )

    @property
    def good_sess(self):
        pipelines: List[MABData]
        pipelines = self.BGM0 + self.BGF1 + self.BGF2
        return pipelines

    @property
    def allsess(self):
        pipelines: List[MABData]
        pipelines = (
            self.BGM0
            + self.BGF1
            + self.BGF2
            + self.BGF3
            + self.BGM5
            + self.Messi
            + self.Neymar
            + self.Son
        )
        return pipelines

    @property
    def BGdataset(self):
        pipelines: List[MABData]
        pipelines = self.BGM0 + self.BGF1 + self.BGF2 + self.BGF3 + self.BGM5
        return pipelines

    @property
    def ACdataset(self):
        pipelines: List[MABData]
        pipelines = self.Messi + self.Neymar + self.Son
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


struc = Struc()
unstruc = Unstruc()
mostly_struc = MostlyStruc()
mostly_unstruc = MostlyUnstruc()


# ------- Files generated with slots -------#
from datetime import datetime


# ---------------------------------------------------------
# Helper class: versioned accessor
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# GroupData main class
# ---------------------------------------------------------
class GroupData:

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

    # -----------------------------------------------------
    # BASENAME DISCOVERY
    # -----------------------------------------------------
    def _discover_basenames(self):
        basenames = set()
        for f in self.path.glob("*.npy"):
            stem = f.stem
            # assume format: basename_YYYYMMDD_HHMMSS
            parts = stem.rsplit("_", 2)  # basename, yyyymmdd, hhmmss
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                basenames.add(parts[0])
        return sorted(basenames)

    # -----------------------------------------------------
    # ATTRIBUTE ACCESS (this gives you autocomplete!)
    # -----------------------------------------------------
    def __getattr__(self, name: str):
        if name in self._basenames:
            return VersionedAccessor(self, name)
        raise AttributeError(f"No attribute or data basename named '{name}'")

    # -----------------------------------------------------
    # FILENAME HELPERS
    # -----------------------------------------------------
    def _versioned_name(self, basename: str):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{basename}_{now}.npy"

    def _all_versions(self, basename: str):
        return sorted(self.path.glob(f"{basename}_*.npy"))

    def _latest_version(self, basename: str):
        files = self._all_versions(basename)
        if not files:
            raise FileNotFoundError(f"No versions found for '{basename}'")
        return files[-1].stem  # no .npy

    # -----------------------------------------------------
    # SAVE WITH VERSIONING + CLEANUP
    # -----------------------------------------------------
    def save(self, data, basename: str, clean: bool = True):
        # convert DataFrame to dict
        if isinstance(data, pd.DataFrame):
            data = data.to_dict()

        filename = self._versioned_name(basename)
        np.save(self.path / filename, {"data": data})
        print(f"[GroupData] Saved: {filename}")

        # register new basename if new
        if basename not in self._basenames:
            self._basenames.append(basename)

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

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------
    def load(self, stem: str):
        data = np.load(self.path / f"{stem}.npy", allow_pickle=True).item()
        try:
            data["data"] = pd.DataFrame(data["data"])
        except Exception:
            pass
        return data

    # -----------------------------------------------------
    # REVERSE MAPPING
    # -----------------------------------------------------
    def filename_to_attr(self, filename: str):
        stem = Path(filename).stem
        for base in self._basenames:
            if stem.startswith(base + "_"):
                return base
        return None
