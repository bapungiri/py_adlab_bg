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
iapath = Path(
    "C:/Users/asheshlab/OneDrive/academia/analyses/adlab/figures/india_alliance"
)

pkpath = Path("C:/Users/asheshlab/OneDrive/academia/analyses/adlab/figures/pk")


class MABData:
    """_summary_
    Notes
    ------
    20-11-2025: Three new attributes added: group_tag, data_tag, lesion_tag to keep track of animal groups and lesion status.

    """

    def __init__(
        self,
        basepath,
        group_tag=None,
        paradigm_tag=None,
        data_tag=None,
        lesion_tag=None,
        sex_tag=None,
    ):
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
        self.paradigm_tag = paradigm_tag
        self.group_paradigm_tag = f"{group_tag}_{paradigm_tag}"
        self.data_tag = data_tag
        self.lesion_tag = lesion_tag
        self.sex_tag = sex_tag
        self.common_kwargs = dict(
            name=self.sub_name,
            group=group_tag,
            paradigm=paradigm_tag,
            dataset=data_tag,
            lesion=lesion_tag,
            sex=sex_tag,
            group_paradigm=self.group_paradigm_tag,
        )

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


class Group:
    group_tag = None

    if os.name == "nt":
        basedir = Path(r"D:\\Data\\mab")
    else:
        basedir = Path("/mnt/pve/Homes/bapun/Data")

    def _process(
        self, rel_path, data_tag=None, paradigm_tag=None, lesion_tag=None, sex_tag=None
    ):
        return [
            MABData(
                self.basedir / rel_path,
                group_tag=self.group_tag,
                data_tag=data_tag,
                paradigm_tag=paradigm_tag,
                lesion_tag=lesion_tag,
                sex_tag=sex_tag,
            )
        ]


@dataclass(frozen=True)
class DatasetCondition:
    data_tag: str
    paradigm: str
    lesion_tag: str
    base_root: str = ""

    @property
    def basedir(self):
        return Path(self.base_root) / self.data_tag / f"Paradigm_{self.paradigm}"

    @property
    def dirstr(self):
        return self.basedir / self.lesion_tag

    @property
    def kwargs(self):
        return {
            "data_tag": self.data_tag,
            "lesion_tag": self.lesion_tag,
            "paradigm_tag": f"{self.paradigm}",
        }


class Datasets:

    class BG:
        P8020_intact = DatasetCondition("BGdataset", "8020", "intact")
        P8020_lesion_mPFC_pre = DatasetCondition("BGdataset", "8020", "lesion_mPFC_pre")
        P8020_lesion_mPFC_post = DatasetCondition(
            "BGdataset", "8020", "lesion_mPFC_post"
        )
        P8020_sham_pre = DatasetCondition("BGdataset", "8020", "sham_pre")
        P8020_sham_post = DatasetCondition("BGdataset", "8020", "sham_post")

    class AC:
        P100_intact = DatasetCondition("ACdataset", "100", "intact")
        P100_lesion_OFC_pre = DatasetCondition("ACdataset", "100", "lesion_OFC_pre")
        P100_lesion_OFC_post = DatasetCondition("ACdataset", "100", "lesion_OFC_post")

        P8020_intact = DatasetCondition("ACdataset", "8020", "intact")
        P8020_lesion_OFC_pre = DatasetCondition("ACdataset", "8020", "lesion_OFC_pre")
        P8020_lesion_OFC_post = DatasetCondition("ACdataset", "8020", "lesion_OFC_post")
        P8020_sham_pre = DatasetCondition("ACdataset", "8020", "sham_pre")
        P8020_sham_post = DatasetCondition("ACdataset", "8020", "sham_post")

    class AS:
        P100_intact = DatasetCondition("ASdataset", "100", "intact")


class Struc(Group):
    group_tag = "struc"
    """_summary_

    Notes
    ----------
    20-11-2025: Updated Bewilderbeast and Gronckle data paths to Aarushi's new data. Commented old paths and animals who were subjected to change of environment (Brat and Grump). Removed "Exp1" suffix from property names.
    """

    def process_wrapper(self, DCinst: DatasetCondition, animal_name: str, sex: str):
        return super()._process(
            DCinst.dirstr / animal_name, sex_tag=sex, **DCinst.kwargs
        )

    # ===================================
    #  Aarushi's dataset
    # ====================================
    # Paradigm 100
    @property
    def p100_intact_Bewilderbeast(self):
        return self.process_wrapper(Datasets.AC.P100_intact, "Bewilderbeast", "male")

    # @property
    # def Bewilderbeast2(self): # Bewilderbeast2 is bad data
    #     # Aarushi's old data path
    #     # return self._process("AAdataset/bewilderbeast/BewilderbeastExp1Structured")
    #     return self._process(
    #         "ACdataset/Bewilderbeast/post_lesion",
    #         data_tag="ACdataset",
    #         lesion_tag="post_lesion_OFC",
    #     )

    @property
    def p100_intact_Aguero(self):
        return self.process_wrapper(Datasets.AC.P100_intact, "Aguero", "male")

    @property
    def p100_intact_Sterling(self):
        return self.process_wrapper(Datasets.AC.P100_intact, "Sterling", "male")

    @property
    def p100_lesion_OFC_post_Aguero(self):
        return self.process_wrapper(Datasets.AC.P100_lesion_OFC_post, "Aguero", "male")

    @property
    def p100_lesion_OFC_pre_Phil(self):
        return self.process_wrapper(Datasets.AC.P100_lesion_OFC_pre, "Phil", "male")

    @property
    def p100_lesion_OFC_pre_Rodri(self):
        return self.process_wrapper(Datasets.AC.P100_lesion_OFC_pre, "Rodri", "male")

    # Paradigm 8020
    @property
    def p8020_intact_Gavi(self):
        return self.process_wrapper(Datasets.AC.P8020_intact, "Gavi", "male")

    @property
    def p8020_intact_Haaland(self):  # male
        return self.process_wrapper(Datasets.AC.P8020_intact, "Haaland", "male")

    @property
    def p8020_intact_Pedri(self):  # female
        return self.process_wrapper(Datasets.AC.P8020_intact, "Pedri", "female")

    @property
    def p8020_intact_Xavi(self):  # male
        return self.process_wrapper(Datasets.AC.P8020_intact, "Xavi", "male")

    @property
    def p8020_lesion_OFC_post_Gavi(self):
        return self.process_wrapper(Datasets.AC.P8020_lesion_OFC_post, "Gavi", "male")

    @property
    def p8020_lesion_OFC_post_Pedri(self):
        return self.process_wrapper(
            Datasets.AC.P8020_lesion_OFC_post, "Pedri", "female"
        )

    @property
    def p8020_lesion_OFC_post_Xavi(self):
        return self.process_wrapper(Datasets.AC.P8020_lesion_OFC_post, "Xavi", "male")

    @property
    def p8020_lesion_OFC_post_Haaland(self):
        return self.process_wrapper(
            Datasets.AC.P8020_lesion_OFC_post, "Haaland", "male"
        )

    # ===================================
    #   Anirudh's dataset
    # ==================================
    # @property
    # def GrumpExp2(self):
    #     return self._process("AAdataset/grump/GrumpExp2Structured")

    # @property
    # def Gronckle_intact(self):
    #     return self.process_wrapper(Datasets.AS.P100_intact, "Gronckle")

    @property
    def p100_intact_Toothless(self):
        return self.process_wrapper(Datasets.AS.P100_intact, "Toothless", "male")

    # @property
    # def bratexp2(self): # bad animal
    #     return self._process("AAdataset/brat/bratexp2structured")

    @property
    def p100_intact_Buffalord(self):
        return self.process_wrapper(Datasets.AS.P100_intact, "Buffalord", "male")

    # ===================================
    #     BG dataset
    # ===================================
    @property
    def p8020_intact_BGM1(self):
        return self.process_wrapper(Datasets.BG.P8020_intact, "BGM1", "male")

    @property
    def p8020_intact_BGF0(self):
        return self.process_wrapper(Datasets.BG.P8020_intact, "BGF0", "female")

    @property
    def p8020_intact_BGM3(self):
        return self.process_wrapper(Datasets.BG.P8020_intact, "BGM3", "male")

    @property
    def p8020_intact_BGM4(self):
        return self.process_wrapper(Datasets.BG.P8020_intact, "BGM4", "male")

    @property
    def p8020_intact_BGF4(self):
        return self.process_wrapper(Datasets.BG.P8020_intact, "BGF4", "female")

    @property
    def p8020_intact_BGM6(self):
        return self.process_wrapper(Datasets.BG.P8020_intact, "BGM6", "male")

    @property
    def p100_intact_sess(self):
        pipelines: List[MABData]
        pipelines = (
            self.P100_intact_Bewilderbeast
            + self.P100_intact_Aguero
            + self.P100_intact_Sterling
            + self.P100_intact_Toothless
            + self.P100_intact_Buffalord
        )
        return pipelines

    @property
    def p100_good_intact_sess(self):
        # same as p100_intact_sess for now since no biased animals are excluded from good_sess in Struc group. If we want to exclude biased animals, we can modify this property accordingly.
        pipelines: List[MABData]
        pipelines = (
            self.p100_intact_Bewilderbeast
            + self.p100_intact_Aguero
            + self.p100_intact_Sterling
            + self.p100_intact_Toothless
            + self.p100_intact_Buffalord
        )
        return pipelines

    @property
    def p100_lesion_OFC_pre_sess(self):
        pipelines: List[MABData]
        pipelines = self.p100_lesion_OFC_pre_Phil + self.p100_lesion_OFC_pre_Rodri
        return pipelines

    @property
    def p100_lesion_OFC_post_sess(self):
        pipelines: List[MABData]
        pipelines = self.p100_lesion_OFC_post_Aguero
        return pipelines

    @property
    def p8020_intact_sess(self):
        pipelines: List[MABData]
        pipelines = (
            self.p8020_intact_BGF0
            + self.p8020_intact_BGM3
            + self.p8020_intact_BGM4
            + self.p8020_intact_BGF4
            + self.p8020_intact_BGM6
            + self.p8020_intact_Gavi
            + self.p8020_intact_Haaland
            + self.p8020_intact_Pedri
            + self.p8020_intact_Xavi
        )
        return pipelines

    @property
    def p8020_good_intact_sess(self):
        # same as p8020_intact_sess for now since no biased animals are excluded from good_sess in Struc group. If we want to exclude biased animals, we can modify this property accordingly.
        pipelines: List[MABData]
        pipelines = (
            self.p8020_intact_BGF0
            + self.p8020_intact_BGM3
            + self.p8020_intact_BGM4
            + self.p8020_intact_BGF4
            + self.p8020_intact_BGM6
            + self.p8020_intact_Gavi
            + self.p8020_intact_Haaland
            + self.p8020_intact_Pedri
            + self.p8020_intact_Xavi
        )
        return pipelines

    @property
    def p8020_lesion_OFC_post_sess(self):
        pipelines: List[MABData]
        pipelines = (
            self.p8020_lesion_OFC_post_Gavi
            + self.p8020_lesion_OFC_post_Haaland
            + self.p8020_lesion_OFC_post_Pedri
            + self.p8020_lesion_OFC_post_Xavi
        )
        return pipelines

    @property
    def all_intact_sess(self):
        return self.p100_intact_sess + self.p8020_intact_sess

    @property
    def all_good_intact_sess(self):
        return self.p100_good_intact_sess + self.p8020_good_intact_sess


class Unstruc(Group):
    group_tag = "unstruc"
    """

    Notes
    ----------
    20-11-2025: Updated Bewilderbeast and Gronckle data paths to Aarushi's new data. Commented old paths and animals who were subjected to change of environment (Brat and Grump). Removed "Exp1" suffix from property names.
    """

    def process_wrapper(self, DCinst: DatasetCondition, animal_name: str, sex: str):
        return super()._process(
            DCinst.dirstr / animal_name, sex_tag=sex, **DCinst.kwargs
        )

    # ====================================
    # ======= Aarushi's dataset =======
    # =====================================
    @property
    def p100_intact_Aggro(self):
        return self.process_wrapper(Datasets.AC.P100_intact, "Aggro", "male")

    @property
    def p100_intact_Auroma(self):
        return self.process_wrapper(Datasets.AC.P100_intact, "Auroma", "male")

    @property
    def p100_intact_Torres(self):
        return self.process_wrapper(Datasets.AC.P100_intact, "Torres", "male")

    @property
    def p100_lesion_OFC_pre_Debruyne(self):
        return self.process_wrapper(Datasets.AC.P100_lesion_OFC_pre, "Debruyne", "male")

    @property
    def p100_lesion_OFC_pre_Kompany(self):
        return self.process_wrapper(Datasets.AC.P100_lesion_OFC_pre, "Kompany", "male")

    @property
    def p100_lesion_OFC_post_Aggro(self):
        return self.process_wrapper(Datasets.AC.P100_lesion_OFC_post, "Aggro", "male")

    @property
    def p8020_intact_Messi(self):  # male
        return self.process_wrapper(Datasets.AC.P8020_intact, "Messi", "male")

    @property
    def p8020_intact_Neymar(self):  # male
        return self.process_wrapper(Datasets.AC.P8020_intact, "Neymar", "male")

    @property
    def p8020_intact_Son(self):  # male
        return self.process_wrapper(Datasets.AC.P8020_intact, "Son", "male")

    @property
    def p8020_lesion_OFC_post_Messi(self):
        return self.process_wrapper(Datasets.AC.P8020_lesion_OFC_post, "Messi", "male")

    @property
    def p8020_lesion_OFC_post_Son(self):
        return self.process_wrapper(Datasets.AC.P8020_lesion_OFC_post, "Son", "male")

    # @property
    # def Gronckle(self):
    #     return self._process("AAdataset/gronckle/GronckleExp2Unstructured")

    # ==================================
    # ======= Anirudh's dataset =======
    # ==========================================
    @property
    def p100_intact_Grump(self):
        return self.process_wrapper(Datasets.AS.P100_intact, "Grump", "male")

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
    # ========================
    # BG dataset
    # ===========================
    @property
    def p8020_intact_BGM0(self):
        return self.process_wrapper(Datasets.BG.P8020_intact, "BGM0", "male")

    # @property
    # def p8020_intact_BGM2(self): # BAD animal
    #     return self.process_wrapper(Datasets.BG.P8020_intact, "BGM2", "male")

    @property
    def p8020_intact_BGF1(self):
        return self.process_wrapper(Datasets.BG.P8020_intact, "BGF1", "female")

    @property
    def p8020_intact_BGF2(self):
        return self.process_wrapper(Datasets.BG.P8020_intact, "BGF2", "female")

    @property
    def p8020_intact_BGF3(self):
        return self.process_wrapper(Datasets.BG.P8020_intact, "BGF3", "female")

    @property
    def p8020_intact_BGM5(self):
        return self.process_wrapper(Datasets.BG.P8020_intact, "BGM5", "male")

    @property
    def p8020_intact_BGF5(self):
        return self.process_wrapper(Datasets.BG.P8020_intact, "BGF5", "female")

    @property
    def p8020_lesion_mPFC_post_BGF2(self):
        return self.process_wrapper(
            Datasets.BG.P8020_lesion_mPFC_post, "BGF2", "female"
        )

    @property
    def p100_intact_sess(self):
        pipelines: List[MABData]
        pipelines = (
            self.p100_intact_Aggro
            + self.p100_intact_Auroma
            + self.p100_intact_Torres
            + self.p100_intact_Grump
        )
        return pipelines

    @property
    def p100_good_intact_sess(self):
        # Same as p100_intact_sess for now since no biased animals are excluded from good_sess in Unstruc group. If we want to exclude biased animals, we can modify this property accordingly.
        pipelines: List[MABData]
        pipelines = (
            self.p100_intact_Aggro
            + self.p100_intact_Auroma
            + self.p100_intact_Torres
            + self.p100_intact_Grump
        )
        return pipelines

    @property
    def p100_lesion_OFC_pre_sess(self):
        pipelines: List[MABData]
        pipelines = self.p100_lesion_OFC_pre_Debruyne + self.p100_lesion_OFC_pre_Kompany
        return pipelines

    @property
    def p100_lesion_OFC_post_sess(self):
        pipelines: List[MABData]
        pipelines = self.p100_lesion_OFC_post_Aggro
        return pipelines

    @property
    def p8020_intact_sess(self):
        pipelines: List[MABData]
        pipelines = (
            self.p8020_intact_BGF1
            + self.p8020_intact_BGF2
            + self.p8020_intact_BGF3  # biased
            + self.p8020_intact_BGM5
            + self.p8020_intact_BGF5
            + self.p8020_intact_Messi
            + self.p8020_intact_Neymar  # biased
            + self.p8020_intact_Son
        )
        return pipelines

    @property
    def p8020_good_intact_sess(self):
        # Biased animals are excluded from good_sess.
        pipelines: List[MABData]
        pipelines = (
            self.p8020_intact_BGF1
            + self.p8020_intact_BGF2
            + self.p8020_intact_BGM5
            + self.p8020_intact_BGF5
            + self.p8020_intact_Messi
            + self.p8020_intact_Son
        )
        return pipelines

    @property
    def p8020_lesion_OFC_post_sess(self):
        pipelines: List[MABData]
        pipelines = self.p8020_lesion_OFC_post_Messi + self.p8020_lesion_OFC_post_Son
        return pipelines

    @property
    def p8020_lesion_mPFC_post_sess(self):
        pipelines: List[MABData]
        pipelines = self.p8020_lesion_mPFC_post_BGF2
        return pipelines

    @property
    def all_intact_sess(self):
        return self.p100_intact_sess + self.p8020_intact_sess

    @property
    def all_good_intact_sess(self):
        return self.p100_good_intact_sess + self.p8020_good_intact_sess


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


struc = Struc()
unstruc = Unstruc()
mostly_struc_short_blocks = MostlyStrucShortBlocks()
mostly_unstruc_short_blocks = MostlyUnstrucShortBlocks()


# ------- Files generated with slots -------#
from datetime import datetime


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

    def _write_stub(self, stub_path: Path | None = None):
        """Write mab_subjects.pyi so VS Code can autocomplete basenames."""
        stub_path = stub_path or Path(__file__).with_suffix(".pyi")
        lines = [
            "from typing import Any",
            "",
            "class VersionedAccessor:",
            "    def __init__(self, parent: 'GroupData', basename: str): ...",
            "    @property",
            "    def all(self) -> Any: ...",
            "    @property",
            "    def latest(self) -> Any: ...",
            "    def __call__(self) -> Any: ...",
            "    @property",
            "    def versions(self) -> list[str]: ...",
            "",
            "class GroupData:",
            "    def __init__(self, keep_versions: int = 3): ...",
            "    def save(self, data: Any, basename: str, clean: bool = True) -> str: ...",
            "    def load(self, stem: str) -> dict: ...",
        ]
        lines += [f"    {b}: VersionedAccessor" for b in sorted(self._basenames)]
        stub_path.write_text("\n".join(lines))
