"""Data-loading mechanism shared by every subject-group class in
mab_subjects.py: how a session's files become a `MABData`, how a
data_tag/paradigm/lesion combination resolves to a path (`DatasetCondition`,
`Datasets`), and the table-driven `AnimalGroup` base class subject groups
build on (`Group` is its lower-level, non-table-driven ancestor, kept for
StrucRNN/UnstrucRNN which enumerate models procedurally rather than from a
hand-maintained table).

None of this changes when an animal is added -- that only ever touches the
`animals` table on a `Struc`/`Unstruc`-style subclass in mab_subjects.py.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

import neuropy
import numpy as np
import pandas as pd
from banditpy.core import Bandit2Arm
from banditpy.models import VanillaRNNFit2Arm


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

        if "Model" in self.sub_name:
            self.b2a: Bandit2Arm = Bandit2Arm.from_csv(
                fp.with_suffix(".csv"),
                probs=["arm1_reward_prob", "arm2_reward_prob"],
                choices="chosen_action",
                rewards="reward",
                session_ids="session_id",
                block_ids="block_id",
                window_ids="window_id",
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

    @property
    def rnn_fit1(self):
        file = self.filePrefix.with_name(
            self.filePrefix.stem + "_RNNfit_N32_LR0.001_E500_Swindow.pt"
        )
        if file.is_file():
            return VanillaRNNFit2Arm.load(file, device="cpu")
        else:
            raise FileNotFoundError(f"No RNN fit file found at {file}")

    @property
    def rnn_fit2(self):
        file = self.filePrefix.with_name(
            self.filePrefix.stem + "_RNNfit_N32_LR0.001_E500_Ssession.pt"
        )
        if file.is_file():
            return VanillaRNNFit2Arm.load(file, device="cpu")
        else:
            raise FileNotFoundError(f"No RNN fit file found at {file}")

    @property
    def rnn_fit3(self):
        file = self.filePrefix.with_name(
            self.filePrefix.stem + "_RNNfit_N48_LR0.001_E500_Ssession.pt"
        )
        if file.is_file():
            return VanillaRNNFit2Arm.load(file, device="cpu")
        else:
            raise FileNotFoundError(f"No RNN fit file found at {file}")

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

    def process_wrapper(self, condition: "DatasetCondition", name: str, sex: str):
        """Turn one (condition, name) pair into its MABData -- shared by
        AnimalGroup (table-driven animal groups) and RNNModelGroup
        (procedurally-named RNN model groups)."""
        return self._process(
            condition.dirstr / name, sex_tag=sex, **condition.kwargs
        )


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
        if self.lesion_tag == "rnn":
            if self.paradigm == "8020":
                # folder = "Train20000_Test1000_LR0.0001_20260606_155227" # bad
                # folder = "Train30000_Test1000_LR0.0001_20260606_155550" # bad
                # folder = "Train40000_Test1000_LR0.0001_20260608_104610" # good
                # folder = "Train50000_Test1000_LR0.0001_20260608_122244" # better
                # folder = "Train60000_Test1000_LR0.0001_20260608_140532"  # even better
                folder = "Train70000_Test1000_LR0.0001_20260608_162810"  # best so far but increment is small from 60000, will check if further training improves performance.
            if self.paradigm == "100":
                folder = "Train70000_Test1000_LR0.0001_20260616_164322"
            if self.paradigm == "9010":
                folder = "Train70000_Test1000_LR0.0001_20260806_171057"
            if self.paradigm == "9505":
                folder = "Train70000_Test1000_LR0.0001_20260806_171030"
            return self.basedir / folder

        else:
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
        P9505_intact = DatasetCondition("BGdataset", "9505", "intact")

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

    class RNN:
        P100 = DatasetCondition("RNNdataset", "100", "rnn")
        P9505 = DatasetCondition("RNNdataset", "9505", "rnn")
        P9010 = DatasetCondition("RNNdataset", "9010", "rnn")
        P8020 = DatasetCondition("RNNdataset", "8020", "rnn")


Lesion = Union[str, Iterable[str]]
Quality = Union[str, Iterable[str], None]


@dataclass(frozen=True)
class Animal:
    name: str
    condition: DatasetCondition
    sex: str
    quality: str = "good"  # "good" | "biased" -- a truly bad animal just has no row


class AnimalGroup(Group):
    """Table-driven Group: subclasses set `animals: list[Animal]`.

    Each animal is one row in `animals`. Per-paradigm/lesion cohorts and
    single-animal lookups are *derived* from that table by filtering
    (`.sess()`/`.animal()`), so there is exactly one place to add an
    animal, and cohort membership can never drift out of sync with it.
    `__init_subclass__` rejects duplicate rows outright.

    Loading stays lazy: `.sess()`/`.animal()` only construct `MABData` for
    the rows that match, at call time -- nothing is loaded up front or
    cached.
    """

    animals: List[Animal] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        seen = set()
        for a in cls.__dict__.get("animals", []):
            key = (
                a.name,
                a.condition.data_tag,
                a.condition.paradigm,
                a.condition.lesion_tag,
            )
            if key in seen:
                raise ValueError(f"{cls.__name__}: duplicate animal entry {key}")
            seen.add(key)

    def _rows(
        self,
        *,
        paradigm: Optional[str] = None,
        lesion: Optional[Lesion] = None,
        quality: Quality = ("good", "biased"),
        names: Optional[Iterable[str]] = None,
    ) -> List[Animal]:
        rows = self.animals
        if paradigm is not None:
            rows = [a for a in rows if a.condition.paradigm == paradigm]
        if lesion is not None:
            lesion_set = {lesion} if isinstance(lesion, str) else set(lesion)
            rows = [a for a in rows if a.condition.lesion_tag in lesion_set]
        if quality is not None:
            quality_set = {quality} if isinstance(quality, str) else set(quality)
            rows = [a for a in rows if a.quality in quality_set]
        if names is not None:
            names_set = set(names)
            rows = [a for a in rows if a.name in names_set]
        return rows

    def names(
        self,
        *,
        paradigm: Optional[str] = None,
        lesion: Optional[Lesion] = None,
        quality: Quality = ("good", "biased"),
    ) -> List[str]:
        """List animal names matching the filter, without loading anything."""
        return [
            a.name
            for a in self._rows(paradigm=paradigm, lesion=lesion, quality=quality)
        ]

    def sess(
        self,
        *,
        paradigm: Optional[str] = None,
        lesion: Optional[Lesion] = None,
        quality: Quality = ("good", "biased"),
        names: Optional[Iterable[str]] = None,
    ) -> List[MABData]:
        """Lazily build MABData for every animal matching the filters (loaded on call, never cached)."""
        out: List[MABData] = []
        for a in self._rows(
            paradigm=paradigm, lesion=lesion, quality=quality, names=names
        ):
            out += self.process_wrapper(a.condition, a.name, a.sex)
        return out

    def animal(
        self,
        name: str,
        *,
        paradigm: Optional[str] = None,
        lesion: Optional[str] = None,
    ) -> MABData:
        """Look up a single animal's data by name (+ paradigm/lesion to disambiguate)."""
        matches = self._rows(
            paradigm=paradigm, lesion=lesion, quality=None, names=[name]
        )
        if not matches:
            raise KeyError(
                f"No animal {name!r} in {type(self).__name__} "
                f"(paradigm={paradigm!r}, lesion={lesion!r})"
            )
        if len(matches) > 1:
            raise KeyError(
                f"{name!r} is ambiguous in {type(self).__name__} "
                f"({len(matches)} matches) -- pass paradigm/lesion to disambiguate"
            )
        a = matches[0]
        return self.process_wrapper(a.condition, a.name, a.sex)[0]

    def pre_post_sess(
        self,
        *,
        paradigm: str,
        lesion_tag: str,
        quality: Quality = ("good", "biased"),
    ) -> List[MABData]:
        """Sessions for animals with BOTH an 'intact' and a `lesion_tag` entry
        -- i.e. their own before/after comparison."""
        lesioned_names = {
            a.name
            for a in self._rows(paradigm=paradigm, lesion=lesion_tag, quality=quality)
        }
        return self.sess(
            paradigm=paradigm,
            lesion=("intact", lesion_tag),
            quality=quality,
            names=lesioned_names,
        )


class RNNModelGroup(Group):
    """Base for RNN-model subject groups: `n_models` procedurally-named
    models per paradigm `DatasetCondition`, with a CSV-curated "good"
    subset. Subclasses set `model_prefix` (used in "BGModel{prefix}{i}"
    names) and `best_models_column` (the column of best_models.csv holding
    this group's good model names).
    """

    model_prefix: str = ""
    best_models_column: str = ""
    n_models: int = 40

    def rnn_sess(self, condition: DatasetCondition) -> List[MABData]:
        return [
            self.process_wrapper(condition, f"BGModel{self.model_prefix}{i}", "rnn")[0]
            for i in range(self.n_models)
        ]

    def rnn_good_sess(self, condition: DatasetCondition) -> List[MABData]:
        """Top-performing models for `condition`, selected by asymptotic performance.

        Best models are identified in ``mab_rnn_train.ipynb`` by evaluating
        mean optimal-choice probability over trials 100-200 and saving the
        top 20 model names to ``best_models.csv`` (column
        `best_models_column`).
        """
        csv_path = self.basedir / condition.dirstr / "best_models.csv"
        best_models = pd.read_csv(csv_path)[self.best_models_column].tolist()
        return [
            self.process_wrapper(condition, model_name, "rnn")[0]
            for model_name in best_models
        ]
