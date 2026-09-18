from mab_data_core import (
    Animal,
    AnimalGroup,
    Datasets,
    DatasetCondition,
    Group,
    MABData,
    RNNModelGroup,
)

# FigPath/figpath*/iapath/pkpath are unrelated to subject-group data -- live
# in mab_paths.py. Re-exported here so existing `mab_subjects.figpath`-style
# access keeps working unchanged.
from mab_paths import FigPath, figpath, iapath, pkpath


class Struc(AnimalGroup):
    group_tag = "struc"

    animals = [
        # --- Aarushi's dataset (AC) ---
        Animal("Bewilderbeast", Datasets.AC.P100_intact, "female"),
        Animal("Aguero", Datasets.AC.P100_intact, "female"),
        Animal("Sterling", Datasets.AC.P100_intact, "female"),
        Animal("Aguero", Datasets.AC.P100_lesion_OFC_post, "female"),
        Animal("Phil", Datasets.AC.P100_lesion_OFC_pre, "female"),
        Animal("Rodri", Datasets.AC.P100_lesion_OFC_pre, "female"),
        Animal("Gavi", Datasets.AC.P8020_intact, "female"),
        Animal("Haaland", Datasets.AC.P8020_intact, "male"),
        Animal("Pedri", Datasets.AC.P8020_intact, "female"),
        Animal("Xavi", Datasets.AC.P8020_intact, "male"),
        Animal("Gavi", Datasets.AC.P8020_lesion_OFC_post, "female"),
        Animal("Pedri", Datasets.AC.P8020_lesion_OFC_post, "female"),
        Animal("Xavi", Datasets.AC.P8020_lesion_OFC_post, "male"),
        Animal("Haaland", Datasets.AC.P8020_lesion_OFC_post, "male"),
        # --- Anirudh's dataset (AS) ---
        # Bad animals (excluded -- never got a row): Grump, Brat
        Animal("Gronckle", Datasets.AS.P100_intact, "female"),
        Animal("Toothless", Datasets.AS.P100_intact, "female"),
        Animal("Buffalord", Datasets.AS.P100_intact, "female"),
        # --- BG dataset ---
        Animal("BGF7", Datasets.BG.P9505_intact, "female"),
        Animal("BGM1", Datasets.BG.P8020_intact, "male"),
        Animal("BGF0", Datasets.BG.P8020_intact, "female"),
        Animal("BGM3", Datasets.BG.P8020_intact, "male"),
        Animal("BGM4", Datasets.BG.P8020_intact, "male"),
        Animal("BGF4", Datasets.BG.P8020_intact, "female"),
        Animal("BGM6", Datasets.BG.P8020_intact, "male"),
        Animal("BGF0", Datasets.BG.P8020_lesion_mPFC_post, "female"),
        Animal("BGM6", Datasets.BG.P8020_lesion_mPFC_post, "male"),
    ]

    @property
    def p100_intact_sess(self):
        return self.sess(paradigm="100", lesion="intact")

    @property
    def p100_good_intact_sess(self):
        return self.sess(paradigm="100", lesion="intact", quality="good")

    @property
    def p100_lesion_OFC_pre_sess(self):
        return self.sess(paradigm="100", lesion="lesion_OFC_pre")

    @property
    def p100_lesion_OFC_post_sess(self):
        return self.sess(paradigm="100", lesion="lesion_OFC_post")

    @property
    def p8020_intact_sess(self):
        return self.sess(paradigm="8020", lesion="intact")

    @property
    def p8020_good_intact_sess(self):
        return self.sess(paradigm="8020", lesion="intact", quality="good")

    @property
    def p8020_lesion_OFC_post_sess(self):
        return self.sess(paradigm="8020", lesion="lesion_OFC_post")

    @property
    def p8020_lesion_OFC_pre_post_sess(self):
        return self.pre_post_sess(paradigm="8020", lesion_tag="lesion_OFC_post")

    @property
    def p8020_lesion_mPFC_post_sess(self):
        return self.sess(paradigm="8020", lesion="lesion_mPFC_post")

    @property
    def p8020_lesion_mPFC_pre_post_sess(self):
        return self.pre_post_sess(paradigm="8020", lesion_tag="lesion_mPFC_post")

    @property
    def all_intact_sess(self):
        return self.p100_intact_sess + self.p8020_intact_sess

    @property
    def all_good_intact_sess(self):
        return self.p100_good_intact_sess + self.p8020_good_intact_sess

    @property
    def all_good_sess(self):
        return (
            self.p100_good_intact_sess
            + self.p8020_good_intact_sess
            + self.p100_lesion_OFC_pre_sess
            + self.p100_lesion_OFC_post_sess
            + self.p8020_lesion_OFC_post_sess
        )


class Unstruc(AnimalGroup):
    group_tag = "unstruc"

    animals = [
        # --- Aarushi's dataset (AC) ---
        # Bad animal (excluded -- never got a row): Torres
        Animal("Aggro", Datasets.AC.P100_intact, "female"),
        Animal("Auroma", Datasets.AC.P100_intact, "female"),
        Animal("Debruyne", Datasets.AC.P100_lesion_OFC_pre, "female"),
        Animal("Kompany", Datasets.AC.P100_lesion_OFC_pre, "female"),
        Animal("Aggro", Datasets.AC.P100_lesion_OFC_post, "female"),
        Animal("Messi", Datasets.AC.P8020_intact, "male"),
        Animal("Neymar", Datasets.AC.P8020_intact, "male", quality="biased"),
        Animal("Son", Datasets.AC.P8020_intact, "male"),
        Animal("Messi", Datasets.AC.P8020_lesion_OFC_post, "male"),
        Animal("Son", Datasets.AC.P8020_lesion_OFC_post, "male"),
        # --- Anirudh's dataset (AS) ---
        # Excluded -- bad data / environment changed post-lesion: Grump, Brat, Gronckle2
        # --- BG dataset ---
        Animal("BGM8", Datasets.BG.P9505_intact, "male"),
        Animal("BGM9", Datasets.BG.P9505_intact, "male"),
        Animal("BGM0", Datasets.BG.P8020_intact, "male"),
        # BGM2 excluded -- bad animal
        Animal("BGF1", Datasets.BG.P8020_intact, "female"),
        Animal("BGF2", Datasets.BG.P8020_intact, "female"),
        Animal("BGF3", Datasets.BG.P8020_intact, "female", quality="biased"),
        Animal("BGM5", Datasets.BG.P8020_intact, "male"),
        Animal("BGF5", Datasets.BG.P8020_intact, "female"),
        Animal("BGM7", Datasets.BG.P8020_intact, "male"),
        Animal("BGF2", Datasets.BG.P8020_lesion_mPFC_post, "female"),
        Animal("BGF5", Datasets.BG.P8020_lesion_mPFC_post, "female"),
        Animal("BGM7", Datasets.BG.P8020_lesion_mPFC_post, "male"),
    ]

    @property
    def p100_intact_sess(self):
        return self.sess(paradigm="100", lesion="intact")

    @property
    def p100_good_intact_sess(self):
        return self.sess(paradigm="100", lesion="intact", quality="good")

    @property
    def p100_lesion_OFC_pre_sess(self):
        return self.sess(paradigm="100", lesion="lesion_OFC_pre")

    @property
    def p100_lesion_OFC_post_sess(self):
        return self.sess(paradigm="100", lesion="lesion_OFC_post")

    @property
    def p8020_intact_sess(self):
        return self.sess(paradigm="8020", lesion="intact")

    @property
    def p8020_good_intact_sess(self):
        return self.sess(paradigm="8020", lesion="intact", quality="good")

    @property
    def p8020_lesion_OFC_post_sess(self):
        return self.sess(paradigm="8020", lesion="lesion_OFC_post")

    @property
    def p8020_lesion_OFC_pre_post_sess(self):
        return self.pre_post_sess(paradigm="8020", lesion_tag="lesion_OFC_post")

    @property
    def p8020_lesion_mPFC_post_sess(self):
        return self.sess(paradigm="8020", lesion="lesion_mPFC_post")

    @property
    def p8020_lesion_mPFC_pre_post_sess(self):
        return self.pre_post_sess(paradigm="8020", lesion_tag="lesion_mPFC_post")

    @property
    def all_intact_sess(self):
        return self.p100_intact_sess + self.p8020_intact_sess

    @property
    def all_good_intact_sess(self):
        return self.p100_good_intact_sess + self.p8020_good_intact_sess

    @property
    def all_good_sess(self):
        return (
            self.p100_good_intact_sess
            + self.p8020_good_intact_sess
            + self.p100_lesion_OFC_pre_sess
            + self.p100_lesion_OFC_post_sess
            + self.p8020_lesion_OFC_post_sess
            + self.p8020_lesion_mPFC_post_sess
        )


class StrucRNN(RNNModelGroup):
    group_tag = "struc"
    model_prefix = "S"
    best_models_column = "struc"

    @property
    def p100_sess(self):
        return self.rnn_sess(Datasets.RNN.P100)

    @property
    def p100_good_sess(self):
        return self.rnn_good_sess(Datasets.RNN.P100)

    @property
    def p9505_sess(self):
        return self.rnn_sess(Datasets.RNN.P9505)

    @property
    def p9505_good_sess(self):
        return self.rnn_good_sess(Datasets.RNN.P9505)

    @property
    def p9010_sess(self):
        return self.rnn_sess(Datasets.RNN.P9010)

    @property
    def p9010_good_sess(self):
        return self.rnn_good_sess(Datasets.RNN.P9010)

    @property
    def p8020_sess(self):
        return self.rnn_sess(Datasets.RNN.P8020)

    @property
    def p8020_good_sess(self):
        return self.rnn_good_sess(Datasets.RNN.P8020)


class UnstrucRNN(RNNModelGroup):
    group_tag = "unstruc"
    model_prefix = "U"
    best_models_column = "unstruc"

    @property
    def p100_sess(self):
        return self.rnn_sess(Datasets.RNN.P100)

    @property
    def p100_good_sess(self):
        return self.rnn_good_sess(Datasets.RNN.P100)

    @property
    def p9505_sess(self):
        return self.rnn_sess(Datasets.RNN.P9505)

    @property
    def p9505_good_sess(self):
        return self.rnn_good_sess(Datasets.RNN.P9505)

    @property
    def p9010_sess(self):
        return self.rnn_sess(Datasets.RNN.P9010)

    @property
    def p9010_good_sess(self):
        return self.rnn_good_sess(Datasets.RNN.P9010)

    @property
    def p8020_sess(self):
        return self.rnn_sess(Datasets.RNN.P8020)

    @property
    def p8020_good_sess(self):
        return self.rnn_good_sess(Datasets.RNN.P8020)


struc = Struc()
unstruc = Unstruc()
struc_rnn = StrucRNN()
unstruc_rnn = UnstrucRNN()

# MostlyStrucShortBlocks/MostlyUnstrucShortBlocks are old/stale -- moved to
# mab_subjects_archive.py. Re-exported here so existing
# `mab_subjects.mostly_struc_short_blocks`-style access keeps working.
# LSTMData/rnn_expsN are NOT re-exported here (no longer part of
# mab_subjects's surface) -- still defined in mab_subjects_archive.py if
# needed directly: `import mab_subjects_archive; mab_subjects_archive.rnn_exps1`.
from mab_subjects_archive import (
    MostlyStrucShortBlocks,
    MostlyUnstrucShortBlocks,
    mostly_struc_short_blocks,
    mostly_unstruc_short_blocks,
)

# GroupData/VersionedAccessor live in mab_group_data.py so their
# self-rewriting stub (see GroupData._write_stub) only ever touches that
# small dedicated file, never this one. Re-exported here so existing
# `from mab_subjects import GroupData` imports keep working unchanged.
from mab_group_data import GroupData, VersionedAccessor
