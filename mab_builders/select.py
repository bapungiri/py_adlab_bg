"""Filter a GroupData product built over all sessions (see 'build_all') down to
the subset a plot needs, instead of rebuilding it per subset."""

import pandas as pd


def _as_set(value):
    return {value} if isinstance(value, str) else set(value)


def select(
    df: pd.DataFrame,
    *,
    group=None,
    paradigm=None,
    lesion=None,
    rnn: bool | None = False,
    names=None,
    paired_lesion: str | None = None,
) -> pd.DataFrame:
    """Rows of 'df' matching every given filter (None = no filter).

    group, paradigm, lesion, names : str or iterable of str
        Match the 'group', 'paradigm', 'lesion' and 'name' columns.
    rnn : bool or None
        False (default) keeps animals only, True keeps RNN models only, None
        keeps both.
    paired_lesion : str, optional
        Keep only animals with both an 'intact' and a 'paired_lesion' entry in
        the same paradigm (their own before/after comparison), and only those
        two lesion conditions -- e.g. "lesion_mPFC_post". Same selection as
        AnimalGroup.intact_post_sess.
    """
    mask = pd.Series(True, index=df.index)
    if rnn is not None:
        is_rnn = df["dataset"] == "RNNdataset"
        mask &= is_rnn if rnn else ~is_rnn
    for col, value in (("group", group), ("paradigm", paradigm), ("lesion", lesion), ("name", names)):
        if value is not None:
            mask &= df[col].isin(_as_set(value))

    out = df[mask]
    if paired_lesion is not None:
        out = out[out["lesion"].isin(["intact", paired_lesion])]
        conditions = out.groupby(["name", "paradigm"])["lesion"].transform("nunique")
        out = out[conditions == 2]
    return out
