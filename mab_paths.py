"""Figure-output path constants, unrelated to subject-group data -- split
out of mab_subjects.py so they don't clutter it.
"""

from pathlib import Path


class FigPath:
    """Namespace of figure-output paths -- accessed on the class itself
    (`FigPath.abstracts`), never instantiated."""

    base: Path = Path("C:/Users/asheshlab/OneDrive/academia/analyses/adlab/figures")
    posters: Path = base / "posters"
    fellowships: Path = base / "fellowships"
    abstracts: Path = base / "abstracts"
    india_alliance: Path = base / "india_alliance"
    pk: Path = base / "pk"


# Back-compat flat names for existing `mab_subjects.figpath`/`iapath`/`pkpath`
# access -- derived from FigPath so each path only has one literal source.
figpath = FigPath.base
iapath = FigPath.india_alliance
pkpath = FigPath.pk
