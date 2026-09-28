from statplotannot.plots import adjust_lightness
import seaborn as sns
from dataclasses import dataclass


def colors_2arm_swap(amount=1):
    """Ordered list [unstruc, unstruc on struc, struc, struc on unstruc]."""
    return list(Palette2Arm(lightness_scale=amount).swap().values())


@dataclass
class Palette2Arm:
    lightness_scale: float = 1.0  # lightness scaling

    # canonical color definitions live inside the class
    unstruc: str = "#f55673"
    struc: str = "#3baaa1"
    unstruc_old: str = "#f58f2a"
    struc_old: str = "#1986ad"
    unstruc_lesion: str = "#ae2846"
    struc_lesion: str = "#007c5c"
    unstruc_on_struc: str = "#ea8c67"
    struc_on_unstruc: str = "#4d89cd"

    def _adjust(self, color):
        return adjust_lightness(color, self.lightness_scale)

    def as_dict(self):
        """
        Return a dict mapping group → adjusted color.
        Use this directly in seaborn (recommended for hue mapping).
        """
        return {
            "unstruc": self._adjust(self.unstruc),
            "struc": self._adjust(self.struc),
        }

    def as_list(self):
        """
        Return a Seaborn palette list (ordered colors).
        Useful when hue order is positional.
        """
        m = self.as_dict()
        return sns.color_palette([m["unstruc"], m["struc"]])

    def old_as_dict(self):
        """
        Return a dict mapping group → adjusted color for old data.
        """
        return {
            "unstruc_old": self._adjust(self.unstruc_old),
            "struc_old": self._adjust(self.struc_old),
        }

    def old_vs_new(self):
        """
        Return a dict mapping group → adjusted color for old vs new comparison.
        """
        return {**self.old_as_dict(), **self.as_dict()}

    def lesion(self, group):
        """
        Return a dict mapping lesion tag → adjusted color for 'group'
        ("unstruc" or "struc"). Intact uses the group color, all lesion tags
        share the lighter lesion shade.
        """
        intact = self._adjust(getattr(self, group))
        lesioned = self._adjust(getattr(self, f"{group}_lesion"))
        return {
            "intact": intact,
            "lesion_mPFC_post": lesioned,
            "lesion_OFC_post": lesioned,
            "lesion_OFC_pre": lesioned,
        }

    def struc_lesion_vs_intact(self):
        """Lesion palette for struc; see 'lesion'."""
        return self.lesion("struc")

    def unstruc_lesion_vs_intact(self):
        """Lesion palette for unstruc; see 'lesion'."""
        return self.lesion("unstruc")

    def swap(self):
        """
        Return a dict mapping condition → adjusted color for environment-swap
        comparisons, ordered unstruc, unstruc on struc, struc, struc on unstruc.
        """
        return {
            "unstruc": self._adjust(self.unstruc),
            "unstruc_on_struc": self._adjust(self.unstruc_on_struc),
            "struc": self._adjust(self.struc),
            "struc_on_unstruc": self._adjust(self.struc_on_unstruc),
        }


# default instance for notebooks: palette=palette_2arm.as_dict()
palette_2arm = Palette2Arm()
