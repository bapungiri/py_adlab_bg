from statplotannot.plots import adjust_lightness
import seaborn as sns
from dataclasses import dataclass


def colors_2arm_swap(amount=1):
    return [
        adjust_lightness("#f55673", amount=amount),  # Unstruc
        adjust_lightness("#ea8c67", amount=amount),  # Unstruc on struc
        adjust_lightness("#3baaa1", amount=amount),  # Struc
        adjust_lightness("#4d89cd", amount=amount),  # Struc on unstruc
    ]


@dataclass
class Palette2Arm:
    lightness_scale: float = 1.0  # lightness scaling

    # canonical color definitions live inside the class
    unstruc: str = "#f55673"
    struc: str = "#3baaa1"

    def as_dict(self):
        """
        Return a dict mapping group → adjusted color.
        Use this directly in seaborn (recommended for hue mapping).
        """
        return {
            "unstruc": adjust_lightness(self.unstruc, self.lightness_scale),
            "struc": adjust_lightness(self.struc, self.lightness_scale),
        }

    def as_list(self):
        """
        Return a Seaborn palette list (ordered colors).
        Useful when hue order is positional.
        """
        m = self.as_dict()
        return sns.color_palette([m["unstruc"], m["struc"]])
