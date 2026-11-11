"""
Shared matplotlib/seaborn styling for every plot the package produces.
Plots are typeset with LaTeX when a TeX installation is available, and fall
back to matplotlib's own serif text otherwise -- so a machine without LaTeX
still gets every figure.
"""

import shutil

import matplotlib.pyplot as plt
import seaborn as sns


def latex_available() -> bool:
    """A TeX installation matplotlib can use for ``text.usetex``."""
    return shutil.which("latex") is not None


def use_latex() -> None:
    """The project's serif plot style: LaTeX (Latin Modern) if available,
    matplotlib's mathtext with Computer Modern otherwise."""
    sns.set_style("whitegrid")
    if latex_available():
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "text.latex.preamble": r"""
                \usepackage[T1]{fontenc}
                \usepackage{lmodern}
            """,
            }
        )
    else:
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": ["DejaVu Serif"],
                "mathtext.fontset": "cm",
            }
        )
