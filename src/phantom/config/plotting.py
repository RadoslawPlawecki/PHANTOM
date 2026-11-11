"""
Shared matplotlib/seaborn styling for every plot the package produces. Lives
under `config/` because it is stage-agnostic presentational configuration — both
`classification/analytics/visualizer.py` and `preprocessing/metadata/visualizer.py`
depend on it.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def use_latex():
    """Switch matplotlib to the project's LaTeX serif style. Requires a working
    TeX install with the `lmodern` package."""
    sns.set_style("whitegrid")
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
