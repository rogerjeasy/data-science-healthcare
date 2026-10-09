"""One chart style for every figure in the project (notebooks, report, slides).

Colors come from a validated colorblind-safe palette: use BLUE for a single
series and ORANGE only as the second series or the one highlighted group.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

from .config import REPORT_FIGURES

# Categorical slots (validated pair: CVD ΔE 24.7, all contrast checks pass)
BLUE = "#2a78d6"
ORANGE = "#eb6834"

# Chart chrome
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL_FILL = "#f0efec"

# Sequential ramp (one hue, light -> dark) for magnitudes such as % missing
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_sequential", BLUE_RAMP)


def use_style() -> None:
    """Apply the project style to matplotlib."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "figure.dpi": 110,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.titlelocation": "left",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelcolor": INK_2,
        "axes.labelsize": 10,
        "axes.prop_cycle": cycler(color=[BLUE, ORANGE]),
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "savefig.facecolor": SURFACE,
    })


def save(fig: plt.Figure, name: str) -> None:
    """Save a figure as reports/figures/<name>.png for the report and slides."""
    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
