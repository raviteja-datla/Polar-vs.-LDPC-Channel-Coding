"""Shared plotting style and CSV I/O helpers for the experiment scripts."""

from __future__ import annotations

import csv
import pathlib

import matplotlib.pyplot as plt

FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "figures"
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "data"

# Validated categorical palette (light mode), used in fixed order per series
# so identity always maps to the same color across plots.
COLOR_BLUE = "#2a78d6"
COLOR_ORANGE = "#eb6834"
COLOR_AQUA = "#1baf7a"
COLOR_YELLOW = "#eda100"
COLOR_MAGENTA = "#e87ba4"
COLOR_VIOLET = "#4a3aa7"
COLOR_RED = "#e34948"

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e3e2dd"

SCHEME_COLORS = {
    "Uncoded": COLOR_BLUE,
    "Repetition(3)": COLOR_ORANGE,
    "Hamming(7,4)": COLOR_AQUA,
    "Polar": COLOR_YELLOW,
    "LDPC": COLOR_MAGENTA,
}


def style_ber_axes(ax, xlabel="Eb/N0 (dB)", ylabel="Bit Error Rate"):
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, color=TEXT_PRIMARY)
    ax.set_ylabel(ylabel, color=TEXT_PRIMARY)
    ax.grid(True, which="both", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT_SECONDARY)
    ax.spines["bottom"].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY)


def save_figure(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_csv(rows: list[dict], name: str):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    if not rows:
        return path
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path
