"""Deliverable 2: short-block vs long-block Polar/LDPC comparison — mirrors
the 5G NR control-channel (short) vs data-channel (long) design split.

Rate is held fixed at ~0.5 for all four curves so block length is the only
independent variable being varied.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "src")

from channel_coding.codes.ldpc.code import LDPCCode
from channel_coding.codes.polar.code import PolarCode
from channel_coding.harness.simulate import monte_carlo_ber
from plot_common import COLOR_BLUE, COLOR_ORANGE, save_csv, save_figure, style_ber_axes

EBN0_DB_RANGE = np.arange(-2.0, 7.0, 0.5)
MIN_ERRORS = 50
MAX_BLOCKS = 3_000
SEED = 2026

SHORT_N = 128
LONG_N = 1024


def build_codes(rng):
    return {
        "Polar short (N=128)": PolarCode(n=SHORT_N, k=SHORT_N // 2, design_ebn0_db=2.0),
        "LDPC short (N=128)": LDPCCode(n=SHORT_N, j=4, k_row_weight=8, rng=rng),
        "Polar long (N=1024)": PolarCode(n=LONG_N, k=LONG_N // 2, design_ebn0_db=2.0),
        "LDPC long (N=1024)": LDPCCode(n=LONG_N, j=4, k_row_weight=8, rng=rng),
    }


def main():
    rng = np.random.default_rng(SEED)
    codes = build_codes(rng)

    fig, ax = plt.subplots(figsize=(8, 6))
    csv_rows = []

    style_map = {
        "Polar short (N=128)": (COLOR_BLUE, "-"),
        "Polar long (N=1024)": (COLOR_BLUE, "--"),
        "LDPC short (N=128)": (COLOR_ORANGE, "-"),
        "LDPC long (N=1024)": (COLOR_ORANGE, "--"),
    }

    for label, code in codes.items():
        points = monte_carlo_ber(code, EBN0_DB_RANGE, rng, min_errors=MIN_ERRORS, max_blocks=MAX_BLOCKS)
        bers = [max(p.ber, 1e-7) for p in points]
        color, linestyle = style_map[label]
        ax.plot(
            EBN0_DB_RANGE,
            bers,
            marker="o",
            markersize=3,
            linewidth=2,
            label=f"{label}  (R={code.rate:.3f})",
            color=color,
            linestyle=linestyle,
            zorder=3,
        )
        for p in points:
            csv_rows.append(
                {
                    "scheme": label,
                    "code_name": code.name,
                    "rate": code.rate,
                    "ebn0_db": p.ebn0_db,
                    "ber": p.ber,
                    "bit_errors": p.bit_errors,
                    "blocks_simulated": p.blocks_simulated,
                }
            )
        print(f"{label}: {code.name} (R={code.rate:.3f}) done")

    style_ber_axes(ax)
    ax.set_title("Polar vs LDPC: Short (Control-like) vs Long (Data-like) Blocks", color="#0b0b0b")
    ax.legend(frameon=False)
    ax.set_ylim(1e-6, 1.0)

    fig_path = save_figure(fig, "short_vs_long_block_comparison.png")
    csv_path = save_csv(csv_rows, "block_length_study.csv")
    print(f"Saved {fig_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
