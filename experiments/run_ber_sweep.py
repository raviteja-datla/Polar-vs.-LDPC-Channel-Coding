"""Deliverable 1: BER vs Eb/N0 headline plot, all five schemes overlaid —
uncoded, repetition, Hamming(7,4), Polar, LDPC — the "coding sophistication
ladder." Also prints/saves each scheme's code rate alongside its curve,
since this is a rate-vs-reliability comparison, not just "lowest BER wins."
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "src")

from channel_coding.codes.hamming74 import Hamming74Code
from channel_coding.codes.ldpc.code import LDPCCode
from channel_coding.codes.polar.code import PolarCode
from channel_coding.codes.repetition import RepetitionCode
from channel_coding.codes.uncoded import UncodedCode
from channel_coding.harness.simulate import monte_carlo_ber
from plot_common import SCHEME_COLORS, save_csv, save_figure, style_ber_axes

EBN0_DB_RANGE = np.arange(-2.0, 9.5, 0.5)
MIN_ERRORS = 100
MAX_BLOCKS = 20_000
SEED = 2026


def build_codes(rng):
    return [
        UncodedCode(k=64),
        RepetitionCode(k=64, factor=3),
        Hamming74Code(),
        PolarCode(n=128, k=64, design_ebn0_db=2.0),
        LDPCCode(n=128, j=4, k_row_weight=8, rng=rng),
    ]


def main():
    rng = np.random.default_rng(SEED)
    codes = build_codes(rng)

    fig, ax = plt.subplots(figsize=(8, 6))
    csv_rows = []
    palette = list(SCHEME_COLORS.values())

    for i, code in enumerate(codes):
        points = monte_carlo_ber(code, EBN0_DB_RANGE, rng, min_errors=MIN_ERRORS, max_blocks=MAX_BLOCKS)
        bers = [max(p.ber, 1e-7) for p in points]  # floor for log-scale plotting
        color = palette[i % len(palette)]
        label = f"{code.name}  (R={code.rate:.3f})"
        ax.plot(EBN0_DB_RANGE, bers, marker="o", markersize=3, linewidth=2, label=label, color=color, zorder=3)

        for p in points:
            csv_rows.append(
                {
                    "code": code.name,
                    "rate": code.rate,
                    "ebn0_db": p.ebn0_db,
                    "ber": p.ber,
                    "bit_errors": p.bit_errors,
                    "blocks_simulated": p.blocks_simulated,
                }
            )
        print(f"{code.name} (R={code.rate:.3f}) done")

    style_ber_axes(ax)
    ax.set_title("BER vs Eb/N0 — Coding Sophistication Ladder", color="#0b0b0b")
    ax.legend(frameon=False)
    ax.set_ylim(1e-6, 1.0)

    fig_path = save_figure(fig, "ber_vs_ebn0_headline.png")
    csv_path = save_csv(csv_rows, "ber_sweep_headline.csv")
    print(f"Saved {fig_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
