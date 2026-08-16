"""Optional depth: visualize the LDPC Tanner graph and the Polar SC decoding
tree, at small sizes chosen purely for legibility (not the sizes used in the
BER experiments).
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "src")

from channel_coding.codes.ldpc.construction import build_regular_ldpc_h
from channel_coding.codes.polar.construction import select_frozen_set
from plot_common import COLOR_AQUA, COLOR_BLUE, COLOR_ORANGE, TEXT_PRIMARY, save_figure


def plot_tanner_graph(n=16, j=3, k=4, seed=3):
    rng = np.random.default_rng(seed)
    h = build_regular_ldpc_h(n, j, k, rng)
    m = h.shape[0]

    fig, ax = plt.subplots(figsize=(8, 6))

    var_x, var_y = 0.0, np.linspace(1, 0, n)
    check_x, check_y = 1.0, np.linspace(1, 0, m)

    for c in range(m):
        for v in np.nonzero(h[c])[0]:
            ax.plot([var_x, check_x], [var_y[v], check_y[c]], color="#c9c8c2", linewidth=0.8, zorder=1)

    ax.scatter([var_x] * n, var_y, s=220, color=COLOR_BLUE, zorder=2, label="variable nodes (bits)")
    ax.scatter([check_x] * m, check_y, s=220, color=COLOR_ORANGE, marker="s", zorder=2, label="check nodes (parity)")

    for v in range(n):
        ax.annotate(f"v{v}", (var_x, var_y[v]), ha="right", va="center", xytext=(-14, 0), textcoords="offset points", fontsize=8, color=TEXT_PRIMARY)
    for c in range(m):
        ax.annotate(f"c{c}", (check_x, check_y[c]), ha="left", va="center", xytext=(14, 0), textcoords="offset points", fontsize=8, color=TEXT_PRIMARY)

    ax.set_title(f"LDPC Tanner Graph (n={n}, m={m}, column weight={j}, row weight={k})", color=TEXT_PRIMARY)
    ax.set_xlim(-0.3, 1.3)
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)

    return save_figure(fig, "ldpc_tanner_graph.png")


def plot_sc_tree(n=16, k=8, design_ebn0_db=2.0):
    info_indices, frozen_indices = select_frozen_set(n, k, design_ebn0_db)
    is_info = np.zeros(n, dtype=bool)
    is_info[info_indices] = True

    depth = int(np.log2(n))
    fig, ax = plt.subplots(figsize=(10, 6))

    # node positions: level 0 = root (single combine spanning all N), level `depth` = leaves (u0..u_{n-1})
    def node_x(level, index_at_level, count_at_level):
        width = n
        slot_width = width / count_at_level
        return (index_at_level + 0.5) * slot_width

    for level in range(depth + 1):
        count = 2 ** level
        y = depth - level
        for idx in range(count):
            x = node_x(level, idx, count)
            if level == depth:
                color = COLOR_ORANGE if is_info[idx] else "#b7b6b0"
                ax.scatter([x], [y], s=140, color=color, zorder=3)
                label = f"u{idx}\n{'info' if is_info[idx] else 'frozen'}"
                ax.annotate(label, (x, y), ha="center", va="top", xytext=(0, -12), textcoords="offset points", fontsize=6.5, color=TEXT_PRIMARY)
            else:
                ax.scatter([x], [y], s=90, color=COLOR_BLUE, zorder=3)
            if level > 0:
                parent_count = 2 ** (level - 1)
                parent_idx = idx // 2
                px = node_x(level - 1, parent_idx, parent_count)
                py = depth - (level - 1)
                branch_color = COLOR_AQUA if idx % 2 == 0 else "#9a8fd1"
                ax.plot([px, x], [py, y], color=branch_color, linewidth=1.2, zorder=1)

    ax.scatter([], [], s=90, color=COLOR_BLUE, label="internal node (f/g combine)")
    ax.scatter([], [], s=140, color=COLOR_ORANGE, label="leaf: information bit")
    ax.scatter([], [], s=140, color="#b7b6b0", label="leaf: frozen bit (=0)")
    ax.plot([], [], color=COLOR_AQUA, label="f-combine (upper/minus branch)")
    ax.plot([], [], color="#9a8fd1", label="g-combine (lower/plus branch)")

    ax.set_title(f"Polar SC Decoding Tree (N={n}, K={k}, design Eb/N0={design_ebn0_db} dB)", color=TEXT_PRIMARY)
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=8)

    return save_figure(fig, "polar_sc_tree.png")


def main():
    p1 = plot_tanner_graph()
    print(f"Saved {p1}")
    p2 = plot_sc_tree()
    print(f"Saved {p2}")


if __name__ == "__main__":
    main()
