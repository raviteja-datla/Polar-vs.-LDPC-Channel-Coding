"""Derive a systematic generator matrix G from a parity-check matrix H via
Gaussian elimination over GF(2).

Goal form: H_std = [P | I_m] (last m columns identity), giving
G = [I_k | P^T] (first k columns identity, k = n - m). Column swaps are
sometimes needed to find a pivot; H_std (the permuted matrix), not the
original H, becomes the parity-check matrix used everywhere downstream
(encoding verification, BP decoding) so G and H stay mutually consistent
with no separate column-index bookkeeping required after this point.

Regular LDPC H matrices built by construction.py are typically NOT full row
rank: with j row-blocks each partitioning all n columns (row weight k), every
block's rows sum (mod 2) to the same all-ones vector, so XORing all rows of
block 1 equals XORing all rows of block 2, etc. — a structural dependency,
not a construction bug. We drop redundant rows via
utils.gf2.select_independent_rows before elimination, which yields a
slightly higher rate code (k = n - rank(H) > n - m) rather than failing.

The full Gauss-Jordan elimination used to compute G XORs rows together,
which destroys H's sparsity — fine for a throwaway working copy used only to
read off G, but not for the H returned for decoding: BP decoding needs a
sparse H with a small, uniform row weight. So H_std is built by applying
only the COLUMN permutation elimination needed (row swaps and column swaps
don't add fill-in; XORing rows does) to the independent-row-selected
(still sparse) H — never the fully-eliminated work copy.
"""

from __future__ import annotations

import numpy as np

from channel_coding.utils.gf2 import select_independent_rows


def derive_generator_matrix(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (G, H_std). H_std is a full-row-rank, sparsity-preserving
    column permutation of H (rows selected for independence, no row-XOR
    fill-in) whose last m columns happen to be invertible; G is the
    systematic generator matrix satisfying (G @ H_std.T) % 2 == 0.
    """
    h_reduced = select_independent_rows(h).astype(np.int64).copy() % 2
    m, n = h_reduced.shape
    k = n - m

    work = h_reduced.copy()
    col_order = np.arange(n)

    for pivot_row in range(m):
        target_col = k + pivot_row

        if work[pivot_row, target_col] == 0:
            row_swap_candidates = np.nonzero(work[pivot_row + 1 :, target_col])[0]
            if len(row_swap_candidates) > 0:
                swap_row = pivot_row + 1 + row_swap_candidates[0]
                work[[pivot_row, swap_row]] = work[[swap_row, pivot_row]]

        if work[pivot_row, target_col] == 0:
            col_swap_candidates = np.nonzero(work[pivot_row, :])[0]
            col_swap_candidates = col_swap_candidates[col_swap_candidates != target_col]
            if len(col_swap_candidates) == 0:
                raise ValueError(
                    "H is not full row rank even after dropping dependent "
                    "rows; this should not happen"
                )
            swap_col = col_swap_candidates[0]
            work[:, [target_col, swap_col]] = work[:, [swap_col, target_col]]
            col_order[[target_col, swap_col]] = col_order[[swap_col, target_col]]

        for r in range(m):
            if r != pivot_row and work[r, target_col] == 1:
                work[r] ^= work[pivot_row]

    p = work[:, :k]
    g = np.hstack([np.eye(k, dtype=np.int64), p.T])
    h_std = h_reduced[:, col_order]
    return g, h_std
