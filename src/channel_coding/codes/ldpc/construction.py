"""Gallager-style regular random LDPC parity-check matrix construction.

Builds an m x n H with exact column weight j and row weight k by stacking j
row-blocks: the first is a deterministic "staircase" (row r has k consecutive
1s at columns [r*k, (r+1)*k)), and the remaining j-1 blocks are independent
random column permutations of the first. This guarantees the exact weights
by construction. To keep the Tanner graph reasonably simple for a small
pedagogical code, attempts that produce duplicate rows or 4-cycles (two rows
sharing 2+ columns) are retried up to max_retries times, keeping the least-
violating attempt if none comes out perfectly clean.
"""

from __future__ import annotations

import warnings

import numpy as np


def _count_violations(h: np.ndarray) -> int:
    """Count row pairs sharing >=2 common 1-columns (4-cycles, which also
    catches exact duplicate rows as a special case)."""
    overlap = h.astype(np.int64) @ h.T
    np.fill_diagonal(overlap, 0)
    return int(np.sum(overlap >= 2) // 2)


def build_regular_ldpc_h(
    n: int,
    j: int,
    k: int,
    rng: np.random.Generator,
    max_retries: int = 30,
) -> np.ndarray:
    """Build an (n*j//k) x n regular LDPC parity-check matrix with column
    weight j and row weight k."""
    if (n * j) % k != 0:
        raise ValueError(f"n*j must be divisible by k: n={n}, j={j}, k={k}")
    m = n * j // k
    if m % j != 0:
        raise ValueError(f"m=n*j/k must be divisible by j: got m={m}, j={j}")
    rows_per_block = m // j

    base = np.zeros((rows_per_block, n), dtype=np.int64)
    for r in range(rows_per_block):
        base[r, r * k : (r + 1) * k] = 1

    best_h = None
    best_violations = None
    for _ in range(max_retries):
        blocks = [base]
        for _ in range(j - 1):
            perm = rng.permutation(n)
            blocks.append(base[:, perm])
        h = np.vstack(blocks)

        violations = _count_violations(h)
        if violations == 0:
            return h
        if best_violations is None or violations < best_violations:
            best_h, best_violations = h, violations

    warnings.warn(
        f"build_regular_ldpc_h: could not fully eliminate 4-cycles/duplicate "
        f"rows in {max_retries} attempts; using best attempt with "
        f"{best_violations} remaining violations.",
        stacklevel=2,
    )
    return best_h
