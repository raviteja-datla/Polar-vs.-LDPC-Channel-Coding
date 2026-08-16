"""Shared linear algebra over GF(2), used by Hamming verification and by
LDPC's rank check / generator-matrix derivation."""

from __future__ import annotations

import numpy as np


def gf2_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.int64) @ b.astype(np.int64)) % 2


def gf2_row_echelon(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    """Full (Gauss-Jordan) row reduction over GF(2) via row swaps + XOR.

    Returns (reduced_matrix, rank). Reduced rows past `rank` are all-zero.
    """
    m = matrix.astype(np.int64).copy() % 2
    rows, cols = m.shape
    pivot_row = 0
    for col in range(cols):
        if pivot_row >= rows:
            break
        pivot = None
        for r in range(pivot_row, rows):
            if m[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != pivot_row:
            m[[pivot_row, pivot]] = m[[pivot, pivot_row]]
        for r in range(rows):
            if r != pivot_row and m[r, col] == 1:
                m[r] ^= m[pivot_row]
        pivot_row += 1
    return m, pivot_row


def gf2_rank(matrix: np.ndarray) -> int:
    _, rank = gf2_row_echelon(matrix)
    return rank


def select_independent_rows(matrix: np.ndarray) -> np.ndarray:
    """Return the (original, still-sparse) rows of `matrix` that form a
    maximal linearly independent subset, in row order. Used to drop
    structurally-redundant parity checks (e.g. regular LDPC H matrices are
    typically rank-deficient by design) while keeping the surviving rows
    exactly as constructed, rather than the fully-reduced echelon form.
    """
    m, n = matrix.shape
    working_basis: list[tuple[int, np.ndarray]] = []
    keep_indices = []
    for idx in range(m):
        row = matrix[idx].astype(np.int64).copy() % 2
        for pivot_col, basis_row in working_basis:
            if row[pivot_col] == 1:
                row = row ^ basis_row
        nonzero = np.nonzero(row)[0]
        if len(nonzero) == 0:
            continue
        working_basis.append((int(nonzero[0]), row))
        keep_indices.append(idx)
    return matrix[keep_indices]
