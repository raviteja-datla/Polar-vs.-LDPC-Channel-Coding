"""Arikan kernel, Kronecker power, bit-reversal, and the Polar generator
matrix.

The generator matrix used here is the raw Kronecker power G_N = F^{(x)
log2(N)}, WITHOUT the B_N bit-reversal permutation some presentations of
Arikan's construction include. B_N is only needed if you want the physical
channel index order to match a particular hardware butterfly layout; it is
not needed for correctness, and dropping it keeps the encoder (a direct
matrix multiply, ground truth) and the recursive SC decoder (decode.py)
using the exact same contiguous-halves index convention, which is what
actually has to match for encode/decode to round-trip. This pairing (no
bit-reversal, contiguous-halves recursion, beta-message propagated up
through the decoder's recursion) was verified by exhaustive zero-noise
round-trip testing at N up to 128 before being adopted; bit_reversal_permutation
is kept as a documented, independently-tested utility, not applied here.

Encoding is done as a direct matrix multiply x = (u @ G_N) % 2 rather than a
hand-rolled recursive encoder: at N up to ~1024 this is trivially cheap and
it sidesteps a source of indexing bugs a recursive encoder would introduce.
"""

from __future__ import annotations

import numpy as np

ARIKAN_F = np.array([[1, 0], [1, 1]], dtype=np.int64)


def _log2_exact(n: int) -> int:
    m = n.bit_length() - 1
    if n <= 0 or (1 << m) != n:
        raise ValueError(f"N must be a power of two, got {n}")
    return m


def kronecker_power(matrix: np.ndarray, power: int) -> np.ndarray:
    """matrix^{(x)power} (Kronecker power), reduced mod 2 at each step."""
    result = np.array([[1]], dtype=np.int64)
    for _ in range(power):
        result = np.kron(result, matrix) % 2
    return result


def bit_reversal_permutation(n: int) -> np.ndarray:
    """perm[i] = bit-reversal of i using log2(n) bits, for i in 0..n-1."""
    num_bits = _log2_exact(n)
    perm = np.empty(n, dtype=np.int64)
    for i in range(n):
        reversed_bits = int(format(i, f"0{num_bits}b")[::-1], 2)
        perm[i] = reversed_bits
    return perm


def build_generator_matrix(n: int) -> np.ndarray:
    m = _log2_exact(n)
    return kronecker_power(ARIKAN_F, m)


def polar_encode(u: np.ndarray, generator_matrix: np.ndarray) -> np.ndarray:
    return (u @ generator_matrix) % 2
