"""Bit-channel reliability ranking via Gaussian Approximation (GA) density
evolution under AWGN, so frozen-bit positions are derived rather than taken
from a hardcoded published table.

GA tracks a single parameter per synthetic channel: the mean of its LLR,
assuming a symmetric Gaussian LLR (variance = 2*mean), which holds exactly
for BPSK/AWGN before any combining and is the standard approximation used
after combining (Trifonov, 2012). The recursion mirrors the SC decoder's own
contiguous-halves recursion tree exactly (see decode.py) so that both index
spaces line up: the "upper" (minus) branch always gets less reliable, the
"lower" (plus) branch always gets more reliable, matching how min-sum (f)
and sum (g) combining behave.
"""

from __future__ import annotations

import numpy as np

from channel_coding.harness.channel import eb_n0_db_to_noise_std


def _phi(x: float) -> float:
    """Trifonov's piecewise fit to the AWGN density-evolution phi function.
    Monotonically decreasing from phi(0)=1 to phi(inf)=0.
    """
    if x <= 0.0:
        return 1.0
    if x < 10.0:
        return float(np.exp(-0.4527 * x ** 0.859 + 0.0218))
    return float(np.sqrt(np.pi / x) * np.exp(-x / 4.0) * (1.0 - 10.0 / (7.0 * x)))


def _phi_inv(y: float, tol: float = 1e-6) -> float:
    """Numeric inverse of phi via bisection (no closed form)."""
    if y >= 1.0:
        return 0.0
    if y <= 0.0:
        y = 1e-12
    lo, hi = 0.0, 1.0
    while _phi(hi) > y:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if hi - lo < tol:
            break
        if _phi(mid) > y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _minus_combine_mean(m: float) -> float:
    """Mean of the LLR resulting from an f-combine (check-node-like, 'upper'
    branch) of two independent channels each with LLR mean m."""
    return _phi_inv(1.0 - (1.0 - _phi(m)) ** 2)


def _plus_combine_mean(m: float) -> float:
    """Mean of the LLR resulting from a g-combine (variable-node-like,
    'lower' branch, genie-aided correct previous decision) of two independent
    channels each with LLR mean m."""
    return 2.0 * m


def channel_means(n: int, m0: float) -> np.ndarray:
    """Recursively compute the GA mean of each of the n final synthetic
    bit-channels, starting from n i.i.d. copies of the base channel (mean
    m0). Recursion structure exactly matches decode.sc_decode's contiguous-
    halves tree: upper half comes from repeated minus-combines, lower half
    from repeated plus-combines, processed upper-subtree-first.
    """
    if n == 1:
        return np.array([m0])
    half = n // 2
    upper_mean = _minus_combine_mean(m0)
    lower_mean = _plus_combine_mean(m0)
    return np.concatenate([channel_means(half, upper_mean), channel_means(half, lower_mean)])


def select_frozen_set(n: int, k: int, design_ebn0_db: float = 2.0):
    """Return (info_indices, frozen_indices), both sorted ascending, ranking
    the n synthetic bit-channels by GA reliability at the given design Eb/N0
    and keeping the k most reliable as information-bearing.
    """
    sigma = eb_n0_db_to_noise_std(design_ebn0_db, rate=1.0)
    m0 = 2.0 / sigma ** 2
    means = channel_means(n, m0)
    order = np.argsort(-means)
    info_indices = np.sort(order[:k])
    frozen_indices = np.sort(order[k:])
    return info_indices, frozen_indices
