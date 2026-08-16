"""BPSK modulation and AWGN channel primitives shared by every coding scheme."""

from __future__ import annotations

import numpy as np


def bits_to_bpsk(bits: np.ndarray) -> np.ndarray:
    """Map bits {0,1} to unit-energy BPSK symbols {+1,-1}: 0 -> +1, 1 -> -1."""
    return 1.0 - 2.0 * bits.astype(np.float64)


def eb_n0_db_to_noise_std(ebn0_db: float, rate: float) -> float:
    """Convert Eb/N0 (dB) to AWGN standard deviation for unit-energy BPSK.

    Es = rate * Eb (a rate-r code spends 1/r symbols per info bit), so at a
    fixed Eb/N0 a lower-rate code sees a lower Es/N0. With unit symbol energy,
    N0 = 1 / (rate * Eb/N0_linear) and sigma = sqrt(N0 / 2). Every code must
    pass its own rate here so multi-rate comparisons stay apples-to-apples.
    """
    ebn0_linear = 10.0 ** (ebn0_db / 10.0)
    esn0_linear = rate * ebn0_linear
    noise_variance = 1.0 / (2.0 * esn0_linear)
    return float(np.sqrt(noise_variance))


def add_awgn(symbols: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise with the given standard deviation."""
    if noise_std == 0.0:
        return symbols.copy()
    return symbols + rng.normal(0.0, noise_std, size=symbols.shape)


def channel_llr(received: np.ndarray, noise_std: float) -> np.ndarray:
    """Log-likelihood ratio log P(y|x=0)/P(y|x=1) for BPSK/AWGN with the
    0->+1, 1->-1 mapping used by bits_to_bpsk. Positive LLR favors bit 0.
    """
    return 2.0 * received / (noise_std ** 2)


def demod_hard(received: np.ndarray) -> np.ndarray:
    """Hard-decision demodulation: negative amplitude -> bit 1, else bit 0."""
    return (received < 0.0).astype(np.int64)
