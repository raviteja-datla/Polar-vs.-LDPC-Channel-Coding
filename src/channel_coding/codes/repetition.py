"""Repetition code: repeat each bit `factor` times, decode by majority vote.

The simplest possible form of redundancy. Used first to validate the AWGN
+ BPSK + BER measurement harness itself before trusting it with anything
more complex.
"""

from __future__ import annotations

import numpy as np

from channel_coding.codes.base import Code


class RepetitionCode(Code):
    def __init__(self, k: int, factor: int = 3):
        if factor % 2 == 0:
            raise ValueError("factor must be odd to avoid majority-vote ties")
        self.name = f"Repetition({factor})"
        self.k = k
        self.n = k * factor
        self.factor = factor
        self.soft_decision = False

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        return np.repeat(info_bits, self.factor)

    def decode(self, received: np.ndarray) -> np.ndarray:
        groups = received.reshape(self.k, self.factor)
        votes = groups.sum(axis=1)
        return (votes > self.factor / 2).astype(np.int64)
