"""Uncoded baseline: transmit bits directly with no redundancy (rate 1)."""

from __future__ import annotations

import numpy as np

from channel_coding.codes.base import Code


class UncodedCode(Code):
    """received is already hard-demodulated bits (soft_decision=False), so
    decoding an uncoded, rate-1 block is the identity."""

    def __init__(self, k: int):
        self.name = "Uncoded"
        self.k = k
        self.n = k
        self.soft_decision = False

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        return info_bits.copy()

    def decode(self, received: np.ndarray) -> np.ndarray:
        return received.copy()
