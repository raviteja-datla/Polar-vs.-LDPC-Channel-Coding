"""PolarCode: ties together G_N construction, GA frozen-bit selection, and
the recursive SC decoder behind the shared Code interface."""

from __future__ import annotations

import numpy as np

from channel_coding.codes.base import Code
from channel_coding.codes.polar.construction import select_frozen_set
from channel_coding.codes.polar.decode import sc_decode
from channel_coding.codes.polar.kernel import build_generator_matrix, polar_encode


class PolarCode(Code):
    def __init__(self, n: int, k: int, design_ebn0_db: float = 2.0):
        self.name = f"Polar({n},{k})"
        self.n = n
        self.k = k
        self.soft_decision = True
        self.generator_matrix = build_generator_matrix(n)
        self.info_indices, self.frozen_indices = select_frozen_set(n, k, design_ebn0_db)

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        u = np.zeros(self.n, dtype=np.int64)
        u[self.info_indices] = info_bits
        return polar_encode(u, self.generator_matrix)

    def decode(self, received: np.ndarray) -> np.ndarray:
        u_hat = sc_decode(received, self.frozen_indices)
        return u_hat[self.info_indices]
