"""LDPCCode: ties together regular random H construction, systematic G
derivation, and min-sum BP decoding behind the shared Code interface."""

from __future__ import annotations

import numpy as np

from channel_coding.codes.base import Code
from channel_coding.codes.ldpc.construction import build_regular_ldpc_h
from channel_coding.codes.ldpc.decode import build_adjacency, min_sum_decode
from channel_coding.codes.ldpc.generator import derive_generator_matrix


class LDPCCode(Code):
    def __init__(
        self,
        n: int,
        j: int,
        k_row_weight: int,
        rng: np.random.Generator,
        max_iters: int = 30,
    ):
        h_random = build_regular_ldpc_h(n, j, k_row_weight, rng)
        self.generator_matrix, self.h = derive_generator_matrix(h_random)

        self.name = f"LDPC({n},{self.generator_matrix.shape[0]})"
        self.n = n
        self.k = self.generator_matrix.shape[0]
        self.soft_decision = True
        self.max_iters = max_iters
        self._adjacency = build_adjacency(self.h)

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        return (info_bits @ self.generator_matrix) % 2

    def decode(self, received: np.ndarray) -> np.ndarray:
        x_hat, _converged, _iters = min_sum_decode(
            received,
            self.h,
            max_iters=self.max_iters,
            adjacency=self._adjacency,
        )
        return x_hat[: self.k]
