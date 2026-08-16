"""Common interface implemented by every coding scheme in this project."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Code(ABC):
    """A block code: k information bits <-> n coded bits.

    soft_decision tells the simulation harness whether decode() expects
    channel LLRs (soft_decision=True) or hard-demodulated bits
    (soft_decision=False).
    """

    name: str
    k: int
    n: int
    soft_decision: bool = False

    @property
    def rate(self) -> float:
        return self.k / self.n

    @abstractmethod
    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """Encode k info bits (0/1 array) into an n-bit codeword."""

    @abstractmethod
    def decode(self, received: np.ndarray) -> np.ndarray:
        """Decode n received values (hard bits or LLRs) into k info bits."""
