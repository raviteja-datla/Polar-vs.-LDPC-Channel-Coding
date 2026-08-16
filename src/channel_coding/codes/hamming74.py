"""Hamming(7,4): the classic single-error-correcting linear block code.

4 data bits -> 7 coded bits via a systematic generator matrix G=[I4|P];
decoding computes a syndrome via H=[P^T|I3] that points directly at the
(at most one) flipped bit position. With 2+ errors in a block the syndrome
can point at the wrong bit (or a valid-looking-but-wrong position), which is
exactly the motivation for moving to iteratively-decoded codes like LDPC.
"""

from __future__ import annotations

import numpy as np

from channel_coding.codes.base import Code

_P = np.array(
    [
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
)

_G = np.hstack([np.eye(4, dtype=np.int64), _P])
_H = np.hstack([_P.T, np.eye(3, dtype=np.int64)])


class Hamming74Code(Code):
    def __init__(self):
        self.name = "Hamming(7,4)"
        self.k = 4
        self.n = 7
        self.soft_decision = False
        self.G = _G
        self.H = _H
        # syndrome (as a tuple of 3 bits) -> index of the column of H it
        # matches, i.e. which single bit position that syndrome implicates.
        self._syndrome_to_error_index = {
            tuple(self.H[:, j]): j for j in range(self.n)
        }

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        return (info_bits @ self.G) % 2

    def decode(self, received: np.ndarray) -> np.ndarray:
        received = received.astype(np.int64)
        syndrome = tuple((self.H @ received) % 2)
        corrected = received.copy()
        if any(syndrome):
            error_index = self._syndrome_to_error_index.get(syndrome)
            if error_index is not None:
                corrected[error_index] ^= 1
        return corrected[: self.k]
