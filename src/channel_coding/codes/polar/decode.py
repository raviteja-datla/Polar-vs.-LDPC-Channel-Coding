"""Successive Cancellation (SC) decoding for Polar codes.

A recursive LLR-domain tree decoder mirroring the encoder's butterfly
structure: at each level the channel LLR array is split into contiguous
halves, an f-combine (min-sum) produces the "upper" branch that is decoded
first, and a g-combine produces the "lower" branch decoded next.

Each recursive call returns two arrays: the decoded u-bits for that subtree
(which is what the caller ultimately wants), and a "beta" array — the local
re-encoding of those decided bits through that subtree's own generator
matrix (beta = decided_bits @ G_subtree mod 2; at a leaf, beta is just the
bit itself). The g-combine for a node's lower branch must be conditioned on
its upper branch's beta (the value upper's decisions actually encode to),
not on the raw upper-branch bit decisions themselves — mixing these up is
the classic off-by-a-level bug in a from-scratch SC decoder. beta is
propagated up the same way encoding combines two halves: beta_upper is
XORed with beta_lower for the first half, beta_lower is copied through for
the second half (mirroring the G_{2N}=[[G_N,0],[G_N,G_N]] block structure).
"""

from __future__ import annotations

import numpy as np


def _f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Min-sum approximation of the box-plus (check-node) combine."""
    return np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def _g(a: np.ndarray, b: np.ndarray, beta_upper: np.ndarray) -> np.ndarray:
    """Variable-node combine, conditioned on the upper branch's beta."""
    return b + (1.0 - 2.0 * beta_upper) * a


def _sc_decode_recursive(llr: np.ndarray, frozen_mask: np.ndarray):
    n = len(llr)
    if n == 1:
        if frozen_mask[0]:
            bit = np.array([0], dtype=np.int64)
        else:
            bit = np.array([0 if llr[0] >= 0 else 1], dtype=np.int64)
        return bit, bit

    half = n // 2
    llr_l = llr[:half]
    llr_r = llr[half:]

    llr_upper = _f(llr_l, llr_r)
    u_upper, beta_upper = _sc_decode_recursive(llr_upper, frozen_mask[:half])

    llr_lower = _g(llr_l, llr_r, beta_upper.astype(np.float64))
    u_lower, beta_lower = _sc_decode_recursive(llr_lower, frozen_mask[half:])

    u_out = np.concatenate([u_upper, u_lower])
    beta_out = np.concatenate([beta_upper ^ beta_lower, beta_lower])
    return u_out, beta_out


def sc_decode(llr: np.ndarray, frozen_indices: np.ndarray) -> np.ndarray:
    """Decode a length-n channel LLR vector into the length-n u vector
    (frozen positions forced to 0, information positions from the tree)."""
    n = len(llr)
    frozen_mask = np.zeros(n, dtype=bool)
    frozen_mask[frozen_indices] = True
    u_hat, _beta = _sc_decode_recursive(llr, frozen_mask)
    return u_hat
