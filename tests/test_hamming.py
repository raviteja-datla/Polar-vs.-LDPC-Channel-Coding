import itertools

import numpy as np

from channel_coding.codes.hamming74 import Hamming74Code
from channel_coding.utils.gf2 import gf2_matmul


def test_h_g_orthogonal():
    code = Hamming74Code()
    # every generated codeword must satisfy all parity checks: H @ c = 0
    check = gf2_matmul(code.H, code.G.T)
    np.testing.assert_array_equal(check, np.zeros_like(check))


def test_rate():
    code = Hamming74Code()
    assert code.k == 4 and code.n == 7


def test_exhaustive_single_error_correction():
    """All 16 messages x (no error + each of 7 single-bit-flip positions)
    must decode back to the exact original message: 128 cases total."""
    code = Hamming74Code()
    cases = 0
    for bits in itertools.product([0, 1], repeat=4):
        info = np.array(bits)
        codeword = code.encode(info)

        # no error
        decoded = code.decode(codeword.copy())
        np.testing.assert_array_equal(decoded, info)
        cases += 1

        # each single-bit flip
        for flip_pos in range(code.n):
            corrupted = codeword.copy()
            corrupted[flip_pos] ^= 1
            decoded = code.decode(corrupted)
            np.testing.assert_array_equal(decoded, info)
            cases += 1

    assert cases == 16 * 8 == 128


def test_two_bit_errors_can_fail_to_decode_correctly():
    """Documents the breakdown point: with 2 errors, the syndrome can point
    at the wrong position, so decoding is not guaranteed to recover the
    original message."""
    code = Hamming74Code()
    info = np.array([1, 0, 1, 1])
    codeword = code.encode(info)

    corrupted = codeword.copy()
    corrupted[[0, 1]] ^= 1  # two-bit error
    decoded = code.decode(corrupted)

    assert not np.array_equal(decoded, info)
