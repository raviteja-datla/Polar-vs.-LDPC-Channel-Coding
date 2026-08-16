import numpy as np

from channel_coding.codes.polar.kernel import (
    ARIKAN_F,
    bit_reversal_permutation,
    build_generator_matrix,
    kronecker_power,
    polar_encode,
)


def test_kronecker_power_n2_is_f_itself():
    np.testing.assert_array_equal(kronecker_power(ARIKAN_F, 1), ARIKAN_F)


def test_kronecker_power_n4_hand_computed():
    expected = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 1, 1],
        ]
    )
    np.testing.assert_array_equal(kronecker_power(ARIKAN_F, 2), expected)


def test_bit_reversal_permutation_hand_values():
    np.testing.assert_array_equal(bit_reversal_permutation(2), [0, 1])
    np.testing.assert_array_equal(bit_reversal_permutation(4), [0, 2, 1, 3])
    np.testing.assert_array_equal(
        bit_reversal_permutation(8), [0, 4, 2, 6, 1, 5, 3, 7]
    )


def test_generator_matrix_n2():
    g2 = build_generator_matrix(2)
    np.testing.assert_array_equal(g2, ARIKAN_F)


def test_generator_matrix_n4_hand_computed():
    # G_4 = F^{(x)2}, no bit-reversal (see kernel.py docstring)
    g4 = build_generator_matrix(4)
    expected = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 1, 1],
        ]
    )
    np.testing.assert_array_equal(g4, expected)


def test_encode_all_zero_is_all_zero():
    g = build_generator_matrix(128)
    u = np.zeros(128, dtype=np.int64)
    x = polar_encode(u, g)
    np.testing.assert_array_equal(x, np.zeros(128, dtype=np.int64))


def test_generator_matrix_rejects_non_power_of_two():
    try:
        build_generator_matrix(100)
        assert False, "expected ValueError"
    except ValueError:
        pass
