import numpy as np

from channel_coding.codes.repetition import RepetitionCode


def test_rate():
    code = RepetitionCode(k=10, factor=3)
    assert code.n == 30
    assert code.rate == 1 / 3


def test_encode_repeats_each_bit():
    code = RepetitionCode(k=4, factor=3)
    info = np.array([0, 1, 1, 0])
    codeword = code.encode(info)
    np.testing.assert_array_equal(codeword, [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0])


def test_single_bit_flip_is_corrected():
    """Hand-verification: one wrong copy out of 3 still majority-votes correctly."""
    code = RepetitionCode(k=1, factor=3)
    original = np.array([1, 1, 1])
    one_flip = original.copy()
    one_flip[0] = 0
    decoded = code.decode(one_flip)
    assert decoded[0] == 1


def test_two_bit_flips_break_majority_vote():
    """Hand-verification of the failure mode: 2 of 3 wrong outvotes the truth."""
    code = RepetitionCode(k=1, factor=3)
    original = np.array([1, 1, 1])
    two_flips = original.copy()
    two_flips[[0, 1]] = 0
    decoded = code.decode(two_flips)
    assert decoded[0] == 0  # incorrect: majority vote fooled by 2 flipped copies


def test_even_factor_rejected():
    try:
        RepetitionCode(k=1, factor=4)
        assert False, "expected ValueError for even factor"
    except ValueError:
        pass
