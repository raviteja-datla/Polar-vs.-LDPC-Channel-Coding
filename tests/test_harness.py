import numpy as np
import pytest
from scipy.special import erfc

from channel_coding.codes.uncoded import UncodedCode
from channel_coding.harness.channel import (
    add_awgn,
    bits_to_bpsk,
    channel_llr,
    demod_hard,
    eb_n0_db_to_noise_std,
)
from channel_coding.harness.simulate import monte_carlo_ber


def test_bpsk_mapping():
    bits = np.array([0, 1, 0, 1])
    symbols = bits_to_bpsk(bits)
    np.testing.assert_array_equal(symbols, [1.0, -1.0, 1.0, -1.0])


def test_eb_n0_db_to_noise_std_hand_values():
    # rate=1, 0 dB -> Eb/N0 linear = 1 -> sigma = sqrt(1/2)
    sigma = eb_n0_db_to_noise_std(0.0, rate=1.0)
    assert np.isclose(sigma, np.sqrt(0.5))

    # halving the rate doubles Es/N0's denominator scale -> sigma grows
    sigma_full_rate = eb_n0_db_to_noise_std(4.0, rate=1.0)
    sigma_half_rate = eb_n0_db_to_noise_std(4.0, rate=0.5)
    assert sigma_half_rate > sigma_full_rate
    assert np.isclose(sigma_half_rate, sigma_full_rate * np.sqrt(2.0))


def test_add_awgn_zero_sigma_is_noop():
    symbols = np.array([1.0, -1.0, 1.0])
    out = add_awgn(symbols, 0.0, np.random.default_rng(0))
    np.testing.assert_array_equal(out, symbols)


def test_llr_sign_matches_hard_decision():
    rng = np.random.default_rng(0)
    received = rng.normal(size=1000)
    llr = channel_llr(received, noise_std=1.0)
    hard = demod_hard(received)
    # positive LLR -> bit 0, negative LLR -> bit 1
    expected_hard = (llr < 0).astype(np.int64)
    np.testing.assert_array_equal(hard, expected_hard)


def test_uncoded_ber_matches_theory():
    rng = np.random.default_rng(42)
    code = UncodedCode(k=1)
    ebn0_db = 4.0
    [point] = monte_carlo_ber(code, [ebn0_db], rng, min_errors=300, max_blocks=2_000_000)

    ebn0_linear = 10 ** (ebn0_db / 10)
    theoretical_ber = 0.5 * erfc(np.sqrt(ebn0_linear))

    assert point.ber == pytest.approx(theoretical_ber, rel=0.15)
