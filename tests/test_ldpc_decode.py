import numpy as np

from channel_coding.codes.ldpc.code import LDPCCode
from channel_coding.harness.channel import add_awgn, bits_to_bpsk, channel_llr


def test_zero_noise_round_trip_converges():
    rng = np.random.default_rng(11)
    code = LDPCCode(n=64, j=3, k_row_weight=4, rng=rng)
    for _ in range(30):
        info_bits = rng.integers(0, 2, size=code.k)
        codeword = code.encode(info_bits)
        symbols = bits_to_bpsk(codeword)
        received = add_awgn(symbols, noise_std=1e-6, rng=rng)
        llr = channel_llr(received, noise_std=1e-6)
        decoded = code.decode(llr)
        np.testing.assert_array_equal(decoded, info_bits)


def test_converged_implies_zero_syndrome():
    from channel_coding.codes.ldpc.decode import min_sum_decode

    rng = np.random.default_rng(13)
    code = LDPCCode(n=64, j=3, k_row_weight=4, rng=rng)
    for _ in range(20):
        info_bits = rng.integers(0, 2, size=code.k)
        codeword = code.encode(info_bits)
        symbols = bits_to_bpsk(codeword)
        received = add_awgn(symbols, noise_std=0.6, rng=rng)
        llr = channel_llr(received, noise_std=0.6)
        x_hat, converged, _iters = min_sum_decode(llr, code.h, max_iters=code.max_iters)
        if converged:
            syndrome = (code.h @ x_hat) % 2
            np.testing.assert_array_equal(syndrome, np.zeros_like(syndrome))


def test_ber_improves_with_snr():
    from channel_coding.harness.simulate import monte_carlo_ber

    rng = np.random.default_rng(17)
    code = LDPCCode(n=64, j=3, k_row_weight=4, rng=rng)
    [low_snr] = monte_carlo_ber(code, [0.0], rng, min_errors=20, max_blocks=2000)
    [high_snr] = monte_carlo_ber(code, [5.0], rng, min_errors=20, max_blocks=2000)
    assert high_snr.ber <= low_snr.ber
