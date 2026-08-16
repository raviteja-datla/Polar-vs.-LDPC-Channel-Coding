import numpy as np

from channel_coding.codes.polar.code import PolarCode


def _noiseless_llr(codeword: np.ndarray, confidence: float = 20.0) -> np.ndarray:
    """Very high-confidence LLRs matching a noiseless channel: bit 0 -> +conf,
    bit 1 -> -conf."""
    return confidence * (1.0 - 2.0 * codeword.astype(np.float64))


def test_zero_noise_round_trip_many_trials():
    rng = np.random.default_rng(7)
    for n, k in [(8, 4), (16, 8), (128, 64)]:
        code = PolarCode(n=n, k=k, design_ebn0_db=2.0)
        for _ in range(200):
            info_bits = rng.integers(0, 2, size=k)
            codeword = code.encode(info_bits)
            llr = _noiseless_llr(codeword)
            decoded = code.decode(llr)
            np.testing.assert_array_equal(decoded, info_bits)


def test_all_zero_round_trip():
    code = PolarCode(n=128, k=64, design_ebn0_db=2.0)
    info_bits = np.zeros(64, dtype=np.int64)
    codeword = code.encode(info_bits)
    llr = _noiseless_llr(codeword)
    decoded = code.decode(llr)
    np.testing.assert_array_equal(decoded, info_bits)


def test_frozen_positions_never_decode_to_one_under_noiseless_all_zero():
    from channel_coding.codes.polar.decode import sc_decode

    code = PolarCode(n=128, k=64, design_ebn0_db=2.0)
    info_bits = np.zeros(64, dtype=np.int64)
    codeword = code.encode(info_bits)
    llr = _noiseless_llr(codeword)
    u_hat = sc_decode(llr, code.frozen_indices)
    assert np.all(u_hat[code.frozen_indices] == 0)


def test_rate():
    code = PolarCode(n=128, k=64)
    assert code.rate == 0.5
