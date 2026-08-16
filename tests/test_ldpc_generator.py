import numpy as np
import pytest

from channel_coding.codes.ldpc.construction import build_regular_ldpc_h
from channel_coding.codes.ldpc.generator import derive_generator_matrix


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_g_h_orthogonal_across_seeds(seed):
    rng = np.random.default_rng(seed)
    h = build_regular_ldpc_h(n=32, j=3, k=4, rng=rng)
    g, h_std = derive_generator_matrix(h)
    check = (g @ h_std.T) % 2
    np.testing.assert_array_equal(check, np.zeros_like(check))


def test_g_systematic_and_h_std_stays_sparse():
    # G is systematic ([I_k | P^T]); H_std keeps H's original row weight
    # (column-permuted only, never row-XORed) so it stays usable for BP.
    rng = np.random.default_rng(5)
    h = build_regular_ldpc_h(n=32, j=3, k=4, rng=rng)
    g, h_std = derive_generator_matrix(h)
    k = g.shape[0]
    np.testing.assert_array_equal(g[:, :k], np.eye(k, dtype=np.int64))
    assert np.all(h_std.sum(axis=1) == 4)  # row weight k=4 preserved, no row-XOR fill-in


def test_random_codewords_satisfy_all_parity_checks():
    rng = np.random.default_rng(7)
    h = build_regular_ldpc_h(n=32, j=3, k=4, rng=rng)
    g, h_std = derive_generator_matrix(h)
    k = g.shape[0]
    for _ in range(50):
        info = rng.integers(0, 2, size=k)
        codeword = (info @ g) % 2
        syndrome = (h_std @ codeword) % 2
        np.testing.assert_array_equal(syndrome, np.zeros_like(syndrome))
