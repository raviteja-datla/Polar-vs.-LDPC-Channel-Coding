import numpy as np
import pytest

from channel_coding.codes.polar.construction import (
    _minus_combine_mean,
    _phi,
    _phi_inv,
    _plus_combine_mean,
    channel_means,
    select_frozen_set,
)


def test_phi_bounds():
    assert _phi(0.0) == 1.0
    assert _phi(50.0) < 0.01


def test_phi_inv_round_trip():
    for x in [0.1, 0.5, 1.0, 3.0, 8.0, 15.0, 30.0]:
        y = _phi(x)
        x_recovered = _phi_inv(y)
        assert x_recovered == pytest.approx(x, rel=1e-2, abs=1e-2)


@pytest.mark.parametrize("m0", [0.5, 1.0, 3.0, 10.0])
def test_minus_le_m0_le_plus_structural_invariant(m0):
    m_minus = _minus_combine_mean(m0)
    m_plus = _plus_combine_mean(m0)
    assert m_minus <= m0 <= m_plus


def test_channel_means_length_and_ordering_extremes():
    means = channel_means(8, m0=2.0)
    assert len(means) == 8
    # index 0 = minus-minus-minus (worst); last index = plus-plus-plus (best)
    assert means[0] == means.min()
    assert means[-1] == means.max()


def test_select_frozen_set_partition_is_complete_and_disjoint():
    for n, k in [(8, 4), (128, 64), (1024, 512)]:
        info_idx, frozen_idx = select_frozen_set(n, k, design_ebn0_db=2.0)
        assert len(info_idx) == k
        assert len(frozen_idx) == n - k
        assert len(set(info_idx.tolist()) & set(frozen_idx.tolist())) == 0
        assert set(info_idx.tolist()) | set(frozen_idx.tolist()) == set(range(n))


def test_higher_design_snr_does_not_decrease_reliability():
    means_low = channel_means(16, m0=1.0)
    means_high = channel_means(16, m0=3.0)
    # higher base mean (higher design SNR) should never decrease any channel's mean
    assert np.all(means_high >= means_low - 1e-9)
