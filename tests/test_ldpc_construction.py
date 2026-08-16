import numpy as np
import pytest

from channel_coding.codes.ldpc.construction import build_regular_ldpc_h


def test_row_and_column_weights_exact(rng):
    n, j, k = 32, 3, 4
    h = build_regular_ldpc_h(n, j, k, rng)
    assert np.all(h.sum(axis=0) == j)  # column weight
    assert np.all(h.sum(axis=1) == k)  # row weight


def test_no_all_zero_rows_or_columns(rng):
    n, j, k = 32, 3, 4
    h = build_regular_ldpc_h(n, j, k, rng)
    assert np.all(h.sum(axis=0) > 0)
    assert np.all(h.sum(axis=1) > 0)


def test_no_duplicate_rows(rng):
    n, j, k = 32, 3, 4
    h = build_regular_ldpc_h(n, j, k, rng)
    rows = [tuple(row) for row in h]
    assert len(rows) == len(set(rows))


def test_dimension_mismatch_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        build_regular_ldpc_h(n=10, j=3, k=4, rng=rng)  # 10*3=30 not divisible by 4
