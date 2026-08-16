"""Iterative message-passing (log-domain Min-Sum) decoding for LDPC codes.

Min-Sum is used instead of full sum-product/tanh-rule belief propagation:
it's numerically simpler (no tanh/atanh over LLR magnitudes that can
overflow at high confidence) and a standard, robust approximation in
practice, at some cost in optimality that full sum-product could recover.

Row weight stays exactly uniform (k) for the H actually used at decode time,
since generator.py only ever drops whole redundant rows. Column weight is
NOT assumed uniform: dropping rows to reach full row rank (see generator.py)
leaves some columns with fewer checks than others, so the variable side of
the Tanner graph is padded to the maximum column degree with a mask that
excludes padding slots from every sum — this keeps both the check-node and
variable-node updates fully vectorized without assuming perfect regularity.
"""

from __future__ import annotations

import numpy as np


def build_adjacency(h: np.ndarray):
    m, n = h.shape

    row_degrees = h.sum(axis=1)
    if not np.all(row_degrees == row_degrees[0]):
        raise ValueError("expected uniform row weight in H (generator.py only drops whole rows)")
    k = int(row_degrees[0])

    check_to_vars = np.array([np.nonzero(h[c])[0] for c in range(m)], dtype=np.int64)

    col_degrees = h.sum(axis=0).astype(np.int64)
    max_deg = int(col_degrees.max())
    var_to_checks = np.zeros((n, max_deg), dtype=np.int64)
    var_mask = np.zeros((n, max_deg), dtype=bool)
    var_pos_in_check = np.zeros((n, max_deg), dtype=np.int64)
    check_pos_in_var = np.zeros((m, k), dtype=np.int64)

    for v in range(n):
        checks = np.nonzero(h[:, v])[0]
        deg = len(checks)
        var_to_checks[v, :deg] = checks
        var_mask[v, :deg] = True
        for slot, c in enumerate(checks):
            i = int(np.nonzero(check_to_vars[c] == v)[0][0])
            var_pos_in_check[v, slot] = i
            check_pos_in_var[c, i] = slot

    return check_to_vars, var_to_checks, var_pos_in_check, check_pos_in_var, var_mask


def _check_update_min_sum(q_compact: np.ndarray) -> np.ndarray:
    """Vectorized min-sum check-node update over a dense (m,k) message array."""
    signs = np.sign(q_compact)
    signs[signs == 0] = 1.0
    abs_vals = np.abs(q_compact)

    order = np.argsort(abs_vals, axis=1)
    min1 = np.take_along_axis(abs_vals, order[:, :1], axis=1)
    min2 = np.take_along_axis(abs_vals, order[:, 1:2], axis=1)
    min1_pos = order[:, 0]

    m = q_compact.shape[0]
    mag = np.broadcast_to(min1, q_compact.shape).copy()
    mag[np.arange(m), min1_pos] = min2[:, 0]

    total_sign = np.prod(signs, axis=1, keepdims=True)
    sign_per_pos = total_sign * signs
    return sign_per_pos * mag


def min_sum_decode(
    channel_llr: np.ndarray,
    h: np.ndarray,
    max_iters: int = 30,
    adjacency=None,
):
    if adjacency is None:
        adjacency = build_adjacency(h)
    check_to_vars, var_to_checks, var_pos_in_check, check_pos_in_var, var_mask = adjacency

    q_compact = channel_llr[check_to_vars]

    x_hat = (channel_llr < 0).astype(np.int64)
    converged = False
    iterations_used = 0

    for iteration in range(max_iters):
        r_compact = _check_update_min_sum(q_compact)

        r_at_var = r_compact[var_to_checks, var_pos_in_check] * var_mask
        total_llr = channel_llr + r_at_var.sum(axis=1)
        q_at_var = total_llr[:, None] - r_at_var
        q_compact = q_at_var[check_to_vars, check_pos_in_var]

        x_hat = (total_llr < 0).astype(np.int64)
        iterations_used = iteration + 1
        if np.all((h @ x_hat) % 2 == 0):
            converged = True
            break

    return x_hat, converged, iterations_used
