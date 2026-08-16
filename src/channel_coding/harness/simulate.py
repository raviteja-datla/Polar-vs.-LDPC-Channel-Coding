"""Monte Carlo BER measurement shared by every coding scheme."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from channel_coding.codes.base import Code
from channel_coding.harness.channel import (
    add_awgn,
    bits_to_bpsk,
    channel_llr,
    demod_hard,
    eb_n0_db_to_noise_std,
)


@dataclass
class BerPoint:
    ebn0_db: float
    ber: float
    bit_errors: int
    blocks_simulated: int
    total_info_bits: int


def simulate_one_point(
    code: Code,
    ebn0_db: float,
    rng: np.random.Generator,
    min_errors: int = 100,
    max_blocks: int = 200_000,
    min_blocks: int = 30,
) -> BerPoint:
    """Transmit random blocks through `code` over AWGN at the given Eb/N0
    until `min_errors` bit errors accumulate or `max_blocks` is reached.

    `min_blocks` is a statistical-confidence floor independent of
    `min_errors`: for a large block size (e.g. N=1024), a single unlucky
    block can already exceed `min_errors`, which would otherwise report a
    BER estimated from just one Monte Carlo trial.
    """
    noise_std = eb_n0_db_to_noise_std(ebn0_db, code.rate)

    bit_errors = 0
    blocks = 0
    total_info_bits = 0

    while (bit_errors < min_errors or blocks < min_blocks) and blocks < max_blocks:
        info_bits = rng.integers(0, 2, size=code.k)
        codeword = code.encode(info_bits)
        symbols = bits_to_bpsk(codeword)
        received = add_awgn(symbols, noise_std, rng)

        if code.soft_decision:
            decoder_input = channel_llr(received, noise_std)
        else:
            decoder_input = demod_hard(received)

        decoded = code.decode(decoder_input)

        bit_errors += int(np.sum(decoded != info_bits))
        blocks += 1
        total_info_bits += code.k

    ber = bit_errors / total_info_bits if total_info_bits else 0.0
    return BerPoint(
        ebn0_db=ebn0_db,
        ber=ber,
        bit_errors=bit_errors,
        blocks_simulated=blocks,
        total_info_bits=total_info_bits,
    )


def monte_carlo_ber(
    code: Code,
    ebn0_db_list,
    rng: np.random.Generator,
    min_errors: int = 100,
    max_blocks: int = 200_000,
    min_blocks: int = 30,
) -> list[BerPoint]:
    return [
        simulate_one_point(
            code, ebn0_db, rng, min_errors=min_errors, max_blocks=max_blocks, min_blocks=min_blocks
        )
        for ebn0_db in ebn0_db_list
    ]
