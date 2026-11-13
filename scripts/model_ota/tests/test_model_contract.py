#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

"""Unit tests for model_contract."""

import pytest
from model_contract import symbol_name_hash


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("nrf_axon_interlayer_buffer", 0x8E1329CF),
        ("nrf_axon_nn_op_extension_tanh", 0xE132B0E6),
        ("", 0x46D2E936),
    ],
)
def test_symbol_name_hash_is_pinned(name: str, expected: int) -> None:
    """The hash is on-flash ABI: firmware bakes it, images have to agree."""
    actual = symbol_name_hash(name)

    assert actual == expected


def test_symbol_name_hash_rejects_non_ascii() -> None:
    name = "nrf_axon_\u00e9"

    with pytest.raises(UnicodeEncodeError):
        symbol_name_hash(name)
