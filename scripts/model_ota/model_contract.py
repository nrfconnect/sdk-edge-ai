#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

"""String hashes shared by the model OTA host tools."""

import hashlib


def _hash32(text: str) -> int:
    """32-bit blake2s (digest_size=4) of an ASCII string, read little-endian; on-flash ABI."""
    return int.from_bytes(hashlib.blake2s(text.encode("ascii"), digest_size=4).digest(), "little")


def symbol_name_hash(name: str) -> int:
    """Hash of an ASCII symbol name; MODEL_OTA_AXON_SYM_HASH in generated headers."""
    return _hash32(name)
