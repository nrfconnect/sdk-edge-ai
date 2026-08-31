#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""The string hashes the host tools need alongside include/model_ota/model_contract.h.

The contract hash itself is *not* computed here. It folds sizeof() of the runtime structs, which
neither the preprocessor nor a host reimplementation can evaluate, so the build reads the
compiler's own value out of a probe object instead (elf_const.py,
lib/model_ota/src/model_ota_contract_probe.c). What is left are the two things the preprocessor
genuinely cannot do, both of which hash a *string*:

  - solution_id_hash(), passed to the stubs as MODEL_OTA_SOLUTION_ID_HASH and mixed into the
    solution contract there,
  - symbol_name_hash(), the Axon binding table's key.

Run as a script, this prints solution_id_hash() for one solution ID, which is how
model_ota_solution_id_hash() in lib/model_ota/cmake/model_ota_common.cmake obtains the value.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

MODEL_IMAGE_FORMAT_VERSION = 12


def _hash32(text: str) -> int:
    """A 32-bit blake2s over an ASCII string.

    digest_size is a blake2s parameter rather than a truncation of a wider digest, so these are
    four bytes the function was asked for, not four bytes picked out of thirty-two.
    """
    return int.from_bytes(
        hashlib.blake2s(text.encode("ascii"), digest_size=4).digest(), "little"
    )


def solution_id_hash(solution_id: str) -> int:
    """Hash of the solution ID; the value CMake passes as MODEL_OTA_SOLUTION_ID_HASH.

    The ID comes from the SOLUTION_ID the build was configured with rather than from the
    generated source's EDGEAI_LAB_SOLUTION_ID_STR, so it is the same string on both sides of an
    update by construction.
    """
    return _hash32(solution_id)


def symbol_name_hash(name: str) -> int:
    """Hash of an ASCII symbol name; the value behind MODEL_OTA_AXON_SYM_HASH in generated headers.

    Deliberately the same function as solution_id_hash(): the two values are never looked up in
    one table, so there is nothing for a domain separator to protect.
    """
    return _hash32(name)


def config_define(path: Path, name: str) -> int | None:
    # TODO: audit/replace regex scraping of generated C headers (#define values); fragile
    # against line continuations, comments, and duplicate macro names.
    match = re.search(
        rf"^\s*#define\s+{re.escape(name)}\s+(0[xX][0-9A-Fa-f]+|\d+)[uUlL]*\s*$",
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return int(match.group(1), 0) if match else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    solution = sub.add_parser(
        "solution-id-hash",
        help="print the decimal MODEL_OTA_SOLUTION_ID_HASH for one solution ID",
    )
    solution.add_argument("solution_id", help="the SOLUTION_ID the build was configured with")
    args = parser.parse_args(argv)

    try:
        print(solution_id_hash(str(args.solution_id)))
    except UnicodeEncodeError as exc:
        # A solution ID also names C identifiers in the generated source, so it is ASCII by
        # construction; refuse instead of silently picking an encoding for it.
        raise SystemExit(f"solution ID must be ASCII: {args.solution_id!r} ({exc})") from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())
