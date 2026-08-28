#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Emit the build-time slot metadata a Neuton model contributes to model_ota_context.json.

Only the contract hash, which is the compiler's own value read back out of the slot's contract
probe (see lib/model_ota/src/model_ota_contract_probe.c). Everything else about a Neuton slot is
known at configure time and comes from model_ota_context_register_slot(). The Axon flavours emit
the same file plus their probe-derived caps and binding table; see emit_axon_context_slot.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from elf_const import contract_hash_from_probe


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        contract_hash = contract_hash_from_probe(args.contract_probe)
    except (ImportError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"contract_hash": contract_hash}, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
