#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Emit build-time slot metadata for model_ota_context.json export.

The contract hash is the compiler's own value, read back out of the slot's contract probe (see
lib/model_ota/src/model_ota_contract_probe.c). With --config, Axon slots also publish caps from
the generated private axon_config.h; with --keep-json they also publish per-slot binding symbols.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from elf_const import contract_hash_from_probe
from model_contract import config_define

OP_EXTENSION_PREFIX = "nrf_axon_nn_op_extension_"


def axon_binding_symbols(keep_json: Path) -> list[str]:
    payload = json.loads(keep_json.read_text(encoding="utf-8"))
    symbols = payload.get("keep_symbols")
    if not isinstance(symbols, list) or not all(isinstance(entry, str) for entry in symbols):
        raise SystemExit(f"{keep_json}: keep_symbols must be a JSON string array")
    return symbols


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        help="Axon private axon_config.h (optional)")
    parser.add_argument("--keep-json", type=Path,
                        help="per-slot Axon keep-symbol manifest (optional)")
    args = parser.parse_args(argv)

    try:
        contract_hash = contract_hash_from_probe(args.contract_probe)
    except (ImportError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    slot = {"contract_hash": contract_hash}

    if args.config is not None:
        if not args.config.is_file():
            raise SystemExit(f"config header not found: {args.config}")
        slot["persistent_vars_cap"] = (
            config_define(args.config, "MODEL_OTA_AXON_PERSISTENT_VARS_CAP") or 0
        )
        slot["packed_output_cap"] = (
            config_define(args.config, "MODEL_OTA_AXON_PACKED_OUTPUT_BYTES") or 0
        )

    if args.keep_json is not None:
        if not args.keep_json.is_file():
            raise SystemExit(f"keep JSON not found: {args.keep_json}")
        slot["binding_symbols"] = axon_binding_symbols(args.keep_json)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(slot, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
