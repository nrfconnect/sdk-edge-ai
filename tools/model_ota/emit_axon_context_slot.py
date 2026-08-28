#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Emit build-time Axon slot metadata for model_ota_context.json export.

The contract hash is the compiler's own value, read back out of the slot's contract probe (see
lib/model_ota/src/model_ota_contract_probe.c); the flavour of the hash - pure Axon or a wrapped
Edge AI Lab solution - is a property of how that probe was compiled, so nothing here has to know
which one it is. The caps and the binding table come from the generated private configuration
header.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from elf_const import contract_hash_from_probe
from model_contract import config_define

OP_EXTENSION_PREFIX = "nrf_axon_nn_op_extension_"


def axon_binding_symbols(config_header: Path) -> list[str]:
    symbols: list[str] = []
    for line in config_header.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*X\(([^)]+)\)", line)
        if match:
            symbols.append(match.group(1))
    return symbols


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract-probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.config.is_file():
        raise SystemExit(f"config header not found: {args.config}")

    try:
        contract_hash = contract_hash_from_probe(args.contract_probe)
    except (ImportError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    slot = {
        "contract_hash": contract_hash,
        "persistent_vars_cap": config_define(args.config, "MODEL_OTA_AXON_PERSISTENT_VARS_CAP")
        or 0,
        "packed_output_cap": config_define(args.config, "MODEL_OTA_AXON_PACKED_OUTPUT_BYTES") or 0,
        "binding_symbols": axon_binding_symbols(args.config),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(slot, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
