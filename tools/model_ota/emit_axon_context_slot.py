#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Emit build-time Axon slot metadata for model_ota_context.json export."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from axon_elf import compiled_model_size_from_probe
from model_contract import axon_contract_from_config, config_define

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
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interlayer-size", type=int, default=0)
    parser.add_argument("--psum-size", type=int, default=0)
    args = parser.parse_args(argv)

    if not args.config.is_file():
        raise SystemExit(f"config header not found: {args.config}")
    if not args.probe.is_file():
        raise SystemExit(f"probe object not found: {args.probe}")

    compiled_size = compiled_model_size_from_probe(args.probe)
    slot = {
        "contract_hash": axon_contract_from_config(
            args.config,
            compiled_model_size=compiled_size,
            interlayer_size=args.interlayer_size,
            psum_size=args.psum_size,
        ),
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
