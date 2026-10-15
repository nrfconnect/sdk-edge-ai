#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Emit model_ota_context.json for a firmware build."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from axon_elf import (
    CONTEXT_BINDINGS_KEY,
    compiled_model_size_from_probe,
    load_binding_symbols,
    read_binding_table,
)
from model_contract import MODEL_IMAGE_FORMAT_VERSION

SLOT_COMMON = (
    "target",
    "name",
    "backend",
    "partition_nodelabel",
    "partition_addr",
    "partition_size",
    "model_image_offset",
    "contract_hash",
)
SLOT_NEUTON = SLOT_COMMON + ("neurons_cap",)
SLOT_AXON = SLOT_COMMON + (
    "persistent_vars_cap",
    "packed_output_cap",
)


def parse_autoconf(path: Path, name: str, default: int = 0) -> int:
    if not path.is_file():
        return default
    import re

    # TODO: audit/replace regex scraping of autoconf.h #defines; use west/kconfig export
    # or another structured Kconfig snapshot instead of parsing the header text.
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"#define\s+{re.escape(name)}\s+(\d+)", text)
    return int(match.group(1)) if match else default


def pick_fields(slot: dict, names: tuple[str, ...]) -> dict:
    return {key: slot[key] for key in names if key in slot}


def load_slot_build(build_dir: Path, target: str) -> dict:
    """Per-slot metadata only known once the build ran, chiefly the compiler's contract hash."""
    path = build_dir / "model_ota" / target / "context_slot.json"
    if not path.is_file():
        raise ValueError("missing build-time slot metadata for %r (%s)" % (target, path))
    return json.loads(path.read_text(encoding="utf-8"))


def compiled_model_size(build_dir: Path, manifest: dict) -> int:
    for slot in manifest.get("slots", []):
        if slot.get("backend") != "axon":
            continue
        probe = build_dir / "model_ota" / slot["target"] / "axon_probe.o"
        if probe.is_file():
            return compiled_model_size_from_probe(probe)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON manifest of slots")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--elf", type=Path, help="zephyr.elf holding the Axon binding table")
    parser.add_argument(
        "--binding-symbols",
        type=Path,
        help="merged Axon binding symbol list from 'axon_elf.py binding-table --symbols-json'",
    )
    parser.add_argument("--autoconf", type=Path, help="zephyr autoconf.h")
    args = parser.parse_args(argv)
    if args.binding_symbols is not None and args.elf is None:
        parser.error("--binding-symbols requires --elf")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    interlayer = parse_autoconf(args.autoconf or Path(), "CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE")
    psum = parse_autoconf(args.autoconf or Path(), "CONFIG_NRF_AXON_PSUM_BUFFER_SIZE")
    compiled_size = compiled_model_size(args.build_dir, manifest)

    context: dict = {
        "format_version": MODEL_IMAGE_FORMAT_VERSION,
        "interlayer_buffer_size": interlayer,
        "psum_buffer_size": psum,
        "compiled_model_size": compiled_size,
    }
    if args.binding_symbols is not None:
        context[CONTEXT_BINDINGS_KEY] = read_binding_table(
            args.elf, load_binding_symbols(args.binding_symbols)
        )
    context["slots"] = []

    for slot in manifest.get("slots", []):
        backend = slot.get("backend", "neuton")
        merged = {**slot, **load_slot_build(args.build_dir, slot["target"])}
        if "contract_hash" not in merged:
            raise ValueError(
                "%s slot %r missing contract_hash" % (backend, slot.get("target"))
            )
        if backend == "neuton":
            entry = pick_fields(merged, SLOT_NEUTON)
        else:
            entry = pick_fields(merged, SLOT_AXON)
        context["slots"].append(entry)

    args.out.write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
    print("Wrote firmware model OTA context -> %s (%u slots)" % (args.out, len(context["slots"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
