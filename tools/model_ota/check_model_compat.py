#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Pre-flash compatibility check for a model partition image against a firmware context.

Exit codes:
  0 — compatible
  1 — incompatible or malformed
  2 — requires firmware update (model exceeds firmware caps)
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

from axon_elf import load_symbol_index, lookup_symbol
from model_contract import MODEL_IMAGE_FORMAT_VERSION, symbol_name_hash
import validate_model_image_layout as layout

PARAMS_AXON = 3
NEUTON_META_NEURONS_NUM_OFFSET = 22
EXIT_OK = 0
EXIT_INCOMPATIBLE = 1
EXIT_NEEDS_FW = 2


def parse_header(bin_path: Path) -> dict:
    data = bin_path.read_bytes()
    if len(data) < layout.HEADER_SIZE:
        raise ValueError("image shorter than header (%u B)" % len(data))
    fields = struct.unpack(layout.HEADER_FMT, data[: layout.HEADER_SIZE])
    (
        magic,
        version,
        params_type,
        _reserved,
        image_size,
        model_version,
        contract_hash,
        crc32,
        name_ptr,
        backend,
    ) = fields
    entry = {
        "magic": magic,
        "format_version": version,
        "params_type": params_type,
        "image_size": image_size,
        "model_version": model_version,
        "contract_hash": contract_hash,
        "crc32": crc32,
        "name_ptr": name_ptr,
    }
    if params_type == PARAMS_AXON:
        model_ptr, packed, persistent, binding_ptr, binding_count = struct.unpack(
            layout.BACKEND_AXON_FMT, backend
        )
        entry.update(
            {
                "model_ptr": model_ptr,
                "axon_packed_output_bytes": packed,
                "persistent_vars_required": persistent,
                "binding_ptr": binding_ptr,
                "binding_count": binding_count,
            }
        )
    else:
        model_ptr, task, p0, p1, p2, decoded = struct.unpack(
            layout.BACKEND_NEUTON_FMT, backend[: struct.calcsize(layout.BACKEND_NEUTON_FMT)]
        )
        entry.update(
            {
                "model_ptr": model_ptr,
                "task": task,
                "decoded_output_ptr": decoded,
            }
        )
    entry["raw"] = data
    return entry


def find_slot(context: dict, slot_name: str | None) -> dict:
    slots = context.get("slots", [])
    if slot_name is not None:
        for slot in slots:
            if slot.get("target") == slot_name or slot.get("name") == slot_name:
                return slot
        raise ValueError(f"slot {slot_name!r} not found in context")
    if len(slots) != 1:
        raise ValueError("--slot required when context has multiple slots")
    return slots[0]


def neuton_neurons_num(hdr: dict, partition_addr: int) -> int | None:
    offset = hdr["model_ptr"] - partition_addr
    end = offset + NEUTON_META_NEURONS_NUM_OFFSET + 2
    if offset < 0 or end > len(hdr["raw"]):
        return None
    return struct.unpack_from("<H", hdr["raw"], offset + NEUTON_META_NEURONS_NUM_OFFSET)[0]


def check_neuton(slot: dict, hdr: dict, image_path: Path) -> int:
    fw_hash = slot.get("contract_hash")
    if fw_hash is None:
        print("context missing contract_hash for Neuton slot", file=sys.stderr)
        return EXIT_INCOMPATIBLE
    if hdr["contract_hash"] != fw_hash:
        print(
            "contract hash mismatch: image 0x%08x != firmware 0x%08x"
            % (hdr["contract_hash"], fw_hash),
            file=sys.stderr,
        )
        return EXIT_INCOMPATIBLE

    partition_size = slot.get("partition_size")
    if partition_size is not None and hdr["image_size"] > partition_size:
        print(
            "image size 0x%x exceeds partition 0x%x" % (hdr["image_size"], partition_size),
            file=sys.stderr,
        )
        return EXIT_INCOMPATIBLE

    neurons_cap = slot.get("neurons_cap")
    if neurons_cap is not None:
        partition_addr = slot.get("partition_addr", 0)
        model_neurons = neuton_neurons_num(hdr, partition_addr)
        if model_neurons is None:
            print("cannot read neuron count from image model meta", file=sys.stderr)
            return EXIT_INCOMPATIBLE
        if model_neurons > neurons_cap:
            print(
                "model needs %u neurons, firmware cap is %u (requires firmware update)"
                % (model_neurons, neurons_cap),
                file=sys.stderr,
            )
            return EXIT_NEEDS_FW

    print("compatible: Neuton image %s matches firmware context" % image_path.name)
    return EXIT_OK


def check_axon(slot: dict, hdr: dict, image_path: Path, elf_path: Path | None) -> int:
    fw_hash = slot.get("contract_hash")
    if fw_hash is None:
        print("context missing contract_hash for Axon slot", file=sys.stderr)
        return EXIT_INCOMPATIBLE
    if hdr["contract_hash"] != fw_hash:
        print(
            "contract hash mismatch: image 0x%08x != firmware 0x%08x"
            % (hdr["contract_hash"], fw_hash),
            file=sys.stderr,
        )
        return EXIT_INCOMPATIBLE

    partition_size = slot.get("partition_size")
    if partition_size is not None and hdr["image_size"] > partition_size:
        print(
            "image size 0x%x exceeds partition 0x%x" % (hdr["image_size"], partition_size),
            file=sys.stderr,
        )
        return EXIT_INCOMPATIBLE

    persistent_cap = slot.get("persistent_vars_cap", 0)
    if hdr["persistent_vars_required"] > persistent_cap:
        print(
            "model needs %u persistent vars, firmware cap is %u (requires firmware update)"
            % (hdr["persistent_vars_required"], persistent_cap),
            file=sys.stderr,
        )
        return EXIT_NEEDS_FW

    packed_cap = slot.get("packed_output_cap", 0)
    if hdr["axon_packed_output_bytes"] > packed_cap:
        print(
            "model needs %u packed-output bytes, firmware cap is %u (requires firmware update)"
            % (hdr["axon_packed_output_bytes"], packed_cap),
            file=sys.stderr,
        )
        return EXIT_NEEDS_FW

    if elf_path is not None and hdr["binding_count"]:
        index = load_symbol_index(elf_path)
        base = slot.get("partition_addr", 0)
        data = hdr["raw"]
        binding_ptr = hdr["binding_ptr"]
        for i in range(hdr["binding_count"]):
            off = binding_ptr - base + i * 8
            if off + 8 > len(data):
                print("binding table extends past image", file=sys.stderr)
                return EXIT_INCOMPATIBLE
            name_hash, address = struct.unpack_from("<II", data, off)
            symbols = slot.get("binding_symbols", [])
            matched = False
            for sym in symbols:
                if symbol_name_hash(sym) != name_hash:
                    continue
                matched = True
                entry = lookup_symbol(elf_path, sym, index)
                if entry is None or entry.is_undefined:
                    print("binding symbol %s missing from firmware elf" % sym, file=sys.stderr)
                    return EXIT_INCOMPATIBLE
                fw_addr = entry.address
                if sym.startswith("nrf_axon_nn_op_extension_"):
                    fw_addr |= 1
                if fw_addr != address:
                    print(
                        "binding mismatch for %s: image 0x%08x != firmware 0x%08x"
                        % (sym, address, fw_addr),
                        file=sys.stderr,
                    )
                    return EXIT_INCOMPATIBLE
                break
            if not matched:
                print("binding hash 0x%08x not in firmware context" % name_hash, file=sys.stderr)
                return EXIT_INCOMPATIBLE

    print("compatible: Axon image %s matches firmware context" % image_path.name)
    return EXIT_OK


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--slot", help="Target/name of the model slot in the context")
    parser.add_argument("--elf", type=Path, help="Firmware zephyr.elf for binding address checks")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print verdict to stderr/stdout but always exit 0 (in-tree build report)",
    )
    args = parser.parse_args(argv)

    if not args.context.is_file():
        parser.error("context not found: %s" % args.context)
    if not args.image.is_file():
        parser.error("image not found: %s" % args.image)

    context = json.loads(args.context.read_text(encoding="utf-8"))
    hdr = parse_header(args.image)
    if hdr["magic"] != layout.MAGIC:
        print("bad magic", file=sys.stderr)
        return EXIT_INCOMPATIBLE

    fw_ver = context.get("format_version", MODEL_IMAGE_FORMAT_VERSION)
    if hdr["format_version"] != fw_ver:
        print(
            "unsupported format version %u (firmware expects %u)"
            % (hdr["format_version"], fw_ver),
            file=sys.stderr,
        )
        return EXIT_INCOMPATIBLE

    slot = find_slot(context, args.slot)
    backend = slot.get("backend", "neuton")
    if backend == "axon" or hdr["params_type"] == PARAMS_AXON:
        rc = check_axon(slot, hdr, args.image, args.elf)
    else:
        rc = check_neuton(slot, hdr, args.image)
    if args.report_only:
        return EXIT_OK
    return rc


if __name__ == "__main__":
    sys.exit(main())
