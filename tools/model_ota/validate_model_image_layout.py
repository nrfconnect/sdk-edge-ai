#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Validate the layout of a linked Neuton model partition image.

Adapted from the Axon validate_model_partition_layout.py. Confirms, after linking:

  - the image was linked at the partition base (__model_image_start == partition addr),
  - the header sits first, at the base,
  - header magic / format_version are correct,
  - header.image_size equals the linker extent (__model_image_end - __model_image_start) and the
    binary size,
  - the header's DIRECT model pointer equals &model_instance_ and lies inside the image, and
  - the crc32 field is non-zero (i.e. patch_image_crc.py ran).

Unlike the Axon validator there is no model_offset arithmetic: the header stores an absolute
flash pointer, which we compare directly against the model symbol address.
"""
import argparse
import re
import struct
import subprocess
import sys
from pathlib import Path

MAGIC = b"NEI5"
PARAMS_AXON = 3

# struct model_image_header (see include/model_ota/model_image.h), little-endian, __packed:
#   magic[4] version:H params_type:B task:B image_size:I crc32:I model:I decoded_output:I
#   name[16] model_version:I
HEADER_FMT = "<4sHBBIIII16sI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def lookup_symbol(nm, elf, symbol):
    out = subprocess.check_output([nm, str(elf)], text=True, errors="replace")
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1] == symbol:
            return int(parts[0], 16)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nm", required=True)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--partition-addr", type=lambda x: int(x, 0), required=True)
    parser.add_argument("--model-symbol", default="model_instance_",
                        help="Expected baked model symbol (Axon: e.g. model_person_det)")
    parser.add_argument("--header-symbol", default="nrf_edgeai_model_image_hdr")
    parser.add_argument("--defs-header", type=Path, default=None,
                        help="model_image.h, to read the expected MODEL_IMAGE_FORMAT_VERSION")
    parser.add_argument("--version", type=int, default=None)
    args = parser.parse_args()

    if not args.elf.is_file():
        print("ELF not found: %s" % args.elf, file=sys.stderr)
        sys.exit(1)
    if not args.bin.is_file():
        print("binary not found: %s" % args.bin, file=sys.stderr)
        sys.exit(1)

    expected_version = args.version
    if expected_version is None and args.defs_header is not None:
        m = re.search(r"#define\s+MODEL_IMAGE_FORMAT_VERSION\s+(\d+)",
                      args.defs_header.read_text(encoding="utf-8"))
        if m is not None:
            expected_version = int(m.group(1))
    if expected_version is None:
        print("expected format version not provided", file=sys.stderr)
        sys.exit(1)

    start = lookup_symbol(args.nm, args.elf, "__model_image_start")
    end = lookup_symbol(args.nm, args.elf, "__model_image_end")
    hdr_sym = lookup_symbol(args.nm, args.elf, args.header_symbol)
    if hdr_sym is None:
        hdr_sym = lookup_symbol(args.nm, args.elf, "model_image_hdr")
    model_sym = lookup_symbol(args.nm, args.elf, args.model_symbol)

    if start is None or end is None:
        print("missing linker anchors __model_image_start/__model_image_end", file=sys.stderr)
        sys.exit(1)

    errors = []

    if start != args.partition_addr:
        errors.append("partition base mismatch: linker start 0x%x != 0x%x"
                      % (start, args.partition_addr))
    if hdr_sym is not None and hdr_sym != start:
        errors.append("header not at image start: hdr 0x%x, start 0x%x" % (hdr_sym, start))

    linker_size = end - start
    if linker_size <= HEADER_SIZE:
        errors.append("image too small: %d bytes" % linker_size)

    header_bytes = args.bin.read_bytes()[:HEADER_SIZE]
    if len(header_bytes) != HEADER_SIZE:
        print("image binary shorter than header", file=sys.stderr)
        sys.exit(1)

    (magic, version, params_type, task, image_size, crc32, model_ptr, decoded_output_ptr,
     name, model_version) = struct.unpack(HEADER_FMT, header_bytes)

    if magic != MAGIC:
        errors.append("magic %r != %r" % (magic, MAGIC))
    if version != expected_version:
        errors.append("format_version %d != expected %d" % (version, expected_version))
    if image_size != linker_size:
        errors.append("header image_size 0x%x != linker extent 0x%x" % (image_size, linker_size))

    bin_size = args.bin.stat().st_size
    if bin_size != linker_size:
        errors.append("binary size 0x%x != linker extent 0x%x" % (bin_size, linker_size))

    if model_ptr < start or model_ptr + 1 > end:
        errors.append("model pointer 0x%x outside image [0x%x, 0x%x)" % (model_ptr, start, end))
    if model_sym is not None and model_ptr != model_sym:
        errors.append("header model 0x%x != &%s 0x%x" % (model_ptr, args.model_symbol, model_sym))

    if params_type == PARAMS_AXON:
        if decoded_output_ptr != 0:
            errors.append("Axon image decoded_output must be NULL, got 0x%x" % decoded_output_ptr)
    elif decoded_output_ptr < start or decoded_output_ptr >= end:
        errors.append("decoded_output pointer 0x%x outside image [0x%x, 0x%x)"
                      % (decoded_output_ptr, start, end))

    if crc32 == 0:
        errors.append("crc32 is 0 (patch_image_crc.py did not run)")

    if errors:
        for e in errors:
            print("layout validation failed: %s" % e, file=sys.stderr)
        sys.exit(1)

    name_str = name.split(b"\x00", 1)[0].decode("ascii", "replace")
    print("model image layout ok: base 0x%x, size 0x%x, model 0x%x (&%s), "
          "params_type %d, task %d, crc32 0x%08x, name '%s' v0x%08x"
          % (start, image_size, model_ptr, args.model_symbol, params_type, task, crc32,
             name_str, model_version))
    return 0


if __name__ == "__main__":
    sys.exit(main())
