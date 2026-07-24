#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Validate the layout of a linked Neuton or Axon model partition image.

Confirms, after linking:

  - the image was linked at the partition base (__model_image_start == partition addr),
  - the header sits first, at the base,
  - header magic / format_version are correct,
  - header.image_size equals the linker extent (__model_image_end - __model_image_start) and the
    binary size, and fits within the partition,
  - the header's DIRECT model pointer equals &<model symbol> and lies inside the image, and
  - the crc32 field is non-zero and matches a recomputed CRC (i.e. patch_image_crc.py ran).

There is no model_offset arithmetic: the header stores an absolute flash pointer, which is
compared directly against the model symbol's address (default `model_instance_` for Neuton;
pass `--model-symbol` for Axon, normally read from the generated config header instead).
"""
import argparse
import re
import struct
import sys
import zlib
from pathlib import Path

from axon_elf import lookup_symbol

MAGIC = b"NEI\x00"
PARAMS_AXON = 3

# struct model_image_header (see include/model_ota/model_image.h), little-endian, __packed:
#   magic[4] version:H params_type:B task:B image_size:I crc32:I model:I decoded_output:I
#   name[16] model_version:I axon_packed_output_bytes:I
HEADER_FMT = "<4sHBBIIII16sII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def symbol(elf, name):
    entry = lookup_symbol(elf, name)
    if entry is None or entry.is_undefined:
        return None
    return entry


def config_define(path, name):
    if path is None:
        return None
    match = re.search(
        rf"^\s*#define\s+{re.escape(name)}\s+([A-Za-z_]\w*|0[xX][0-9A-Fa-f]+|\d+)[uUlL]*\s*$",
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return match.group(1) if match is not None else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--partition-addr", type=lambda x: int(x, 0), required=True)
    parser.add_argument("--partition-size", type=lambda x: int(x, 0))
    parser.add_argument("--params-type", type=lambda x: int(x, 0))
    parser.add_argument("--model-symbol", default="model_instance_",
                        help="Expected baked model symbol (Axon: e.g. model_person_det)")
    parser.add_argument("--header-symbol", default="nrf_edgeai_model_image_hdr")
    parser.add_argument("--config-header", type=Path,
                        help="Generated Axon private configuration header")
    parser.add_argument("--defs-header", type=Path, default=None,
                        help="model_image.h, to read the expected MODEL_IMAGE_FORMAT_VERSION")
    parser.add_argument("--version", type=int, default=None)
    args = parser.parse_args(argv)

    configured_model = config_define(args.config_header, "MODEL_OTA_AXON_MODEL_SYM")
    if configured_model is not None:
        args.model_symbol = configured_model
    configured_packed = config_define(
        args.config_header, "MODEL_OTA_AXON_PACKED_OUTPUT_BYTES"
    )

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

    start_sym = symbol(args.elf, "__model_image_start")
    end_sym = symbol(args.elf, "__model_image_end")
    hdr_sym = symbol(args.elf, args.header_symbol)
    if hdr_sym is None:
        hdr_sym = symbol(args.elf, "model_image_hdr")
    model_sym = symbol(args.elf, args.model_symbol)

    if start_sym is None or end_sym is None:
        print("missing linker anchors __model_image_start/__model_image_end", file=sys.stderr)
        sys.exit(1)

    start = start_sym.address
    end = end_sym.address
    errors = []

    if start != args.partition_addr:
        errors.append("partition base mismatch: linker start 0x%x != 0x%x"
                      % (start, args.partition_addr))
    if hdr_sym is None:
        errors.append("missing model image header symbol")
    elif hdr_sym.address != start:
        errors.append(
            "header not at image start: hdr 0x%x, start 0x%x" % (hdr_sym.address, start)
        )

    linker_size = end - start
    if linker_size <= HEADER_SIZE:
        errors.append("image too small: %d bytes" % linker_size)

    header_bytes = args.bin.read_bytes()[:HEADER_SIZE]
    if len(header_bytes) != HEADER_SIZE:
        print("image binary shorter than header", file=sys.stderr)
        sys.exit(1)

    (magic, version, params_type, task, image_size, crc32, model_ptr, decoded_output_ptr,
     name, model_version, axon_packed_output_bytes) = struct.unpack(HEADER_FMT, header_bytes)

    if magic != MAGIC:
        errors.append("magic %r != %r" % (magic, MAGIC))
    if version != expected_version:
        errors.append("format_version %d != expected %d" % (version, expected_version))
    if image_size != linker_size:
        errors.append("header image_size 0x%x != linker extent 0x%x" % (image_size, linker_size))

    bin_size = args.bin.stat().st_size
    if bin_size != linker_size:
        errors.append("binary size 0x%x != linker extent 0x%x" % (bin_size, linker_size))
    if args.partition_size is not None and image_size > args.partition_size:
        errors.append(
            "image size 0x%x exceeds partition size 0x%x"
            % (image_size, args.partition_size)
        )

    model_extent = model_sym.size if model_sym is not None and model_sym.size > 0 else 1
    if model_ptr < start or model_ptr + model_extent > end:
        errors.append("model pointer 0x%x outside image [0x%x, 0x%x)" % (model_ptr, start, end))
    if model_sym is None:
        errors.append("missing model symbol %s" % args.model_symbol)
    elif model_ptr != model_sym.address:
        errors.append(
            "header model 0x%x != &%s 0x%x"
            % (model_ptr, args.model_symbol, model_sym.address)
        )

    if args.params_type is not None and params_type != args.params_type:
        errors.append(
            "params_type %d != expected %d" % (params_type, args.params_type)
        )

    if params_type == PARAMS_AXON:
        if decoded_output_ptr != 0:
            errors.append("Axon image decoded_output must be NULL, got 0x%x" % decoded_output_ptr)
    elif decoded_output_ptr < start or decoded_output_ptr >= end:
        errors.append("decoded_output pointer 0x%x outside image [0x%x, 0x%x)"
                      % (decoded_output_ptr, start, end))

    if crc32 == 0:
        errors.append("crc32 is 0 (patch_image_crc.py did not run)")
    else:
        crc_data = bytearray(args.bin.read_bytes())
        struct.pack_into("<I", crc_data, 12, 0)
        computed_crc = zlib.crc32(crc_data) & 0xFFFFFFFF
        if crc32 != computed_crc:
            errors.append(
                "crc32 0x%08x != computed 0x%08x" % (crc32, computed_crc)
            )

    if configured_packed is not None:
        expected_packed = int(configured_packed, 0)
        if axon_packed_output_bytes != expected_packed:
            errors.append(
                "Axon packed output %d != expected %d"
                % (axon_packed_output_bytes, expected_packed)
            )

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
