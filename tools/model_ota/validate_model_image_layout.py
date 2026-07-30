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
    binary size, and fits within the partition payload area (partition_size - model_image_offset),
  - the header's DIRECT model pointer equals &<model symbol> and lies inside the image,
  - the feature-scaling block is self-consistent and its pointers lie inside the image, and
  - the crc32 field is non-zero and matches a recomputed CRC (i.e. patch_image_crc.py ran).

The header's contract_hash is reported but not checked here: it is only meaningful against a
firmware context, which check_model_compat.py has and this tool does not.

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
#   magic[4] version:H params_type:B reserved:B image_size:I model_version:I
#   contract_hash:I crc32:I name:I backend[20] edgeai_params[36]
# backend Neuton (first 4 B): model:I, rest of the union slot zeroed
# backend Axon (20 B): model:I packed_output:I persistent_required:I binding:I binding_count:I
HEADER_FMT = "<4sHBBIIIII20s36s"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
BACKEND_NEUTON_FMT = "<I"
BACKEND_AXON_FMT = "<IIIII"

# struct model_image_edgeai_params (zeroed when the image carries no parameters):
#   scale union (12 B): p_min:I p_max:I p_arguments:I (the third word is DSP-features only)
#   decoded_output (16 B): nrf_edgeai_decoded_output_t, laid out per task
#   p_extraction_mask:I (DSP solutions only; verified against the app's at load time)
#   scale_num:H scale_elem_size:B reserved:B
PARAMS_FMT = "<IIIIIIIIHBB"
PARAMS_SIZE = struct.calcsize(PARAMS_FMT)
CRC32_OFFSET = 20


def params_scale_layout(params_bytes):
    """(scale_num, scale_elem_size) out of an edgeai_params block; both 0 when it carries none.

    Lives here so that PARAMS_FMT and the field offsets into it stay in one place.
    """
    fields = struct.unpack(PARAMS_FMT, params_bytes)
    return fields[8], fields[9]


def name_in_image(name_ptr, image_bytes, start, end):
    if name_ptr < start or name_ptr >= end:
        return None
    off = name_ptr - start
    if off < 0 or off >= len(image_bytes):
        return None
    chars = bytearray()
    for i in range(off, len(image_bytes)):
        byte = image_bytes[i]
        if byte == 0:
            return bytes(chars).decode("ascii", "replace")
        chars.append(byte)
    return None


def validate_params(params_bytes, start, end, errors):
    """Check the header's nrf_edgeai_t parameter block.

    Returns (scale_num, scale_elem_size), both 0 when the image carries no parameters (a pure
    Axon model, or a solution whose parameters the application keeps compiled in).
    """
    fields = struct.unpack(PARAMS_FMT, params_bytes)
    p_min, p_max, p_args = fields[0:3]
    decode_words = fields[3:7]
    p_mask = fields[7]
    num, elem_size = params_scale_layout(params_bytes)
    reserved = fields[10]

    def check_ptr(label, ptr, span):
        if ptr < start or ptr + span > end:
            errors.append("edgeai_params %s 0x%x + %u B outside image [0x%x, 0x%x)"
                          % (label, ptr, span, start, end))

    if reserved != 0:
        errors.append("edgeai_params reserved byte %d != 0" % reserved)

    if num == 0:
        # No parameters carried, so the whole block must be empty.
        if any(field != 0 for field in fields):
            errors.append("edgeai_params scale_num 0 but block is not empty: %r" % (fields,))
        return 0, 0

    if elem_size == 0:
        errors.append("edgeai_params scale_num %d with elem_size 0" % num)
    else:
        span = num * elem_size
        check_ptr("scale p_min", p_min, span)
        check_ptr("scale p_max", p_max, span)
        if p_args != 0:
            # Pipeline-implicit length, so only the base is bounded.
            check_ptr("scale p_arguments", p_args, 1)

    if p_mask != 0:
        # Length is INPUT_UNIQ_FEATURES_NUM, which the header does not record (it is folded into
        # the contract hash), so only the base is bounded here; the loader bounds the full span
        # against the count it gets from the application's pipeline.
        check_ptr("p_extraction_mask", p_mask, 8)

    # Which words of the baked nrf_edgeai_decoded_output_t are pointers depends on the task, which
    # the image does not record as a field (it is folded into the contract hash). Check the
    # property that does not need it: whatever the union member, a decode word is either runtime
    # state left at its initial value or a pointer into this image, never a pointer out of it.
    # Word 0 is exempt because it holds a scalar for every task (score / outputs_num /
    # predicted class + num_classes), and those are not all zero.
    for index, word in enumerate(decode_words):
        if index != 0 and word != 0:
            check_ptr("decoded_output word %d" % index, word, 1)

    return num, elem_size


def symbol(elf, name):
    entry = lookup_symbol(elf, name)
    if entry is None or entry.is_undefined:
        return None
    return entry


def config_define(path, name):
    if path is None:
        return None
    # TODO: audit/replace regex scraping of generated C headers; share one implementation
    # with model_contract.config_define() or read values from structured build metadata.
    match = re.search(
        rf"^\s*#define\s+{re.escape(name)}\s+([A-Za-z_]\w*|0[xX][0-9A-Fa-f]+|\d+)[uUlL]*\s*$",
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return match.group(1) if match is not None else None


def max_payload_size(partition_size: int | None, model_image_offset: int) -> int | None:
    if partition_size is None:
        return None
    return partition_size - model_image_offset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--image-link-addr", type=lambda x: int(x, 0), required=True,
                        help="Flash address where the model image is linked")
    parser.add_argument("--partition-size", type=lambda x: int(x, 0))
    parser.add_argument("--model-image-offset", type=lambda x: int(x, 0), default=32,
                        help="Bytes from partition base to the linked model image")
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
    configured_persistent = config_define(
        args.config_header, "MODEL_OTA_AXON_PERSISTENT_VARS_REQUIRED"
    )

    if not args.elf.is_file():
        print("ELF not found: %s" % args.elf, file=sys.stderr)
        sys.exit(1)
    if not args.bin.is_file():
        print("binary not found: %s" % args.bin, file=sys.stderr)
        sys.exit(1)

    expected_version = args.version
    if args.defs_header is not None:
        text = args.defs_header.read_text(encoding="utf-8")
        if expected_version is None:
            # TODO: audit/replace regex scraping of model_image.h; import
            # MODEL_IMAGE_FORMAT_VERSION from model_contract instead.
            m = re.search(r"#define\s+MODEL_IMAGE_FORMAT_VERSION\s+(\d+)", text)
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

    if start != args.image_link_addr:
        errors.append("image link addr mismatch: linker start 0x%x != 0x%x"
                      % (start, args.image_link_addr))
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

    image_bytes = args.bin.read_bytes()

    (magic, version, params_type, _reserved, image_size, model_version, contract_hash, crc32,
     name_ptr, backend_bytes, params_bytes) = struct.unpack(HEADER_FMT, header_bytes)

    if magic != MAGIC:
        errors.append("magic %r != %r" % (magic, MAGIC))
    if version != expected_version:
        errors.append("format_version %d != expected %d"
                      % (version, expected_version))
    if _reserved != 0:
        errors.append("reserved byte %d != 0" % _reserved)
    if image_size != linker_size:
        errors.append("header image_size 0x%x != linker extent 0x%x" % (image_size, linker_size))

    name_str = name_in_image(name_ptr, image_bytes, start, end)
    if name_str is None:
        errors.append("name pointer 0x%x outside image or not NUL-terminated before end"
                      % name_ptr)

    if params_type == PARAMS_AXON:
        (model_ptr, axon_packed_output_bytes, persistent_required, binding_ptr,
         binding_count) = struct.unpack(BACKEND_AXON_FMT, backend_bytes)
        if binding_count > 0:
            binding_bytes = binding_count * 8
            if binding_ptr < start or binding_ptr + binding_bytes > end:
                errors.append(
                    "binding pointer 0x%x + %u B outside image [0x%x, 0x%x)"
                    % (binding_ptr, binding_bytes, start, end)
                )
        elif binding_ptr != 0:
            errors.append("binding_count 0 but binding pointer 0x%x != 0" % binding_ptr)
    else:
        neuton_size = struct.calcsize(BACKEND_NEUTON_FMT)
        (model_ptr,) = struct.unpack(BACKEND_NEUTON_FMT, backend_bytes[:neuton_size])
        if any(backend_bytes[neuton_size:]):
            errors.append("Neuton backend must leave the rest of the union slot 0, got %r"
                          % (backend_bytes[neuton_size:],))

    bin_size = args.bin.stat().st_size
    if bin_size != linker_size:
        errors.append("binary size 0x%x != linker extent 0x%x" % (bin_size, linker_size))
    payload_cap = max_payload_size(args.partition_size, args.model_image_offset)
    if payload_cap is not None and payload_cap < 0:
        errors.append(
            "model_image_offset 0x%x exceeds partition size 0x%x"
            % (args.model_image_offset, args.partition_size)
        )
    elif payload_cap is not None and image_size > payload_cap:
        errors.append(
            "image size 0x%x exceeds partition payload cap 0x%x "
            "(partition 0x%x - model_image_offset 0x%x)"
            % (image_size, payload_cap, args.partition_size, args.model_image_offset)
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

    if crc32 == 0:
        errors.append("crc32 is 0 (patch_image_crc.py did not run)")
    else:
        crc_data = bytearray(image_bytes)
        struct.pack_into("<I", crc_data, CRC32_OFFSET, 0)
        computed_crc = zlib.crc32(crc_data) & 0xFFFFFFFF
        if crc32 != computed_crc:
            errors.append(
                "crc32 0x%08x != computed 0x%08x" % (crc32, computed_crc)
            )

    if configured_packed is not None:
        expected_packed = int(configured_packed, 0)
        if params_type != PARAMS_AXON:
            errors.append("packed output config given for non-Axon image")
        elif axon_packed_output_bytes != expected_packed:
            errors.append(
                "Axon packed output %d != expected %d"
                % (axon_packed_output_bytes, expected_packed)
            )

    if configured_persistent is not None:
        expected_persistent = int(configured_persistent, 0)
        if params_type != PARAMS_AXON:
            errors.append("persistent vars config given for non-Axon image")
        elif persistent_required != expected_persistent:
            errors.append(
                "Axon persistent vars %d != expected %d"
                % (persistent_required, expected_persistent)
            )

    scale_num, scale_elem = validate_params(params_bytes, start, end, errors)

    if errors:
        for e in errors:
            print("layout validation failed: %s" % e, file=sys.stderr)
        sys.exit(1)

    print("model image layout ok: base 0x%x, size 0x%x, model 0x%x (&%s), "
          "params_type %d, contract 0x%08x, crc32 0x%08x, name '%s' v0x%08x, "
          "scale %ux%uB"
          % (start, image_size, model_ptr, args.model_symbol, params_type,
             contract_hash, crc32, name_str if name_str is not None else "?", model_version,
             scale_num, scale_elem))
    return 0


if __name__ == "__main__":
    sys.exit(main())
