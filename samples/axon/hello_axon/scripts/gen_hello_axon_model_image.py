#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Generate hello_axon model partition image from the compiler header."""

from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path

MAGIC = 0x4E4F5841
VERSION = 2
MODEL_CONST_SIZE = 420
CMD_BUFFER_LEN = 69
MODEL_NAME_MAX_LEN = 32

CMD_PTR_INTERLAYER = 0xFFFFFFFF
CMD_PTR_MODEL_CONST_BASE = 0x80000000

MODEL_CONST_FIELDS = (
    ("l00_weights", 0),
    ("l00_biasp", 16),
    ("l01_weights", 80),
    ("l01_biasp", 336),
    ("l02_weights", 400),
    ("l02_biasp", 416),
)

METADATA_FIELDS = (
    ("compiler_version", "<I"),
    ("interlayer_buffer_needed", "<I"),
    ("psum_buffer_needed", "<I"),
    ("min_driver_version_required", "<I"),
    ("output_dequant_mult", "<I"),
    ("input_quant_mult", "<I"),
    ("input_height", "<H"),
    ("input_width", "<H"),
    ("input_channel_cnt", "<H"),
    ("output_height", "<H"),
    ("output_width", "<H"),
    ("output_channel_cnt", "<H"),
    ("input_byte_width", "<B"),
    ("input_quant_round", "<B"),
    ("input_quant_zp", "<b"),
    ("input_stride", "<H"),
    ("output_byte_width", "<B"),
    ("output_dequant_round", "<B"),
    ("output_dequant_zp", "<b"),
    ("output_stride", "<H"),
    ("input_cnt", "<B"),
    ("external_input_ndx", "<b"),
    ("is_external", "<B"),
    ("is_layer_model", "<B"),
    ("extra_output_cnt", "<H"),
)


def parse_model_const(header_text: str) -> bytes:
    values: list[int] = []

    for field, _offset in MODEL_CONST_FIELDS:
        match = re.search(rf"\.{field} = \{{([^}}]*)\}}", header_text)
        if match is None:
            raise ValueError(f"Could not find field {field}")

        nums = [int(item.strip()) for item in match.group(1).split(",") if item.strip()]

        if field.endswith("weights"):
            values.extend([(num + 256) % 256 for num in nums])
            continue

        for num in nums:
            values.extend(struct.pack("<i", num))

    if len(values) != MODEL_CONST_SIZE:
        raise ValueError(f"Unexpected model const size: {len(values)}")

    return bytes(values)


def encode_cmd_word(token: str) -> int:
    token = token.strip()

    if not token:
        raise ValueError("Empty cmd buffer token")

    if "nrf_axon_interlayer_buffer" in token:
        return CMD_PTR_INTERLAYER

    for field, offset in MODEL_CONST_FIELDS:
        if f"axon_model_const_hello_axon.{field}" in token:
            return CMD_PTR_MODEL_CONST_BASE | offset

    if token.startswith("0x"):
        return int(token, 16)

    return int(token)


def parse_cmd_buffer(header_text: str) -> list[int]:
    match = re.search(
        r"cmd_buffer_hello_axon\[(\d+)\] = \{([^;]*)\};",
        header_text,
        re.S,
    )
    if match is None:
        raise ValueError("Could not find cmd_buffer_hello_axon")

    length = int(match.group(1))
    body = match.group(2)
    body = re.sub(r"//.*", "", body)
    tokens = [item.strip() for item in body.split(",") if item.strip()]

    words = [encode_cmd_word(token) for token in tokens]
    if len(words) != length:
        raise ValueError(f"Unexpected cmd buffer length: {len(words)} != {length}")

    return words


def resolve_value(header_text: str, value: str) -> int:
    value = value.strip()
    if value in {"true", "false"}:
        return int(value == "true")
    if value.startswith("0x"):
        return int(value, 16)
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)

    define_match = re.search(rf"#define\s+{re.escape(value)}\s+(\d+)", header_text)
    if define_match is not None:
        return int(define_match.group(1))

    raise ValueError(f"Could not resolve value {value}")


def parse_scalar_in_model(header_text: str, field: str) -> int:
    match = re.search(
        rf"const nrf_axon_nn_compiled_model_s model_hello_axon = \{{.*?\.{field} = ([^,\n}}]+)",
        header_text,
        re.S,
    )
    if match is None:
        raise ValueError(f"Could not find model metadata field {field}")

    return resolve_value(header_text, match.group(1))


def parse_metadata(header_text: str) -> bytes:
    input_match = re.search(
        r"\.inputs = \{\s*\{[^}]*\.dimensions = \{\s*"
        r"\.height = (\d+),\s*\.width = (\d+),\s*\.channel_cnt = (\d+),\s*"
        r"\.byte_width = (\d+),\s*\},\s*"
        r"\.quant_mult = (\d+),\s*\.stride = (\d+),\s*"
        r"\.quant_round = (\d+),\s*\.quant_zp = (-?\d+),\s*"
        r"\.is_external = (true|false),",
        header_text,
        re.S,
    )
    if input_match is None:
        raise ValueError("Could not parse input descriptor")

    output_match = re.search(
        r"\.output_dimensions = \{\s*"
        r"\.height = (\d+),\s*\.width = (\d+),\s*\.channel_cnt = (\d+),\s*"
        r"\.byte_width = (\d+),\s*\},",
        header_text,
        re.S,
    )
    if output_match is None:
        raise ValueError("Could not parse output dimensions")

    values = {
        "compiler_version": parse_scalar_in_model(header_text, "compiler_version"),
        "interlayer_buffer_needed": parse_scalar_in_model(header_text, "interlayer_buffer_needed"),
        "psum_buffer_needed": parse_scalar_in_model(header_text, "psum_buffer_needed"),
        "min_driver_version_required": parse_scalar_in_model(header_text, "min_driver_version_required"),
        "output_dequant_mult": parse_scalar_in_model(header_text, "output_dequant_mult"),
        "input_quant_mult": int(input_match.group(5)),
        "input_height": int(input_match.group(1)),
        "input_width": int(input_match.group(2)),
        "input_channel_cnt": int(input_match.group(3)),
        "output_height": int(output_match.group(1)),
        "output_width": int(output_match.group(2)),
        "output_channel_cnt": int(output_match.group(3)),
        "input_byte_width": int(input_match.group(4)),
        "input_quant_round": int(input_match.group(7)),
        "input_quant_zp": int(input_match.group(8)),
        "input_stride": int(input_match.group(6)),
        "output_byte_width": int(output_match.group(4)),
        "output_dequant_round": parse_scalar_in_model(header_text, "output_dequant_round"),
        "output_dequant_zp": parse_scalar_in_model(header_text, "output_dequant_zp"),
        "output_stride": parse_scalar_in_model(header_text, "output_stride"),
        "input_cnt": parse_scalar_in_model(header_text, "input_cnt"),
        "external_input_ndx": parse_scalar_in_model(header_text, "external_input_ndx"),
        "is_external": int(input_match.group(9) == "true"),
        "is_layer_model": parse_scalar_in_model(header_text, "is_layer_model"),
        "extra_output_cnt": parse_scalar_in_model(header_text, "extra_output_cnt"),
    }

    packed = b"".join(struct.pack(fmt, values[name]) for name, fmt in METADATA_FIELDS)
    return packed


def parse_model_name(header_text: str) -> str:
    match = re.search(
        r'const nrf_axon_nn_compiled_model_s model_hello_axon = \{.*?\.model_name = "([^"]*)"',
        header_text,
        re.S,
    )
    if match is None:
        raise ValueError("Could not find model_name")

    return match.group(1)


def build_image(header_text: str) -> bytes:
    model_const = parse_model_const(header_text)
    cmd_buffer = parse_cmd_buffer(header_text)
    metadata = parse_metadata(header_text)
    model_name = parse_model_name(header_text).encode("ascii") + b"\0"

    if len(model_name) > MODEL_NAME_MAX_LEN:
        raise ValueError("Model name too long for partition image")

    model_name_padded = model_name + bytes(MODEL_NAME_MAX_LEN - len(model_name))

    header_size = 32
    metadata_offset = header_size
    model_const_offset = metadata_offset + len(metadata)
    cmd_buffer_offset = model_const_offset + len(model_const)
    model_name_offset = cmd_buffer_offset + len(cmd_buffer) * 4

    header = struct.pack(
        "<8I",
        MAGIC,
        VERSION,
        MODEL_CONST_SIZE,
        CMD_BUFFER_LEN,
        metadata_offset,
        model_const_offset,
        cmd_buffer_offset,
        model_name_offset,
    )

    cmd_buffer_blob = b"".join(struct.pack("<I", word) for word in cmd_buffer)

    return header + metadata + model_const + cmd_buffer_blob + model_name_padded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    image = build_image(args.header.read_text())
    args.output.write_bytes(image)


if __name__ == "__main__":
    main()
