#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Generate a separately linkable hello_axon model image C source."""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_hello_axon_model_image import (
    CMD_BUFFER_LEN,
    MAGIC,
    METADATA_FIELDS,
    MODEL_CONST_FIELDS,
    MODEL_CONST_SIZE,
    MODEL_NAME_MAX_LEN,
    parse_cmd_buffer,
    parse_metadata,
    parse_model_const,
    parse_model_name,
    resolve_value,
)

VERSION = 3


def c_int_array(data: bytes, signed: bool = True) -> str:
    values = list(data)

    if not signed:
        values = [value % 256 for value in values]

    return "{" + ",".join(str(value) for value in values) + "," + "}"


def cmd_token_to_c(token: str) -> str:
    token = token.strip()

    if "nrf_axon_interlayer_buffer" in token:
        return "IL_PTR"

    for field, _offset in MODEL_CONST_FIELDS:
        if f"axon_model_const_hello_axon.{field}" in token:
            return f"MC_PTR({field})"

    if token.startswith("0x"):
        return token

    return token


def parse_cmd_buffer_c(header_text: str) -> str:
    match = re.search(
        r"cmd_buffer_hello_axon\[(\d+)\] = \{([^;]*)\};",
        header_text,
        re.S,
    )
    if match is None:
        raise ValueError("Could not find cmd_buffer_hello_axon")

    body = re.sub(r"//.*", "", match.group(2))
    tokens = [item.strip() for item in body.split(",") if item.strip()]
    words = [cmd_token_to_c(token) for token in tokens]

    if len(words) != CMD_BUFFER_LEN:
        raise ValueError(f"Unexpected cmd buffer length: {len(words)} != {CMD_BUFFER_LEN}")

    return ",\n\t\t".join(words)


def metadata_to_c_initializer(metadata_blob: bytes) -> str:
    offset = 0
    fields: list[str] = []

    for name, fmt in METADATA_FIELDS:
        size = struct.calcsize(fmt)
        value = struct.unpack(fmt, metadata_blob[offset : offset + size])[0]
        offset += size
        fields.append(f".{name} = {value},")

    return "\n\t\t".join(fields)


def model_const_to_c_initializer(model_const: bytes) -> str:
    chunks: list[str] = []
    offset = 0

    for field, field_offset in MODEL_CONST_FIELDS:
        if field.endswith("weights"):
            if field == "l00_weights":
                length = 16
            elif field == "l01_weights":
                length = 256
            else:
                length = 16

            values = list(model_const[offset : offset + length])
            chunks.append(f".{field} = {{{','.join(str(value) for value in values)},}},")
            offset += length
            continue

        if field == "l00_biasp":
            length = 16
        elif field == "l01_biasp":
            length = 16
        else:
            length = 1

        values = []
        for index in range(length):
            value = struct.unpack_from("<i", model_const, offset + index * 4)[0]
            values.append(str(value))

        chunks.append(f".{field} = {{{','.join(values)},}},")
        offset += length * 4

    if offset != MODEL_CONST_SIZE:
        raise ValueError(f"Unexpected model const size: {offset}")

    return "\n\t\t".join(chunks)


def image_offsets() -> tuple[int, int, int, int, int]:
    header_size = 32
    metadata_offset = header_size
    model_const_offset = metadata_offset + 52
    cmd_buffer_offset = model_const_offset + MODEL_CONST_SIZE
    model_name_offset = cmd_buffer_offset + CMD_BUFFER_LEN * 4
    return metadata_offset, model_const_offset, cmd_buffer_offset, model_name_offset


def build_source(header_text: str) -> str:
    model_const = parse_model_const(header_text)
    metadata_blob = parse_metadata(header_text)
    model_name = parse_model_name(header_text)
    cmd_buffer_body = parse_cmd_buffer_c(header_text)
    metadata_offset, model_const_offset, cmd_buffer_offset, model_name_offset = image_offsets()

    model_name_padded = model_name.encode("ascii") + b"\0"
    if len(model_name_padded) > MODEL_NAME_MAX_LEN:
        raise ValueError("Model name too long for partition image")

    model_name_literal = "{" + ",".join(str(byte) for byte in model_name_padded) + ","
    model_name_literal += "0," * (MODEL_NAME_MAX_LEN - len(model_name_padded)) + "}"

    return f"""/*
 * Auto-generated hello_axon model partition image source.
 * Linked separately into axon_model_partition with absolute cmd buffer pointers.
 */

#include <stddef.h>
#include <stdint.h>

#include "hello_axon_model_image_defs.h"

#ifndef HELLO_AXON_INTERLAYER_BUFFER_ADDR
#error "HELLO_AXON_INTERLAYER_BUFFER_ADDR must be defined when linking the model image"
#endif

#ifndef HELLO_AXON_MODEL_PARTITION_ADDR
#error "HELLO_AXON_MODEL_PARTITION_ADDR must be defined when linking the model image"
#endif

#define IL_PTR ((uint32_t)HELLO_AXON_INTERLAYER_BUFFER_ADDR)
#define MC_PTR(field) ((uint32_t)(HELLO_AXON_MODEL_PARTITION_ADDR + \\
	offsetof(struct hello_axon_model_image_blob, model_const) + \\
	offsetof(struct hello_axon_model_const_layout, field)))

__attribute__((section(".model_image"), used))
const struct hello_axon_model_image_blob hello_axon_model_image = {{
	.header = {{
		.magic = {hex(MAGIC)},
		.version = {VERSION},
		.model_const_size = {MODEL_CONST_SIZE},
		.cmd_buffer_len = {CMD_BUFFER_LEN},
		.metadata_offset = {metadata_offset},
		.model_const_offset = {model_const_offset},
		.cmd_buffer_offset = {cmd_buffer_offset},
		.model_name_offset = {model_name_offset},
	}},
	.metadata = {{
		{metadata_to_c_initializer(metadata_blob)}
	}},
	.model_const = {{
		{model_const_to_c_initializer(model_const)}
	}},
	.cmd_buffer = {{
		{cmd_buffer_body}
	}},
	.model_name = {model_name_literal},
}};
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.write_text(build_source(args.header.read_text()), encoding="ascii")


if __name__ == "__main__":
    main()
