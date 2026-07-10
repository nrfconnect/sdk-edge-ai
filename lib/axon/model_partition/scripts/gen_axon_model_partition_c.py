#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Generate a separately linkable Axon model partition image C source."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MAGIC = 0x4E4F5841
VERSION = 4

CAST_PREFIX_RE = re.compile(
    r"^\((?:NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE|int8_t\s*\*|const\s+void\s*\*|void\s*\*|uint\d+_t)\)\s*",
    re.S,
)

APP_SYM_RE = re.compile(
    r"\b(nrf_axon_nn_op_extension_\w+|axon_model_\w+_packed_output_buf)\b"
)
IL_OFF_RE = re.compile(
    r"nrf_axon_interlayer_buffer\s*\)\s*\+\s*(0x[0-9A-Fa-f]+)"
)


def strip_casts(expr: str) -> str:
    value = expr.strip()

    while True:
        match = CAST_PREFIX_RE.match(value)
        if match is None:
            break
        value = value[match.end():].strip()

    return value


def strip_preprocessor_blocks(text: str) -> str:
    text = re.sub(
        r"#if\s+NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER.*?#else\s*(.*?)#endif",
        r"\1",
        text,
        flags=re.S,
    )
    return text


def resolve_model_macros(text: str, header_text: str) -> str:
    for match in re.finditer(
        r"\b(NRF_AXON_MODEL_\w+_(?:MAX_IL_BUFFER_USED|MAX_PSUM_BUFFER_USED))\b", text
    ):
        name = match.group(1)
        define_match = re.search(rf"#define\s+{re.escape(name)}\s+(\d+)", header_text)
        if define_match is not None:
            text = text.replace(name, define_match.group(1))

    return text


def collect_symbols_from_header(header_text: str) -> list[str]:
    text = strip_preprocessor_blocks(header_text)
    symbols = set(APP_SYM_RE.findall(text))
    symbols.add("nrf_axon_interlayer_buffer")
    return sorted(symbols)


def collect_app_symbols(text: str) -> list[str]:
    symbols = set(APP_SYM_RE.findall(text))
    symbols.add("nrf_axon_interlayer_buffer")
    return sorted(symbols)


def transform_pointer_expr(expr: str, model_const_name: str) -> str:
    value = strip_casts(expr.strip())

    if value in {"NULL", "0"}:
        return "0"

    if value.startswith("0x") or value.isdigit() or (
        value.startswith("-") and value[1:].isdigit()
    ):
        return value

    match = IL_OFF_RE.search(value)
    if match is not None:
        return f"IL_OFF({match.group(1)})"

    if "nrf_axon_interlayer_buffer" in value:
        return "IL_OFF(0)"

    match = re.search(rf"{re.escape(model_const_name)}\.(\w+)", value)
    if match is not None:
        return f"MC_PTR({match.group(1)})"

    match = APP_SYM_RE.search(value)
    if match is not None:
        return f"APP_SYM({match.group(1)})"

    raise ValueError(f"Unsupported pointer expression: {expr}")


def tokenize_initializer(body: str) -> list[str]:
    body = re.sub(r"//.*", "", body)
    tokens: list[str] = []
    current: list[str] = []
    depth = 0

    for char in body:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)

        if char == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            current = []
            continue

        current.append(char)

    token = "".join(current).strip()
    if token:
        tokens.append(token)

    return tokens


def extract_block(header: str, pattern: str, label: str) -> re.Match[str]:
    match = re.search(pattern, header, re.S)
    if match is None:
        raise ValueError(f"Could not find {label} in model header")
    return match


def transform_cmd_buffer(body: str, model_const_name: str) -> str:
    tokens = tokenize_initializer(body)
    transformed = [transform_pointer_expr(token, model_const_name) for token in tokens]
    return ",\n\t\t".join(transformed)


def transform_extra_outputs(body: str, model_const_name: str) -> str:
    output = body

    for match in re.finditer(r"\.ptr\s*=\s*([^,}]+)", body):
        original = match.group(1).strip()
        replacement = transform_pointer_expr(original, model_const_name)
        output = output.replace(f".ptr = {original}", f".ptr = (int8_t *)(uintptr_t){replacement}")

    return output


def transform_model_block(body: str, model_const_name: str, cmd_buffer_name: str,
                          extra_outputs_name: str | None, model_name: str,
                          image_symbol: str, header_text: str) -> str:
    output = body

    output = re.sub(
        rf"\.model_name\s*=\s*\"{re.escape(model_name)}\"",
        ".model_name = (const char *)PART_MEMBER_PTR(model_name_str)",
        output,
    )

    output = re.sub(
        rf"\b{re.escape(cmd_buffer_name)}\b",
        "(const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *)PART_MEMBER_PTR(cmd_buffer)",
        output,
    )

    output = re.sub(
        rf"&{re.escape(model_const_name)}\b",
        "(const void *)PART_MEMBER_PTR(model_const)",
        output,
    )

    if extra_outputs_name is not None:
        output = re.sub(
            rf"\b{re.escape(extra_outputs_name)}\b",
            "PART_MEMBER_PTR(extra_outputs)",
            output,
        )

    for field in ("output_ptr", "packed_output_buf"):
        for match in re.finditer(rf"\.{field}\s*=\s*([^,}}\n]+)", output):
            original = match.group(1).strip()
            if original in {"NULL", "0"}:
                continue
            replacement = transform_pointer_expr(original, model_const_name)
            cast = "int8_t *" if field == "output_ptr" else "int8_t *"
            output = output.replace(
                f".{field} = {original}",
                f".{field} = ({cast})(uintptr_t){replacement}",
            )

    for match in re.finditer(r"\.ptr\s*=\s*([^,}]+)", output):
        original = match.group(1).strip()
        if original in {"NULL", "0"}:
            continue
        replacement = transform_pointer_expr(original, model_const_name)
        output = output.replace(
            f".ptr = {original}",
            f".ptr = (int8_t *)(uintptr_t){replacement}",
        )

    for match in re.finditer(r"\.buf_ptr\s*=\s*([^,}]+)", output):
        original = match.group(1).strip()
        if original in {"NULL", "0"}:
            continue
        replacement = transform_pointer_expr(original, model_const_name)
        output = output.replace(
            f".buf_ptr = {original}",
            f".buf_ptr = (int8_t *)(uintptr_t){replacement}",
        )

    for match in re.finditer(r"\.vars\s*=\s*([^,}]+)", output):
        original = match.group(1).strip()
        if original in {"NULL", "0"}:
            continue
        if original.startswith("&"):
            output = output.replace(
                f".vars = {original}",
                ".vars = (const nrf_axon_nn_model_persistent_var_s *)PART_MEMBER_PTR(persistent_vars)",
            )

    output = strip_preprocessor_blocks(output)
    output = resolve_model_macros(output, header_text)
    output = re.sub(
        rf"sizeof\({re.escape(model_const_name)}\)",
        f"sizeof({image_symbol}_const)",
        output,
    )

    return output


def parse_model_name(model_block: str) -> str:
    match = re.search(r'\.model_name\s*=\s*"([^"]*)"', model_block)
    if match is None:
        raise ValueError("Could not find model_name in compiled model struct")
    return match.group(1)


def build_source(header_text: str, image_symbol: str) -> tuple[str, list[str]]:
    model_const_match = extract_block(
        header_text,
        r"const static struct \{(.*?)\} (axon_model_const_\w+) = \{(.*?)\};",
        "model constants",
    )
    model_const_fields = model_const_match.group(1).strip()
    model_const_name = model_const_match.group(2)
    model_const_init = model_const_match.group(3).strip()

    cmd_buffer_match = extract_block(
        header_text,
        r"const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE (cmd_buffer_\w+)\[(\d+)\] = \{(.*?)\};",
        "command buffer",
    )
    cmd_buffer_name = cmd_buffer_match.group(1)
    cmd_buffer_len = int(cmd_buffer_match.group(2))
    cmd_buffer_body = transform_cmd_buffer(cmd_buffer_match.group(3), model_const_name)

    extra_outputs_name = None
    extra_outputs_field = ""
    extra_outputs_init = ""
    extra_outputs_match = re.search(
        r"const nrf_axon_compiled_model_output_s (\w+)\[\] = \{(.*?)\};",
        header_text,
        re.S,
    )
    if extra_outputs_match is not None:
        extra_outputs_name = extra_outputs_match.group(1)
        extra_outputs_body = transform_extra_outputs(
            extra_outputs_match.group(2), model_const_name
        )
        extra_output_cnt = len(re.findall(r"\.ptr\s*=", extra_outputs_body))
        extra_outputs_field = (
            f"\tnrf_axon_compiled_model_output_s extra_outputs[{extra_output_cnt}];"
        )
        extra_outputs_init = f"""
\t.extra_outputs = {{
\t\t{extra_outputs_body}
\t}},"""

    model_match = extract_block(
        header_text,
        r"const nrf_axon_nn_compiled_model_s (model_\w+) = \{(.*?)\};",
        "compiled model struct",
    )
    model_name = parse_model_name(model_match.group(2))
    model_body = transform_model_block(
        model_match.group(2),
        model_const_name,
        cmd_buffer_name,
        extra_outputs_name,
        model_name,
        image_symbol,
        header_text,
    )

    app_symbols = collect_app_symbols(
        cmd_buffer_body + model_body + (extra_outputs_body if extra_outputs_name else "")
    )

    model_name_literal = "{" + ",".join(str(ord(ch)) for ch in model_name) + ",0}"
    model_name_size = len(model_name) + 1

    source = f"""/*
 * Auto-generated Axon model partition image source.
 * Linked separately into flash with absolute pointers.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>
#include <axon/nrf_axon_model_partition_defs.h>

#ifndef NRF_AXON_MODEL_PARTITION_ADDR
#error "NRF_AXON_MODEL_PARTITION_ADDR must be defined when linking the model image"
#endif

#define IL_OFF(off) ((uintptr_t)(AXON_INTERLAYER_BUFFER_ADDR + (off)))
#define APP_SYM(name) ((uintptr_t)AXON_SYM_##name)
#define PART_MEMBER_PTR(member) \
	((void *)(NRF_AXON_MODEL_PARTITION_ADDR + offsetof(struct {image_symbol}, member)))
#define MC_PTR(field) ((uintptr_t)(NRF_AXON_MODEL_PARTITION_ADDR + \\
	offsetof(struct {image_symbol}, model_const) + \\
	offsetof({image_symbol}_const, field)))

typedef struct {{
{model_const_fields}
}} {image_symbol}_const;

struct {image_symbol} {{
	struct nrf_axon_model_partition_header header;
	{image_symbol}_const model_const;
	uint32_t cmd_buffer[{cmd_buffer_len}];
	char model_name_str[{model_name_size}];
{extra_outputs_field}
	nrf_axon_nn_compiled_model_s model;
}};

__attribute__((section(".model_image"), used))
const struct {image_symbol} {image_symbol} = {{
\t.header = {{
\t\t.magic = {hex(MAGIC)},
\t\t.version = {VERSION},
\t\t.model_offset = offsetof(struct {image_symbol}, model),
\t\t.image_size = sizeof(struct {image_symbol}),
\t}},
\t.model_const = {{
\t\t{model_const_init}
\t}},
\t.cmd_buffer = {{
\t\t{cmd_buffer_body}
\t}},
\t.model_name_str = {model_name_literal},{extra_outputs_init}
\t.model = {{
\t\t{model_body}
\t}},
}};
"""

    return source, app_symbols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--symbols", type=Path, required=False)
    parser.add_argument("--image-symbol", default="axon_model_partition_image")
    parser.add_argument("--symbols-only", action="store_true")
    args = parser.parse_args()

    header_text = args.header.read_text(encoding="utf-8")

    if args.symbols_only:
        if args.symbols is None:
            raise ValueError("--symbols is required with --symbols-only")
        args.symbols.write_text(
            "\n".join(collect_symbols_from_header(header_text)) + "\n",
            encoding="ascii",
        )
        return

    if args.symbols is None or args.output is None:
        raise ValueError("--output and --symbols are required unless --symbols-only is set")

    source, app_symbols = build_source(header_text, args.image_symbol)
    args.output.write_text(source, encoding="ascii")
    args.symbols.write_text("\n".join(app_symbols) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
