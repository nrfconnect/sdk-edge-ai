#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Parse Axon model headers for OTA: app RAM symbols, persistent-var storage, model symbol name."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

APP_SYM_RE = re.compile(
    r"\b(nrf_axon_nn_op_extension_\w+|axonpro_\w+|axon_model_\w+_packed_output_buf)\b"
)
PERSISTENT_VARS_RE = re.compile(
    r"(?:extern )?int32_t (axon_model_\w+_persistent_vars)\[(\d+)\];"
)
PACKED_OUTPUT_BUF_RE = re.compile(
    r"uint32_t (axon_model_\w+_packed_output_buf)\[([^\]]+)\];"
)


def extract_block(header: str, pattern: str, label: str) -> re.Match[str]:
    match = re.search(pattern, header, re.S)
    if match is None:
        raise ValueError(f"Could not find {label} in model header")
    return match


def parse_model_symbol(header_text: str) -> str:
    model_match = extract_block(
        header_text,
        r"const nrf_axon_nn_compiled_model_s (model_\w+) = \{(.*?)\};",
        "compiled model struct",
    )
    return model_match.group(1)


def collect_persistent_var_symbols(header_text: str) -> list[str]:
    return [match.group(1) for match in PERSISTENT_VARS_RE.finditer(header_text)]


def build_app_persistent_src(header_text: str) -> str:
    blocks = []
    defines = set()
    for name, size in PERSISTENT_VARS_RE.findall(header_text):
        blocks.append(f"int32_t {name}[{size}];")
    for name, size_expr in PACKED_OUTPUT_BUF_RE.findall(header_text):
        for macro in re.findall(r"NRF_AXON_MODEL_\w+_PACKED_OUTPUT_SIZE", size_expr):
            def_match = re.search(rf"#define {macro}\s+(\d+)", header_text)
            if def_match is not None:
                defines.add(f"#define {macro} {def_match.group(1)}")
        blocks.append(f"uint32_t {name}[{size_expr}];")
    if not blocks:
        return ""
    return "\n".join([
        "/* Auto-generated Axon OTA persistent-var storage (application RAM). */",
        "#include <stdint.h>",
        "",
        *sorted(defines),
        "",
        *blocks,
        "",
    ])


def collect_app_symbols(header_text: str) -> list[str]:
    symbols = set(APP_SYM_RE.findall(header_text))
    symbols.update(collect_persistent_var_symbols(header_text))
    symbols.add("nrf_axon_interlayer_buffer")
    return sorted(symbols)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--symbols", type=Path, required=False)
    parser.add_argument("--app-persistent-src", type=Path, required=False)
    parser.add_argument("--print-symbols", action="store_true")
    parser.add_argument("--print-model-symbol", action="store_true")
    args = parser.parse_args()

    header_text = args.header.read_text(encoding="utf-8")

    if args.print_model_symbol:
        print(parse_model_symbol(header_text), end="")
        return

    if args.print_symbols:
        print("\n".join(collect_app_symbols(header_text)), end="")
        return

    if args.app_persistent_src is not None:
        args.app_persistent_src.write_text(build_app_persistent_src(header_text), encoding="ascii")
        return

    if args.symbols is None:
        raise ValueError("--symbols is required unless --print-symbols or --print-model-symbol")

    symbol_lines = "\n".join(collect_app_symbols(header_text)) + "\n"
    args.symbols.write_text(symbol_lines, encoding="ascii")


if __name__ == "__main__":
    main()
