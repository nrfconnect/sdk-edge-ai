#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Parse Axon model headers for OTA: app RAM symbols and model metadata."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

APP_SYM_RE = re.compile(
    r"\b(nrf_axon_nn_op_extension_\w+|axonpro_\w+|axon_model_\w+_packed_output_buf)\b"
)
_STORAGE_QUAL = r"(?:NRF_AXON_MODEL_APP_STORAGE\s+)?"
PERSISTENT_VARS_DEF_RE = re.compile(
    rf"^{_STORAGE_QUAL}int32_t (axon_model_\w+_persistent_vars)\[(\d+)\];", re.M
)
PACKED_OUTPUT_BUF_RE = re.compile(
    rf"{_STORAGE_QUAL}uint32_t (axon_model_\w+_packed_output_buf)\["
)


def strip_packed_output_blocks(header: str) -> str:
    """Drop optional packed-output buffer decls (not used when macro is unset)."""
    return re.sub(
        r"#if\s+NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER.*?#endif",
        "",
        header,
        flags=re.S,
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


def parse_persistent_vars(header_text: str) -> tuple[str, int] | None:
    match = PERSISTENT_VARS_DEF_RE.search(header_text)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def parse_packed_output(header_text: str) -> tuple[str, int] | None:
    buf_match = PACKED_OUTPUT_BUF_RE.search(header_text)
    if buf_match is None:
        return None
    sym = buf_match.group(1)
    model_token = sym.removeprefix("axon_model_").removesuffix("_packed_output_buf")
    size_match = re.search(
        rf"#define NRF_AXON_MODEL_{re.escape(model_token.upper())}_PACKED_OUTPUT_SIZE\s+(\d+)",
        header_text,
    )
    if size_match is None:
        raise ValueError(f"packed output buffer {sym} found but size macro is missing")
    return sym, int(size_match.group(1))


def collect_persistent_var_symbols(header_text: str) -> list[str]:
    parsed = parse_persistent_vars(header_text)
    return [parsed[0]] if parsed is not None else []


def collect_app_symbols(header_text: str) -> list[str]:
    symbols = set(APP_SYM_RE.findall(strip_packed_output_blocks(header_text)))
    symbols.update(collect_persistent_var_symbols(header_text))
    symbols.add("nrf_axon_interlayer_buffer")
    return sorted(symbols)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--symbols", type=Path, required=False)
    parser.add_argument("--print-symbols", action="store_true")
    parser.add_argument("--print-model-symbol", action="store_true")
    parser.add_argument("--print-persistent-vars", action="store_true")
    parser.add_argument("--print-packed-output", action="store_true")
    args = parser.parse_args()

    header_text = args.header.read_text(encoding="utf-8")

    if args.print_model_symbol:
        print(parse_model_symbol(header_text), end="")
        return

    if args.print_persistent_vars:
        parsed = parse_persistent_vars(header_text)
        if parsed is not None:
            print(f"{parsed[0]} {parsed[1]}", end="")
        return

    if args.print_packed_output:
        parsed = parse_packed_output(header_text)
        if parsed is not None:
            print(f"{parsed[0]} {parsed[1]}", end="")
        return

    if args.print_symbols:
        print("\n".join(collect_app_symbols(header_text)), end="")
        return

    if args.symbols is None:
        raise ValueError(
            "--symbols is required unless --print-symbols, --print-model-symbol, "
            "--print-persistent-vars or --print-packed-output is used"
        )

    symbol_lines = "\n".join(collect_app_symbols(header_text)) + "\n"
    args.symbols.write_text(symbol_lines, encoding="ascii")


if __name__ == "__main__":
    main()
