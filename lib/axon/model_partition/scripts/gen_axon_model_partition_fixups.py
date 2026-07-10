#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Generate Axon model partition fixups header and symbol list for the stub image build."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

APP_SYM_RE = re.compile(
    r"\b(nrf_axon_nn_op_extension_\w+|axon_model_\w+_packed_output_buf)\b"
)


def strip_preprocessor_blocks(text: str) -> str:
    text = re.sub(
        r"#if\s+NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER.*?#else\s*(.*?)#endif",
        r"\1",
        text,
        flags=re.S,
    )
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


def build_fixups_header(header_text: str, model_sym: str, header_path: Path) -> str:
    text = strip_preprocessor_blocks(header_text)
    app_symbols = collect_app_symbols(text)

    lines = [
        "/* Auto-generated Axon model partition fixup macros. */",
        "#ifndef NRF_AXON_MODEL_IMAGE_FIXUPS_H_",
        "#define NRF_AXON_MODEL_IMAGE_FIXUPS_H_",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "#include <stdbool.h>",
        "#include <assert.h>",
        "",
        "#include <drivers/axon/nrf_axon_driver.h>",
        "#include <drivers/axon/nrf_axon_nn_infer.h>",
        "#include <axon/nrf_axon_platform.h>",
        "",
        "#ifndef NRF_AXON_MODEL_PARTITION_ADDR",
        '#error "NRF_AXON_MODEL_PARTITION_ADDR must be defined when building the model image"',
        "#endif",
        "",
        "#undef NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER",
        "#define NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER 0",
        "",
    ]

    for sym in app_symbols:
        if sym.startswith("axon_model_") and sym.endswith("_packed_output_buf"):
            lines.append(f"#define {sym} ((uint8_t *)0)")

    lines.extend([
        "",
        "#ifndef NRF_AXON_MODEL_IMAGE_MODEL_SYM",
        f"#define NRF_AXON_MODEL_IMAGE_MODEL_SYM {model_sym}",
        "#endif",
        "",
        f'#include "{header_path.name}"',
        "",
        "#endif /* NRF_AXON_MODEL_IMAGE_FIXUPS_H_ */",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--symbols", type=Path, required=True)
    parser.add_argument("--fixups-header", type=Path, required=False)
    parser.add_argument("--symbols-only", action="store_true")
    args = parser.parse_args()

    header_text = args.header.read_text(encoding="utf-8")

    if args.symbols_only:
        args.symbols.write_text(
            "\n".join(collect_symbols_from_header(header_text)) + "\n",
            encoding="ascii",
        )
        return

    if args.fixups_header is None:
        raise ValueError("--fixups-header is required unless --symbols-only is set")

    model_sym = parse_model_symbol(header_text)
    text = strip_preprocessor_blocks(header_text)
    app_symbols = collect_app_symbols(
        text + f"const nrf_axon_nn_compiled_model_s {model_sym} = {{}};"
    )

    args.fixups_header.write_text(
        build_fixups_header(header_text, model_sym, args.header),
        encoding="ascii",
    )
    args.symbols.write_text("\n".join(app_symbols) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
