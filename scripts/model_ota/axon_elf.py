#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

"""Inspect Axon probe ELFs and emit OTA slot headers.

NAMING_CONTRACT
---------------
The inspect subcommand assumes the Axon compiler emits the following symbols for a model whose
logical name is ``<name>`` (lowercase ASCII identifier, for example ``tinyml_kws``):

- ``axon_model_const_<name>`` — file-local STT_OBJECT used only to recover ``<name>`` from the
  probe.
- ``model_<name>`` — global read-only STT_OBJECT, the ``nrf_axon_nn_compiled_model_s`` instance.
- ``<name>_input_vector_list`` — global app storage for input vector pointers.
- ``axon_model_<name>_persistent_vars`` — optional global app storage for persistent variables.
- ``<name>_packed_output_buf`` — optional global app storage for the packed output buffer.
- ``nrf_axon_interlayer_buffer`` — firmware symbol every model references.

Generated public headers re-export ``NRF_AXON_MODEL_<TOKEN>_PACKED_OUTPUT_SIZE``, where ``<TOKEN>``
is ``<name>`` uppercased. Compiler owners should confirm this naming is stable across compiler
versions.
"""

import argparse
import contextlib
import os
import re
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from elf_symbols import ElfSymbol, SymbolIndex, load_symbol_index
from model_contract import symbol_name_hash

CONFIG_SCHEMA_VERSION = 0

MODEL_CONST_PREFIX = "axon_model_const_"
MODEL_NAME_PATTERN = re.compile(r"[a-z_][a-z0-9_]*")
PERSISTENT_VARS_PREFIX = "axon_model_"
PERSISTENT_VARS_SUFFIX = "_persistent_vars"
PACKED_OUTPUT_SUFFIX = "_packed_output_buf"
INPUT_VECTOR_LIST_SUFFIX = "_input_vector_list"

INTERLAYER_BUFFER_SYMBOL = "nrf_axon_interlayer_buffer"

INT32_SIZE = 4
POINTER_SIZE = 4

_UNDEFINED_ALLOWLIST = frozenset(
    {
        "__model_image_end",
        "__aeabi_read_tp",
        "__stack_chk_fail",
        "__stack_chk_guard",
        "_GLOBAL_OFFSET_TABLE_",
    }
)


@dataclass(frozen=True)
class ProbeMetadata:
    """Storage and model symbols discovered in one Axon probe object.

    keep_symbols lists the firmware symbols the model image references; the firmware must keep
    and bind each of them. persistent_required and persistent_cap count int32_t entries,
    packed_output_bytes is in bytes, and input_vector_cnt counts pointers.
    """

    model_name: str
    model_symbol: str
    keep_symbols: tuple[str, ...]
    persistent_required: int
    persistent_cap: int
    persistent_symbol: str | None
    packed_output_bytes: int
    packed_output_symbol: str | None
    packed_output_allocated: bool
    input_vector_cnt: int
    input_vector_list_symbol: str


def discover_model_name(index: SymbolIndex) -> str:
    """Recover the Axon model name from the sole ``axon_model_const_<name>`` object."""
    matches = [
        entry
        for entry in index.symbols
        if not entry.is_undefined
        and entry.is_object
        and entry.is_local
        and entry.name.startswith(MODEL_CONST_PREFIX)
    ]
    if len(matches) != 1:
        names = ", ".join(sorted(entry.name for entry in matches)) or "none"
        raise ValueError(
            f"probe must define exactly one local {MODEL_CONST_PREFIX}* object (found: {names})"
        )
    entry = matches[0]
    if entry.size <= 0:
        raise ValueError(f"{entry.name} must be a non-empty object")
    model_name = entry.name[len(MODEL_CONST_PREFIX) :]
    if not MODEL_NAME_PATTERN.fullmatch(model_name):
        raise ValueError(
            f"model name {model_name!r} from {entry.name} is not a lowercase ASCII identifier"
        )
    return model_name


def discover_model_symbol(index: SymbolIndex, model_name: str) -> str:
    """Check the compiled-model symbol ``model_<model_name>`` and return its name."""
    expected = f"model_{model_name}"
    try:
        entry = index.unique(expected)
    except ValueError as exc:
        raise ValueError(f"model symbol {expected!r}: {exc}") from exc
    if entry is None:
        raise ValueError(f"probe is missing defined global model symbol {expected!r}")
    if not entry.is_global or not entry.is_object:
        raise ValueError(f"{expected} must be a defined global STT_OBJECT")
    if entry.read_only is not True:  # None (no section info) is rejected too
        raise ValueError(f"{expected} must live in a read-only section")
    if entry.size <= 0:
        raise ValueError(f"{expected} must be a non-empty object")
    return expected


def _undefined_global_symbols(index: SymbolIndex) -> set[str]:
    """Collect the undefined global symbols the model needs from the firmware.

    Toolchain helpers in the allowlist are left out.
    """
    return {
        entry.name
        for entry in index.symbols
        if entry.is_undefined and entry.is_global and entry.name not in _UNDEFINED_ALLOWLIST
    }


def _lookup_storage(index: SymbolIndex, symbol_name: str) -> ElfSymbol | None:
    """Look up and check the app-owned storage object symbol_name; return None if absent."""
    try:
        entry = index.unique(symbol_name)
    except ValueError as exc:
        raise ValueError(f"storage symbol {symbol_name!r}: {exc}") from exc
    if entry is None:
        return None
    if not entry.is_global or not entry.is_object:
        raise ValueError(f"storage symbol {symbol_name!r} must be a defined global STT_OBJECT")
    if entry.read_only is not False:  # None (no section info) is rejected too
        raise ValueError(f"storage symbol {symbol_name!r} must live in a writable section")
    return entry


def _input_vector_list(index: SymbolIndex, model_name: str) -> ElfSymbol:
    """Look up the required input vector list and check it is a non-empty pointer array."""
    name = f"{model_name}{INPUT_VECTOR_LIST_SUFFIX}"
    entry = _lookup_storage(index, name)
    if entry is None:
        raise ValueError(f"probe is missing required symbol {name!r}")
    if entry.size == 0 or entry.size % POINTER_SIZE != 0:
        raise ValueError(f"{name} size {entry.size} is not a non-zero multiple of the pointer size")
    return entry


def _persistent_vars(
    index: SymbolIndex, model_name: str, cap: int | None
) -> tuple[ElfSymbol | None, int, int]:
    """Look up the optional persistent-vars array and compute its required and cap int32_t counts.

    The capacity defaults to the required count; a smaller or negative capacity is an error.
    """
    name = f"{PERSISTENT_VARS_PREFIX}{model_name}{PERSISTENT_VARS_SUFFIX}"
    entry = _lookup_storage(index, name)
    required = 0
    if entry is not None:
        if entry.size % INT32_SIZE != 0:
            raise ValueError(f"{name} size {entry.size} is not a multiple of sizeof(int32_t)")
        required = entry.size // INT32_SIZE
    if cap is None:
        cap = required
    if cap < 0:
        raise ValueError("persistent-vars cap cannot be negative")
    if required > cap:
        raise ValueError(f"persistent-vars cap {cap} < required {required}")
    return entry, required, cap


def _packed_output(index: SymbolIndex, model_name: str, allocate: bool) -> ElfSymbol | None:
    """Look up the optional uint32_t packed-output buffer; require it when allocate is set."""
    name = f"{model_name}{PACKED_OUTPUT_SUFFIX}"
    entry = _lookup_storage(index, name)
    if entry is None:
        if allocate:
            raise ValueError("allocate_packed_output requested but no packed-output buffer found")
        return None
    if entry.size % INT32_SIZE != 0:
        raise ValueError(f"{name} size {entry.size} is not a multiple of sizeof(uint32_t)")
    return entry


def inspect_symbols(
    index: SymbolIndex,
    *,
    persistent_vars_cap: int | None = None,
    allocate_packed_output: bool = False,
) -> ProbeMetadata:
    """Extract OTA metadata for the Axon model in index; ValueError on contract violations.

    persistent_vars_cap is the firmware's persistent-vars capacity in int32_t entries and
    defaults to what the model requires. With allocate_packed_output, the packed-output buffer
    becomes firmware-owned storage the image binds to; without it, the image links with
    packed_output_buf NULL.
    """
    model_name = discover_model_name(index)
    model_symbol = discover_model_symbol(index, model_name)
    input_vectors = _input_vector_list(index, model_name)
    persistent, persistent_required, persistent_cap = _persistent_vars(
        index, model_name, persistent_vars_cap
    )
    packed = _packed_output(index, model_name, allocate_packed_output)

    keep_symbols = _undefined_global_symbols(index)
    keep_symbols.add(INTERLAYER_BUFFER_SYMBOL)
    keep_symbols.add(input_vectors.name)
    if persistent is not None:
        keep_symbols.add(persistent.name)
    if allocate_packed_output and packed is not None:
        keep_symbols.add(packed.name)

    return ProbeMetadata(
        model_name=model_name,
        model_symbol=model_symbol,
        keep_symbols=tuple(sorted(keep_symbols)),
        persistent_required=persistent_required,
        persistent_cap=persistent_cap,
        persistent_symbol=persistent.name if persistent else None,
        packed_output_bytes=packed.size if packed else 0,
        packed_output_symbol=packed.name if packed else None,
        packed_output_allocated=allocate_packed_output,
        input_vector_cnt=input_vectors.size // POINTER_SIZE,
        input_vector_list_symbol=input_vectors.name,
    )


def _c_string(value: str) -> str:
    """Quote value as a C string literal; ValueError on control characters."""
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise ValueError(f"{value!r} contains control characters")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def sanitize_model_id(model_id: str) -> str:
    """Turn a free-form OTA slot ID into a C macro token."""
    token = re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_").upper()
    if not token:
        raise ValueError("model ID must contain an ASCII letter or digit")
    if token[0].isdigit():
        token = "_" + token
    return token


def axon_model_token(model_name: str) -> str:
    """Derive the token used in the compiler's ``NRF_AXON_MODEL_<TOKEN>_*`` macros."""
    return model_name.upper()


def render_private_header(header_name: str, metadata: ProbeMetadata) -> str:
    """Render the header the model image build includes.

    Defines MODEL_OTA_AXON_* macros for the model header and symbol, the storage capacities and
    symbols, and the keep-symbol list with its name hashes (MODEL_OTA_AXON_KEEP_REFS).
    """
    lines = [
        "/* Generated by axon_elf.py; do not edit. */",
        "#pragma once",
        f"#define MODEL_OTA_AXON_CONFIG_VERSION {CONFIG_SCHEMA_VERSION}",
        f"#define MODEL_OTA_AXON_HEADER {_c_string(header_name)}",
        f"#define MODEL_OTA_AXON_MODEL_SYM {metadata.model_symbol}",
        f"#define MODEL_OTA_AXON_PERSISTENT_VARS_REQUIRED {metadata.persistent_required}",
        f"#define MODEL_OTA_AXON_PERSISTENT_VARS_CAP {metadata.persistent_cap}",
        f"#define MODEL_OTA_AXON_PACKED_OUTPUT_BYTES {metadata.packed_output_bytes}",
        f"#define MODEL_OTA_AXON_PACKED_OUTPUT_ALLOC {int(metadata.packed_output_allocated)}",
        f"#define MODEL_OTA_AXON_INPUT_VECTOR_CAP {metadata.input_vector_cnt}",
        f"#define MODEL_OTA_AXON_INPUT_VECTOR_LIST_SYM {metadata.input_vector_list_symbol}",
    ]
    if metadata.persistent_symbol:
        lines.append(f"#define MODEL_OTA_AXON_PERSISTENT_VARS_SYM {metadata.persistent_symbol}")
    if metadata.packed_output_allocated and metadata.packed_output_symbol:
        lines.append(f"#define MODEL_OTA_AXON_PACKED_OUTPUT_SYM {metadata.packed_output_symbol}")
    lines.append(f"#define MODEL_OTA_AXON_KEEP_SYMBOL_COUNT {len(metadata.keep_symbols)}")
    for symbol in metadata.keep_symbols:
        lines.append(f"#define MODEL_OTA_AXON_SYM_HASH_{symbol} 0x{symbol_name_hash(symbol):08X}u")
    lines.append("#define MODEL_OTA_AXON_SYM_HASH(sym) MODEL_OTA_AXON_SYM_HASH_##sym")
    lines.append("#define MODEL_OTA_AXON_KEEP_REFS(X) \\")
    for pos, symbol in enumerate(metadata.keep_symbols):
        continuation = " \\" if pos + 1 < len(metadata.keep_symbols) else ""
        lines.append(f"\tX({symbol}){continuation}")
    return "\n".join(lines) + "\n"


def render_public_header(model_id: str, metadata: ProbeMetadata) -> str:
    """Render the header the application includes for one OTA slot.

    Defines MODEL_OTA_AXON_<SLOT>_PACKED_OUTPUT_BYTES and _PERSISTENT_VARS_CAP, and re-exports
    NRF_AXON_MODEL_<TOKEN>_PACKED_OUTPUT_SIZE, since the application no longer compiles the
    model header.
    """
    slot_token = sanitize_model_id(model_id)
    model_token = axon_model_token(metadata.model_name)
    lines = [
        "/* Generated by axon_elf.py; do not edit. */",
        "#pragma once",
        f"/* Firmware capacities for OTA slot {model_id}. An image may require less. */",
        f"#define MODEL_OTA_AXON_{slot_token}_PACKED_OUTPUT_BYTES {metadata.packed_output_bytes}",
        f"#define MODEL_OTA_AXON_{slot_token}_PERSISTENT_VARS_CAP {metadata.persistent_cap}",
        "",
        "/* Re-exported from the Axon model header, which an OTA-wired app does not compile. */",
        f"#ifndef NRF_AXON_MODEL_{model_token}_PACKED_OUTPUT_SIZE",
        f"#define NRF_AXON_MODEL_{model_token}_PACKED_OUTPUT_SIZE \\",
        f"\tMODEL_OTA_AXON_{slot_token}_PACKED_OUTPUT_BYTES",
        "#endif",
    ]
    return "\n".join(lines) + "\n"


def atomic_write_if_changed(path: Path, data: str) -> None:
    """Write path with data atomically, leaving it untouched when the contents match.

    Unchanged files keep their timestamp, so the build does not recompile their users.
    """
    payload = data.encode("utf-8")
    try:
        if path.read_bytes() == payload:
            return
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def cmd_inspect(args: argparse.Namespace) -> int:
    """Run the inspect subcommand: write the private and public headers for one probe."""
    if "*/" in args.model_id or not args.model_id.isprintable():
        raise ValueError(f"model ID {args.model_id!r} contains illegal characters")
    if not args.probe.is_file():
        raise ValueError(f"probe object not found: {args.probe}")
    metadata = inspect_symbols(
        load_symbol_index(args.probe),
        persistent_vars_cap=args.persistent_vars_cap,
        allocate_packed_output=args.allocate_packed_output,
    )
    atomic_write_if_changed(args.private_header, render_private_header(args.header_name, metadata))
    atomic_write_if_changed(args.public_header, render_public_header(args.model_id, metadata))
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Build the command-line parser, with one subparser per subcommand."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    inspect = sub.add_parser(
        "inspect", help="inspect one Axon probe and emit C headers", allow_abbrev=False
    )
    inspect.add_argument(
        "--probe",
        type=Path,
        required=True,
        help="probe object compiled from the Axon model header",
    )
    inspect.add_argument(
        "--header-name",
        required=True,
        help="Axon model header file name, written as MODEL_OTA_AXON_HEADER",
    )
    inspect.add_argument(
        "--model-id",
        required=True,
        help="OTA slot ID, used for the MODEL_OTA_AXON_<ID>_* macros in the public header",
    )
    inspect.add_argument(
        "--private-header",
        type=Path,
        required=True,
        help="output header for the model image build",
    )
    inspect.add_argument(
        "--public-header",
        type=Path,
        required=True,
        help="output header with the slot capacities for the application",
    )
    inspect.add_argument(
        "--persistent-vars-cap",
        type=int,
        help="persistent-vars capacity in int32_t entries reserved by the firmware "
        "(default: what the probe model requires)",
    )
    inspect.add_argument(
        "--allocate-packed-output",
        action="store_true",
        help="allocate app-owned storage for the model's packed-output buffer and wire it "
        "into the linked partition image (default: image links with packed_output_buf "
        "NULL)",
    )
    inspect.set_defaults(func=cmd_inspect)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
