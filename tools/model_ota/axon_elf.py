#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Inspect Axon probe ELFs and resolve app symbols for model images."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

OP_EXTENSION_PREFIX = "nrf_axon_nn_op_extension_"
INTERLAYER_BUFFER_SYMBOL = "nrf_axon_interlayer_buffer"
MODEL_SIZE_MARKER = "model_ota_axon_compiled_model_size"
CONFIG_SCHEMA_VERSION = 1

# Undefined linker/compiler implementation symbols are not app dependencies.
# Keep this explicit: generated model symbols beginning with "__" must not be
# silently discarded.
_UNDEFINED_ALLOWLIST = frozenset(
    {
        "__model_image_end",
        "__aeabi_read_tp",
        "__stack_chk_fail",
        "__stack_chk_guard",
        "_GLOBAL_OFFSET_TABLE_",
    }
)


def _bootstrap_site_packages() -> None:
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    base = Path(sys.executable).resolve().parent.parent / "lib" / f"python{ver}" / "site-packages"
    if base.is_dir() and str(base) not in sys.path:
        sys.path.insert(0, str(base))


_bootstrap_site_packages()

try:
    from elftools.elf.elffile import ELFFile
    from elftools.elf.sections import SymbolTableSection
except ImportError as exc:  # pragma: no cover - depends on host Python
    ELFFile = None  # type: ignore[misc, assignment]
    SymbolTableSection = None  # type: ignore[misc, assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class ElfSymbol:
    __slots__ = ("name", "address", "size", "bind", "type", "section_index", "read_only")

    def __init__(
        self,
        name: str,
        address: int,
        size: int,
        bind: str,
        type: str,
        section_index: int | str,
        read_only: bool | None = None,
    ) -> None:
        self.name = name
        self.address = address
        self.size = size
        self.bind = bind
        self.type = type
        self.section_index = section_index
        self.read_only = read_only

    @property
    def is_undefined(self) -> bool:
        return self.section_index in ("SHN_UNDEF", "UND")

    @property
    def is_global(self) -> bool:
        return self.bind in ("STB_GLOBAL", "GLOBAL")

    @property
    def is_function(self) -> bool:
        return self.type in ("STT_FUNC", "FUNC")

    @property
    def is_object(self) -> bool:
        return self.type in ("STT_OBJECT", "OBJECT")


@dataclass(frozen=True)
class SymbolIndex:
    symbols: tuple[ElfSymbol, ...]
    by_name: dict[str, tuple[ElfSymbol, ...]]

    @classmethod
    def build(cls, symbols: Iterable[ElfSymbol]) -> "SymbolIndex":
        items = tuple(symbols)
        grouped: dict[str, list[ElfSymbol]] = {}
        for symbol in items:
            grouped.setdefault(symbol.name, []).append(symbol)
        return cls(items, {name: tuple(entries) for name, entries in grouped.items()})

    def defined(self, name: str) -> list[ElfSymbol]:
        return [entry for entry in self.by_name.get(name, ()) if not entry.is_undefined]


@dataclass(frozen=True)
class ProbeMetadata:
    model_symbol: str
    persistent_required: int
    persistent_cap: int
    persistent_symbol: str | None
    packed_output_bytes: int
    packed_output_symbol: str | None
    packed_output_allocated: bool
    keep_symbols: tuple[str, ...]


def require_pyelftools() -> None:
    if ELFFile is None:
        raise ImportError("pyelftools is required (use NCS toolchain Python)") from _IMPORT_ERROR


def load_symbol_index(elf_path: Path) -> SymbolIndex:
    """Read .symtab once, including section writability where available."""
    require_pyelftools()
    with elf_path.open("rb") as handle:
        elffile = ELFFile(handle)
        symtab = elffile.get_section_by_name(".symtab")
        if symtab is None or not isinstance(symtab, SymbolTableSection):
            raise ValueError(f"{elf_path}: missing ELF .symtab")
        symbols: list[ElfSymbol] = []
        for entry in symtab.iter_symbols():
            if not entry.name:
                continue
            section_index = entry["st_shndx"]
            read_only: bool | None = None
            if isinstance(section_index, int) and 0 <= section_index < elffile.num_sections():
                section = elffile.get_section(section_index)
                read_only = not bool(int(section["sh_flags"]) & 0x1)  # SHF_WRITE
            symbols.append(
                ElfSymbol(
                    entry.name,
                    int(entry["st_value"]),
                    int(entry["st_size"]),
                    str(entry["st_info"]["bind"]),
                    str(entry["st_info"]["type"]),
                    section_index,
                    read_only,
                )
            )
    return SymbolIndex.build(symbols)


def lookup_symbol(
    elf_path: Path, symbol: str, index: SymbolIndex | None = None
) -> ElfSymbol | None:
    entries = (index or load_symbol_index(elf_path)).by_name.get(symbol, ())
    return next((entry for entry in entries if not entry.is_undefined), entries[0] if entries else None)


def _undefined_global_symbols(index: SymbolIndex) -> list[str]:
    return sorted(
        {
            entry.name
            for entry in index.symbols
            if entry.is_undefined
            and entry.is_global
            and entry.name not in _UNDEFINED_ALLOWLIST
        }
    )


def collect_app_symbols_from_object(
    elf_path: Path, index: SymbolIndex | None = None
) -> list[str]:
    symbols = set(_undefined_global_symbols(index or load_symbol_index(elf_path)))
    symbols.add(INTERLAYER_BUFFER_SYMBOL)
    return sorted(symbols)


def _defined_storage(
    index: SymbolIndex, suffix: str, description: str
) -> ElfSymbol | None:
    matches = [
        entry
        for entry in index.symbols
        if not entry.is_undefined
        and entry.is_global
        and entry.is_object
        and entry.size > 0
        and entry.name.startswith("axon_model_")
        and entry.name.endswith(suffix)
    ]
    if len(matches) > 1:
        names = ", ".join(sorted(entry.name for entry in matches))
        raise ValueError(f"multiple {description} symbols in probe: {names}")
    return matches[0] if matches else None


def _known_storage_symbol(entry: ElfSymbol) -> bool:
    return (
        entry.name == INTERLAYER_BUFFER_SYMBOL
        or entry.name.startswith("axon_model_")
        and entry.name.endswith(("_persistent_vars", "_packed_output_buf"))
    )


def discover_model_symbol(index: SymbolIndex, override: str | None = None) -> str:
    markers = index.defined(MODEL_SIZE_MARKER)
    if len(markers) != 1:
        raise ValueError(
            f"probe must define exactly one global {MODEL_SIZE_MARKER} marker"
        )
    marker = markers[0]
    if not marker.is_global or not marker.is_object or marker.size <= 0:
        raise ValueError(
            f"{MODEL_SIZE_MARKER} must be a non-empty defined global STT_OBJECT"
        )

    candidates = [
        entry
        for entry in index.symbols
        if not entry.is_undefined
        and entry.is_global
        and entry.is_object
        and entry.size == marker.size
        and entry.name != MODEL_SIZE_MARKER
        and not _known_storage_symbol(entry)
    ]
    read_only = [entry for entry in candidates if entry.read_only is True]
    if read_only:
        candidates = read_only
    unique = candidates[0] if len(candidates) == 1 else None

    if override is not None:
        override_entries = index.defined(override)
        if len(override_entries) != 1:
            raise ValueError(f"model symbol override {override!r} is not uniquely defined")
        selected = override_entries[0]
        if not selected.is_global or not selected.is_object or selected.size != marker.size:
            raise ValueError(
                f"model symbol override {override!r} must be a defined global STT_OBJECT "
                f"of size {marker.size}"
            )
        if unique is not None and selected.name != unique.name:
            raise ValueError(
                f"model symbol override {override!r} does not match discovered "
                f"symbol {unique.name!r}"
            )
        return selected.name

    if unique is not None:
        return unique.name
    names = ", ".join(sorted(entry.name for entry in candidates)) or "none"
    raise ValueError(
        f"could not uniquely discover model symbol (size {marker.size}; candidates: {names}); "
        "pass --model-sym"
    )


def inspect_symbols(
    symbols: Iterable[ElfSymbol],
    *,
    persistent_vars_cap: int | None = None,
    model_sym: str | None = None,
    allocate_packed_output: bool = False,
) -> ProbeMetadata:
    index = SymbolIndex.build(symbols)
    persistent = _defined_storage(index, "_persistent_vars", "persistent-vars")
    packed = _defined_storage(index, "_packed_output_buf", "packed-output")
    if persistent is not None and persistent.size % 4:
        raise ValueError(
            f"{persistent.name} size {persistent.size} is not a multiple of sizeof(int32_t)"
        )
    if packed is not None and packed.size % 4:
        raise ValueError(
            f"{packed.name} size {packed.size} is not a multiple of sizeof(uint32_t)"
        )

    required = persistent.size // 4 if persistent else 0
    cap = required if persistent_vars_cap is None else persistent_vars_cap
    if cap < 0:
        raise ValueError("persistent-vars cap cannot be negative")
    if required > cap:
        raise ValueError(f"persistent-vars cap {cap} < required {required}")
    if allocate_packed_output and packed is None:
        raise ValueError("allocate_packed_output requested but no packed-output buffer found")

    # By default (allocate_packed_output=False), packed.name (axon_model_*_packed_output_buf)
    # is excluded from app_symbols/keep_symbols: the OTA image is linked without
    # NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER, so its packed_output_buf field is NULL and
    # no app storage for it is ever referenced. Only packed_output_bytes (a size, not a symbol)
    # is exposed, via the public header. Callers that opt in get real app-owned storage kept
    # alive and referenced by the linked model's packed_output_buf field.
    packed_bytes = packed.size if packed else 0
    app_symbols = set(_undefined_global_symbols(index))
    app_symbols.add(INTERLAYER_BUFFER_SYMBOL)
    if persistent is not None:
        app_symbols.add(persistent.name)
    if allocate_packed_output and packed is not None:
        app_symbols.add(packed.name)
    return ProbeMetadata(
        model_symbol=discover_model_symbol(index, model_sym),
        persistent_required=required,
        persistent_cap=cap,
        persistent_symbol=persistent.name if persistent else None,
        packed_output_bytes=packed_bytes,
        packed_output_symbol=packed.name if packed else None,
        packed_output_allocated=allocate_packed_output,
        keep_symbols=tuple(sorted(app_symbols)),
    )


def inspect_probe(
    probe: Path,
    *,
    persistent_vars_cap: int | None = None,
    model_sym: str | None = None,
    allocate_packed_output: bool = False,
) -> ProbeMetadata:
    return inspect_symbols(
        load_symbol_index(probe).symbols,
        persistent_vars_cap=persistent_vars_cap,
        model_sym=model_sym,
        allocate_packed_output=allocate_packed_output,
    )


def _c_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def sanitize_model_id(model_id: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_").upper()
    if not token:
        raise ValueError("model ID must contain an ASCII letter or digit")
    if token[0].isdigit():
        token = "_" + token
    return token


def render_private_header(header_name: str, metadata: ProbeMetadata) -> str:
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
    ]
    if metadata.persistent_symbol:
        lines.append(
            f"#define MODEL_OTA_AXON_PERSISTENT_VARS_SYM {metadata.persistent_symbol}"
        )
    if metadata.packed_output_allocated and metadata.packed_output_symbol:
        lines.append(
            f"#define MODEL_OTA_AXON_PACKED_OUTPUT_SYM {metadata.packed_output_symbol}"
        )
    if metadata.keep_symbols:
        lines.append("#define MODEL_OTA_AXON_KEEP_REFS(X) \\")
        for pos, symbol in enumerate(metadata.keep_symbols):
            continuation = " \\" if pos + 1 < len(metadata.keep_symbols) else ""
            lines.append(f"\tX({symbol}){continuation}")
    else:
        lines.append("#define MODEL_OTA_AXON_KEEP_REFS(X)")
    return "\n".join(lines) + "\n"


def render_public_header(model_id: str, packed_output_bytes: int) -> str:
    token = sanitize_model_id(model_id)
    return (
        "/* Generated by axon_elf.py; do not edit. */\n"
        "#pragma once\n"
        f"#define MODEL_OTA_AXON_{token}_PACKED_OUTPUT_BYTES {packed_output_bytes}\n"
    )


def atomic_write_if_changed(path: Path, data: str | bytes) -> bool:
    payload = data.encode("utf-8") if isinstance(data, str) else data
    try:
        if path.read_bytes() == payload:
            return False
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
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return True


def build_provide_script(
    elf_path: Path,
    symbols: Sequence[str],
    index: SymbolIndex | None = None,
) -> tuple[str, list[str]]:
    symbol_index = index or load_symbol_index(elf_path)
    lines: list[str] = []
    missing: list[str] = []
    for symbol in symbols:
        entry = lookup_symbol(elf_path, symbol, symbol_index)
        if entry is None or entry.is_undefined:
            missing.append(symbol)
            continue
        address = entry.address
        if symbol.startswith(OP_EXTENSION_PREFIX) and entry.is_function:
            address |= 1
        lines.append(f"PROVIDE({symbol} = 0x{address:X});")
    return "\n".join(lines) + ("\n" if lines else ""), missing


def cmd_inspect(args: argparse.Namespace) -> int:
    if not args.probe.is_file():
        raise ValueError(f"probe object not found: {args.probe}")
    metadata = inspect_probe(
        args.probe,
        persistent_vars_cap=args.persistent_vars_cap,
        model_sym=args.model_sym,
        allocate_packed_output=args.allocate_packed_output,
    )
    private = render_private_header(args.header_name, metadata)
    public = render_public_header(args.model_id, metadata.packed_output_bytes)
    atomic_write_if_changed(args.private_header, private)
    atomic_write_if_changed(args.public_header, public)
    return 0


def cmd_provide(args: argparse.Namespace) -> int:
    if not args.object.is_file():
        raise ValueError(f"object not found: {args.object}")
    if not args.elf.is_file():
        raise ValueError(f"ELF not found: {args.elf}")
    object_index = load_symbol_index(args.object)
    symbols = collect_app_symbols_from_object(args.object, object_index)
    script, missing = build_provide_script(args.elf, symbols)
    if missing:
        raise ValueError(f"symbols not found in {args.elf}: {', '.join(missing)}")
    atomic_write_if_changed(args.output, script)
    return 0


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    inspect = sub.add_parser("inspect", help="inspect one Axon probe and emit C headers")
    inspect.add_argument("--probe", type=Path, required=True)
    inspect.add_argument("--header-name", required=True)
    inspect.add_argument("--model-id", required=True)
    inspect.add_argument("--private-header", type=Path, required=True)
    inspect.add_argument("--public-header", type=Path, required=True)
    inspect.add_argument("--persistent-vars-cap", type=int)
    inspect.add_argument("--model-sym")
    inspect.add_argument(
        "--allocate-packed-output",
        action="store_true",
        help="allocate app-owned storage for the model's packed-output buffer and wire it "
             "into the linked partition image (default: image links with packed_output_buf "
             "NULL)",
    )
    inspect.set_defaults(func=cmd_inspect)

    provide = sub.add_parser("provide", help="emit a PROVIDE() linker fragment")
    provide.add_argument("--object", type=Path, required=True)
    provide.add_argument("--elf", type=Path, required=True)
    provide.add_argument("-o", "--output", type=Path, required=True)
    provide.set_defaults(func=cmd_provide)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ImportError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
