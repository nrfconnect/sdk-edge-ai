#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Read compile-time constants back out of a compiled object.

The contract hash (include/model_ota/model_contract.h) mixes sizeof() of the runtime structs, and
neither the preprocessor nor a host reimplementation can fold that: only the compiler knows the
target layout. So the build compiles lib/model_ota/src/model_ota_contract_probe.c with the
application's own flags and the host reads the finished word out of that object, which is what
keeps model_ota_context.json carrying the compiler's value rather than a second implementation of
the hash.

Kept apart from axon_elf.py because every flavour needs it, including Neuton, which has nothing to
do with Axon model inspection. axon_elf.py takes its pyelftools access from here.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

#: The constant model_ota_contract_probe.c defines.
CONTRACT_HASH_SYMBOL = "model_ota_contract_hash"

_UNDEFINED_SECTIONS = ("SHN_UNDEF", "UND")


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
    _IMPORT_ERROR: ImportError | None = exc
else:
    _IMPORT_ERROR = None


def require_pyelftools() -> None:
    if ELFFile is None:
        raise ImportError("pyelftools is required (use NCS toolchain Python)") from _IMPORT_ERROR


def read_u32(elf_path: Path, symbol: str) -> int:
    """Value of a 4-byte object defined in @p elf_path.

    Handles a relocatable object, where st_value is the offset into the symbol's section, as well
    as a linked ELF, where it is an address and sh_addr the section base.
    """
    require_pyelftools()
    with elf_path.open("rb") as handle:
        elffile = ELFFile(handle)
        symtab = elffile.get_section_by_name(".symtab")
        if symtab is None or not isinstance(symtab, SymbolTableSection):
            raise ValueError(f"{elf_path}: missing ELF .symtab")
        defined = [
            entry
            for entry in symtab.iter_symbols()
            if entry.name == symbol and entry["st_shndx"] not in _UNDEFINED_SECTIONS
        ]
        if len(defined) != 1:
            raise ValueError(
                f"{elf_path}: expected exactly one defined {symbol}, found {len(defined)}"
            )
        entry = defined[0]
        size = int(entry["st_size"])
        if size != 4:
            raise ValueError(f"{elf_path}: {symbol} is {size} B, expected 4")
        section_index = entry["st_shndx"]
        if not isinstance(section_index, int):
            raise ValueError(f"{elf_path}: {symbol} is not in a normal section ({section_index})")
        section = elffile.get_section(section_index)
        offset = int(entry["st_value"]) - int(section["sh_addr"])
        data = section.data()[offset : offset + 4]
        if len(data) != 4:
            raise ValueError(
                f"{elf_path}: {symbol} at offset {offset} lies outside section {section.name}"
            )
    return int(struct.unpack("<I", data)[0])


def contract_hash_from_probe(probe: Path) -> int:
    """The contract hash the compiler folded for one model slot."""
    if not probe.is_file():
        raise ValueError(f"contract probe object not found: {probe}")
    return read_u32(probe, CONTRACT_HASH_SYMBOL)
