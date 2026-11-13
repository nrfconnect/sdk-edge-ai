#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

"""Shared ELF access layer for the model OTA host tools.

Wraps pyelftools to index the .symtab of an object or linked ELF and to read the bytes of a
defined symbol back out of its section. Every model flavor uses it, so it holds no Axon-specific
knowledge.
"""

import typing
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from elftools.elf.constants import SH_FLAGS
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection


@dataclass(frozen=True, slots=True)
class ElfSymbol:
    """One named .symtab entry, with pyelftools' names for binding, type and section."""

    name: str
    address: int
    size: int
    bind: str
    sym_type: str
    section_index: int | str
    read_only: bool | None = None

    @property
    def is_undefined(self) -> bool:
        """True when the symbol has no section in this ELF."""
        return self.section_index == "SHN_UNDEF"

    @property
    def is_global(self) -> bool:
        """True for global binding."""
        return self.bind == "STB_GLOBAL"

    @property
    def is_function(self) -> bool:
        """True for function symbols."""
        return self.sym_type == "STT_FUNC"

    @property
    def is_object(self) -> bool:
        """True for data object symbols."""
        return self.sym_type == "STT_OBJECT"


@dataclass(frozen=True)
class SymbolIndex:
    """All named symbols of one ELF, grouped by name."""

    symbols: tuple[ElfSymbol, ...]
    by_name: dict[str, tuple[ElfSymbol, ...]]

    @classmethod
    def build(cls, symbols: Iterable[ElfSymbol]) -> typing.Self:
        """Index symbols by name, keeping their order."""
        items = tuple(symbols)
        grouped: dict[str, list[ElfSymbol]] = {}
        for symbol in items:
            grouped.setdefault(symbol.name, []).append(symbol)
        return cls(items, {name: tuple(entries) for name, entries in grouped.items()})

    def defined(self, name: str) -> list[ElfSymbol]:
        """Defined entries called name."""
        return [entry for entry in self.by_name.get(name, ()) if not entry.is_undefined]

    def unique(self, name: str) -> ElfSymbol | None:
        """The one defined entry called name, or None; ValueError if there are several."""
        entries = self.defined(name)
        if len(entries) > 1:
            raise ValueError(f"expected at most one defined {name}, found {len(entries)}")
        return entries[0] if entries else None


@contextmanager
def _open_elf(elf_path: Path) -> Iterator[tuple[ELFFile, SymbolIndex]]:
    """Open elf_path and yield it with its symbol index. pyelftools reads lazily, so stay open."""
    with elf_path.open("rb") as handle:
        elffile = ELFFile(handle)
        symtab = elffile.get_section_by_name(".symtab")
        if not isinstance(symtab, SymbolTableSection):
            raise ValueError(f"{elf_path}: missing ELF .symtab")
        symbols: list[ElfSymbol] = []
        for entry in symtab.iter_symbols():
            if not entry.name:
                continue
            section_index = entry["st_shndx"]
            read_only: bool | None = None
            if isinstance(section_index, int):
                section = elffile.get_section(section_index)
                read_only = not section["sh_flags"] & SH_FLAGS.SHF_WRITE
            symbols.append(
                ElfSymbol(
                    entry.name,
                    entry["st_value"],
                    entry["st_size"],
                    entry["st_info"]["bind"],
                    entry["st_info"]["type"],
                    section_index,
                    read_only,
                )
            )
        yield elffile, SymbolIndex.build(symbols)


def load_symbol_index(elf_path: Path) -> SymbolIndex:
    """Read .symtab once, including section writability where available."""
    with _open_elf(elf_path) as (_, index):
        return index


def lookup_symbol(elf_path: Path, symbol: str) -> ElfSymbol | None:
    """The one defined entry for symbol, or None; ValueError if there are several."""
    try:
        return load_symbol_index(elf_path).unique(symbol)
    except ValueError as exc:
        raise ValueError(f"{elf_path}: {exc}") from exc


def read_symbol_bytes(elf_path: Path, symbol: str) -> bytes:
    """Contents of the object symbol defined in elf_path, st_size bytes long.

    Works on both a compiled object (.o) and a linked ELF. In a .o, a symbol's st_value is
    already its offset within the section and the section's sh_addr is 0; in a linked ELF,
    st_value is an absolute address and sh_addr is where the section starts. So
    st_value - sh_addr is the offset into the section data in both cases.
    """
    with _open_elf(elf_path) as (elffile, index):
        try:
            entry = index.unique(symbol)
        except ValueError as exc:
            raise ValueError(f"{elf_path}: {exc}") from exc
        if entry is None:
            raise ValueError(f"{elf_path}: no defined {symbol}")
        if not isinstance(entry.section_index, int):
            raise ValueError(
                f"{elf_path}: {symbol} is not in a normal section ({entry.section_index})"
            )
        if not entry.is_object:
            raise ValueError(f"{elf_path}: {symbol} is not a data object ({entry.sym_type})")
        section = elffile.get_section(entry.section_index)
        if section["sh_type"] == "SHT_NOBITS":
            raise ValueError(f"{elf_path}: {symbol} is in {section.name}, which has no file data")
        offset = entry.address - section["sh_addr"]
        data = section.data()
        if offset < 0 or offset + entry.size > len(data):
            raise ValueError(
                f"{elf_path}: {symbol} at offset {offset} lies outside section {section.name}"
            )
    return data[offset : offset + entry.size]
