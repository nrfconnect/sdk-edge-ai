#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

"""Unit tests for elf_symbols.

TestSymbolIndex works on hand-built symbols. TestCompiledElf works on Cortex-M33 objects and
linked ELFs built by $ARM_CC, else arm-zephyr-eabi-gcc (NCS toolchain), else arm-none-eabi-gcc.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from elf_symbols import (
    ElfSymbol,
    SymbolIndex,
    load_symbol_index,
    lookup_symbol,
    read_symbol_bytes,
)

ARM_CC = (
    os.environ.get("ARM_CC")
    or shutil.which("arm-zephyr-eabi-gcc")
    or shutil.which("arm-none-eabi-gcc")
)


class TestSymbolIndex:
    """SymbolIndex behavior that needs no compiler."""

    @staticmethod
    def _symbol(name: str, section_index: int | str = 1) -> ElfSymbol:
        return ElfSymbol(name, 0, 4, "STB_GLOBAL", "STT_OBJECT", section_index)

    def test_unique_rejects_duplicate_definitions(self) -> None:
        index = SymbolIndex.build([self._symbol("model_ota_dup"), self._symbol("model_ota_dup", 2)])

        with pytest.raises(ValueError, match="found 2"):
            index.unique("model_ota_dup")

    def test_unique_ignores_undefined_entries(self) -> None:
        defined = self._symbol("model_ota_mixed", 1)
        index = SymbolIndex.build([self._symbol("model_ota_mixed", "SHN_UNDEF"), defined])

        entry = index.unique("model_ota_mixed")

        assert entry is defined


@pytest.mark.skipif(not ARM_CC and "CI" not in os.environ, reason="no ARM cross compiler available")
class TestCompiledElf:
    """Reading objects and linked ELFs built from PROBE_C."""

    CFLAGS = ["-mcpu=cortex-m33", "-mthumb", "-Os", "-fdata-sections", "-ffunction-sections"]

    TEXT_BASE = 0x10000
    DATA_BASE = 0x20000000
    EXTERNAL_ADDRESS = 0x20001000

    PROBE_C = """
#include <stdint.h>
const uint32_t model_ota_zero = 0u;
const uint32_t model_ota_high_bit = 0xbfd5cb73u;
const uint16_t model_ota_small = 1;
uint32_t model_ota_writable = 7u;
uint32_t model_ota_bss;
__attribute__((common)) uint32_t model_ota_common;
__attribute__((used)) static const uint32_t model_ota_local = 5u;
extern uint32_t model_ota_external;
uint32_t model_ota_use(void) { return model_ota_external; }
"""

    LINKER_SCRIPT = f"""
SECTIONS
{{
    . = {TEXT_BASE:#x};
    .text : {{ *(.text*) }}
    .rodata : {{ *(.rodata*) }}
    . = {DATA_BASE:#x};
    .data : {{ *(.data*) }}
    .bss : {{ *(.bss*) *(COMMON) }}
}}
"""

    @classmethod
    def _compile(cls, sources: Path, output: Path, *flags: str) -> Path:
        subprocess.run(
            [ARM_CC, *cls.CFLAGS, *flags, str(sources / "probe.c"), "-o", str(output)],
            check=True,
            capture_output=True,
        )
        return output

    @classmethod
    def _link_flags(cls, sources: Path) -> list[str]:
        return [
            "-nostdlib",
            "-T",
            str(sources / "probe.ld"),
            "-Wl,-e,0",
            f"-Wl,--defsym=model_ota_external={cls.EXTERNAL_ADDRESS:#x}",
        ]

    @pytest.fixture(name="sources", scope="class")
    def sources_fixture(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        """Directory holding probe.c and probe.ld, shared by every build below."""
        if not ARM_CC:
            pytest.fail("no ARM cross compiler found; set ARM_CC or install arm-none-eabi-gcc")
        directory = tmp_path_factory.mktemp("sources")
        (directory / "probe.c").write_text(self.PROBE_C, encoding="utf-8")
        (directory / "probe.ld").write_text(self.LINKER_SCRIPT, encoding="utf-8")
        return directory

    @pytest.fixture(name="obj", scope="class")
    def obj_fixture(self, sources: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
        """Relocatable .o: sections at address 0, model_ota_external undefined."""
        return self._compile(sources, tmp_path_factory.mktemp("obj") / "probe.o", "-c")

    @pytest.fixture(name="linked", scope="class")
    def linked_fixture(self, sources: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
        """Linked at TEXT_BASE/DATA_BASE, model_ota_external absolute at EXTERNAL_ADDRESS."""
        output = tmp_path_factory.mktemp("linked") / "probe.elf"
        return self._compile(sources, output, *self._link_flags(sources))

    @pytest.fixture(name="stripped", scope="class")
    def stripped_fixture(self, sources: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
        """The linked ELF without .symtab."""
        output = tmp_path_factory.mktemp("stripped") / "probe.elf"
        return self._compile(sources, output, *self._link_flags(sources), "-Wl,-s")

    @pytest.fixture(name="elf", params=["obj", "linked"])
    def elf_fixture(self, request: pytest.FixtureRequest) -> Path:
        """Each of the .o and the linked ELF in turn."""
        return request.getfixturevalue(request.param)

    def test_reads_a_zero_value(self, elf: Path) -> None:
        """GCC emits a zero as `.space 4`; it is still four bytes of section data."""
        symbol = "model_ota_zero"

        data = read_symbol_bytes(elf, symbol)

        assert data == b"\0\0\0\0"

    @pytest.mark.parametrize(
        ("symbol", "expected"),
        [
            ("model_ota_high_bit", (0xBFD5CB73).to_bytes(4, "little")),
            ("model_ota_small", b"\x01\x00"),
            ("model_ota_writable", (7).to_bytes(4, "little")),
        ],
    )
    def test_reads_symbol_bytes(self, elf: Path, symbol: str, expected: bytes) -> None:
        data = read_symbol_bytes(elf, symbol)

        assert data == expected

    def test_missing_symbol_is_rejected(self, elf: Path) -> None:
        symbol = "model_ota_not_there"

        with pytest.raises(ValueError, match="no defined model_ota_not_there"):
            read_symbol_bytes(elf, symbol)

    def test_undefined_symbol_is_rejected(self, obj: Path) -> None:
        symbol = "model_ota_external"

        with pytest.raises(ValueError, match="no defined model_ota_external"):
            read_symbol_bytes(obj, symbol)

    def test_absolute_symbol_is_rejected(self, linked: Path) -> None:
        symbol = "model_ota_external"

        with pytest.raises(ValueError, match="not in a normal section"):
            read_symbol_bytes(linked, symbol)

    def test_common_symbol_is_rejected(self, obj: Path) -> None:
        symbol = "model_ota_common"

        with pytest.raises(ValueError, match="not in a normal section"):
            read_symbol_bytes(obj, symbol)

    def test_bss_symbol_is_rejected(self, elf: Path) -> None:
        symbol = "model_ota_bss"

        with pytest.raises(ValueError, match="no file data"):
            read_symbol_bytes(elf, symbol)

    def test_linked_common_symbol_is_rejected(self, linked: Path) -> None:
        """The linker places COMMON into .bss, so it has no file data either."""
        symbol = "model_ota_common"

        with pytest.raises(ValueError, match="no file data"):
            read_symbol_bytes(linked, symbol)

    def test_function_symbol_is_rejected(self, elf: Path) -> None:
        """A Thumb function's address has bit 0 set, so its bytes would be misaligned."""
        symbol = "model_ota_use"

        with pytest.raises(ValueError, match="not a data object"):
            read_symbol_bytes(elf, symbol)

    def test_missing_symtab_is_rejected(self, stripped: Path) -> None:
        with pytest.raises(ValueError, match="missing ELF .symtab"):
            load_symbol_index(stripped)

    def test_linked_symbol_has_its_address(self, linked: Path) -> None:
        symbol = "model_ota_writable"

        entry = lookup_symbol(linked, symbol)

        assert entry is not None
        assert entry.address == self.DATA_BASE

    def test_linked_thumb_function_has_bit_0_set(self, linked: Path) -> None:
        symbol = "model_ota_use"

        entry = lookup_symbol(linked, symbol)

        assert entry is not None
        assert entry.address == self.TEXT_BASE | 1
        assert entry.is_function
        assert not entry.is_object

    def test_linked_absolute_symbol_has_no_writability(self, linked: Path) -> None:
        symbol = "model_ota_external"

        entry = lookup_symbol(linked, symbol)

        assert entry is not None
        assert entry.address == self.EXTERNAL_ADDRESS
        assert entry.read_only is None

    def test_index_groups_by_name(self, obj: Path) -> None:
        defined, undefined, missing = (
            "model_ota_high_bit",
            "model_ota_external",
            "model_ota_not_there",
        )

        index = load_symbol_index(obj)

        assert [entry.name for entry in index.defined(defined)] == [defined]
        assert index.defined(undefined) == []
        assert index.defined(missing) == []
        assert undefined in {entry.name for entry in index.symbols}

    def test_lookup_defined_symbol(self, elf: Path) -> None:
        symbol = "model_ota_high_bit"

        entry = lookup_symbol(elf, symbol)

        assert entry is not None
        assert entry.name == symbol
        assert entry.is_global
        assert entry.is_object
        assert not entry.is_function
        assert entry.size == 4

    def test_lookup_local_symbol(self, elf: Path) -> None:
        symbol = "model_ota_local"

        entry = lookup_symbol(elf, symbol)

        assert entry is not None
        assert not entry.is_global

    def test_lookup_undefined_symbol_is_none(self, obj: Path) -> None:
        symbol = "model_ota_external"

        entry = lookup_symbol(obj, symbol)

        assert entry is None

    def test_lookup_missing_symbol_is_none(self, elf: Path) -> None:
        symbol = "model_ota_not_there"

        entry = lookup_symbol(elf, symbol)

        assert entry is None

    def test_read_only_follows_section_flags(self, elf: Path) -> None:
        index = load_symbol_index(elf)

        read_only = index.defined("model_ota_high_bit")[0].read_only
        writable = index.defined("model_ota_writable")[0].read_only

        assert read_only is True
        assert writable is False
