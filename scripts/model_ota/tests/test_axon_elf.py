#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

"""Unit tests for axon_elf.

Every test works on hand-built symbols for a model called demo; FULL_PROBE holds every symbol
the naming contract allows, and _probe() derives variants from it.
"""

import dataclasses
import os
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import axon_elf
import pytest
from elf_symbols import ElfSymbol, SymbolIndex
from model_contract import symbol_name_hash


def _symbol(
    name: str,
    size: int,
    *,
    bind: str = "STB_GLOBAL",
    sym_type: str = "STT_OBJECT",
    section_index: int | str = 1,
    read_only: bool | None = True,
) -> ElfSymbol:
    return ElfSymbol(name, 0, size, bind, sym_type, section_index, read_only)


def _undefined(name: str, sym_type: str = "STT_NOTYPE", bind: str = "STB_GLOBAL") -> ElfSymbol:
    return _symbol(name, 0, bind=bind, sym_type=sym_type, section_index="SHN_UNDEF", read_only=None)


def _storage(name: str, size: int, **kwargs) -> ElfSymbol:
    return _symbol(name, size, read_only=False, **kwargs)


HEADER_NAME = "nrf_axon_model_demo_.h"

# Sizes are chosen so the derived counts differ: 8 B is 2 pointers, 16 B is 4 int32_t entries.
MODEL_CONST = _symbol("axon_model_const_demo", 8, bind="STB_LOCAL")
MODEL = _symbol("model_demo", 32)
INPUT_VECTORS = _storage("demo_input_vector_list", 8)
PERSISTENT = _storage("axon_model_demo_persistent_vars", 16)
PACKED = _storage("demo_packed_output_buf", 20)
DRIVER_CALL = _undefined("driver_call", "STT_FUNC")
IMAGE_END = _undefined("__model_image_end")

FULL_PROBE = [MODEL_CONST, MODEL, INPUT_VECTORS, PERSISTENT, PACKED, DRIVER_CALL, IMAGE_END]
MINIMAL_PROBE = [MODEL_CONST, MODEL, INPUT_VECTORS]
STORAGE = (INPUT_VECTORS, PERSISTENT, PACKED)

FULL_METADATA = axon_elf.ProbeMetadata(
    model_name="demo",
    model_symbol=MODEL.name,
    keep_symbols=(
        PERSISTENT.name,
        INPUT_VECTORS.name,
        DRIVER_CALL.name,
        axon_elf.INTERLAYER_BUFFER_SYMBOL,
    ),
    persistent_required=4,
    persistent_cap=8,
    persistent_symbol=PERSISTENT.name,
    packed_output_bytes=20,
    packed_output_symbol=PACKED.name,
    packed_output_allocated=False,
    input_vector_cnt=2,
    input_vector_list_symbol=INPUT_VECTORS.name,
)


def _probe(
    base: list[ElfSymbol] = FULL_PROBE,
    *,
    drop: list[str] | None = None,
    add: list[ElfSymbol] | None = None,
) -> SymbolIndex:
    """base without the symbols named in drop, plus add."""
    drop = drop or []
    return SymbolIndex.build([entry for entry in base if entry.name not in drop] + (add or []))


def _replace(symbol: ElfSymbol) -> SymbolIndex:
    """FULL_PROBE with the symbol of the same name swapped for symbol."""
    return _probe(drop=[symbol.name], add=[symbol])


class TestDiscoverModelName:
    """discover_model_name and the axon_model_const_<name> contract."""

    def test_name_is_the_const_suffix(self) -> None:
        index = _probe([_symbol("axon_model_const_tinyml_kws", 4, bind="STB_LOCAL")])

        model_name = axon_elf.discover_model_name(index)

        assert model_name == "tinyml_kws"

    @pytest.mark.parametrize(
        "symbols",
        (
            pytest.param([], id="missing"),
            pytest.param([_symbol(MODEL_CONST.name, 8)], id="global"),
            pytest.param(
                [
                    _symbol("axon_model_const_a", 4, bind="STB_LOCAL"),
                    _symbol("axon_model_const_b", 4, bind="STB_LOCAL"),
                ],
                id="duplicate",
            ),
        ),
    )
    def test_requires_exactly_one_local_const(self, symbols: list[ElfSymbol]) -> None:
        index = _probe(symbols)

        with pytest.raises(ValueError, match=r"exactly one local axon_model_const_\*"):
            axon_elf.discover_model_name(index)

    def test_rejects_empty_const(self) -> None:
        index = _probe([_symbol(MODEL_CONST.name, 0, bind="STB_LOCAL")])

        with pytest.raises(ValueError, match="axon_model_const_demo must be a non-empty object"):
            axon_elf.discover_model_name(index)

    @pytest.mark.parametrize("name", ("", "Demo", "9demo", "demo-v2"))
    def test_rejects_names_outside_contract(self, name: str) -> None:
        index = _probe([_symbol(f"axon_model_const_{name}", 4, bind="STB_LOCAL")])

        with pytest.raises(ValueError, match="not a lowercase ASCII identifier"):
            axon_elf.discover_model_name(index)


class TestDiscoverModelSymbol:
    """discover_model_symbol and the model_<name> contract."""

    def test_returns_model_name_symbol(self) -> None:
        index = _probe([MODEL])

        model_symbol = axon_elf.discover_model_symbol(index, "demo")

        assert model_symbol == MODEL.name

    @pytest.mark.parametrize(
        ("symbol", "match"),
        (
            pytest.param(None, "missing defined global model symbol 'model_demo'", id="missing"),
            pytest.param(
                _symbol(MODEL.name, 32, sym_type="STT_FUNC"), "global STT_OBJECT", id="func"
            ),
            pytest.param(
                _symbol(MODEL.name, 32, bind="STB_LOCAL"), "global STT_OBJECT", id="local"
            ),
            pytest.param(
                _symbol(MODEL.name, 32, read_only=False), "read-only section", id="writable"
            ),
            pytest.param(
                _symbol(MODEL.name, 32, read_only=None), "read-only section", id="no-section"
            ),
            pytest.param(_symbol(MODEL.name, 0), "non-empty object", id="empty"),
        ),
    )
    def test_rejects_invalid_model_symbol(self, symbol: ElfSymbol | None, match: str) -> None:
        index = _probe([] if symbol is None else [symbol])

        with pytest.raises(ValueError, match=match):
            axon_elf.discover_model_symbol(index, "demo")

    def test_wraps_duplicate_model_symbol_error(self) -> None:
        index = _probe([MODEL, _symbol(MODEL.name, 32, section_index=2)])

        with pytest.raises(ValueError, match=r"^model symbol 'model_demo': ") as excinfo:
            axon_elf.discover_model_symbol(index, "demo")

        assert isinstance(excinfo.value.__cause__, ValueError)


class TestInspectSymbols:
    """inspect_symbols on whole probes."""

    def test_full_probe(self) -> None:
        index = _probe()

        metadata = axon_elf.inspect_symbols(index, persistent_vars_cap=8)

        assert metadata == FULL_METADATA

    def test_minimal_probe(self) -> None:
        index = _probe(MINIMAL_PROBE)

        metadata = axon_elf.inspect_symbols(index)

        assert metadata.keep_symbols == (INPUT_VECTORS.name, axon_elf.INTERLAYER_BUFFER_SYMBOL)
        assert metadata.persistent_symbol is None
        assert (metadata.persistent_required, metadata.persistent_cap) == (0, 0)
        assert metadata.packed_output_symbol is None
        assert metadata.packed_output_bytes == 0

    def test_keep_symbols_leave_out_allowlisted_and_local_undefined(self) -> None:
        index = _probe(
            MINIMAL_PROBE,
            add=[DRIVER_CALL, IMAGE_END, _undefined("local_helper", bind="STB_LOCAL")],
        )

        metadata = axon_elf.inspect_symbols(index)

        assert metadata.keep_symbols == (
            INPUT_VECTORS.name,
            DRIVER_CALL.name,
            axon_elf.INTERLAYER_BUFFER_SYMBOL,
        )

    def test_allocate_packed_output_keeps_the_buffer(self) -> None:
        index = _probe()

        metadata = axon_elf.inspect_symbols(index, allocate_packed_output=True)

        assert metadata.packed_output_allocated
        assert PACKED.name in metadata.keep_symbols

    def test_allocate_packed_output_requires_the_buffer(self) -> None:
        index = _probe(MINIMAL_PROBE)

        with pytest.raises(ValueError, match="no packed-output buffer found"):
            axon_elf.inspect_symbols(index, allocate_packed_output=True)

    def test_missing_model_is_reported_before_storage(self) -> None:
        index = _probe(drop=[MODEL.name, INPUT_VECTORS.name])

        with pytest.raises(ValueError, match="model symbol 'model_demo'"):
            axon_elf.inspect_symbols(index)


class TestStorage:
    """Type, size and capacity checks on the app-owned storage symbols."""

    def test_input_vector_list_is_required(self) -> None:
        index = _probe(drop=[INPUT_VECTORS.name])

        with pytest.raises(ValueError, match="missing required symbol 'demo_input_vector_list'"):
            axon_elf.inspect_symbols(index)

    @pytest.mark.parametrize(
        ("symbol", "match"),
        (
            pytest.param(
                _storage(INPUT_VECTORS.name, 0), "non-zero multiple of the pointer", id="iv-0"
            ),
            pytest.param(
                _storage(INPUT_VECTORS.name, 6), "non-zero multiple of the pointer", id="iv-6"
            ),
            pytest.param(_storage(PERSISTENT.name, 6), r"sizeof\(int32_t\)", id="persistent"),
            pytest.param(_storage(PACKED.name, 6), r"sizeof\(uint32_t\)", id="packed"),
        ),
    )
    def test_rejects_bad_sizes(self, symbol: ElfSymbol, match: str) -> None:
        index = _replace(symbol)

        with pytest.raises(ValueError, match=match):
            axon_elf.inspect_symbols(index)

    @pytest.mark.parametrize("name", tuple(entry.name for entry in STORAGE))
    @pytest.mark.parametrize(
        ("overrides", "match"),
        (
            pytest.param({"bind": "STB_LOCAL"}, "defined global STT_OBJECT", id="local"),
            pytest.param({"sym_type": "STT_FUNC"}, "defined global STT_OBJECT", id="func"),
            pytest.param({"read_only": True}, "writable section", id="read-only"),
            pytest.param({"read_only": None}, "writable section", id="no-section"),
        ),
    )
    def test_rejects_bad_types(self, name: str, overrides: dict, match: str) -> None:
        index = _replace(_symbol(name, 8, **{"read_only": False, **overrides}))

        with pytest.raises(ValueError, match=rf"^storage symbol '{name}' .*{match}"):
            axon_elf.inspect_symbols(index)

    def test_wraps_duplicate_storage_error(self) -> None:
        index = _probe(add=[_storage(PERSISTENT.name, 8, section_index=2)])

        with pytest.raises(ValueError, match=rf"^storage symbol '{PERSISTENT.name}': ") as excinfo:
            axon_elf.inspect_symbols(index)

        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_persistent_cap_defaults_to_required(self) -> None:
        index = _probe()

        metadata = axon_elf.inspect_symbols(index)

        assert metadata.persistent_cap == 4

    @pytest.mark.parametrize(
        ("cap", "match"),
        (
            pytest.param(-1, "cannot be negative", id="negative"),
            pytest.param(3, "cap 3 < required 4", id="too-small"),
        ),
    )
    def test_rejects_bad_persistent_cap(self, cap: int, match: str) -> None:
        index = _probe()

        with pytest.raises(ValueError, match=match):
            axon_elf.inspect_symbols(index, persistent_vars_cap=cap)


class TestTokens:
    """Macro tokens for the slot ID and the Axon model name."""

    def test_model_token_is_uppercase_name(self) -> None:
        token = axon_elf.axon_model_token("tinyml_kws")

        assert token == "TINYML_KWS"

    @pytest.mark.parametrize(
        ("model_id", "expected"),
        (
            pytest.param("door bell-v2", "DOOR_BELL_V2", id="separators"),
            pytest.param("2nd", "_2ND", id="leading-digit"),
        ),
    )
    def test_slot_token(self, model_id: str, expected: str) -> None:
        token = axon_elf.sanitize_model_id(model_id)

        assert token == expected

    def test_slot_token_needs_a_letter_or_digit(self) -> None:
        with pytest.raises(ValueError, match="ASCII letter or digit"):
            axon_elf.sanitize_model_id("--")


class TestRenderPrivateHeader:
    """Contents of the header the model image build includes."""

    def test_defines(self) -> None:
        expected = (
            "#define MODEL_OTA_AXON_CONFIG_VERSION 0",
            '#define MODEL_OTA_AXON_HEADER "nrf_axon_model_demo_.h"',
            "#define MODEL_OTA_AXON_MODEL_SYM model_demo",
            "#define MODEL_OTA_AXON_PERSISTENT_VARS_REQUIRED 4",
            "#define MODEL_OTA_AXON_PERSISTENT_VARS_CAP 8",
            "#define MODEL_OTA_AXON_PERSISTENT_VARS_SYM axon_model_demo_persistent_vars",
            "#define MODEL_OTA_AXON_PACKED_OUTPUT_BYTES 20",
            "#define MODEL_OTA_AXON_PACKED_OUTPUT_ALLOC 0",
            "#define MODEL_OTA_AXON_INPUT_VECTOR_CAP 2",
            "#define MODEL_OTA_AXON_INPUT_VECTOR_LIST_SYM demo_input_vector_list",
            "#define MODEL_OTA_AXON_KEEP_SYMBOL_COUNT 4",
            f"#define MODEL_OTA_AXON_SYM_HASH_driver_call 0x{symbol_name_hash('driver_call'):08X}u",
            "#define MODEL_OTA_AXON_SYM_HASH(sym) MODEL_OTA_AXON_SYM_HASH_##sym",
        )

        lines = axon_elf.render_private_header(HEADER_NAME, FULL_METADATA).splitlines()

        for line in expected:
            assert line in lines

    def test_keep_refs(self) -> None:
        lines = axon_elf.render_private_header(HEADER_NAME, FULL_METADATA).splitlines()

        start = lines.index("#define MODEL_OTA_AXON_KEEP_REFS(X) \\")

        assert lines[start + 1 :] == [
            "\tX(axon_model_demo_persistent_vars) \\",
            "\tX(demo_input_vector_list) \\",
            "\tX(driver_call) \\",
            "\tX(nrf_axon_interlayer_buffer)",
        ]

    def test_omits_unallocated_packed_output(self) -> None:
        header = axon_elf.render_private_header(HEADER_NAME, FULL_METADATA)

        assert PACKED.name not in header

    def test_allocated_packed_output(self) -> None:
        metadata = dataclasses.replace(
            FULL_METADATA,
            packed_output_allocated=True,
            keep_symbols=(*FULL_METADATA.keep_symbols, PACKED.name),
        )

        lines = axon_elf.render_private_header(HEADER_NAME, metadata).splitlines()

        assert "#define MODEL_OTA_AXON_PACKED_OUTPUT_ALLOC 1" in lines
        assert "#define MODEL_OTA_AXON_PACKED_OUTPUT_SYM demo_packed_output_buf" in lines

    def test_rejects_control_characters(self) -> None:
        with pytest.raises(ValueError, match="control characters"):
            axon_elf.render_private_header("bad\nname.h", FULL_METADATA)


class TestRenderPublicHeader:
    """Contents of the header the application includes for one OTA slot."""

    def test_defines(self) -> None:
        capacities = """\
#define MODEL_OTA_AXON_DOOR_BELL_V2_PACKED_OUTPUT_BYTES 20
#define MODEL_OTA_AXON_DOOR_BELL_V2_PERSISTENT_VARS_CAP 8
"""
        reexport = """\
#ifndef NRF_AXON_MODEL_DEMO_PACKED_OUTPUT_SIZE
#define NRF_AXON_MODEL_DEMO_PACKED_OUTPUT_SIZE \\
\tMODEL_OTA_AXON_DOOR_BELL_V2_PACKED_OUTPUT_BYTES
#endif
"""

        header = axon_elf.render_public_header("door bell-v2", FULL_METADATA)

        assert capacities in header
        assert reexport in header


class TestAtomicWriteIfChanged:
    """atomic_write_if_changed on real files under tmp_path."""

    def test_creates_missing_file_and_parents(self, tmp_path: Path) -> None:
        path = tmp_path / "out" / "header.h"

        axon_elf.atomic_write_if_changed(path, "new\n")

        assert path.read_text(encoding="utf-8") == "new\n"

    def test_replaces_changed_file(self, tmp_path: Path) -> None:
        path = tmp_path / "header.h"
        path.write_text("old\n", encoding="utf-8")
        inode = path.stat().st_ino

        axon_elf.atomic_write_if_changed(path, "new\n")

        assert path.read_text(encoding="utf-8") == "new\n"
        assert path.stat().st_ino != inode

    def test_leaves_unchanged_file_alone(self, tmp_path: Path) -> None:
        path = tmp_path / "header.h"
        path.write_text("same\n", encoding="utf-8")
        inode = path.stat().st_ino

        with patch.object(axon_elf.tempfile, "mkstemp", wraps=axon_elf.tempfile.mkstemp) as mkstemp:
            axon_elf.atomic_write_if_changed(path, "same\n")

        mkstemp.assert_not_called()
        assert path.stat().st_ino == inode

    def test_removes_temporary_file_on_failure(self, tmp_path: Path) -> None:
        path = tmp_path / "header.h"
        path.write_text("old\n", encoding="utf-8")

        with (
            patch.object(axon_elf.os, "replace", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            axon_elf.atomic_write_if_changed(path, "new\n")

        assert os.listdir(tmp_path) == ["header.h"]
        assert path.read_text(encoding="utf-8") == "old\n"


class CliPaths(NamedTuple):
    """Files the inspect subcommand reads and writes."""

    probe: Path
    private: Path
    public: Path


class TestCli:
    """The inspect subcommand: argument wiring, with every collaborator mocked."""

    @pytest.fixture(name="paths")
    def paths_fixture(self, tmp_path: Path) -> CliPaths:
        """An empty probe file, which must exist to pass the is_file() check, and the outputs."""
        probe = tmp_path / "probe.o"
        probe.touch()
        return CliPaths(probe, tmp_path / "private.h", tmp_path / "public.h")

    @pytest.fixture(name="mocks")
    def mocks_fixture(self) -> Iterator[dict[str, MagicMock]]:
        """Every axon_elf function cmd_inspect calls, autospecced, keyed by name."""
        names = (
            "load_symbol_index",
            "inspect_symbols",
            "render_private_header",
            "render_public_header",
            "atomic_write_if_changed",
        )
        patchers = [patch.object(axon_elf, name, autospec=True) for name in names]
        mocks = dict(zip(names, (patcher.start() for patcher in patchers), strict=True))
        mocks["render_private_header"].return_value = "private"
        mocks["render_public_header"].return_value = "public"
        yield mocks
        for patcher in patchers:
            patcher.stop()

    @staticmethod
    def _argv(paths: CliPaths, *extra: str) -> tuple[str, ...]:
        return (
            "inspect",
            "--probe",
            str(paths.probe),
            "--header-name",
            HEADER_NAME,
            "--model-id",
            "demo",
            "--private-header",
            str(paths.private),
            "--public-header",
            str(paths.public),
            *extra,
        )

    def test_wires_arguments_through(self, mocks: dict[str, MagicMock], paths: CliPaths) -> None:
        argv = self._argv(paths, "--persistent-vars-cap", "8", "--allocate-packed-output")

        result = axon_elf.main(argv)

        assert result == 0
        mocks["load_symbol_index"].assert_called_once_with(paths.probe)
        mocks["inspect_symbols"].assert_called_once_with(
            mocks["load_symbol_index"].return_value,
            persistent_vars_cap=8,
            allocate_packed_output=True,
        )
        metadata = mocks["inspect_symbols"].return_value
        mocks["render_private_header"].assert_called_once_with(HEADER_NAME, metadata)
        mocks["render_public_header"].assert_called_once_with("demo", metadata)
        assert mocks["atomic_write_if_changed"].call_args_list == [
            ((paths.private, "private"),),
            ((paths.public, "public"),),
        ]

    def test_defaults(self, mocks: dict[str, MagicMock], paths: CliPaths) -> None:
        argv = self._argv(paths)

        axon_elf.main(argv)

        mocks["inspect_symbols"].assert_called_once_with(
            mocks["load_symbol_index"].return_value,
            persistent_vars_cap=None,
            allocate_packed_output=False,
        )

    @pytest.mark.parametrize(
        "model_id",
        (
            pytest.param("slot */ x", id="comment-end"),
            pytest.param("slot\nx", id="newline"),
        ),
    )
    def test_rejects_model_id_breaking_comment(
        self, mocks: dict[str, MagicMock], paths: CliPaths, model_id: str
    ) -> None:
        argv = self._argv(paths, "--model-id", model_id)

        with pytest.raises(SystemExit) as excinfo:
            axon_elf.main(argv)

        assert excinfo.value.code == 2
        mocks["load_symbol_index"].assert_not_called()

    def test_missing_probe_exits_with_usage_error(
        self, mocks: dict[str, MagicMock], paths: CliPaths
    ) -> None:
        paths.probe.unlink()
        argv = self._argv(paths)

        with pytest.raises(SystemExit) as excinfo:
            axon_elf.main(argv)

        assert excinfo.value.code == 2
        mocks["load_symbol_index"].assert_not_called()

    def test_contract_error_exits_with_usage_error(
        self,
        mocks: dict[str, MagicMock],
        paths: CliPaths,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mocks["inspect_symbols"].side_effect = ValueError("mocked error")
        argv = self._argv(paths)

        with pytest.raises(SystemExit) as excinfo:
            axon_elf.main(argv)

        assert excinfo.value.code == 2
        assert "mocked error" in capsys.readouterr().err
        mocks["atomic_write_if_changed"].assert_not_called()
