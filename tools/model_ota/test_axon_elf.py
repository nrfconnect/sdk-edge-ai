#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Focused unit tests for Axon OTA Python tooling."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import axon_elf
from axon_elf import ElfSymbol, ProbeMetadata, SymbolIndex


def symbol(
    name: str,
    size: int,
    *,
    bind: str = "GLOBAL",
    type: str = "OBJECT",
    section: int | str = 1,
    read_only: bool | None = True,
    address: int = 0,
) -> ElfSymbol:
    return ElfSymbol(name, address, size, bind, type, section, read_only)


def basic_probe_symbols() -> list[ElfSymbol]:
    return [
        symbol(axon_elf.MODEL_SIZE_MARKER, 32),
        symbol("generated_model", 32),
        symbol("axon_model_demo_persistent_vars", 16, read_only=False),
        symbol("axon_model_demo_packed_output_buf", 20, read_only=False),
        symbol("driver_call", 0, section="UND", type="FUNC", read_only=None),
        symbol("__model_image_end", 0, section="UND", read_only=None),
    ]


class InspectOutputTests(unittest.TestCase):
    def test_inspect_emits_private_public_and_keep_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe = root / "probe.o"
            private = root / "private.h"
            public = root / "public.h"
            keep_json = root / "keep.json"
            probe.write_bytes(b"not read due to mock")
            index = SymbolIndex.build(basic_probe_symbols())
            with patch("axon_elf.load_symbol_index", return_value=index):
                result = axon_elf.main(
                    [
                        "inspect",
                        "--probe",
                        str(probe),
                        "--header-name",
                        "nrf_axon_model_demo.h",
                        "--model-id",
                        "door bell-v2",
                        "--private-header",
                        str(private),
                        "--public-header",
                        str(public),
                        "--keep-json",
                        str(keep_json),
                        "--persistent-vars-cap",
                        "8",
                        "--partition-addr",
                        "0x102000",
                    ]
                )
            self.assertEqual(result, 0)
            private_text = private.read_text()
            self.assertIn("#define MODEL_OTA_AXON_CONFIG_VERSION 1", private_text)
            self.assertIn(
                '#define MODEL_OTA_AXON_HEADER "nrf_axon_model_demo.h"', private_text
            )
            self.assertIn("#define MODEL_OTA_AXON_MODEL_SYM generated_model", private_text)
            self.assertIn("#define MODEL_OTA_AXON_PERSISTENT_VARS_REQUIRED 4", private_text)
            self.assertIn("#define MODEL_OTA_AXON_PERSISTENT_VARS_CAP 8", private_text)
            self.assertIn(
                "#define MODEL_OTA_AXON_PACKED_OUTPUT_BYTES 20", private_text
            )
            self.assertIn("\tX(driver_call) \\", private_text)
            self.assertIn("\tX(axon_model_demo_persistent_vars) \\", private_text)
            self.assertIn("#define MODEL_OTA_AXON_KEEP_SYMBOL_COUNT 3", private_text)
            self.assertIn("#define MODEL_OTA_AXON_SYM_HASH_driver_call", private_text)
            self.assertNotIn("__model_image_end", private_text)
            self.assertIn("#define MODEL_OTA_AXON_PACKED_OUTPUT_ALLOC 0", private_text)
            self.assertNotIn("axon_model_demo_packed_output_buf", private_text)
            self.assertNotIn("MODEL_OTA_AXON_PACKED_OUTPUT_SYM", private_text)

            public_text = public.read_text()
            self.assertIn(
                "#define MODEL_OTA_AXON_DOOR_BELL_V2_PACKED_OUTPUT_BYTES 20",
                public_text,
            )
            self.assertIn(
                "#define MODEL_OTA_AXON_DOOR_BELL_V2_PERSISTENT_VARS_CAP 8",
                public_text,
            )
            self.assertIn("#define NRF_AXON_MODEL_DEMO_PACKED_OUTPUT_SIZE", public_text)
            self.assertNotIn("model_contract.h", public_text)
            self.assertNotIn("IMAGE_BASE", public_text)
            self.assertNotIn("KEEP_LABEL", public_text)
            self.assertNotIn("CONTRACT_HASH", public_text)

            payload = json.loads(keep_json.read_text())
            self.assertEqual(payload["target"], "door bell-v2")
            self.assertEqual(
                payload["keep_symbols"],
                [
                    "axon_model_demo_persistent_vars",
                    "driver_call",
                    "nrf_axon_interlayer_buffer",
                ],
            )

    def test_inspect_edgeai_public_header_is_app_facing_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe = root / "probe.o"
            private = root / "private.h"
            public = root / "public.h"
            probe.write_bytes(b"not read due to mock")
            index = SymbolIndex.build(basic_probe_symbols())
            with patch("axon_elf.load_symbol_index", return_value=index):
                result = axon_elf.main(
                    [
                        "inspect",
                        "--probe",
                        str(probe),
                        "--header-name",
                        "nrf_axon_model_demo.h",
                        "--model-id",
                        "demo",
                        "--private-header",
                        str(private),
                        "--public-header",
                        str(public),
                        "--partition-addr",
                        "0x102000",
                        "--edgeai",
                    ]
                )
            self.assertEqual(result, 0)
            public_text = public.read_text()
            self.assertIn("#define MODEL_OTA_AXON_DEMO_PACKED_OUTPUT_BYTES 20", public_text)
            self.assertNotIn("CONTRACT_HASH", public_text)
            self.assertNotIn("IMAGE_BASE", public_text)

    def test_inspect_allocate_packed_output_wires_app_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe = root / "probe.o"
            private = root / "private.h"
            public = root / "public.h"
            probe.write_bytes(b"not read due to mock")
            index = SymbolIndex.build(basic_probe_symbols())
            with patch("axon_elf.load_symbol_index", return_value=index):
                result = axon_elf.main(
                    [
                        "inspect",
                        "--probe",
                        str(probe),
                        "--header-name",
                        "nrf_axon_model_demo.h",
                        "--model-id",
                        "demo",
                        "--private-header",
                        str(private),
                        "--public-header",
                        str(public),
                        "--persistent-vars-cap",
                        "8",
                        "--partition-addr",
                        "0x102000",
                        "--allocate-packed-output",
                    ]
                )
            self.assertEqual(result, 0)
            private_text = private.read_text()
            self.assertIn("#define MODEL_OTA_AXON_PACKED_OUTPUT_ALLOC 1", private_text)
            self.assertIn(
                "#define MODEL_OTA_AXON_PACKED_OUTPUT_SYM axon_model_demo_packed_output_buf",
                private_text,
            )
            self.assertIn("\tX(axon_model_demo_packed_output_buf) \\", private_text)

    def test_allocate_packed_output_without_buffer_fails(self) -> None:
        symbols = [
            symbol(axon_elf.MODEL_SIZE_MARKER, 32),
            symbol("generated_model", 32),
            symbol("driver_call", 0, section="UND", type="FUNC", read_only=None),
        ]
        with self.assertRaisesRegex(ValueError, "no packed-output buffer found"):
            axon_elf.inspect_symbols(symbols, allocate_packed_output=True)

    def test_duplicate_storage_candidates_fail(self) -> None:
        symbols = basic_probe_symbols() + [
            symbol("axon_model_other_persistent_vars", 8, read_only=False)
        ]
        with self.assertRaisesRegex(ValueError, "multiple persistent-vars"):
            axon_elf.inspect_symbols(symbols)


class BindingTableTests(unittest.TestCase):
    def test_binding_table_merges_and_dedups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keep_a = root / "a.json"
            keep_b = root / "b.json"
            out = root / "binding.h"
            keep_a.write_text(
                json.dumps(
                    {
                        "target": "a",
                        "keep_symbols": [
                            "nrf_axon_interlayer_buffer",
                            "driver_call",
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            keep_b.write_text(
                json.dumps(
                    {
                        "target": "b",
                        "keep_symbols": [
                            "nrf_axon_interlayer_buffer",
                            "axon_model_demo_persistent_vars",
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = axon_elf.main(
                [
                    "binding-table",
                    "--keep-json",
                    str(keep_a),
                    "--keep-json",
                    str(keep_b),
                    "-o",
                    str(out),
                ]
            )
            self.assertEqual(result, 0)
            text = out.read_text()
            self.assertIn("#define MODEL_OTA_AXON_BINDING_SYMBOL_COUNT 3", text)
            self.assertIn("MODEL_OTA_AXON_BINDING_REFS(X)", text)
            self.assertIn("X(nrf_axon_interlayer_buffer)", text)

    def test_binding_table_hash_collision_fails(self) -> None:
        metadata = ProbeMetadata(
            model_symbol="model_a",
            persistent_required=0,
            persistent_cap=0,
            persistent_symbol=None,
            packed_output_bytes=0,
            packed_output_symbol=None,
            packed_output_allocated=False,
            keep_symbols=("sym_a", "sym_b"),
        )
        with patch("axon_elf.symbol_name_hash", side_effect=[1, 1]):
            with self.assertRaisesRegex(ValueError, "hash collision"):
                axon_elf.render_binding_table_header(list(metadata.keep_symbols))


class ModelSymbolDiscoveryTests(unittest.TestCase):
    def test_read_only_candidate_is_preferred(self) -> None:
        symbols = [
            symbol(axon_elf.MODEL_SIZE_MARKER, 24),
            symbol("writable_decoy", 24, read_only=False),
            symbol("compiled_model", 24, read_only=True),
        ]
        self.assertEqual(
            axon_elf.discover_model_symbol(SymbolIndex.build(symbols)),
            "compiled_model",
        )

    def test_ambiguous_candidates_require_override(self) -> None:
        symbols = [
            symbol(axon_elf.MODEL_SIZE_MARKER, 24),
            symbol("model_a", 24),
            symbol("model_b", 24),
        ]
        index = SymbolIndex.build(symbols)
        with self.assertRaisesRegex(ValueError, "pass --model-sym"):
            axon_elf.discover_model_symbol(index)
        self.assertEqual(axon_elf.discover_model_symbol(index, "model_b"), "model_b")

    def test_invalid_override_type_fails(self) -> None:
        symbols = [
            symbol(axon_elf.MODEL_SIZE_MARKER, 24),
            symbol("only_model", 24),
            symbol("bad", 24, type="FUNC"),
        ]
        with self.assertRaisesRegex(ValueError, "defined global STT_OBJECT"):
            axon_elf.discover_model_symbol(SymbolIndex.build(symbols), "bad")

    def test_override_must_match_unique_discovery(self) -> None:
        symbols = [
            symbol(axon_elf.MODEL_SIZE_MARKER, 24),
            symbol("read_only_model", 24, read_only=True),
            symbol("writable_override", 24, read_only=False),
        ]
        with self.assertRaisesRegex(ValueError, "does not match discovered"):
            axon_elf.discover_model_symbol(
                SymbolIndex.build(symbols), "writable_override"
            )

    def test_derive_axon_model_token_from_storage(self) -> None:
        metadata = axon_elf.inspect_symbols(basic_probe_symbols(), persistent_vars_cap=8)
        self.assertEqual(
            axon_elf.derive_axon_model_token(metadata.model_symbol, metadata),
            "DEMO",
        )


class ProvideTests(unittest.TestCase):
    def test_data_symbol_and_thumb_bit(self) -> None:
        index = SymbolIndex.build(
            [
                symbol(
                    "nrf_axon_interlayer_buffer",
                    4096,
                    address=0x200003C0,
                    read_only=False,
                ),
                symbol(
                    "nrf_axon_nn_op_extension_relu",
                    0,
                    type="FUNC",
                    address=0x1234,
                ),
            ]
        )
        script, missing = axon_elf.build_provide_script(
            Path("unused.elf"),
            ["nrf_axon_interlayer_buffer", "nrf_axon_nn_op_extension_relu"],
            index,
        )
        self.assertEqual(missing, [])
        self.assertIn("PROVIDE(nrf_axon_interlayer_buffer = 0x200003C0);", script)
        self.assertIn("PROVIDE(nrf_axon_nn_op_extension_relu = 0x1235);", script)


if __name__ == "__main__":
    unittest.main()
