#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Unit tests for model OTA compatibility checking."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import check_model_compat as compat
import validate_model_image_layout as layout
from model_contract import (
    MODEL_IMAGE_OFFSET_MCUBOOT,
    contract_hash_neuton,
    neuton_pipeline_hash,
    symbol_name_hash,
)


class CompatCheckerTests(unittest.TestCase):
    def _neuton_image(self, directory: Path, contract_hash: int, neurons_num: int = 10) -> Path:
        model_off = layout.HEADER_SIZE
        name_off = model_off + 28
        image_size = name_off + 5
        name_ptr = 0x102000 + name_off
        model_ptr = 0x102000 + model_off
        meta = bytearray(28)
        struct.pack_into("<H", meta, 20, 1)
        struct.pack_into("<H", meta, 22, neurons_num)
        backend = struct.pack(
            layout.BACKEND_NEUTON_FMT, model_ptr, 2, 0, 0, 0, model_ptr + 16
        )
        header = struct.pack(
            layout.HEADER_FMT,
            layout.MAGIC,
            5,
            0,
            0,
            image_size,
            0x10000,
            contract_hash,
            0,
            name_ptr,
            backend + b"\0" * (20 - len(backend)),
        )
        data = bytearray(header + meta + b"gear\0")
        struct.pack_into("<I", data, layout.CRC32_OFFSET, zlib.crc32(data) & 0xFFFFFFFF)
        path = directory / "gear_anomaly_model_image.bin"
        path.write_bytes(data)
        return path

    def test_compatible_neuton(self) -> None:
        pipeline = neuton_pipeline_hash(0, 128, 128, 2, 0, 1)
        fw_hash = contract_hash_neuton(
            task=3,
            params_type=1,
            outputs_cap=10,
            inputs_num=2,
            neurons_cap=20,
            solution_id="90360",
            pipeline_hash=pipeline,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = self._neuton_image(root, fw_hash)
            context = {
                "format_version": 5,
                "slots": [
                    {
                        "target": "gear_anomaly",
                        "backend": "neuton",
                        "partition_addr": 0x102000,
                        "partition_size": 32768,
                        "model_image_offset": 0,
                        "contract_hash": fw_hash,
                        "neurons_cap": 20,
                    }
                ],
            }
            ctx = root / "model_ota_context.json"
            ctx.write_text(json.dumps(context), encoding="utf-8")
            self.assertEqual(
                compat.main(["--context", str(ctx), "--image", str(image), "--slot", "gear_anomaly"]),
                compat.EXIT_OK,
            )

    def test_neurons_cap_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = self._neuton_image(root, 1, neurons_num=25)
            context = {
                "format_version": 5,
                "slots": [
                    {
                        "target": "gear_anomaly",
                        "backend": "neuton",
                        "partition_addr": 0x102000,
                        "model_image_offset": 0,
                        "contract_hash": 1,
                        "neurons_cap": 20,
                    }
                ],
            }
            ctx = root / "model_ota_context.json"
            ctx.write_text(json.dumps(context), encoding="utf-8")
            self.assertEqual(
                compat.main(["--context", str(ctx), "--image", str(image), "--slot", "gear_anomaly"]),
                compat.EXIT_NEEDS_FW,
            )

    def test_contract_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = self._neuton_image(root, 0xDEADBEEF)
            context = {
                "format_version": 5,
                "slots": [
                    {
                        "target": "gear_anomaly",
                        "backend": "neuton",
                        "model_image_offset": 0,
                        "contract_hash": 1,
                    }
                ],
            }
            ctx = root / "model_ota_context.json"
            ctx.write_text(json.dumps(context), encoding="utf-8")
            self.assertEqual(
                compat.main(["--context", str(ctx), "--image", str(image), "--slot", "gear_anomaly"]),
                compat.EXIT_INCOMPATIBLE,
            )

    def test_missing_model_image_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = self._neuton_image(root, 1)
            context = {
                "format_version": 5,
                "slots": [
                    {
                        "target": "gear_anomaly",
                        "backend": "neuton",
                        "partition_addr": 0x102000,
                        "contract_hash": 1,
                    }
                ],
            }
            ctx = root / "model_ota_context.json"
            ctx.write_text(json.dumps(context), encoding="utf-8")
            self.assertEqual(
                compat.main(["--context", str(ctx), "--image", str(image), "--slot", "gear_anomaly"]),
                compat.EXIT_INCOMPATIBLE,
            )

    def test_axon_bindings_with_model_image_offset(self) -> None:
        partition_addr = 0xF7000
        link_addr = partition_addr + MODEL_IMAGE_OFFSET_MCUBOOT
        symbols = [
            "axon_model_axon_user_instance_36711_persistent_vars",
            "nrf_axon_interlayer_buffer",
        ]
        fw_hash = 554036569
        binding_entries = [
            (symbol_name_hash(symbols[0]), 0x20002FE4),
            (symbol_name_hash(symbols[1]), 0x20000778),
        ]
        binding_off = layout.HEADER_SIZE + 8
        binding_ptr = link_addr + binding_off
        name_off = binding_off + len(binding_entries) * 8
        image_size = name_off + 4
        name_ptr = link_addr + name_off
        model_ptr = link_addr + layout.HEADER_SIZE + 32
        backend = struct.pack(
            layout.BACKEND_AXON_FMT,
            model_ptr,
            4,
            1160,
            binding_ptr,
            len(binding_entries),
        )
        header = struct.pack(
            layout.HEADER_FMT,
            layout.MAGIC,
            5,
            3,
            0,
            image_size,
            0x10000,
            fw_hash,
            0,
            name_ptr,
            backend,
        )
        data = bytearray(header + b"\0" * (binding_off - layout.HEADER_SIZE))
        for name_hash, address in binding_entries:
            data.extend(struct.pack("<II", name_hash, address))
        data.extend(b"ww\x00\x00")
        struct.pack_into("<I", data, layout.CRC32_OFFSET, zlib.crc32(data) & 0xFFFFFFFF)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "ww_model_image.bin"
            image.write_bytes(data)
            slot = {
                "target": "ww",
                "backend": "axon",
                "partition_addr": partition_addr,
                "partition_size": 65536,
                "model_image_offset": MODEL_IMAGE_OFFSET_MCUBOOT,
                "contract_hash": fw_hash,
                "persistent_vars_cap": 1160,
                "packed_output_cap": 4,
                "binding_symbols": symbols,
            }
            context = {"format_version": 5, "slots": [slot]}
            ctx = root / "model_ota_context.json"
            ctx.write_text(json.dumps(context), encoding="utf-8")

            self.assertEqual(compat.image_link_addr(slot), link_addr)
            self.assertEqual(
                compat.main(["--context", str(ctx), "--image", str(image), "--slot", "ww"]),
                compat.EXIT_OK,
            )


    def test_image_size_exceeds_payload_cap_with_offset(self) -> None:
        partition_addr = 0xF7000
        link_addr = partition_addr + MODEL_IMAGE_OFFSET_MCUBOOT
        symbols = ["nrf_axon_interlayer_buffer"]
        fw_hash = 554036569
        image_size = layout.HEADER_SIZE + 16
        name_ptr = link_addr + image_size
        model_ptr = link_addr + layout.HEADER_SIZE
        backend = struct.pack(
            layout.BACKEND_AXON_FMT,
            model_ptr,
            4,
            1160,
            0,
            0,
        )
        header = struct.pack(
            layout.HEADER_FMT,
            layout.MAGIC,
            5,
            3,
            0,
            image_size,
            0x10000,
            fw_hash,
            0,
            name_ptr,
            backend,
        )
        data = bytearray(header + b"\0" * 16)
        struct.pack_into("<I", data, layout.CRC32_OFFSET, zlib.crc32(data) & 0xFFFFFFFF)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "ww_model_image.bin"
            image.write_bytes(data)
            context = {
                "format_version": 5,
                "slots": [
                    {
                        "target": "ww",
                        "backend": "axon",
                        "partition_addr": partition_addr,
                        "partition_size": image_size,
                        "model_image_offset": MODEL_IMAGE_OFFSET_MCUBOOT,
                        "contract_hash": fw_hash,
                        "persistent_vars_cap": 1160,
                        "packed_output_cap": 4,
                        "binding_symbols": symbols,
                    }
                ],
            }
            ctx = root / "model_ota_context.json"
            ctx.write_text(json.dumps(context), encoding="utf-8")
            self.assertEqual(
                compat.main(["--context", str(ctx), "--image", str(image), "--slot", "ww"]),
                compat.EXIT_INCOMPATIBLE,
            )


if __name__ == "__main__":
    unittest.main()
