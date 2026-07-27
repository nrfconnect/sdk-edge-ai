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
from model_contract import contract_hash_neuton, neuton_pipeline_hash


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
                "slots": [{"target": "gear_anomaly", "backend": "neuton", "contract_hash": 1}],
            }
            ctx = root / "model_ota_context.json"
            ctx.write_text(json.dumps(context), encoding="utf-8")
            self.assertEqual(
                compat.main(["--context", str(ctx), "--image", str(image), "--slot", "gear_anomaly"]),
                compat.EXIT_INCOMPATIBLE,
            )


if __name__ == "__main__":
    unittest.main()
