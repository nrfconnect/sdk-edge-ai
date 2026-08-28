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
from model_contract import MODEL_IMAGE_FORMAT_VERSION

# Any value works: the checker compares the image header against the context, and neither side
# recomputes the contract hash any more (it is the compiler's, read out of the slot's probe).
CONTRACT_HASH = 0x5F3A21C4


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
        backend = struct.pack(layout.BACKEND_NEUTON_FMT, model_ptr)
        header = struct.pack(
            layout.HEADER_FMT,
            layout.MAGIC,
            MODEL_IMAGE_FORMAT_VERSION,
            0,
            0,
            image_size,
            0x10000,
            contract_hash,
            0,
            name_ptr,
            backend + b"\0" * (20 - len(backend)),
            b"\0" * layout.PARAMS_SIZE,
        )
        data = bytearray(header + meta + b"gear\0")
        struct.pack_into("<I", data, layout.CRC32_OFFSET, zlib.crc32(data) & 0xFFFFFFFF)
        path = directory / "gear_anomaly_model_image.bin"
        path.write_bytes(data)
        return path

    def test_compatible_neuton(self) -> None:
        fw_hash = CONTRACT_HASH
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = self._neuton_image(root, fw_hash)
            context = {
                "format_version": MODEL_IMAGE_FORMAT_VERSION,
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
                "format_version": MODEL_IMAGE_FORMAT_VERSION,
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
                "format_version": MODEL_IMAGE_FORMAT_VERSION,
                "slots": [{"target": "gear_anomaly", "backend": "neuton", "contract_hash": 1}],
            }
            ctx = root / "model_ota_context.json"
            ctx.write_text(json.dumps(context), encoding="utf-8")
            argv = ["--context", str(ctx), "--image", str(image), "--slot", "gear_anomaly"]
            self.assertEqual(compat.main(argv), compat.EXIT_INCOMPATIBLE)
            # --report-only makes the capacity verdicts a report; the hash is still a build gate.
            self.assertEqual(compat.main(argv + ["--report-only"]), compat.EXIT_INCOMPATIBLE)

    def test_report_only_allows_cap_overrun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = self._neuton_image(root, 1, neurons_num=25)
            context = {
                "format_version": MODEL_IMAGE_FORMAT_VERSION,
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
                compat.main(
                    ["--context", str(ctx), "--image", str(image), "--slot", "gear_anomaly",
                     "--report-only"]
                ),
                compat.EXIT_OK,
            )


if __name__ == "__main__":
    unittest.main()
