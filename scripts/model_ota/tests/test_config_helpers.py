#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

"""Unit tests for config_helpers."""

from pathlib import Path

import pytest
from config_helpers import config_define


def _header(directory: Path, text: str) -> Path:
    path = directory / "axon_config.h"
    path.write_text(text, encoding="utf-8")
    return path


def test_config_define_reads_decimal(tmp_path: Path) -> None:
    header = _header(tmp_path, "#define MODEL_OTA_AXON_PERSISTENT_VARS_CAP 12\n")

    actual = config_define(header, "MODEL_OTA_AXON_PERSISTENT_VARS_CAP")

    assert actual == 12


def test_config_define_reads_hex_with_suffix(tmp_path: Path) -> None:
    header = _header(tmp_path, "#define MODEL_OTA_AXON_PACKED_OUTPUT_BYTES 0x40u\n")

    actual = config_define(header, "MODEL_OTA_AXON_PACKED_OUTPUT_BYTES")

    assert actual == 0x40


def test_config_define_missing_is_none(tmp_path: Path) -> None:
    header = _header(tmp_path, "#define SOMETHING_ELSE 1\n")

    actual = config_define(header, "MODEL_OTA_AXON_PERSISTENT_VARS_CAP")

    assert actual is None


def test_config_define_reads_indented_with_ul_suffix(tmp_path: Path) -> None:
    header = _header(tmp_path, "  #define MODEL_OTA_AXON_PERSISTENT_VARS_CAP 0UL\n")

    actual = config_define(header, "MODEL_OTA_AXON_PERSISTENT_VARS_CAP")

    assert actual == 0


def test_config_define_does_not_match_a_longer_name(tmp_path: Path) -> None:
    header = _header(tmp_path, "#define MODEL_OTA_AXON_CAP_BYTES 8\n#define MODEL_OTA_AXON_CAP 4\n")

    actual = config_define(header, "MODEL_OTA_AXON_CAP")

    assert actual == 4


@pytest.mark.parametrize(
    "line",
    [
        "#define MODEL_OTA_AXON_CAP 010",
        "#define MODEL_OTA_AXON_CAP 4 /* bytes */",
        "#define MODEL_OTA_AXON_CAP (4)",
    ],
)
def test_config_define_unsupported_forms_are_none(tmp_path: Path, line: str) -> None:
    header = _header(tmp_path, line + "\n")

    actual = config_define(header, "MODEL_OTA_AXON_CAP")

    assert actual is None
