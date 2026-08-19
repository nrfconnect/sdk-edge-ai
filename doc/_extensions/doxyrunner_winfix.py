# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Windows-safe Doxygen output sync for Sphinx doc builds."""

from __future__ import annotations

import filecmp
import shutil
import sys
import time
from pathlib import Path

import zephyr.doxyrunner as doxyrunner


def _rmtree_retry(path: Path, retries: int = 10, delay_s: float = 0.25) -> None:
    for attempt in range(retries):
        if not path.exists():
            return

        try:
            shutil.rmtree(path)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(delay_s)


def _replace_dir(src: Path, dst: Path) -> None:
    if sys.platform == "win32":
        _rmtree_retry(dst)
        shutil.move(str(src), str(dst))
        return

    if dst.exists():
        shutil.rmtree(dst)
    src.rename(dst)


def sync_doxygen(doxyfile: str, new: Path, prev: Path) -> None:
    generate_html = doxyrunner.get_doxygen_option(doxyfile, "GENERATE_HTML")
    if generate_html[0] == "YES":
        html_output = doxyrunner.get_doxygen_option(doxyfile, "HTML_OUTPUT")
        if not html_output:
            raise ValueError("No HTML_OUTPUT set in Doxyfile")

        _replace_dir(new / html_output[0], prev / html_output[0])

    xml_output = doxyrunner.get_doxygen_option(doxyfile, "XML_OUTPUT")
    if not xml_output:
        raise ValueError("No XML_OUTPUT set in Doxyfile")

    new_xmldir = new / xml_output[0]
    prev_xmldir = prev / xml_output[0]

    if prev_xmldir.exists():
        dcmp = filecmp.dircmp(new_xmldir, prev_xmldir)

        for file in dcmp.right_only:
            (Path(dcmp.right) / file).unlink()

        for file in dcmp.left_only + dcmp.diff_files:
            shutil.copy(Path(dcmp.left) / file, Path(dcmp.right) / file)

        shutil.rmtree(new_xmldir)
    else:
        _replace_dir(new_xmldir, prev_xmldir)


doxyrunner.sync_doxygen = sync_doxygen


def setup(app):
    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
