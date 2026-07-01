#!/usr/bin/env bash
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Build a single self-contained Linux executable for the Data Forwarder Host.
#
# The result is ONE file, dist/data-forwarder-host, that bundles the Python
# interpreter and every dependency (PySide6/Qt6, numpy, bleak, ...). Copy it to
# any reasonably recent x86_64 Ubuntu machine and run it directly — nothing has
# to be installed there.
#
# Usage:
#   ./scripts/build_linux_binary.sh
#
# Notes:
#   * Run on the OLDEST Ubuntu you intend to support: glibc is forward- but not
#     backward-compatible, so a binary built on Ubuntu 22.04 runs on 22.04+ but
#     a binary built on 24.04 may not run on 22.04.
#   * Requires a working Python 3.12 with `venv` + internet access for the
#     one-time dependency download.
set -euo pipefail

# Resolve the package directory (parent of this scripts/ dir) regardless of CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PKG_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILD_VENV="${BUILD_VENV:-$PKG_DIR/.build-venv}"

echo ">> Using interpreter: $($PYTHON_BIN --version)"

# 1. Isolated build environment (kept out of git via .gitignore).
#    A venv is only reused if it actually works: a leftover/foreign venv (e.g.
#    one copied from another machine) has launcher scripts whose shebang points
#    at a path that does not exist here ("pip: cannot execute: required file not
#    found"), so we validate `python -m pip` and rebuild from scratch otherwise.
venv_is_usable() {
    [ -x "$BUILD_VENV/bin/python" ] && "$BUILD_VENV/bin/python" -m pip --version >/dev/null 2>&1
}
if ! venv_is_usable; then
    if [ -e "$BUILD_VENV" ]; then
        echo ">> Existing build venv is unusable (stale/foreign) — recreating"
        rm -rf "$BUILD_VENV"
    fi
    echo ">> Creating build venv at $BUILD_VENV"
    # Prefer stdlib venv; fall back to `virtualenv` on distros that ship Python
    # without the venv/ensurepip module (e.g. minimal Debian/Ubuntu installs).
    if "$PYTHON_BIN" -c "import venv, ensurepip" 2>/dev/null; then
        "$PYTHON_BIN" -m venv "$BUILD_VENV"
    elif command -v virtualenv >/dev/null 2>&1; then
        virtualenv -p "$PYTHON_BIN" "$BUILD_VENV"
    else
        echo "!! Neither 'python -m venv' nor 'virtualenv' is available." >&2
        echo "   Install one of:  apt install python3-venv   |   pip install --user virtualenv" >&2
        exit 1
    fi
fi
# shellcheck disable=SC1091
source "$BUILD_VENV/bin/activate"

# 2. Install the app's runtime deps (from pyproject.toml) + PyInstaller.
echo ">> Installing runtime dependencies and PyInstaller"
python -m pip install --upgrade pip wheel >/dev/null
python -m pip install "." "pyinstaller>=6.6"

# 3. Clean previous artefacts and build from the spec.
#    -I (isolated): the package root contains a `platform/` subpackage; without
#    isolation, CWD on sys.path shadows stdlib `platform` and PyInstaller dies
#    with "module 'platform' has no attribute 'system'" / missing `win32_ver`.
echo ">> Building single-file binary"
rm -rf build dist
python -I -m PyInstaller --noconfirm --clean data_forwarder_host.spec

BIN="$PKG_DIR/dist/data-forwarder-host"
if [ -x "$BIN" ]; then
    echo ""
    echo ">> Done. Single-file binary:"
    echo "     $BIN"
    ls -lh "$BIN" | awk '{print "     size: "$5}'
    echo ">> Run it with:  $BIN"
else
    echo "!! Build finished but $BIN was not produced." >&2
    exit 1
fi
