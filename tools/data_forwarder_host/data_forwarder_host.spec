# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# PyInstaller spec for the Data Forwarder Host standalone binary.
#
# Produces ONE self-contained executable that bundles the CPython interpreter,
# every third-party dependency (PySide6/Qt6 incl. QtCharts, numpy, bleak,
# pyserial, cbor2, cobs, psutil, platformdirs) and the application package.
# The end user runs a single file; nothing has to be installed.
#
# Build with:
#     pyinstaller data_forwarder_host.spec
#
# (Normally driven by scripts/build_linux_binary.sh, which provisions an
# isolated venv with the runtime deps + PyInstaller first.)

import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# The package import name ``data_forwarder_host`` maps to *this* directory, so
# its PARENT (``tools/``) must be importable for ``import data_forwarder_host``
# to resolve. ``SPECPATH`` is the directory containing this spec file.
#
# IMPORTANT: only the PARENT (``tools/``) goes on ``pathex`` — never this package
# directory itself. Putting the package dir on the search path makes PyInstaller
# collect the package's internal subpackages (``platform``, ``core``, ``gui`` …)
# *also* as top-level modules, so a bare stdlib ``import platform`` (reached via
# ``uuid``) would resolve to ``data_forwarder_host/platform`` and explode with
# ``module 'platform' has no attribute 'system'``. Keeping only ``tools/`` on the
# path means the subpackages are reachable solely as ``data_forwarder_host.*``.
PKG_DIR = os.path.abspath(SPECPATH)
TOOLS_DIR = os.path.dirname(PKG_DIR)
ENTRY = os.path.join(PKG_DIR, "scripts", "freeze_entry.py")

# ---------------------------------------------------------------------------
# Dependency collection
# ---------------------------------------------------------------------------
hiddenimports: list[str] = []
datas: list = []
binaries: list = []

# Some dependencies pick their OS backend dynamically, so PyInstaller's static
# import analysis misses it: bleak loads the BlueZ/D-Bus stack on Linux, and
# pyserial imports its platform-specific port-listing module
# (serial.tools.list_ports_linux/osx/windows) by name. collect_all pulls in each
# package's code + data + every submodule so the frozen binary is complete.
for pkg in ("bleak", "serial"):
    b, d, h = collect_all(pkg)
    binaries += b
    datas += d
    hiddenimports += h

# The acquisition child is launched by `spawn`, which imports the target by
# dotted path. Make sure every app submodule is present even if a future lazy
# import hides one from static analysis.
hiddenimports += collect_submodules("data_forwarder_host")

# PySide6/Qt is handled by PyInstaller's bundled hooks (they include QtCharts as
# long as it is imported, which it is). No manual Qt plugin wrangling needed.

block_cipher = None

a = Analysis(
    [ENTRY],
    pathex=[TOOLS_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Trim large, unused Qt modules to keep the binary smaller. The app only
    # uses QtWidgets/QtGui/QtCore/QtCharts.
    excludes=[
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebEngineQuick",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DRender",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "PySide6.QtQuick3D",
        "PySide6.QtDesigner",
        "PySide6.QtPdf",
        "tkinter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="data-forwarder-host",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI app: no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
