# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Build a single self-contained Windows executable for the Data Forwarder Host.
#
# The result is ONE file, dist\data-forwarder-host.exe, that bundles the Python
# interpreter and every dependency (PySide6/Qt6, numpy, bleak, ...). Copy it to
# another Windows machine and run it directly - nothing has to be installed there.
#
# Usage (from the package directory or anywhere):
#   powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_binary.ps1
#
# Notes:
#   * Build on the OLDEST Windows version you intend to support (e.g. Windows 10
#     or 11). Test on a clean VM before distributing broadly.
#   * Requires a standard Windows CPython 3.12+ on PATH (python.org or winget).
#     Do not use the nrfutil/NCS toolchain Python - use $env:PYTHON_BIN to point
#     at a normal install, e.g.  py -3.12
#   * Internet access for the one-time dependency download.
#   * Unsigned executables may trigger SmartScreen on first run; code-sign for
#     external distribution.
#   * BLE on Windows is implemented but not yet validated in project docs - test
#     UART and BLE after building.

#Requires -Version 5.1
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PkgDir = Split-Path -Parent $ScriptDir
Set-Location $PkgDir

$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$BuildVenv = if ($env:BUILD_VENV) { $env:BUILD_VENV } else { Join-Path $PkgDir ".build-venv" }
$VenvPython = Join-Path $BuildVenv "Scripts\python.exe"

Write-Host ">> Using interpreter: $(& $PythonBin --version)"
$InterpreterPath = & $PythonBin -c "import sys; print(sys.executable)"
Write-Host ">> Interpreter path:   $InterpreterPath"
$Platform = & $PythonBin -c "import sys; print(sys.platform)"
if ($Platform -ne "win32") {
    Write-Error "!! Expected Windows CPython (sys.platform=win32), got '$Platform'. Use a standard python.org install."
}

function Test-VenvUsable {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        return $false
    }
    & $VenvPython -m pip --version *> $null
    return $LASTEXITCODE -eq 0
}

# 1. Isolated build environment (kept out of git via .gitignore).
#    Reuse only if python -m pip works; stale/copied venvs get recreated.
if (-not (Test-VenvUsable)) {
    if (Test-Path -LiteralPath $BuildVenv) {
        Write-Host ">> Existing build venv is unusable (stale/foreign) - recreating"
        Remove-Item -LiteralPath $BuildVenv -Recurse -Force
    }
    Write-Host ">> Creating build venv at $BuildVenv"
    & $PythonBin -m venv $BuildVenv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "!! Failed to create venv. Install Python 3.12+ with the venv module."
    }
}

# 2. Install the app's runtime deps (from pyproject.toml) + PyInstaller.
Write-Host ">> Installing runtime dependencies and PyInstaller"
& $VenvPython -m pip install --upgrade pip wheel *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Error "!! pip upgrade failed."
}
# pywin32-ctypes is required by PyInstaller on Windows (imported from compat.py).
& $VenvPython -m pip install "." "pyinstaller>=6.6" "pywin32-ctypes>=0.2"
if ($LASTEXITCODE -ne 0) {
    Write-Error "!! Dependency install failed."
}

# 3. Clean previous artefacts and build from the spec.
#    -I (isolated): the package root contains a `platform/` subpackage; without
#    isolation, CWD on sys.path shadows stdlib `platform` and PyInstaller dies
#    with "module 'platform' has no attribute 'win32_ver'".
Write-Host ">> Building single-file binary"
foreach ($dir in @("build", "dist")) {
    if (Test-Path -LiteralPath $dir) {
        Remove-Item -LiteralPath $dir -Recurse -Force
    }
}
& $VenvPython -I -m PyInstaller --noconfirm --clean data_forwarder_host.spec
if ($LASTEXITCODE -ne 0) {
    Write-Error "!! PyInstaller build failed."
}

$Bin = Join-Path $PkgDir "dist\data-forwarder-host.exe"
if (Test-Path -LiteralPath $Bin) {
    $sizeMb = [math]::Round((Get-Item -LiteralPath $Bin).Length / 1MB, 1)
    Write-Host ""
    Write-Host ">> Done. Single-file binary:"
    Write-Host "     $Bin"
    Write-Host "     size: ${sizeMb} MB"
    Write-Host ">> Run it with:  $Bin"
} else {
    Write-Error "!! Build finished but $Bin was not produced."
}
