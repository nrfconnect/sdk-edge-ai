<!--
Copyright (c) 2026 Nordic Semiconductor ASA
SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
-->

# Data Forwarder Host

`data_forwarder_host` is a cross-platform desktop GUI that receives, visualises,
records and exports sensor data forwarded from an nRF device over **UART** or
**BLE NUS** (Nordic UART Service). Frames use the device's COBS + CBOR v1
framing; multiple capture sessions can run side by side in their own tabs.

The matching device-side firmware is the
[`data_forwarder` sample](../../samples/data_forwarder) (`edge-ai/samples/data_forwarder`),
which produces the COBS/CBOR `sensor-data` stream this host decodes. For BLE it
advertises the Nordic UART Service as `nRF DataFwd`. Without that firmware
running on a connected device there is no data to receive.

## Requirements

- Python **3.12** or newer.
- Linux, Windows or macOS.
- An nRF device running the [`data_forwarder` firmware
  sample](../../samples/data_forwarder) (`edge-ai/samples/data_forwarder`).
- For UART sources: permission to access the serial port (on Linux this usually
  means membership of the `dialout` group).

## Install

> **Ubuntu / Debian:** Qt 6 (PySide6) needs the XCB cursor library at runtime.
> If the app fails to start with an `xcb` platform-plugin error
> (`Could not load the Qt platform plugin "xcb"`), install it first:
>
> ```bash
> sudo apt install libxcb-cursor0
> ```
>
> On a minimal install you may also need the other XCB runtime libraries
> (`libxcb-xinerama0`, `libxkbcommon-x11-0`); these are present on a standard
> Ubuntu desktop.

From this directory (`data_forwarder_host/`), install the requirements:

```bash
pip install -r requirements.txt
```

`requirements.txt` uses `pyproject.toml` as the single source of truth, so the
dependency list lives in exactly one place. Alternatively, install the package
directly with pip's editable mode:

```bash
pip install -e .
```

PySide6 (Qt 6, with QtCharts) is a base dependency and is always installed — the
application is GUI-only.

## Run

The application is a pure GUI — there is no command-line interface. Launch it in
any of these equivalent ways:

```bash
data-forwarder-host          # console script (installed entry point)
python -m data_forwarder_host
python main.py
```

By default, source acquisition (byte reading + COBS/CBOR decode) runs in a
separate child process so the GUI stays responsive regardless of backend load.
Set `DFH_SINGLE_PROCESS=1` to fall back to the legacy in-GUI acquisition path.

## First-time use

1. Launch the application — it starts blank, with no session open and no tabs.
2. Open `Session → New session…`.
3. Pick a **data source**: **UART** (live serial device) or **BLE NUS** (Nordic
   UART Service). For BLE NUS, the dialog scans for advertising devices and
   lets you pick one to connect to (requires a working Bluetooth adapter). If
   the BLE stream drops frames or runs slower than expected, see
   [docs/ble_throughput_tuning.md](docs/ble_throughput_tuning.md) for an
   ordered, OS-aware speed-up procedure.
4. Set any other session details, then click **Create session** to open a
   per-session tab; streaming starts automatically.
5. Enter a recording **label**, choose an output directory, then press
   **Record** to capture data. Recordings are buffered in RAM and dumped to
   a CSV on **Stop**; when the output directory is left blank they are written
   to `recordings/` next to the package. Each recording writes a `{stem}.csv`
   data file plus a paired `{stem}.txt` metadata sidecar (host, transport,
   timing, device session info, channels and errors). A banner shows the last
   saved CSV with shortcuts to open the file or its containing folder.

The **View** menu toggles the error and log panels, the per-tab visualisation
panels (Combined View, Individual Channels, Channel ASCII, Decoded Frames),
the theme (System / Light / Dark) and the log level. Session configuration can
be saved and reopened via the **Session** menu. Live bandwidth, host metrics
and First Failure Data Capture (FFDC) details are surfaced on each tab.

## Architecture (short)

The application is strictly layered. The acquisition/transport layers are fully
**Qt-free** (no PySide6 imports), so they can run inside the headless child
acquisition process; the data-model, session and GUI layers run in the GUI
process and use Qt (PySide6 `QObject`/`Signal`):

- `platform/` — *(Qt-free)* OS-specific adapters (serial-port enumeration,
  access diagnostics).
- `source/` — *(Qt-free)* byte sources: `uart` (pyserial) and `ble_nus`
  (BLE NUS, via bleak).
- `protocol/` — *(Qt-free)* frame decoders (COBS + CBOR v1) and the framing
  primitives.
- `pipeline/` — *(Qt-free)* out-of-process acquisition: a `spawn` child process
  reads the source and decodes frames, while the GUI process drains decoded
  envelopes over an IPC queue (toggle with `DFH_SINGLE_PROCESS`).
- `core/` — per-session data model, rolling channel buffers, decimation, the
  recorder, the CSV writer, the metadata sidecar and the metrics (bandwidth,
  host, transfer, pipeline). *Uses Qt:* `data_model`, `recorder` and
  `error_log` are `QObject`s that emit signals to the GUI; the CSV writer,
  sidecar and the metrics dataclasses are plain Python.
- `session/` — *(uses Qt)* `ForwardingSession` (source + decoder + I/O),
  `SessionController` (per-tab façade) and `SessionManager` (app-wide registry).
- `utils/` — standard paths, logging, version and small helpers (plus one
  optional Qt-based debug-stream helper).
- `gui/` — *(uses Qt)* PySide6 main window, menus, dialogs (`gui/dialogs/`) and
  per-session tab widgets (`gui/widgets/`, including the QtCharts plots).

## License

All files carry the SPDX header:

```
Copyright (c) 2026 Nordic Semiconductor ASA
SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
```
