# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""New-session dialog — builds a validated :class:`SessionConfig`.

The user chooses and configures a single data source (UART or BLE NUS) and
defines the recording ``label`` and per-session output directory. There is no
protocol selection (the single COBS/CBOR v1 protocol is fixed) and no sink
configuration (output is a single CSV file).
"""

from __future__ import annotations

import uuid
from typing import Any

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from data_forwarder_host.gui.scan_worker import ScanWorker
from data_forwarder_host.platform.base import PlatformAdapter
from data_forwarder_host.session.config import ConfigError, SessionConfig, SourceSpec
from data_forwarder_host.source import SOURCE_KINDS, source_for_kind
from data_forwarder_host.source.base import ConfigField, RoleMode
from data_forwarder_host.source.ble_nus import DEFAULT_DEVICE_NAME
from data_forwarder_host.utils.logging import log_user_action

# Default source kind: BLE NUS is preselected when the dialog opens.
DEFAULT_SOURCE_KIND = "ble"

# Process-wide registry of scan threads that did not finish within the join
# timeout on dialog teardown. Holding a Python reference here keeps the QThread
# (and its scanner) alive until it actually finishes, so it is never garbage-
# collected or destroyed while still running — which would abort the process.
_PENDING_SCAN_THREADS: set[QThread] = set()


def _detach_running_scan_thread(thread: QThread, scanner: Any) -> None:
    """Keep a still-running scan thread alive until it finishes, then delete it.

    Called when ``QThread.wait`` times out. Destroying a running ``QThread``
    aborts the process, so instead we retain references and schedule deletion
    via the thread's ``finished`` signal.
    """
    _PENDING_SCAN_THREADS.add(thread)

    def _cleanup() -> None:
        _PENDING_SCAN_THREADS.discard(thread)
        try:
            thread.deleteLater()
        except RuntimeError:
            pass
        if scanner is not None:
            try:
                scanner.deleteLater()
            except RuntimeError:
                pass

    try:
        thread.finished.connect(_cleanup)
    except RuntimeError:
        pass


# How often the live BLE device list is repainted from coalesced detections.
# A fixed 1 s cadence keeps rows from toggling/re-sorting several times a second.
_DEVICE_REFRESH_MS = 1000

# Default bandwidth-averaging window for a new session (seconds). Independent of
# the plot window; adjustable later from the session window.
DEFAULT_BANDWIDTH_WINDOW_SECONDS = 1.0

# Hint shown next to a disabled OK button until a usable data source is ready.
_HINT_PICK_KIND = "Pick a data source."
_HINT_BLE_SELECT = "Choose a data source: select a Bluetooth device from the list."
_HINT_BLE_CONNECTING = (
    "Press Connect to connect to the selected device. "
    "Create session enables once connected."
)


def compute_ok_state(
    kind: str,
    *,
    ble_selected: bool = False,
    ble_connected: bool = False,
) -> tuple[bool, str]:
    """Decide whether OK is enabled, and the hint to show while it is not.

    Pure (no Qt) so the gating policy is isolated. For BLE the user must
    select a device *and* the connection attempted in the dialog must have
    succeeded (connect-in-dialog hand-off). UART is gated only by
    having chosen a kind; its port is validated on accept.
    """
    if not kind:
        return False, _HINT_PICK_KIND
    if kind == "ble":
        if not ble_selected:
            return False, _HINT_BLE_SELECT
        if not ble_connected:
            return False, _HINT_BLE_CONNECTING
        return True, ""
    return True, ""


def compute_scan_status(
    *,
    device_count: int,
    has_selection: bool,
    selected_name: str | None = None,
) -> tuple[str, str]:
    """Decide the idle BLE status line (icon state, text) — pure, no Qt.

    This is the line refreshed periodically while scanning. It must *reinforce*
    the current guidance rather than clobber it: once the user has selected a
    device the line keeps telling them to press Connect (instead of reverting to
    the generic device-count prompt on the next scan tick).
    """
    if device_count <= 0:
        return (
            "checking",
            "Looking for Bluetooth devices… make sure the device is advertising.",
        )
    if has_selection:
        name = selected_name or "the selected device"
        return (
            "on",
            f"Selected {name}. Press Connect to connect "
            f"({device_count} device(s) found).",
        )
    return (
        "on",
        f"Bluetooth is on. {device_count} device(s) found — select one to connect.",
    )


def compute_connect_button(
    *,
    selected_id: str | None,
    connected_id: str | None,
    connecting: bool = False,
    canceling: bool = False,
) -> tuple[str, bool]:
    """Decide the Connect/Disconnect toggle label and enabled state (pure).

    Selecting a device never connects or disconnects anything; the
    button is the *only* control that changes the connection, and it is a
    toggle that re-targets whatever device is currently selected:

    - the selected device **is** the connected one → ``"Disconnect"`` (enabled):
      pressing it tears the live connection down;
    - any other (or no) connection for the selected device → ``"Connect"``
      (enabled): pressing it disconnects the previously connected device, if
      any, and connects the selected one.

    The button is disabled **only** when there is nothing whatsoever to act on
    (no device selected **and** nothing connected). It stays enabled while a
    connect/cancel is in flight and is **never** disabled merely because a
    device is connected — an active connection always shows an enabled control
    so the user can disconnect at any moment. The ``connecting`` and
    ``canceling`` flags are accepted for call-site symmetry but no longer gate
    the enabled state; in-flight clicks are blocked by the modal progress popup,
    not by greying the button out.
    """
    is_connected_selected = selected_id is not None and selected_id == connected_id
    has_connection = connected_id is not None
    # The toggle targets the connected device for "Disconnect" when it is the
    # selected one or when nothing else is selected; otherwise it offers to
    # connect the currently-selected device.
    target_is_disconnect = is_connected_selected or (
        selected_id is None and has_connection
    )
    label = "Disconnect" if target_is_disconnect else "Connect"
    enabled = selected_id is not None or has_connection
    return label, enabled


# Braille spinner frames used to animate "Connecting…"/"Disconnecting…" states.
_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def spinner_frame(tick: int) -> str:
    """Return the spinner glyph for animation step *tick* (pure, no Qt)."""
    return _SPINNER_FRAMES[tick % len(_SPINNER_FRAMES)]


def order_ble_devices(infos):
    """Order discovered BLE devices for the selection list (ordering only).

    Devices are arranged into three ordering groups — there is no visual
    grouping, only sort order:

    1. devices advertising the target sample name ``"nRF DataFwd"``;
    2. other *named* devices;
    3. *unnamed* devices.

    Within a group the secondary key is **signal strength** (RSSI), strongest
    first; a missing RSSI sorts last. The sort is stable, so devices that are
    otherwise equal keep their discovery order.
    """
    def group(info) -> int:
        name = (info.details.get("name") or "").strip()
        if name == DEFAULT_DEVICE_NAME:
            return 0
        if name:
            return 1
        return 2

    def rssi(info) -> float:
        value = info.details.get("rssi")
        return float(value) if isinstance(value, (int, float)) else float("-inf")

    return sorted(infos, key=lambda i: (group(i), -rssi(i)))


class _BusyDialog(QDialog):
    """Tiny non-modal popup with an animated spinner, e.g. while disconnecting.

    It is intentionally non-blocking: the GUI stays responsive (the actual
    disconnect runs on a worker thread) and the popup is closed by the caller
    once the operation finishes.
    """

    def __init__(self, parent, text: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Please wait")
        self.setModal(False)
        # Drop the close button — the popup is dismissed programmatically.
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint
        )
        row = QHBoxLayout(self)
        self._spin = QLabel(spinner_frame(0))
        self._label = QLabel(text)
        row.addWidget(self._spin)
        row.addWidget(self._label)
        self._tick = 0
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._advance)
        self._timer.start()

    def _advance(self) -> None:
        self._tick += 1
        self._spin.setText(spinner_frame(self._tick))

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self._timer.stop()
        super().closeEvent(event)


class _ConnectProgressDialog(QDialog):
    """Application-modal popup shown while a BLE connection is established.

    It grabs input focus so the New Session form is inaccessible during the
    connect (the operation can take seconds and must not be interrupted by
    further clicks). It offers **Cancel**; cancellation is handled by the
    caller, which keeps the device's Connect button disabled until the
    in-flight attempt has been safely torn down (avoiding the "too many
    connections right after cancelling" device fault).
    """

    cancel_requested = Signal()

    def __init__(self, parent, device_name: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Connecting")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        # No close/help button — Cancel is the only way out.
        self.setWindowFlags(
            self.windowFlags()
            & ~Qt.WindowType.WindowCloseButtonHint
            & ~Qt.WindowType.WindowContextHelpButtonHint
        )
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self._spin = QLabel(spinner_frame(0))
        self._label = QLabel(f"Connecting to {device_name}…")
        self._label.setWordWrap(True)
        row.addWidget(self._spin)
        row.addWidget(self._label, 1)
        layout.addLayout(row)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._cancel = QPushButton("Cancel")
        self._cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel)
        layout.addLayout(btn_row)

        self._tick = 0
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._advance)
        self._timer.start()

    def _advance(self) -> None:
        self._tick += 1
        self._spin.setText(spinner_frame(self._tick))

    def _on_cancel(self) -> None:
        # Reflect that cancellation is underway; the caller closes the popup
        # once the in-flight connection has been safely released.
        log_user_action("Clicked Cancel in the BLE connecting popup")
        self._cancel.setEnabled(False)
        self._label.setText("Cancelling… please wait")
        self.cancel_requested.emit()

    def reject(self) -> None:  # noqa: D401 - intercept Esc to behave like Cancel
        if self._cancel.isEnabled():
            self._on_cancel()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self._timer.stop()
        super().closeEvent(event)


class NewSessionDialog(QDialog):
    """Dialog that returns a validated :class:`SessionConfig`."""

    def __init__(self, platform: PlatformAdapter, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New session")
        self.resize(540, 520)
        self._platform = platform
        self._config: SessionConfig | None = None
        self._source_field_widgets: dict[str, Any] = {}

        # Connect-in-dialog state: the BLE source is opened here
        # and handed to the new session; OK is gated on a live connection.
        self._prepared_source: Any = None
        self._ble_selected_info: Any = None
        self._ble_connected = False
        self._threads: list[QThread] = []
        self._workers: list[Any] = []
        # Live (continuous) BLE scan that streams detections while the dialog is
        # open — no Refresh button, no fixed scan wait. ``_seen_rows`` maps a
        # device address to its row so detections upsert in place (keeping the
        # current selection and the connected-row highlight stable).
        self._scanner: Any = None
        self._scanner_thread: QThread | None = None
        self._seen_rows: dict[str, int] = {}
        # All discovered devices, keyed by address, so the streamed detections
        # can be kept in the sort order (nRF first, then named, then
        # unnamed; RSSI strongest-first within a group) as the list updates.
        self._device_infos: dict[str, Any] = {}
        # Detections arrive faster than is useful to render; they are coalesced
        # into ``_device_infos`` and the table is refreshed on a fixed 1 s timer
        # so rows do not flicker/re-sort several times a second.
        self._devices_dirty = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(_DEVICE_REFRESH_MS)
        self._refresh_timer.timeout.connect(self._flush_device_rows)
        # The BLE scan is started only *after* the dialog is first shown so the
        # window pops up immediately and the Bluetooth work happens afterwards.
        self._shown = False
        # Set just before accept(): distinguishes a confirmed hand-off (keep the
        # prepared source open) from a cancel/close (release the connection so
        # the device starts advertising again and reappears on reopen).
        self._accepted = False
        # Each device selection bumps this token; an in-flight connection whose
        # token is stale (the user clicked another device) is discarded. The
        # source being connected is tracked so it can be aborted on supersede.
        self._connect_token = 0
        self._connecting_source: Any = None
        # Explicit connect/cancel lifecycle: connection is only
        # attempted when the user clicks Connect, behind a modal progress popup.
        self._connected_info: Any = None
        self._connecting = False
        self._canceling = False
        self._progress: _ConnectProgressDialog | None = None

        # Status-line spinner animation (connecting/disconnecting feedback).
        self._spin_tick = 0
        self._spin_base = ""
        self._spin_tone = "neutral"
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(120)
        self._spin_timer.timeout.connect(self._on_spin_tick)
        self._disconnect_popup: _BusyDialog | None = None

        outer = QVBoxLayout(self)
        outer.setSpacing(6)

        # ── Session identity ──────────────────────────────────────────
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(6)
        outer.addLayout(form)

        self._tag = QLineEdit()
        self._tag.setPlaceholderText("auto-generated if left blank")
        form.addRow("Session tag:", self._tag)

        self._plot_secs = QDoubleSpinBox()
        self._plot_secs.setRange(1.0, 600.0)
        self._plot_secs.setValue(10.0)
        self._plot_secs.setSuffix(" s")
        form.addRow("Plot window:", self._plot_secs)

        # ── Data source ───────────────────────────────────────────────
        source_box = QGroupBox("Data source")
        slayout = QFormLayout(source_box)
        self._source_layout = slayout
        outer.addWidget(source_box)

        self._source_kind = QComboBox()
        for kind in SOURCE_KINDS:
            self._source_kind.addItem(kind, kind)
        slayout.addRow("Kind:", self._source_kind)

        # ── BLE controls (read-only Bluetooth status + live device list) ──
        # Built once; shown only while the BLE kind is selected. The Bluetooth
        # state is a *read-only* reflection of the host OS setting — the app
        # never turns the adapter on/off.
        self._ble_host = QWidget()
        ble_layout = QVBoxLayout(self._ble_host)
        ble_layout.setContentsMargins(0, 0, 0, 0)
        ble_layout.setSpacing(4)

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(6)
        self._bt_icon = QLabel("")
        self._bt_icon.setToolTip(
            "Read-only Bluetooth status, reflecting your host OS setting. The "
            "app does not change it; enable Bluetooth in your operating system."
        )
        status_row.addWidget(self._bt_icon)
        self._bt_status = QLabel("")
        self._bt_status.setWordWrap(True)
        status_row.addWidget(self._bt_status, 1)
        ble_layout.addLayout(status_row)

        # Guidance shown only when Bluetooth is off/unavailable.
        self._bt_hint = QLabel(
            "Bluetooth is turned off. Enable Bluetooth in your host operating "
            "system settings, then reopen this dialog."
        )
        self._bt_hint.setWordWrap(True)
        self._bt_hint.setStyleSheet("color:#AA8800;")
        self._bt_hint.setVisible(False)
        ble_layout.addWidget(self._bt_hint)

        self._ble_list = QTableWidget(0, 3)
        self._ble_list.setHorizontalHeaderLabels(["Name", "Address", "Signal"])
        self._ble_list.setMinimumHeight(140)
        self._ble_list.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._ble_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._ble_list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._ble_list.verticalHeader().setVisible(False)
        # Name and Address are user-resizable (drag the header dividers); the
        # Signal column keeps its content-based default width, which is good.
        header = self._ble_list.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(False)
        self._ble_list.setColumnWidth(0, 240)
        self._ble_list.setColumnWidth(1, 180)
        self._ble_list.itemSelectionChanged.connect(self._on_ble_device_selected)
        ble_layout.addWidget(self._ble_list)

        # Explicit Connect action — selection no longer auto-connects so the
        # GUI never appears to "halt" the moment a device is clicked. OK stays
        # disabled until Connect succeeds.
        connect_row = QHBoxLayout()
        connect_row.setContentsMargins(0, 0, 0, 0)
        connect_row.addStretch(1)
        self._connect_btn = QPushButton("Connect")
        self._connect_btn.setToolTip(
            "Connect to the selected device. A progress window appears while the "
            "connection is established; you can cancel it there."
        )
        self._connect_btn.setEnabled(False)
        self._connect_btn.clicked.connect(self._on_connect_clicked)
        connect_row.addWidget(self._connect_btn)
        ble_layout.addLayout(connect_row)

        slayout.addRow(self._ble_host)

        # ── UART discovery button + schema form ───────────────────────
        self._discover = QPushButton("Discover…")
        self._discover.clicked.connect(self._on_discover)
        slayout.addRow("", self._discover)

        self._source_form_host = QWidget()
        self._source_form = QFormLayout(self._source_form_host)
        slayout.addRow(self._source_form_host)

        # ── Protocol configuration ────────────────────────────────────
        # Protocol-specific options live in their own section, separate from
        # the data-source selection.
        protocol_box = QGroupBox("Protocol configuration")
        protocol_layout = QFormLayout(protocol_box)
        self._expect_crc = QCheckBox("Frames carry CRC-16 trailer")
        self._expect_crc.setChecked(True)
        protocol_layout.addRow("", self._expect_crc)
        outer.addWidget(protocol_box)

        self._ble_note = QLabel("")
        self._ble_note.setWordWrap(True)
        self._ble_note.setVisible(False)
        outer.addWidget(self._ble_note)

        # ── Buttons ───────────────────────────────────────────────────
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        self._ok_button = bb.button(QDialogButtonBox.StandardButton.Ok)
        if self._ok_button is not None:
            # "OK" is not self-explanatory for this dialog; spell out the action.
            self._ok_button.setText("Create session")
        outer.addWidget(bb)

        # Inline hint shown next to the buttons while OK is disabled.
        self._ok_hint = QLabel("")
        self._ok_hint.setWordWrap(True)
        self._ok_hint.setStyleSheet("color:#AA8800;")
        outer.addWidget(self._ok_hint)

        self._source_kind.currentIndexChanged.connect(self._on_source_kind_changed)
        # BLE is the default source kind.
        default_idx = self._source_kind.findData(DEFAULT_SOURCE_KIND)
        if default_idx >= 0:
            self._source_kind.setCurrentIndex(default_idx)
        self._rebuild_source_fields()
        log_user_action("Opened the New Session dialog")


    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def result_config(self) -> SessionConfig | None:
        return self._config

    def reject(self) -> None:  # noqa: D401 - log Cancel / Esc on the dialog
        log_user_action("Cancelled the New Session dialog")
        self._stop_live_scan()
        self._release_unconfirmed_connection()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        log_user_action("Closed the New Session dialog")
        self._stop_live_scan()
        self._release_unconfirmed_connection()
        super().closeEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        # Show the window first, then kick off the Bluetooth scan on the next
        # event-loop turn so the dialog pops up immediately (the bleak backend
        # import + scan no longer block the first paint).
        super().showEvent(event)
        if not self._shown:
            self._shown = True
            if self._source_kind.currentData() == "ble":
                QTimer.singleShot(0, self._start_live_scan)

    def _release_unconfirmed_connection(self) -> None:
        """Close a still-open BLE connection when the dialog is cancelled/closed.

        Without this, a device connected in the dialog stays connected after a
        cancel/close; a connected peripheral stops advertising, so it would not
        reappear in the scan on reopen. Skipped when the session was confirmed
        (the live source is handed off and must stay open).
        """
        if self._accepted:
            return
        self._abort_pending_connection()
        self._discard_prepared_source()

    def prepared_source(self) -> Any:
        """The live source opened in the dialog, handed to the new session.

        Non-``None`` only for BLE once a device has been selected and connected.
        UART currently reconnects at session start, so this is
        ``None`` for UART.
        """
        return self._prepared_source

    # ------------------------------------------------------------------
    # Source sub-form
    # ------------------------------------------------------------------

    def _on_source_kind_changed(self) -> None:
        kind = self._source_kind.currentData()
        log_user_action("Selected data source kind: %s", kind)
        # Drop any half-prepared BLE connection when leaving BLE.
        self._discard_prepared_source()
        self._rebuild_source_fields()

    def _rebuild_source_fields(self) -> None:
        while self._source_form.count():
            item = self._source_form.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._source_field_widgets.clear()

        kind = self._source_kind.currentData()
        is_ble = kind == "ble"

        # Stop any live BLE scan; it is (re)started only for the BLE kind below.
        self._stop_live_scan()

        # BLE uses its bespoke connect-in-dialog UI; UART uses the schema form.
        self._ble_host.setVisible(is_ble)
        self._source_form_host.setVisible(not is_ble)
        self._discover.setVisible(not is_ble)
        self._ble_note.setVisible(False)

        if is_ble:
            self._ble_list.setRowCount(0)
            self._seen_rows = {}
            self._device_infos = {}
            self._ble_selected_info = None
            self._ble_connected = False
            self._connected_info = None
            self._connecting = False
            self._canceling = False
            self._connect_btn.setEnabled(False)
            self._update_ok_state()
            # Live, continuous scan: devices stream into the list as the dialog
            # is open (no Refresh button, no fixed scan wait). The very first
            # scan is deferred to showEvent so the window pops up immediately;
            # later kind switches (dialog already shown) start it right away.
            if self._shown:
                self._start_live_scan()
            return

        if not kind:
            self._update_ok_state()
            return

        cls = source_for_kind(kind)
        schema = cls.config_schema()
        for f in schema.fields:
            widget = self._make_widget(f)
            self._source_field_widgets[f.name] = (f, widget)
            self._source_form.addRow(f"{f.label}{' *' if f.required else ''}:", widget)

        if kind == "uart":
            self._auto_fill_uart_port()
        self._update_ok_state()


    def _auto_fill_uart_port(self) -> None:
        try:
            cls = source_for_kind("uart")
            infos = cls.discover(self._platform)
        except Exception:
            return
        if not infos:
            return
        candidates = sorted(
            infos,
            key=lambda i: (0 if i.details.get("looks_like_nrf") else 1, i.id),
        )
        port_widget = self._source_field_widgets.get("port")
        if port_widget:
            _, w = port_widget
            w.setText(candidates[0].id)

    def _make_widget(self, f: ConfigField):
        if f.kind == "int":
            sb = QSpinBox()
            sb.setRange(-(2**31), 2**31 - 1)
            if isinstance(f.default, int):
                sb.setValue(f.default)
            return sb
        if f.kind == "float":
            ds = QDoubleSpinBox()
            ds.setDecimals(4)
            ds.setRange(0.0, 10_000.0)
            if isinstance(f.default, (int, float)):
                ds.setValue(float(f.default))
            return ds
        if f.kind == "bool":
            cb = QCheckBox()
            cb.setChecked(bool(f.default))
            return cb
        le = QLineEdit()
        le.setText(str(f.default) if f.default is not None else "")
        if f.help:
            le.setToolTip(f.help)
        return le

    def _read_value(self, f: ConfigField, widget: Any) -> Any:
        if f.kind == "int":
            return int(widget.value())
        if f.kind == "float":
            return float(widget.value())
        if f.kind == "bool":
            return bool(widget.isChecked())
        return widget.text().strip()

    # ------------------------------------------------------------------
    # Discover (UART only)
    # ------------------------------------------------------------------

    def _on_discover(self) -> None:
        kind = self._source_kind.currentData()
        log_user_action("Clicked Discover for source kind %r", kind)
        if kind == "uart":
            self._discover_uart()
        elif kind == "ble":
            self._discover_ble()
        else:
            QMessageBox.information(self, "Discover", "Discovery is not supported for this source.")

    def _discover_uart(self) -> None:
        cls = source_for_kind("uart")
        infos = cls.discover(self._platform)
        if not infos:
            QMessageBox.information(self, "Discover", "No serial ports detected.")
            return
        candidates = sorted(
            infos, key=lambda i: (0 if i.details.get("looks_like_nrf") else 1, i.id)
        )
        items = [
            f"{'★  ' if i.details.get('looks_like_nrf') else '   '}{i.display}"
            for i in candidates
        ]
        item, ok = QInputDialog.getItem(
            self, "Discover serial ports", "Select port:", items, 0, False
        )
        if not ok:
            return
        chosen = candidates[items.index(item)]
        log_user_action("Selected serial port %s from discovery", chosen.id)
        port_widget = self._source_field_widgets.get("port")
        if port_widget:
            _, w = port_widget
            w.setText(chosen.id)

    def _discover_ble(self) -> None:
        # BLE no longer uses the Discover… button; selection happens via the
        # Bluetooth toggle + live list. Kept as a no-op guard for safety.
        QMessageBox.information(
            self, "Bluetooth", "Turn Bluetooth on and pick a device from the list."
        )

    # ------------------------------------------------------------------
    # BLE: Bluetooth toggle + live list + connect-in-dialog
    # ------------------------------------------------------------------

    def _run_off_thread(self, fn, on_done, on_failed) -> None:
        """Run a blocking *fn* on a worker thread; deliver result on GUI thread.

        The result/failure handlers must run on the **GUI thread** (they touch
        widgets). Connecting the worker signals to *bound methods of self* — a
        QObject living on the GUI thread — makes Qt use a queued cross-thread
        connection, so the handlers run on the GUI thread. Connecting to a bare
        lambda instead would run them on the worker thread (direct connection)
        and crash the app the moment a widget is touched.
        """
        thread = QThread(self)
        worker = ScanWorker(fn)
        worker.moveToThread(thread)
        # Stash per-run state on the worker so the shared dispatch slots can
        # recover it via sender().
        worker._df_on_done = on_done
        worker._df_on_failed = on_failed
        worker._df_thread = thread
        thread.started.connect(worker.run)
        worker.done.connect(self._on_worker_done)
        worker.failed.connect(self._on_worker_failed)
        self._threads.append(thread)
        self._workers.append(worker)
        thread.start()

    def _on_worker_done(self, result) -> None:
        worker = self.sender()
        try:
            worker._df_on_done(result)
        finally:
            self._finish_worker(worker)

    def _on_worker_failed(self, exc) -> None:
        worker = self.sender()
        try:
            worker._df_on_failed(exc)
        finally:
            self._finish_worker(worker)

    def _finish_worker(self, worker) -> None:
        thread = worker._df_thread
        thread.quit()
        thread.wait()
        worker.deleteLater()
        if thread in self._threads:
            self._threads.remove(thread)
        if worker in self._workers:
            self._workers.remove(worker)

    def _set_bt_indicator(self, state: str, text: str) -> None:
        """Drive the read-only Bluetooth status icon/text and the guidance note.

        *state* is one of ``"on"``, ``"off"``, ``"checking"``. The app never
        changes the adapter; when off it only guides the user to host OS
        settings.
        """
        glyph = {"on": "🔵", "off": "⚪", "checking": "⏳"}.get(state, "⚪")
        color = {"on": "#2EA043", "off": "#9E9E9E", "checking": "#9E9E9E"}.get(state, "#9E9E9E")
        self._bt_icon.setText(glyph)
        self._bt_status.setText(text)
        self._bt_status.setStyleSheet(f"color:{color};")
        self._bt_hint.setVisible(state == "off")

    def _start_live_scan(self, *, clear: bool = True) -> None:
        """Begin a continuous BLE scan that streams detections into the list.

        With *clear* the list and dedup map are reset first (a fresh scan); when
        resuming after a connect attempt (*clear=False*) the already-discovered
        rows, the current selection and the connected highlight are preserved.
        """
        if self._scanner_thread is not None:
            return  # already scanning
        if clear:
            self._seen_rows = {}
            self._device_infos = {}
            self._ble_list.setRowCount(0)
            self._set_bt_indicator("checking", "Looking for Bluetooth devices…")
            self._update_ok_state()

        from data_forwarder_host.gui.ble_live_scanner import BleLiveScanner

        # No Qt parent: the dialog must never own (and therefore never destroy)
        # a running scan thread. Lifetime is managed explicitly via the
        # ``self._scanner_thread`` reference and the detach registry on teardown.
        thread = QThread()
        scanner = BleLiveScanner()
        scanner.moveToThread(thread)
        # Bound-method (GUI-thread QObject) slots → queued cross-thread delivery,
        # so all widget updates happen on the main thread.
        scanner.device_found.connect(self._on_live_device_found)
        scanner.state_changed.connect(self._on_live_scan_state)
        thread.started.connect(scanner.run)
        self._scanner = scanner
        self._scanner_thread = thread
        thread.start()
        # Repaint the list on a fixed cadence rather than per advertisement.
        self._refresh_timer.start()

    def _stop_live_scan(self) -> None:
        """Stop the live BLE scan and tear down its worker thread.

        Crash-safety: a still-running ``QThread`` must **never** be destroyed
        (Qt aborts the process with "QThread: Destroyed while thread is still
        running"). If the worker does not finish within the join timeout — e.g.
        a wedged Bluetooth backend — we detach the thread from this dialog and
        keep it alive in a process-wide registry until it finishes, instead of
        deleting it out from under itself.
        """
        self._refresh_timer.stop()
        scanner = self._scanner
        thread = self._scanner_thread
        self._scanner = None
        self._scanner_thread = None
        if scanner is not None:
            try:
                scanner.device_found.disconnect(self._on_live_device_found)
                scanner.state_changed.disconnect(self._on_live_scan_state)
            except (RuntimeError, TypeError):
                pass
            scanner.stop()
        if thread is None:
            return
        thread.quit()
        if thread.wait(6000):
            thread.deleteLater()
            if scanner is not None:
                scanner.deleteLater()
            return
        # Did not stop in time — keep it alive (detached) until it finishes so
        # destroying the dialog cannot abort the process.
        _detach_running_scan_thread(thread, scanner)

    def _on_live_device_found(self, info) -> None:
        """Coalesce a streamed detection; the list repaints on the 1 s timer."""
        addr = str(info.id)
        self._device_infos[addr] = info
        # Keep the cached selected/connected info fresh.
        if self._ble_selected_info is not None and str(self._ble_selected_info.id) == addr:
            self._ble_selected_info = info
        if self._connected_info is not None and str(self._connected_info.id) == addr:
            self._connected_info = info
        self._devices_dirty = True

    def _ordered_devices(self) -> list:
        """Devices in display order: connected first, then the default order."""
        ordered = order_ble_devices(list(self._device_infos.values()))
        cid = (
            str(self._connected_info.id)
            if (self._ble_connected and self._connected_info is not None)
            else None
        )
        if cid is not None and any(str(i.id) == cid for i in ordered):
            ordered = [i for i in ordered if str(i.id) == cid] + [
                i for i in ordered if str(i.id) != cid
            ]
        return ordered

    def _flush_device_rows(self) -> None:
        """Repaint the device list from coalesced detections (1 s cadence).

        Re-renders in sorted order (connected pinned on top) when the order
        changes; otherwise updates the visible rows in place. The current
        selection and the connected-row highlight are preserved.
        """
        if not self._devices_dirty:
            return
        self._devices_dirty = False
        ordered = self._ordered_devices()
        desired = [str(i.id) for i in ordered]
        if desired != self._current_row_order():
            self._render_device_rows(ordered)
        else:
            for info in ordered:
                self._update_device_row_cells(str(info.id), info)
        self._refresh_connected_highlight()
        self._update_scan_status()

    def _reorder_device_rows(self) -> None:
        """Force an immediate repaint (e.g. after connect/disconnect)."""
        if self._device_infos:
            self._devices_dirty = True
            self._flush_device_rows()

    def _current_row_order(self) -> list[str]:
        """Addresses in current table-row order (top to bottom)."""
        order: list[str | None] = [None] * self._ble_list.rowCount()
        for addr, row in self._seen_rows.items():
            if 0 <= row < len(order):
                order[row] = addr
        return [a for a in order if a is not None]

    def _render_device_rows(self, ordered) -> None:
        """Rebuild the table in *ordered* sequence, preserving the selection."""
        selected_addr = (
            str(self._ble_selected_info.id)
            if self._ble_selected_info is not None
            else None
        )
        # Block selection signals while rebuilding so the passive-selection
        # handler does not fire spuriously during the repopulate.
        self._ble_list.blockSignals(True)
        try:
            self._ble_list.setRowCount(0)
            self._seen_rows = {}
            select_row = None
            for info in ordered:
                addr = str(info.id)
                name, address, signal = self._device_columns(info)
                row = self._ble_list.rowCount()
                self._ble_list.insertRow(row)
                name_item = QTableWidgetItem(name)
                name_item.setData(Qt.ItemDataRole.UserRole, info)
                self._ble_list.setItem(row, 0, name_item)
                self._ble_list.setItem(row, 1, QTableWidgetItem(address))
                self._ble_list.setItem(row, 2, QTableWidgetItem(signal))
                self._seen_rows[addr] = row
                if addr == selected_addr:
                    select_row = row
        finally:
            self._ble_list.blockSignals(False)
        if select_row is not None:
            self._ble_list.selectRow(select_row)

    def _update_device_row_cells(self, addr: str, info) -> None:
        """Update one already-present row in place (no reordering)."""
        row = self._seen_rows.get(addr)
        if row is None:
            return
        name, address, signal = self._device_columns(info)
        name_item = self._ble_list.item(row, 0)
        if name_item is not None:
            name_item.setData(Qt.ItemDataRole.UserRole, info)
            name_item.setText(name)
        addr_item = self._ble_list.item(row, 1)
        if addr_item is not None:
            addr_item.setText(address)
        sig_item = self._ble_list.item(row, 2)
        if sig_item is not None:
            sig_item.setText(signal)

    def _on_live_scan_state(self, state: str) -> None:
        if state == "on":
            self._update_scan_status()
        elif state == "off":
            self._set_bt_indicator(
                "off",
                "Bluetooth is off or unavailable. Enable Bluetooth in your host "
                "operating system, then reopen this dialog.",
            )
        else:
            self._set_bt_indicator(
                "off",
                "Bluetooth status could not be determined. Enable Bluetooth in "
                "your host operating system, then reopen this dialog.",
            )
        self._update_ok_state()

    def _update_scan_status(self) -> None:
        """Refresh the live device-count line (unless a connect status is shown)."""
        # Don't clobber the green "Connected"/connecting/cancelling status line.
        if self._connecting or self._canceling or self._ble_connected:
            return
        has_selection = self._ble_selected_info is not None
        selected_name = (
            (self._ble_selected_info.details.get("name") or self._ble_selected_info.id)
            if has_selection
            else None
        )
        state, text = compute_scan_status(
            device_count=self._ble_list.rowCount(),
            has_selection=has_selection,
            selected_name=selected_name,
        )
        self._set_bt_indicator(state, text)

    @staticmethod
    def _device_columns(info) -> tuple[str, str, str]:
        """Return the (name, address, signal) cell texts for a discovered device."""
        star = "★ " if info.details.get("looks_like_nrf") else ""
        name = f"{star}{info.details.get('name') or '(unnamed)'}"
        rssi = info.details.get("rssi")
        signal = f"{rssi} dBm" if rssi is not None else "—"
        return name, str(info.id), signal

    def _set_connect_status(self, text: str, *, tone: str = "neutral") -> None:
        """Show a *static* connection status line (stops any spinner).

        *tone* selects the colour: ``"neutral"`` (default, e.g. while
        connecting — neither green nor red), ``"ok"`` (connected) or
        ``"error"`` (failure, red).
        """
        self._stop_status_spinner()
        color = {"neutral": "#9E9E9E", "ok": "#2EA043", "error": "#E5534B"}.get(
            tone, "#9E9E9E"
        )
        self._bt_status.setText(text)
        self._bt_status.setStyleSheet(f"color:{color};")
        self._bt_hint.setVisible(False)

    # -- animated status spinner -------------------------------------------

    def _start_status_spinner(self, base_text: str, *, tone: str = "neutral") -> None:
        """Animate *base_text* with a trailing spinner (e.g. while connecting)."""
        self._spin_base = base_text
        self._spin_tone = tone
        self._spin_tick = 0
        self._apply_spinner_text()
        if not self._spin_timer.isActive():
            self._spin_timer.start()

    def _on_spin_tick(self) -> None:
        self._spin_tick += 1
        self._apply_spinner_text()

    def _apply_spinner_text(self) -> None:
        color = {"neutral": "#9E9E9E", "ok": "#2EA043", "error": "#E5534B"}.get(
            self._spin_tone, "#9E9E9E"
        )
        self._bt_status.setText(f"{self._spin_base}… {spinner_frame(self._spin_tick)}")
        self._bt_status.setStyleSheet(f"color:{color};")
        self._bt_hint.setVisible(False)

    def _stop_status_spinner(self) -> None:
        if self._spin_timer.isActive():
            self._spin_timer.stop()

    # -- disconnecting popup ------------------------------------------------

    def _show_disconnecting_popup(self) -> None:
        if self._disconnect_popup is None:
            self._disconnect_popup = _BusyDialog(
                self, "Disconnecting from the current device…"
            )
        self._disconnect_popup.show()
        self._disconnect_popup.raise_()

    def _close_disconnecting_popup(self) -> None:
        if self._disconnect_popup is not None:
            self._disconnect_popup.close()
            self._disconnect_popup.deleteLater()
            self._disconnect_popup = None

    @staticmethod
    def _safe_close(src) -> None:
        try:
            src.close()
        except Exception:
            pass

    def _abort_pending_connection(self) -> None:
        """Best-effort stop of an in-flight connection that has been superseded."""
        src = self._connecting_source
        self._connecting_source = None
        if src is not None:
            self._safe_close(src)

    def _on_ble_device_selected(self) -> None:
        row = self._ble_list.currentRow()
        if row < 0:
            return
        name_item = self._ble_list.item(row, 0)
        if name_item is None:
            return
        info = name_item.data(Qt.ItemDataRole.UserRole)

        same = (
            self._ble_selected_info is not None
            and self._ble_selected_info.id == info.id
        )
        self._ble_selected_info = info
        if not same:
            log_user_action(
                "Selected BLE device %r (%s)", info.details.get("name"), info.id
            )

        # Selection NEVER connects or disconnects anything. The active
        # connection (if any) is left fully intact; selecting only re-targets the
        # Connect/Disconnect button and the OK gating onto the chosen device.
        if self._selected_is_connected():
            self._set_connect_status(
                f"Connected to {self._connected_name()}. Ready to start.", tone="ok"
            )
        else:
            # Route the "press Connect" guidance through the shared idle-status
            # path so the periodic scan refresh reinforces (never clobbers) it.
            self._update_scan_status()
        self._update_connect_button()
        self._update_ok_state()

    def _connected_name(self) -> str:
        """Display name of the currently connected device (or a fallback)."""
        info = self._connected_info
        if info is None:
            return "device"
        return info.details.get("name") or info.id

    def _selected_is_connected(self) -> bool:
        """True when the selected device is the one that is currently connected."""
        return (
            self._ble_connected
            and self._connected_info is not None
            and self._ble_selected_info is not None
            and self._ble_selected_info.id == self._connected_info.id
        )

    def _on_connect_clicked(self) -> None:
        if self._connecting or self._canceling:
            return
        # The button is a Connect/Disconnect toggle. Disconnect when
        # the connected device is the toggle's target: it is the selected one,
        # or nothing else is selected.
        if self._ble_connected and (
            self._selected_is_connected() or self._ble_selected_info is None
        ):
            log_user_action("Clicked Disconnect in the New Session dialog")
            self._begin_disconnect()
            return
        info = self._ble_selected_info
        if info is None:
            return
        log_user_action("Clicked Connect in the New Session dialog")
        self._connect_token += 1
        self._begin_connect(self._connect_token, info)

    def _begin_disconnect(self) -> None:
        """Tear down the live connection (Disconnect button)."""
        src = self._prepared_source
        # Invalidate any in-flight resolution and drop all connection state.
        self._connect_token += 1
        self._connecting_source = None
        self._prepared_source = None
        self._ble_connected = False
        self._connected_info = None
        self._set_connect_status("Disconnected. Press Connect to reconnect.")
        if src is not None:
            self._show_disconnecting_popup()
            # Keep the button enabled: the modal disconnecting popup
            # already blocks interaction, so the control is never greyed out.
            self._update_connect_button()
            self._run_off_thread(
                lambda: self._safe_close(src),
                lambda _r: self._on_disconnect_done(),
                lambda _e: self._on_disconnect_done(),
            )
        else:
            self._update_connect_button()
        self._update_ok_state()

    def _on_disconnect_done(self) -> None:
        self._close_disconnecting_popup()
        self._update_connect_button()
        self._update_ok_state()
        self._reorder_device_rows()

    def _begin_connect(self, token: int, info) -> None:
        """Open the modal progress popup and connect to *info* off-thread.

        When another device is already connected, switching to a new one
        releases the previous source as part of this same connect (still behind
        the modal popup), so selection itself stayed side-effect-free
        and the disconnect happens only on the explicit Connect press.
        """
        if token != self._connect_token:
            return  # a newer action superseded this one

        name = info.details.get("name") or info.id

        # Pause the live scan for the duration of the connect: a scan and a
        # connect contending for the same adapter can stall or fail. The list is
        # kept (rows/selection/highlight) and streaming resumes once the connect
        # attempt resolves.
        self._stop_live_scan()

        # Release whatever was connected/half-connected to a different device.
        old_sources = [
            s for s in (self._connecting_source, self._prepared_source) if s is not None
        ]
        self._connecting_source = None
        self._prepared_source = None
        self._ble_connected = False
        self._connected_info = None
        self._refresh_connected_highlight()

        self._connecting = True
        self._update_connect_button()
        self._set_connect_status(f"Connecting to {name}.")

        self._progress = _ConnectProgressDialog(self, name)
        self._progress.cancel_requested.connect(
            lambda t=token: self._on_connect_cancel(t)
        )
        self._progress.show()
        self._progress.raise_()

        from data_forwarder_host.source.ble_nus import BleNusSource

        src = BleNusSource(address=info.id, name=info.details.get("name") or "")
        self._connecting_source = src

        def _connect():
            for old in old_sources:
                self._safe_close(old)
            src.open()
            return src

        self._run_off_thread(
            _connect,
            lambda r: self._on_ble_connect_done(token, r),
            lambda e: self._on_ble_connect_failed(token, e),
        )

    def _close_progress(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress.deleteLater()
            self._progress = None

    def _on_connect_cancel(self, token: int) -> None:
        """User pressed Cancel: invalidate the attempt and tear it down safely.

        The in-flight ``open()`` cannot be interrupted mid-flight, so we mark it
        stale (bump the token) and keep Connect disabled until the worker
        resolves and the source is released. This serialises connect/cancel and
        prevents the "too many connections right after cancelling" device fault.
        """
        if token != self._connect_token:
            return  # already resolved
        self._connect_token += 1
        self._connecting = False
        self._canceling = True
        self._update_connect_button()
        log_user_action("Cancelled BLE connection to selected device")
        # _on_ble_connect_done/_failed will fire for the stale token and call
        # _finish_cancel_if_pending(), which closes the popup and re-enables UI.

    def _finish_cancel_if_pending(self) -> None:
        if not self._canceling:
            return
        self._canceling = False
        self._connecting_source = None
        self._close_progress()
        self._set_connect_status("Connection cancelled. Press Connect to retry.")
        self._update_connect_button()
        self._update_ok_state()
        self._resume_live_scan()

    def _resume_live_scan(self) -> None:
        """Restart streaming detections after a connect attempt resolves."""
        if self._source_kind.currentData() == "ble" and self.isVisible():
            self._start_live_scan(clear=False)

    def _on_ble_connect_done(self, token: int, src) -> None:
        if token != self._connect_token:
            # Stale (superseded or cancelled) — release and resolve cancellation.
            self._safe_close(src)
            self._finish_cancel_if_pending()
            return
        self._connecting = False
        self._connecting_source = None
        self._prepared_source = src
        self._ble_connected = True
        self._connected_info = self._ble_selected_info
        info = self._ble_selected_info
        name = (info.details.get("name") or info.id) if info else "device"
        log_user_action("Connected to BLE device %s", name)
        self._close_progress()
        self._set_connect_status(f"Connected to {name}. Ready to start.", tone="ok")
        self._update_connect_button()
        self._update_ok_state()
        self._reorder_device_rows()
        self._resume_live_scan()

    def _on_ble_connect_failed(self, token: int, exc) -> None:
        if token != self._connect_token:
            self._finish_cancel_if_pending()
            return
        self._connecting = False
        self._connecting_source = None
        self._ble_connected = False
        self._discard_prepared_source(keep_selection=True)
        log_user_action("BLE connection failed: %s", exc)
        self._close_progress()
        self._set_connect_status(
            f"Connection failed: {exc}\nPress Connect to retry.", tone="error"
        )
        self._update_connect_button()
        self._update_ok_state()
        self._resume_live_scan()

    def _update_connect_button(self) -> None:
        """Drive the Connect/Disconnect toggle (never permanently disabled)."""
        selected_id = (
            self._ble_selected_info.id if self._ble_selected_info is not None else None
        )
        connected_id = (
            self._connected_info.id
            if (self._ble_connected and self._connected_info is not None)
            else None
        )
        label, enabled = compute_connect_button(
            selected_id=selected_id,
            connected_id=connected_id,
            connecting=self._connecting,
            canceling=self._canceling,
        )
        self._connect_btn.setText(label)
        self._connect_btn.setEnabled(enabled)
        self._refresh_connected_highlight()

    def _refresh_connected_highlight(self) -> None:
        """Mark which device row is currently connected.

        The connected device is shown with a check mark, bold text and a soft
        green row background so the user has clear, persistent confirmation of
        which device the live connection is on (independent of the selection).
        """
        connected_id = (
            self._connected_info.id
            if (self._ble_connected and self._connected_info is not None)
            else None
        )
        # Translucent green tint (not an opaque light fill) so the row stays
        # readable on both light and dark themes — it shades the existing base
        # colour instead of replacing it with a light background under light text.
        ok_bg = QBrush(QColor(46, 160, 67, 70))
        clear_bg = QBrush()
        for row in range(self._ble_list.rowCount()):
            name_item = self._ble_list.item(row, 0)
            if name_item is None:
                continue
            info = name_item.data(Qt.ItemDataRole.UserRole)
            base_name, _address, _signal = self._device_columns(info)
            is_connected = (
                connected_id is not None
                and info is not None
                and info.id == connected_id
            )
            name_item.setText(f"✓ {base_name}" if is_connected else base_name)
            for col in range(self._ble_list.columnCount()):
                cell = self._ble_list.item(row, col)
                if cell is None:
                    continue
                cell.setBackground(ok_bg if is_connected else clear_bg)
                font = cell.font()
                font.setBold(is_connected)
                cell.setFont(font)


    def _discard_prepared_source(self, *, keep_selection: bool = False) -> None:
        src = self._prepared_source
        self._prepared_source = None
        self._ble_connected = False
        if not keep_selection:
            self._ble_selected_info = None
        if src is not None:
            try:
                src.close()
            except Exception:
                pass

    def _update_ok_state(self) -> None:
        kind = self._source_kind.currentData()
        enabled, hint = compute_ok_state(
            kind,
            ble_selected=self._ble_selected_info is not None,
            ble_connected=self._selected_is_connected(),
        )
        self._ok_button.setEnabled(enabled)
        self._ok_hint.setText(hint)
        self._ok_hint.setVisible(bool(hint))


    # ------------------------------------------------------------------
    # Accept
    # ------------------------------------------------------------------

    def _accept(self) -> None:
        kind = self._source_kind.currentData()
        if not kind:
            QMessageBox.warning(self, "Session", "Pick a data source.")
            return

        params: dict[str, Any] = {}
        for fname, (f, widget) in self._source_field_widgets.items():
            v = self._read_value(f, widget)
            if f.required and (v is None or v == ""):
                QMessageBox.warning(self, "Session", f"Source field {f.label!r} is required.")
                return
            if v is not None and v != "":
                params[fname] = v

        # BLE: identity comes from the device selected + connected in this
        # dialog. The live source is handed off via prepared_source(); the
        # config stores the address (+ name) so a reopened session can
        # reconnect.
        if kind == "ble":
            info = self._ble_selected_info
            if info is None or not self._selected_is_connected():
                QMessageBox.warning(self, "Session", _HINT_BLE_SELECT)
                return
            params["address"] = info.id
            name = info.details.get("name") or ""
            if name:
                params["name"] = name

        # Normalize UART port path: if it looks like a ttyACM/ttyUSB but is missing
        # the leading slash, add it (e.g. "dev/ttyACM1" → "/dev/ttyACM1").
        if kind == "uart" and "port" in params:
            port = str(params["port"]).strip()
            if port and not port.startswith("/") and not port.startswith("COM"):
                if port.startswith("dev/"):
                    params["port"] = "/" + port
                    # Inform user of the normalization
                    port_widget = self._source_field_widgets.get("port")
                    if port_widget:
                        _, w = port_widget
                        w.setText(params["port"])

        tag = self._tag.text().strip() or f"{kind}-{uuid.uuid4().hex[:6]}"
        source_spec = SourceSpec(kind=kind, role=RoleMode.LISTENER, params=params)

        # The recording label and output directory are defined in the session
        # window (control panel), not here — a session can be created without a
        # label; the label is required only to *start* recording.
        cfg = SessionConfig(
            tag=tag,
            source=source_spec,
            expect_crc=self._expect_crc.isChecked(),
            plot_window_seconds=float(self._plot_secs.value()),
            # The bandwidth window is independent of the plot window: it starts
            # at a fixed 1 s default and is adjustable only from the session
            # window.
            bandwidth_window_seconds=DEFAULT_BANDWIDTH_WINDOW_SECONDS,
        )
        try:
            cfg.validate()
        except ConfigError as exc:
            QMessageBox.critical(self, "Invalid configuration", str(exc))
            return
        self._config = cfg
        log_user_action(
            "Confirmed new session: tag=%s source=%s", cfg.tag, source_spec.kind
        )
        self._accepted = True
        self._stop_live_scan()
        self.accept()
