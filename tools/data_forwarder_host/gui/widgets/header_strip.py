# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Top-of-tab strip showing session identity and live status."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy

from data_forwarder_host.gui.widgets.bandwidth_details import BandwidthDetailsDialog
from data_forwarder_host.session.controller import SessionController
from data_forwarder_host.session.states import SessionState
from data_forwarder_host.utils.logging import log_user_action


class HeaderStrip(QFrame):
    def __init__(self, controller: SessionController, parent=None) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self.setFrameShape(QFrame.Shape.StyledPanel)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)

        self._tag = QLabel(f"<b>{controller.config.tag}</b>")
        self._source = QLabel(controller.describe_source())
        self._protocol = QLabel(controller.describe_protocol())
        # Single combined status label (state + recording).
        self._status = QLabel()

        for w in (self._tag, self._source, self._protocol, self._status):
            w.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            lay.addWidget(w)
            lay.addSpacing(12)
        lay.addStretch(1)

        # Live throughput readout, right-aligned after the stretch.
        self._bandwidth = QLabel()
        self._bandwidth.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._bandwidth.setToolTip(
            "Live throughput over the bandwidth measurement window"
        )
        lay.addWidget(self._bandwidth)

        # Button: opens the detailed bandwidth / reception sub-window.
        # Uses a text label (an icon glyph rendered as an empty box
        # on some platforms).
        self._bandwidth_info = QPushButton("Bandwidth details")
        self._bandwidth_info.setToolTip("More data-transfer details")
        self._bandwidth_info.clicked.connect(self._open_bandwidth_details)
        lay.addWidget(self._bandwidth_info)
        self._bandwidth_dialog: BandwidthDetailsDialog | None = None

        # Timer that fires every second while recording to update the counter.
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._refresh_status)

        # Timer that continuously refreshes the bandwidth readout.
        self._bandwidth_timer = QTimer(self)
        self._bandwidth_timer.setInterval(500)
        self._bandwidth_timer.timeout.connect(self._refresh_bandwidth)
        self._bandwidth_timer.start()

        controller.state_changed.connect(lambda _s: self._refresh_status())
        controller.recording_changed.connect(self._on_recording_changed)
        controller.session_phase_changed.connect(lambda _p: self._refresh_status())

        self._refresh_status()
        self._refresh_bandwidth()

    def _open_bandwidth_details(self) -> None:
        log_user_action("Clicked Bandwidth details")
        if self._bandwidth_dialog is None:
            self._bandwidth_dialog = BandwidthDetailsDialog(self._ctrl, self)
        self._bandwidth_dialog.show()
        self._bandwidth_dialog.raise_()
        self._bandwidth_dialog.activateWindow()

    def _on_recording_changed(self, recording: bool) -> None:
        if recording:
            self._elapsed_timer.start()
        else:
            self._elapsed_timer.stop()
        self._refresh_status()

    def _refresh_bandwidth(self) -> None:
        s = self._ctrl.bandwidth_sample()
        self._bandwidth.setText(
            f'<span style="color:#6B7280">'
            f"{_fmt_bytes_per_s(s.bytes_per_second)} · "
            f"{s.messages_per_second:.0f} msg/s · "
            f"{s.channels_per_second:.0f} ch/s</span>"
        )

    def _refresh_status(self) -> None:
        state = self._ctrl.state()
        recording = self._ctrl.is_recording()
        if not recording and self._elapsed_timer.isActive():
            self._elapsed_timer.stop()

        if recording:
            elapsed = self._ctrl.recorder.elapsed_seconds()
            if elapsed is not None:
                m = int(elapsed) // 60
                s = int(elapsed) % 60
                elapsed_str = f"  {m}:{s:02d}"
            else:
                elapsed_str = ""
            self._status.setText(
                f'<span style="color:#d62728">⏺ Recording{elapsed_str}</span>'
            )
        elif state == SessionState.RUNNING:
            phase = self._ctrl.session_phase()
            if phase == "awaiting device metadata":
                self._status.setText('<span style="color:#6B7280">◔ Awaiting Metadata</span>')
            elif phase == "ready to record":
                self._status.setText('<span style="color:#2ca02c">▶ Ready To Record</span>')
            elif phase == "stopping":
                self._status.setText('<span style="color:#6B7280">◷ Stopping</span>')
            else:
                self._status.setText('<span style="color:#2ca02c">▶ Streaming</span>')
        elif state == SessionState.ERROR:
            self._status.setText('<span style="color:#d62728">✕ Error</span>')
        elif state == SessionState.STOPPED:
            self._status.setText('<span style="color:#888888">◼ Stopped</span>')
        else:
            self._status.setText('<span style="color:#888888">○ Configured</span>')


def _fmt_bytes_per_s(value: float) -> str:
    """Human-readable bytes/second with a binary-ish scale."""
    units = ("B/s", "KB/s", "MB/s", "GB/s")
    v = float(value)
    idx = 0
    while v >= 1000.0 and idx < len(units) - 1:
        v /= 1000.0
        idx += 1
    if idx == 0:
        return f"{v:.0f} {units[idx]}"
    return f"{v:.1f} {units[idx]}"
