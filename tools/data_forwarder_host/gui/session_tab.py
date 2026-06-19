# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Per-session tab assembled from the reusable widgets."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QLabel,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from data_forwarder_host.gui.widgets.charts_panel import ChartsPanel
from data_forwarder_host.gui.widgets.control_panel import ControlPanel
from data_forwarder_host.gui.widgets.error_panel import ErrorPanel
from data_forwarder_host.gui.widgets.header_strip import HeaderStrip
from data_forwarder_host.gui.widgets.recording_label_strip import RecordingLabelStrip
from data_forwarder_host.gui.widgets.save_banner import SaveBanner
from data_forwarder_host.gui.widgets.session_log_panel import SessionLogPanel
from data_forwarder_host.session.controller import SessionController
from data_forwarder_host.session.states import SessionState


class SessionTab(QWidget):
    def __init__(self, controller: SessionController, parent=None) -> None:
        super().__init__(parent)
        self._ctrl = controller

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # ── Header strip ─────────────────────────────────────────────
        outer.addWidget(HeaderStrip(controller))

        self._no_data_banner = QLabel("No data received yet")
        self._no_data_banner.setStyleSheet(
            "QLabel { background:palette(alternate-base); color:palette(text); "
            "border:1px solid palette(mid); border-radius:4px; padding:6px 10px; "
            "font-weight:600; }"
        )
        self._no_data_banner.setVisible(False)
        outer.addWidget(self._no_data_banner)

        self._sensor_data_seen = False

        # ── Main horizontal split: narrow controls | scrollable content ─
        self._body = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(self._body, 1)

        ctrl_panel = ControlPanel(controller)
        # No maximum width: the panel must stay freely resizable so the full
        # values in "Session Configuration" and "Session Info" can be read. A
        # capped width would let the splitter only shrink the panel, never
        # enlarge it.
        self._body.addWidget(ctrl_panel)

        # ── One common scroll area: charts on top, log consoles below ─
        # ChartsPanel has no internal scroll.  SessionLogPanel log
        # consoles each carry their own internal scrollbar.
        right_content = QWidget()
        right_layout = QVBoxLayout(right_content)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        # Prominent recording-label strip sits at the top-right, beside/above
        # the charts and separate from the control-panel form.
        self._label_strip = RecordingLabelStrip(controller)
        self._label_strip.label_changed.connect(ctrl_panel.refresh_record_state)
        right_layout.addWidget(self._label_strip)
        self._charts_panel = ChartsPanel(controller.data_model)
        right_layout.addWidget(self._charts_panel)
        self._log_panel = SessionLogPanel(controller)
        right_layout.addWidget(self._log_panel)
        right_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(right_content)
        self._body.addWidget(scroll)

        self._body.setStretchFactor(0, 0)   # controls: no extra horizontal growth
        self._body.setStretchFactor(1, 1)   # charts+logs: take all extra space

        # ── Last-saved CSV banner (bottom of the tab) ────────
        self._save_banner = SaveBanner()
        outer.addWidget(self._save_banner)

        # ── Saving progress bar (bottom of the session window) ─
        # Shown only while a recording-stop CSV dump is being written. The dump
        # runs incrementally off the event loop, so this bar animates while the
        # rest of the UI stays responsive.
        self._saving_bar = QProgressBar()
        self._saving_bar.setTextVisible(True)
        self._saving_bar.setFormat("Saving recording to CSV…")
        self._saving_bar.setVisible(False)
        outer.addWidget(self._saving_bar)

        # ── Error panel (collapsible, at the bottom) ──────────────────
        self._error_panel = ErrorPanel(controller.error_log, controller=controller)
        # Hidden by default; revealed via View ▸ Show error panel.
        self._error_panel.setVisible(False)
        outer.addWidget(self._error_panel)

        controller.state_changed.connect(self._on_state_changed)
        controller.message_received.connect(self._on_message_received)
        controller.recording_saved.connect(self._save_banner.show_saved)
        controller.recording_save_started.connect(self._on_save_started)
        controller.recording_save_progress.connect(self._on_save_progress)
        controller.recording_saved.connect(self._on_save_done)
        controller.recording_save_failed.connect(self._on_save_done)

    @property
    def controller(self) -> SessionController:
        return self._ctrl

    def focus_recording_label(self) -> None:
        """Move keyboard focus to the recording-label entry.

        Invoked right after the tab is created so the cursor waits in the
        label field for immediate typing.
        """
        self._label_strip.focus_label()

    def set_panel_visible(self, key: str, on: bool) -> None:
        """Show/hide one of the session panels (View ▸ Panels)."""
        if key == "combined":
            self._charts_panel.set_combined_visible(on)
        elif key == "individual":
            self._charts_panel.set_individual_visible(on)
        elif key == "channel_ascii":
            self._log_panel.set_channel_ascii_visible(on)
        elif key == "decoded_frames":
            self._log_panel.set_decoded_frames_visible(on)

    def _on_state_changed(self, state: object) -> None:
        running = state == SessionState.RUNNING
        if running:
            self._sensor_data_seen = False
            self._no_data_banner.setVisible(True)
        else:
            self._no_data_banner.setVisible(False)

    def _on_message_received(self, msg: object) -> None:
        kind = getattr(msg, "kind", "")
        if kind == "sensor_data":
            self._sensor_data_seen = True
            self._no_data_banner.setVisible(False)

    # ------------------------------------------------------------------
    # Saving progress bar
    # ------------------------------------------------------------------

    def _on_save_started(self) -> None:
        # Range 0..0 renders an indeterminate (busy) bar until the first
        # progress update tells us the total row count (unknown for spilled
        # recordings, which stay indeterminate throughout).
        self._saving_bar.setRange(0, 0)
        self._saving_bar.setFormat("Saving recording to CSV…")
        self._saving_bar.setVisible(True)

    def _on_save_progress(self, rows_written: int, total: int) -> None:
        if total > 0:
            self._saving_bar.setRange(0, total)
            self._saving_bar.setValue(min(rows_written, total))
            self._saving_bar.setFormat(f"Saving recording to CSV… %p% ({rows_written}/{total})")
        else:
            self._saving_bar.setRange(0, 0)
            self._saving_bar.setFormat(f"Saving recording to CSV… ({rows_written} rows)")

    def _on_save_done(self, *_args: object) -> None:
        self._saving_bar.setVisible(False)
        self._saving_bar.reset()

    # ------------------------------------------------------------------
    # Layout state persistence
    # ------------------------------------------------------------------

    def get_layout_state(self) -> dict:
        """Return a serialisable snapshot of the current layout state."""
        state = self._log_panel.get_layout_state()
        sizes = self._body.sizes()
        total = sum(sizes) if sizes else 0
        if total > 0:
            state["body_splitter_ratio"] = sizes[0] / total
        return state

    def apply_layout_state(self, state: dict) -> None:
        """Restore layout from a previously saved state dict."""
        self._log_panel.apply_layout_state(state)
        ratio = state.get("body_splitter_ratio")
        if ratio is not None:
            # Defer until after the widget has been shown and laid out.
            def _restore_splitter() -> None:
                total = self._body.width()
                if total > 0:
                    left = int(total * float(ratio))
                    self._body.setSizes([left, total - left])
            QTimer.singleShot(100, _restore_splitter)
