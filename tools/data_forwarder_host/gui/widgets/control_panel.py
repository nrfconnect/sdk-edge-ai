# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Per-tab control panel.

Session model
-------------
The device channel is opened automatically when the session tab is created, so
``session_info`` is observed and the live plots update without user action.
Two buttons drive CSV capture on top of that stream:

* **Record** — begin capturing samples to the output CSV. Enabled only once a
  valid recording label is defined *and* the device has sent
  ``session_info`` (metadata gate). If the channel previously errored, pressing
  Record clears the error and reopens it first.
* **Stop**   — stop capturing and finalise the CSV file. The device channel
  stays open and listening so the user can immediately record again.

There is no sink subsystem: recorded data is written to a single CSV file when
recording stops.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from data_forwarder_host.session.controller import SessionController
from data_forwarder_host.session.states import SessionState
from data_forwarder_host.utils.logging import log_user_action

log = logging.getLogger(__name__)


class ControlPanel(QWidget):
    def __init__(self, controller: SessionController, parent=None) -> None:
        super().__init__(parent)
        self._ctrl = controller

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # ── Recording ─────────────────────────────────────────────────
        outer.addWidget(self._section_label("Recording"))
        hint = QLabel(
            "<small style='color:gray;'>The device channel opens automatically. "
            "Record starts CSV capture once session_info arrives; Stop ends "
            "capture and writes the CSV (the channel keeps listening).</small>"
        )
        hint.setWordWrap(True)
        outer.addWidget(hint)

        rec_form = QFormLayout()
        rec_form.setContentsMargins(0, 0, 0, 0)

        out_row = QWidget()
        out_lay = QHBoxLayout(out_row)
        out_lay.setContentsMargins(0, 0, 0, 0)
        self._output_dir = QLineEdit()
        self._output_dir.setText(controller.config.output_dir)
        self._output_dir.setPlaceholderText("default: recordings")
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(32)
        out_lay.addWidget(self._output_dir)
        out_lay.addWidget(btn_browse)
        rec_form.addRow("Output dir:", out_row)
        outer.addLayout(rec_form)

        row = QHBoxLayout()
        self._btn_record = QPushButton("Record")
        self._btn_record.setMinimumHeight(44)
        self._btn_record.setStyleSheet(
            "QPushButton { font-size: 13px; font-weight: bold; padding: 6px 12px; }"
        )
        self._btn_record.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setMinimumHeight(44)
        self._btn_stop.setStyleSheet(
            "QPushButton { font-size: 13px; font-weight: bold; padding: 6px 12px; }"
        )
        self._btn_stop.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        row.addWidget(self._btn_record)
        row.addWidget(self._btn_stop)
        outer.addLayout(row)

        self._label_hint = QLabel()
        self._label_hint.setWordWrap(True)
        self._label_hint.setStyleSheet("color:#AA8800; font-size: 11px;")
        outer.addWidget(self._label_hint)

        # ── Session configuration ─────────────────────────────────────
        outer.addWidget(self._section_label("Session Configuration"))
        cfg_form = QFormLayout()
        cfg_form.setContentsMargins(0, 0, 0, 0)
        self._plot_secs = QDoubleSpinBox()
        self._plot_secs.setRange(1.0, 600.0)
        self._plot_secs.setValue(controller.data_model.plot_window_seconds)
        self._plot_secs.setSuffix(" s")
        self._plot_secs.setToolTip("Rolling plot window length. Change takes effect immediately.")
        cfg_form.addRow("Plot window:", self._plot_secs)

        self._bandwidth_secs = QDoubleSpinBox()
        self._bandwidth_secs.setRange(0.1, 600.0)
        self._bandwidth_secs.setSingleStep(0.1)
        self._bandwidth_secs.setValue(controller.bandwidth.window_seconds)
        self._bandwidth_secs.setSuffix(" s")
        self._bandwidth_secs.setToolTip(
            "Trailing window for the live bandwidth readout. Change takes effect immediately."
        )
        cfg_form.addRow("Bandwidth window:", self._bandwidth_secs)
        outer.addLayout(cfg_form)

        outer.addWidget(self._section_label("Session Info"))
        self._phase = QLabel("awaiting device metadata...")
        self._phase.setStyleSheet("color:palette(mid); font-size:11px; font-weight:bold;")
        outer.addWidget(self._phase)

        # Structured metadata fields (created dynamically on first session_info)
        self._metadata_form = QFormLayout()
        self._metadata_form.setContentsMargins(0, 0, 0, 0)
        self._metadata_form.setSpacing(4)
        self._metadata_fields: dict[str, QLineEdit] = {}
        outer.addLayout(self._metadata_form)

        self._metadata_status = QLabel()
        self._metadata_status.setStyleSheet("color:palette(mid); font-size:10px; font-style:italic;")
        self._metadata_status.setText("waiting for session_info from device...")
        outer.addWidget(self._metadata_status)

        outer.addStretch(1)

        # ── Wiring ────────────────────────────────────────────────────
        self._btn_record.clicked.connect(self._on_record)
        self._btn_stop.clicked.connect(self._on_stop)
        btn_browse.clicked.connect(self._on_browse_output_dir)
        self._output_dir.textChanged.connect(self._on_output_dir_changed)
        self._plot_secs.valueChanged.connect(
            lambda v: controller.data_model.set_plot_window_seconds(v)
        )
        self._bandwidth_secs.valueChanged.connect(
            lambda v: controller.set_bandwidth_window_seconds(v)
        )
        # Keep this control in sync when the window is changed elsewhere
        # (e.g. the Bandwidth details sub-window).
        controller.bandwidth_window_changed.connect(self._on_bandwidth_window_changed)

        controller.state_changed.connect(self._on_state_changed)
        controller.recording_changed.connect(self._on_recording_changed)
        controller.recording_empty.connect(self._on_recording_empty)
        # While a recording is being written to CSV, both Record and Stop stay
        # locked; Record re-enables only once the save completes (Feature 1).
        controller.recording_save_started.connect(self._refresh_buttons)
        controller.recording_saved.connect(self._refresh_buttons)
        controller.recording_save_failed.connect(self._refresh_buttons)
        controller.session_info_received.connect(self._on_session_info_received)
        controller.session_info_mismatch.connect(self._on_session_info_mismatch)
        controller.session_phase_changed.connect(self._on_session_phase_changed)

        self._refresh_buttons()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(f"<b>{text}</b>")
        lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
        return lbl

    def _on_bandwidth_window_changed(self, seconds: float) -> None:
        """Reflect an external bandwidth-window change without re-emitting."""
        if abs(self._bandwidth_secs.value() - seconds) < 1e-9:
            return
        blocked = self._bandwidth_secs.blockSignals(True)
        self._bandwidth_secs.setValue(seconds)
        self._bandwidth_secs.blockSignals(blocked)

    def _refresh_buttons(self, *_args: object) -> None:
        state = self._ctrl.state()
        recording = self._ctrl.is_recording()
        can_record = self._ctrl.can_record()
        metadata_ready = self._ctrl.metadata_ready()
        saving = self._ctrl.is_saving()

        # Record starts CSV capture. The channel is already streaming (opened on
        # tab creation); enable only when metadata has arrived, a valid label is
        # defined, we are not already capturing, and no previous recording is
        # still being written to CSV (Feature 1).
        self._btn_record.setEnabled(
            can_record and metadata_ready and not recording and not saving
        )
        # Stop ends the active capture (the channel keeps listening afterwards).
        # It is also locked while a CSV save is draining.
        self._btn_stop.setEnabled(recording and not saving)

        # Output dir is locked while a capture is live or a save is draining.
        # (The recording label is owned by the prominent RecordingLabelStrip
        # beside the charts — which manages its own disable state.)
        self._output_dir.setEnabled(not recording and not saving)

        if saving:
            self._label_hint.setText("Saving recording to CSV…")
        elif not can_record:
            self._label_hint.setText("Define a valid recording label to enable Record.")
        elif not metadata_ready:
            self._label_hint.setText(
                "Awaiting session_info from device; Record enables once metadata arrives."
            )
        elif state == SessionState.ERROR:
            self._label_hint.setText("Channel errored — press Record to reconnect.")
        else:
            self._label_hint.setText("")

    def refresh_record_state(self, *_args: object) -> None:
        """Re-evaluate the Record button after the external label strip changes
        the recording label."""
        self._refresh_buttons()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_state_changed(self, state: SessionState) -> None:
        self._refresh_buttons()

    def _on_recording_changed(self, recording: bool) -> None:
        self._refresh_buttons()

    def _on_output_dir_changed(self, text: str) -> None:
        self._ctrl.set_output_dir(text.strip())

    def _on_browse_output_dir(self) -> None:
        log_user_action("Clicked browse for output directory")
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            log_user_action("Selected output directory %s", path)
            self._output_dir.setText(path)

    def _on_record(self) -> None:
        log_user_action("Clicked Record")
        # Clear a prior error transparently so we can recover.
        if self._ctrl.state() == SessionState.ERROR:
            self._ctrl.reset()
        # The channel is opened automatically when the tab opens; if it is not
        # currently streaming (e.g. it errored or was stopped), reopen it first.
        if self._ctrl.state() != SessionState.RUNNING:
            try:
                self._ctrl.start()
            except Exception as exc:
                QMessageBox.warning(self, "Recording", f"Could not open channel: {exc}")
                return
        try:
            self._ctrl.start_recording()
        except Exception as exc:
            QMessageBox.warning(self, "Recording", f"Could not start: {exc}")

    def _on_stop(self) -> None:
        log_user_action("Clicked Stop")
        # Stop capture and write the CSV; the channel stays open/listening so
        # the user can immediately record again within the same session.
        if self._ctrl.is_recording():
            self._ctrl.stop_recording()

    def _on_recording_empty(self) -> None:
        QMessageBox.information(
            self,
            "Recording",
            "No data was recorded yet. No CSV file was generated.",
        )

    def _on_session_info_received(self, raw: object) -> None:
        payload = raw if isinstance(raw, dict) else {}
        d = payload.get("d") if isinstance(payload.get("d"), dict) else {}

        # The structured fields are populated once from the first session_info
        # and must never accept user input. On every subsequent session_info
        # the read-only fields are refreshed to reflect the latest device
        # values: most fields (hz, st, name, dr) may change freely;
        # sid and ch_n cannot change without terminating the session,
        # so refreshing them here is a harmless no-op in practice.
        if self._metadata_fields:
            self._refresh_metadata_fields(d)
            return

        # Extract fields and display in structured form. Each entry carries a
        # tooltip mirroring the "Session Configuration" pattern.
        field_defs = [
            ("sid", "Session ID", str(d.get("sid", "—")),
             "Unique identifier the device assigned to this streaming session."),
            ("hz", "Sampling Rate (Hz)", str(d.get("hz", "—")),
             "Per-channel sampling frequency reported by the device, in Hz."),
            ("ch_n", "Channels",
             ", ".join(d.get("ch_n", [])) if d.get("ch_n") else "—",
             "Names of the channels streamed by the device."),
            ("st", "Sensor Type", str(d.get("st", "—")),
             "Sensor type reported by the device."),
            ("dr", "Drop Count", str(d.get("dr", "—")),
             "Frames the device reports it dropped before transmission."),
            ("name", "Device Name", str(d.get("name", "—")),
             "Human-readable device name reported in session_info."),
        ]

        for key, label, value, tooltip in field_defs:
            field = QLineEdit()
            field.setText(value)
            field.setReadOnly(True)
            # Make read-only Session Info fields visually distinct from the
            # editable Session Configuration inputs: flat (no frame), muted
            # transparent background, and non-focusable so they don't look or
            # behave like text inputs.
            field.setFrame(False)
            field.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            field.setCursorPosition(0)
            field.setStyleSheet(
                "QLineEdit { background: transparent; border: none; "
                "color: palette(text); padding: 0; }"
            )
            field.setToolTip(tooltip)
            self._metadata_fields[key] = field
            label_widget = QLabel(f"{label}:")
            label_widget.setToolTip(tooltip)
            self._metadata_form.addRow(label_widget, field)

        self._metadata_status.setText(f"received at {raw.get('t', '?')} = session_info")
        self._refresh_buttons()  # Update Record button state now that metadata is ready

    @staticmethod
    def _format_metadata_value(key: str, d: dict) -> str:
        if key == "ch_n":
            return ", ".join(d.get("ch_n", [])) if d.get("ch_n") else "—"
        return str(d.get(key, "—"))

    def _refresh_metadata_fields(self, d: dict) -> None:
        """Refresh the read-only Session Info fields from the latest payload."""
        for key, field in self._metadata_fields.items():
            if key in d:
                field.setText(self._format_metadata_value(key, d))

    def _on_session_phase_changed(self, phase: str) -> None:
        # Format phase for display: "awaiting device metadata" → "Awaiting Device Metadata"
        display_phase = " ".join(word.capitalize() for word in phase.split())
        self._phase.setText(display_phase)
        # A phase transition (notably "ready to record" once session_info has
        # arrived) changes recording availability; refresh the buttons so the
        # Record action is enabled regardless of the order in which the label
        # was entered vs. the metadata arriving.
        self._refresh_buttons()

    def _on_session_info_mismatch(self, detail: str, has_buffered_data: bool) -> None:
        if not has_buffered_data:
            QMessageBox.critical(
                self,
                "Session metadata changed",
                f"{detail}.\n\nSession stopped with error.",
            )
            return

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Session metadata changed")
        box.setText(
            "Session metadata changed. Recording was automatically stopped.\n\n"
            "Do you want to save buffered data to CSV?"
        )
        save_btn = box.addButton("Save buffered data", QMessageBox.ButtonRole.AcceptRole)
        drop_btn = box.addButton("Drop buffered data", QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(save_btn)
        box.exec()

        if box.clickedButton() == save_btn:
            try:
                self._ctrl.save_pending_mismatch_recording()
            except Exception as exc:
                QMessageBox.warning(self, "Recording", f"Could not save buffered data: {exc}")
        elif box.clickedButton() == drop_btn:
            self._ctrl.drop_pending_mismatch_recording()
