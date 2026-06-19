# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Prominent recording-label strip shown beside the charts.

The recording label is a key field, so it is presented in its own clearly
captioned strip adjacent to the charts rather than buried in the control-panel
form. It writes through to :meth:`SessionController.set_label` and is disabled
while a recording is active.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QWidget

from data_forwarder_host.session.controller import SessionController


class RecordingLabelStrip(QWidget):
    """A captioned, prominent entry for the recording label."""

    #: Emitted whenever the label text changes (so the Record button can be
    #: re-evaluated by the control panel).
    label_changed = Signal(str)

    def __init__(self, controller: SessionController, parent=None) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self.setObjectName("recordingLabelStrip")
        self.setStyleSheet(
            "#recordingLabelStrip { background:palette(alternate-base); "
            "border:1px solid palette(mid); border-radius:6px; }"
        )

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(10)

        caption = QLabel("Recording label")
        caption.setStyleSheet("font-weight:bold; color:palette(highlight);")
        lay.addWidget(caption)

        self._edit = QLineEdit()
        self._edit.setText(controller.config.label)
        self._edit.setPlaceholderText("required before recording")
        self._edit.setToolTip(
            "Recording label. Builds the CSV filename '{label}_{session}.csv'. "
            "Allowed: letters, digits, '_', '-', '.', up to 64 chars."
        )
        self._edit.setMinimumHeight(36)
        self._edit.setStyleSheet(
            "QLineEdit { font-size: 14px; font-weight: bold; padding: 4px 6px; }"
        )
        lay.addWidget(self._edit, 1)

        self._edit.textChanged.connect(self._on_text_changed)
        controller.recording_changed.connect(self._on_recording_changed)

    def focus_label(self) -> None:
        """Give keyboard focus to the label entry.

        Used so that when a session window opens, the text cursor waits in the
        recording-label field and the user can type the label immediately
        without first clicking it. Existing text (if any) is left intact.
        """
        self._edit.setFocus(Qt.FocusReason.OtherFocusReason)

    def _on_text_changed(self, text: str) -> None:
        self._ctrl.set_label(text.strip())
        self.label_changed.emit(text)

    def _on_recording_changed(self, recording: bool) -> None:
        # The label is locked while a capture is live.
        self._edit.setEnabled(not recording)
