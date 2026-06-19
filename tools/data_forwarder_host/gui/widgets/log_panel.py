# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""App-wide rotating log view (bound via ``install_qt_handler``)."""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget

from data_forwarder_host.utils.logging import install_qt_handler


class _LogBridge(QObject):
    """Marshals log records onto the GUI thread via a signal."""

    record = Signal(str)


class LogPanel(QWidget):
    MAX_LINES = 2000

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(self.MAX_LINES)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._text)

        self._bridge = _LogBridge()
        self._bridge.record.connect(self._append)
        install_qt_handler(self._on_record)

    def _on_record(self, _record: logging.LogRecord, formatted: str) -> None:
        # Always cross to the GUI thread.
        self._bridge.record.emit(formatted)

    def _append(self, line: str) -> None:
        self._text.appendPlainText(line)
