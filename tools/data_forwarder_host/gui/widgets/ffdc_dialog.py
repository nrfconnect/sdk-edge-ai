# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""First Failure Data Capture (FFDC) sub-window.

Opened from the **First Failure Data Capture** button in the Error & Loss
Analysis panel. It snapshots the per-session error journal and, for each error
category, shows the *first* failure that occurred with its diagnostic detail:

* a Python error trace (when the failure carried an exception);
* real issue analysis for the frame — the raw bytestream as hex and as ASCII,
  plus what was expected versus what actually arrived.

Each category is a collapsible section, and within it each diagnostic view is a
nested collapsible section, so the user can expand only what they want to focus
on. The heavy lifting (selecting first failures, formatting bytestreams) lives
in the GUI-free :mod:`data_forwarder_host.core.ffdc` module.
"""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from data_forwarder_host.core.error_log import ErrorLog
from data_forwarder_host.core.ffdc import FfdcEntry, build_ffdc
from data_forwarder_host.gui.widgets.collapsible_section import CollapsibleSection
from data_forwarder_host.utils.logging import log_user_action

_EMPTY = "No failures have been captured in this session yet."


def _mono_view(text: str) -> QPlainTextEdit:
    view = QPlainTextEdit()
    view.setReadOnly(True)
    view.setPlainText(text)
    view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
    font = QFont("monospace")
    font.setStyleHint(QFont.StyleHint.Monospace)
    view.setFont(font)
    view.setMinimumHeight(80)
    return view


class FfdcDialog(QDialog):
    """Expandable First Failure Data Capture report for one session."""

    def __init__(self, error_log: ErrorLog, parent=None) -> None:
        super().__init__(parent)
        self._log = error_log

        self.setWindowTitle("First Failure Data Capture")
        self.setMinimumSize(640, 480)
        self.resize(820, 600)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(6)

        intro = QLabel(
            "The first occurrence of each error category is captured below. "
            "Expand a category, then expand the view you want to focus on."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        # Scrollable stack of per-category captures.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        outer.addWidget(self._scroll, 1)
        self._host = QWidget()
        self._host_lay = QVBoxLayout(self._host)
        self._host_lay.setContentsMargins(0, 0, 0, 0)
        self._host_lay.setSpacing(4)
        self._scroll.setWidget(self._host)

        btn_row = QHBoxLayout()
        self._refresh_btn = QPushButton("Refresh capture")
        self._refresh_btn.setToolTip("Re-snapshot the session error journal")
        self._refresh_btn.clicked.connect(self._on_refresh_clicked)
        btn_row.addWidget(self._refresh_btn)
        btn_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._on_close_clicked)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

        self.refresh()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_refresh_clicked(self) -> None:
        log_user_action("Clicked Refresh capture in the FFDC window")
        self.refresh()

    def _on_close_clicked(self) -> None:
        log_user_action("Clicked Close in the FFDC window")
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        log_user_action("Closed the FFDC window")
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Rebuild the report from the current error journal."""
        self._clear()
        entries = build_ffdc(self._log.events())
        if not entries:
            self._host_lay.addWidget(QLabel(_EMPTY))
            self._host_lay.addStretch(1)
            return
        for entry in entries:
            self._host_lay.addWidget(self._build_entry(entry))
        self._host_lay.addStretch(1)

    def _build_entry(self, entry: FfdcEntry) -> CollapsibleSection:
        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(8, 4, 4, 4)
        body_lay.setSpacing(4)

        header = QLabel(f"<b>{entry.detail}</b><br><i>first seen {entry.timestamp}</i>")
        header.setWordWrap(True)
        body_lay.addWidget(header)

        if entry.panels:
            for i, panel in enumerate(entry.panels):
                # Expand the first diagnostic view by default; leave the rest
                # collapsed so the user opts in to what they want to see.
                sub = CollapsibleSection(
                    panel.title, _mono_view(panel.text), expanded=(i == 0)
                )
                body_lay.addWidget(sub)
        else:
            body_lay.addWidget(QLabel("No additional diagnostic context was captured."))

        return CollapsibleSection(
            f"{entry.category} — first failure", body, expanded=False
        )

    def _clear(self) -> None:
        while self._host_lay.count():
            item = self._host_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
