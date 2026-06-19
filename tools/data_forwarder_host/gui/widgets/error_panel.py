# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""ErrorPanel — live event list + summary table + incomplete banner."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from data_forwarder_host.core.error_log import (
    ErrorCategory,
    ErrorEvent,
    ErrorLog,
    ErrorSummary,
)
from data_forwarder_host.gui.widgets.ffdc_dialog import FfdcDialog
from data_forwarder_host.utils.logging import log_user_action

# Coalesce live-event-list updates to <=20 Hz. A transport flood can confirm many
# losses in a single burst; appending one widget item and calling scrollToBottom()
# for every event would stall the GUI thread (the appends/scrolls happen inside
# the inbox drain loop). Buffering and flushing in batches keeps per-event cost
# O(1) and the flush cost bounded regardless of the loss rate.
_EVENT_FLUSH_MS = 50
_MAX_LIVE_EVENTS = 1000


class ErrorPanel(QWidget):
    def __init__(self, error_log: ErrorLog, controller=None, parent=None) -> None:
        super().__init__(parent)
        self._log = error_log
        self._controller = controller

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        self._banner = QLabel()
        self._banner.setStyleSheet(
            "QLabel { background:#d62728; color:white; padding:6px; font-weight:bold; }"
        )
        self._banner.setVisible(False)
        outer.addWidget(self._banner)

        # Loss confirmation window: how long a missing sensor_data sequence
        # number is awaited before it is counted as a TRANSPORT loss.
        if controller is not None:
            cfg_row = QHBoxLayout()
            cfg_row.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel("Loss confirmation window:")
            lbl.setToolTip(
                "How long a missing sensor_data sequence number (d/seq) is "
                "awaited before it is confirmed a transport loss. Frames that "
                "arrive late/out of order within this window are not losses."
            )
            cfg_row.addWidget(lbl)
            self._loss_window = QDoubleSpinBox()
            self._loss_window.setDecimals(2)
            self._loss_window.setRange(0.05, 30.0)
            self._loss_window.setSingleStep(0.25)
            self._loss_window.setSuffix(" s")
            self._loss_window.setValue(controller.loss_confirmation_window_seconds())
            self._loss_window.valueChanged.connect(self._on_loss_window_changed)
            cfg_row.addWidget(self._loss_window)
            cfg_row.addStretch(1)
            outer.addLayout(cfg_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)

        # Left: live error event list
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.addWidget(QLabel("<b>⚠ Live Error Events</b>"))
        self._events = QListWidget()
        left_lay.addWidget(self._events)
        splitter.addWidget(left)

        # Right: per-category loss & error analysis table
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        header_row = QHBoxLayout()
        header_row.addWidget(QLabel("<b>📊 Error &amp; Loss Analysis</b>"))
        header_row.addStretch(1)
        self._ffdc_btn = QPushButton("First Failure Data Capture")
        self._ffdc_btn.setToolTip(
            "Inspect the first captured failure of each category "
            "(bytestream, expected vs. actual, Python trace)"
        )
        self._ffdc_btn.clicked.connect(self._open_ffdc)
        header_row.addWidget(self._ffdc_btn)
        right_lay.addLayout(header_row)
        self._table = QTableWidget(len(ErrorCategory), 2)
        self._table.setHorizontalHeaderLabels(["Count", "% of messages"])
        self._table.setVerticalHeaderLabels([c.name for c in ErrorCategory])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        right_lay.addWidget(self._table)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        error_log.event_added.connect(self._on_event)
        error_log.summary_changed.connect(self._on_summary)
        error_log.cleared.connect(self._on_cleared)
        # Seed display.
        self._on_summary(error_log.summary())

        # Buffered live-event rendering: _on_event only appends to this list; the
        # QListWidget is updated in batches by _flush_events on a timer.
        self._pending_events: list[ErrorEvent] = []
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(_EVENT_FLUSH_MS)
        self._flush_timer.timeout.connect(self._flush_events)
        self._flush_timer.start()

        self._ffdc_dialog: FfdcDialog | None = None

    def _open_ffdc(self) -> None:
        """Open (or re-focus) the First Failure Data Capture sub-window."""
        log_user_action("Clicked First Failure Data Capture")
        if self._ffdc_dialog is None:
            self._ffdc_dialog = FfdcDialog(self._log, self)
        else:
            self._ffdc_dialog.refresh()
        self._ffdc_dialog.show()
        self._ffdc_dialog.raise_()
        self._ffdc_dialog.activateWindow()

    def _on_loss_window_changed(self, seconds: float) -> None:
        """Apply a new transport loss confirmation window from the spinner."""
        log_user_action("Changed the loss confirmation window to %.2f s", float(seconds))
        if self._controller is not None:
            self._controller.set_loss_confirmation_window_seconds(float(seconds))

    def _on_cleared(self) -> None:
        """Reset the live-event list and banner when a new recording starts."""
        self._pending_events.clear()
        self._events.clear()
        self._banner.setVisible(False)

    def _on_event(self, evt: ErrorEvent) -> None:
        # Buffer only; the widget is updated in batches by _flush_events so a
        # burst of confirmed losses cannot stall the GUI thread with one addItem
        # + scrollToBottom per event.
        self._pending_events.append(evt)

    def _flush_events(self) -> None:
        """Append all buffered events to the list in one batch (timer-driven)."""
        if not self._pending_events:
            return
        pending = self._pending_events
        self._pending_events = []
        for evt in pending:
            item = QListWidgetItem(
                f"[{evt.t_host_utc}] {evt.category.name} ({evt.session_state.name}): {evt.detail}"
            )
            item.setForeground(QColor("#d62728"))
            self._events.addItem(item)
        # Cap list to a bounded number of items.
        overflow = self._events.count() - _MAX_LIVE_EVENTS
        for _ in range(max(0, overflow)):
            self._events.takeItem(0)
        self._events.scrollToBottom()

    def _on_summary(self, summary: ErrorSummary) -> None:
        for row, cat in enumerate(ErrorCategory):
            cnt = summary.counts.get(cat, 0)
            pct = summary.percentages.get(cat, 0.0)
            self._table.setItem(row, 0, QTableWidgetItem(str(cnt)))
            self._table.setItem(row, 1, QTableWidgetItem(f"{pct:.4f}"))

        if summary.incomplete:
            total_lost = summary.counts.get(ErrorCategory.RECORDER_OVERFLOW, 0)
            denom = summary.total_messages + total_lost
            pct = (100.0 * total_lost / denom) if denom else 0.0
            self._banner.setText(f"Recording is incomplete ({pct:.4f}% lost)")
            self._banner.setVisible(True)
        else:
            self._banner.setVisible(False)
