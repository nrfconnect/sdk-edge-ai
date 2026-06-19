# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Status-bar label that periodically shows host CPU / RAM usage."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QWidget

from data_forwarder_host.core.host_metrics import (
    format_host_metrics,
    format_host_metrics_tooltip,
    sample_host_metrics,
)


class HostMetricsLabel(QLabel):
    """A compact, self-refreshing host resource indicator.

    Owns a single-shot-less :class:`QTimer` that re-samples every
    ``interval_ms``. Sampling is non-blocking; any failure (e.g. ``psutil``
    missing) simply blanks the label rather than disrupting the UI.
    """

    def __init__(self, parent: QWidget | None = None, *, interval_ms: int = 2000) -> None:
        super().__init__(parent)
        self.setObjectName("HostMetricsLabel")
        # Prime cpu_percent so the first displayed value is meaningful rather
        # than the 0.0 that the very first psutil.cpu_percent() call returns.
        try:
            sample_host_metrics()
        except Exception:  # noqa: BLE001 - resource probing must never crash the UI
            pass

        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()
        self._refresh()

    def _refresh(self) -> None:
        try:
            metrics = sample_host_metrics()
        except Exception:  # noqa: BLE001 - tolerate missing/unavailable psutil
            self.setText("")
            self.setToolTip("")
            return
        self.setText(format_host_metrics(metrics))
        self.setToolTip(format_host_metrics_tooltip(metrics))
