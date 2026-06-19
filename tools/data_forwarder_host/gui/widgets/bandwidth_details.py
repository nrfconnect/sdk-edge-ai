# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Bandwidth details sub-window.

A non-modal dialog opened from the "Bandwidth details" button next to the
header throughput readout. It shows more of the data-transfer picture than the
one-line header:

* the live rates (bytes/s, messages/s, channels/s) over the measurement window;
* the bandwidth measurement window, editable here;
* a table of received-frame categories with two counts per row: the
  session-cumulative ``Total count`` and a window-scoped ``Since reset`` count
  that the user can zero with a local **Reset** button.

The table columns are user-resizable with the mouse.
"""

from __future__ import annotations

import math
from collections import deque
from time import monotonic

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QMargins, QPointF, Qt, QTimer
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from data_forwarder_host.core.error_log import ErrorCategory, ErrorSummary
from data_forwarder_host.core.pipeline_metrics import (
    PipelineSnapshot,
    build_host_pipeline_snapshot,
    per_stage_drop_totals,
)
from data_forwarder_host.core.transfer_stats import (
    baseline_counts,
    breakdown_since,
    byte_baseline,
    bytes_breakdown_since,
)
from data_forwarder_host.gui.theme import color_for_channel
from data_forwarder_host.gui.widgets.pipeline_flow import PipelineFlowWidget
from data_forwarder_host.protocol.base import DecodeStats
from data_forwarder_host.session.controller import SessionController
from data_forwarder_host.session.forwarding import ForwardingSession
from data_forwarder_host.utils.logging import log_user_action
from data_forwarder_host.utils.slow_span import slow_span, slow_span_fn

_COLUMNS = (
    "Category",
    "Total count",
    "Since reset",
    "Total bytes",
    "Bytes since reset",
)

# Label for the window-scoped reset button. It now clears more than the
# since-reset counters — it also zeroes each pipeline stage's drop/issue counter
# and removes the red node-warning borders.
# The ampersand is doubled so Qt renders a literal "&" instead of treating the
# following character as a mnemonic (which shows as an underline).
CLEAR_BUTTON_LABEL = "Clear counters && node warnings"

# Error & loss categories surfaced here, mirroring the Error & Loss Analysis
# panel. The cumulative PRODUCER_DROP_TOTAL is intentionally omitted (the
# since-reset PRODUCER_DROP is the meaningful figure in this view).
_ERROR_CATEGORIES = tuple(
    c for c in ErrorCategory if c is not ErrorCategory.PRODUCER_DROP_TOTAL
)


def _fmt_size(n: int) -> str:
    units = ("B", "KB", "MB", "GB")
    v = float(n)
    idx = 0
    while v >= 1000.0 and idx < len(units) - 1:
        v /= 1000.0
        idx += 1
    return f"{int(n)} B" if idx == 0 else f"{v:.1f} {units[idx]}"


def _fmt_bytes_per_s(value: float) -> str:
    units = ("B/s", "KB/s", "MB/s", "GB/s")
    v = float(value)
    idx = 0
    while v >= 1000.0 and idx < len(units) - 1:
        v /= 1000.0
        idx += 1
    return f"{v:.0f} {units[idx]}" if idx == 0 else f"{v:.1f} {units[idx]}"


def _pretty_category(name: str) -> str:
    """Humanise an ``ErrorCategory`` enum name for display."""
    return name.replace("_", " ").title()


def compute_expected_step_points(
    history: list[tuple[float, float]],
    times: list[float],
    window_seconds: float,
) -> list[tuple[float, float]]:
    """Piecewise-constant *expected msg/s* line, smoothed like a low-pass filter.

    The expected rate is not a single flat line: it changes whenever the
    producer's reported sampling frequency changes. Each change must take effect
    **from its own timepoint forward** (a step), not be redrawn flat across the
    whole graph. The step transition is smoothed with a
    first-order (EMA) response whose time constant is the bandwidth measurement
    window — so a level change ramps in over roughly that window, mirroring how
    the measured rate itself reacts.

    Args:
        history: ``(t_seconds, hz)`` expected-rate changes in chronological
            order (``t_seconds`` on the same axis as *times*).
        times: x positions (seconds) at which to evaluate the line — typically
            the timestamps of the actual-rate samples currently on screen.
        window_seconds: the bandwidth window; used as the smoothing time
            constant ``tau``.

    Returns:
        ``(t, value)`` points for the expected series, or ``[]`` if there is
        nothing to draw.
    """
    if not history or not times:
        return []
    tau = window_seconds if window_seconds and window_seconds > 0 else 1e-3

    def level_at(t: float) -> float:
        level = history[0][1]
        for t_change, hz in history:
            if t_change <= t:
                level = hz
            else:
                break
        return level

    lo, hi = times[0], times[-1]
    # Evaluate at the sample times *and* at every step change inside the window,
    # so each sub-interval has a single constant level and each step begins
    # exactly at its change timepoint.
    grid = sorted(set(times) | {t for t, _ in history if lo <= t <= hi})
    out: list[tuple[float, float]] = []
    prev_t = grid[0]
    smoothed = level_at(grid[0])
    out.append((grid[0], smoothed))
    for t in grid[1:]:
        dt = t - prev_t
        if dt < 0.0:
            dt = 0.0
        # The level is constant on [prev_t, t); use the level in effect at the
        # interval's start so a change at time ``t`` only ramps in afterwards.
        target = level_at(prev_t)
        alpha = 1.0 - math.exp(-dt / tau) if tau > 0 else 1.0
        smoothed += alpha * (target - smoothed)
        out.append((t, smoothed))
        prev_t = t
    return out


def error_rows_since(
    counts: dict,
    baseline: dict,
    categories: tuple,
) -> list[tuple[str, int, int]]:
    """Build ``(label, total, since_reset)`` rows for the error/loss categories.

    *since_reset* is the window-local delta ``total - baseline`` (never negative)
    so the bandwidth-details Reset can zero these counts **without touching the
    shared error log** — they are secondary, window-scoped figures derived from
    the general counters but living only in this sub-window.
    """
    rows: list[tuple[str, int, int]] = []
    for cat in categories:
        total = int(counts.get(cat, 0))
        base = int(baseline.get(cat, 0))
        since = total - base
        if since < 0:
            since = 0
        rows.append((_pretty_category(cat.name), total, since))
    return rows


class _MsgRateChart(QChartView):
    """Rolling line chart of the message rate (msg/s) over time.

    Styled to match the session's individual-channel plots (QtCharts
    ``QChartView``, no animation, full-bleed plot area). The dialog feeds it a
    new ``msg/s`` sample on every rates refresh; the chart keeps a rolling
    window of recent samples and auto-ranges the y axis.
    """

    _WINDOW_SECONDS: float = 60.0

    def __init__(self, parent=None) -> None:
        chart = QChart()
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        chart.setMargins(QMargins(0, 0, 0, 0))
        chart.layout().setContentsMargins(0, 0, 0, 0)
        super().__init__(chart, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumHeight(160)

        self._series = QLineSeries()
        self._series.setName("actual msg/s")
        pen = QPen(_to_qcolor(color_for_channel(0)))
        pen.setWidth(2)
        self._series.setPen(pen)
        chart.addSeries(self._series)

        # Dynamic "expected" line: the producer's sampling frequency (msg/s)
        # from session_info. Drawn flat across the visible window and updated
        # whenever the frequency changes.
        self._expected_series = QLineSeries()
        self._expected_series.setName("expected msg/s")
        exp_pen = QPen(_to_qcolor("#C62828"))
        exp_pen.setWidth(2)
        exp_pen.setStyle(Qt.PenStyle.DashLine)
        self._expected_series.setPen(exp_pen)
        chart.addSeries(self._expected_series)
        self._expected: float | None = None
        # History of (t_seconds, hz) expected-rate changes so the line steps at
        # each change timepoint; smoothing time constant comes
        # from the bandwidth window.
        self._expected_history: list[tuple[float, float]] = []
        self._window_seconds: float = 1.0

        self._axis_x = QValueAxis()
        self._axis_x.setLabelFormat("%.0f")
        self._axis_x.setTitleText("Time (s)")
        self._axis_y = QValueAxis()
        self._axis_y.setLabelFormat("%.0f")
        self._axis_y.setTitleText("msg/s")
        chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)
        self._series.attachAxis(self._axis_x)
        self._series.attachAxis(self._axis_y)
        self._expected_series.attachAxis(self._axis_x)
        self._expected_series.attachAxis(self._axis_y)

        self._t0 = monotonic()
        self._points: deque[tuple[float, float]] = deque()

    def set_expected(self, msg_per_second: float | None) -> None:
        """Set/clear the expected-rate line (producer frequency, msg/s).

        Records a step change at the current time so the line changes level only
        from this timepoint forward, rather than redrawing flat.
        """
        value = (
            float(msg_per_second) if msg_per_second and msg_per_second > 0 else None
        )
        self._expected = value
        if value is None:
            self._expected_history.clear()
        elif not self._expected_history or self._expected_history[-1][1] != value:
            self._expected_history.append((monotonic() - self._t0, value))
        self._refresh()

    def set_bandwidth_window(self, seconds: float) -> None:
        """Update the smoothing time constant (the bandwidth window)."""
        if seconds and seconds > 0:
            self._window_seconds = float(seconds)
            self._refresh()

    def add_sample(self, msg_per_second: float) -> None:
        now = monotonic() - self._t0
        self._points.append((now, float(msg_per_second)))
        cutoff = now - self._WINDOW_SECONDS
        while self._points and self._points[0][0] < cutoff:
            self._points.popleft()
        self._refresh()

    def _refresh(self) -> None:
        self._series.replace([QPointF(t, v) for t, v in self._points])
        if not self._points:
            if self._expected is not None:
                self._expected_series.replace(
                    [QPointF(0.0, self._expected), QPointF(1.0, self._expected)]
                )
                self._axis_x.setRange(0.0, 1.0)
                self._axis_y.setRange(0.0, self._expected * 1.1)
            return
        x_min = self._points[0][0]
        x_max = self._points[-1][0]
        if x_max <= x_min:
            x_max = x_min + 1.0
        self._axis_x.setRange(x_min, x_max)
        y_max = max(v for _t, v in self._points)
        exp_points = compute_expected_step_points(
            self._expected_history,
            [t for t, _ in self._points],
            self._window_seconds,
        )
        if exp_points:
            self._expected_series.replace([QPointF(t, v) for t, v in exp_points])
            y_max = max(y_max, max(v for _t, v in exp_points))
        else:
            self._expected_series.replace([])
        # Keep headroom for the latest expected level so the axis stays stable
        # while the smoothed step ramps toward a new level.
        if self._expected is not None:
            y_max = max(y_max, self._expected)
        # Always start the rate axis at zero and leave a little headroom.
        self._axis_y.setRange(0.0, (y_max * 1.1) if y_max > 0 else 1.0)


def _to_qcolor(name: str):
    from PySide6.QtGui import QColor

    return QColor(name)


class BandwidthDetailsDialog(QDialog):
    """Detailed data-transfer / reception statistics for one session."""

    def __init__(self, controller: SessionController, parent=None) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self._stats = DecodeStats()
        # Window-scoped counter baselines; empty means "since reset" == total.
        self._reset_baseline: dict[str, int] = {}
        self._reset_bytes_baseline: dict[str, int] = {}
        # Window-local baseline for the merged error/loss categories. These are
        # secondary, sub-window-scoped figures: resetting them never touches the
        # shared error log / general counters.
        self._error_reset_baseline: dict = {}
        self._summary: ErrorSummary | None = None
        # Latest device frame rate (Hz) reported via session_info, shown as the
        # Device pipeline stage's production rate.
        self._device_rate: float = 0.0

        self.setWindowTitle(f"Bandwidth details — {controller.config.tag}")
        # Open at a comfortable size rather than collapsing to the hint. Larger
        # now that the window also carries the session info and the error table.
        self.setMinimumSize(560, 520)
        self.resize(780, 720)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        # Session info: sampling rate (Hz) and channel count, from session_info.
        self._sampling_label = QLabel("Sampling rate: — Hz")
        self._channels_label = QLabel("Channels: —")
        info_row = QHBoxLayout()
        info_row.addWidget(self._sampling_label)
        info_row.addSpacing(20)
        info_row.addWidget(self._channels_label)
        info_row.addStretch(1)
        lay.addLayout(info_row)

        lay.addWidget(QLabel("<b>Live throughput</b>"))
        self._rates = QLabel()
        lay.addWidget(self._rates)

        # Rolling chart of the message rate over time (msg/s), styled like the
        # session's individual-channel plots. The dashed line shows the expected
        # rate (producer frequency from session_info).
        lay.addWidget(QLabel("Message rate over time"))
        self._rate_chart = _MsgRateChart()
        lay.addWidget(self._rate_chart)

        # Per-stage pipeline flow: source → decode → in-flight gate → model →
        # recorder, coloured by utilisation with drop badges and a highlighted
        # bottleneck. Fed a fresh snapshot on each rates refresh.
        lay.addWidget(QLabel("<b>Live throughput</b>"))
        self._pipeline = PipelineFlowWidget()
        lay.addWidget(self._pipeline)

        # Measurement-window control.
        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("Bandwidth window:"))
        self._window_spin = QDoubleSpinBox()
        self._window_spin.setRange(0.1, 600.0)
        self._window_spin.setDecimals(1)
        self._window_spin.setSingleStep(0.5)
        self._window_spin.setSuffix(" s")
        self._window_spin.setValue(controller.current_config().bandwidth_window_seconds)
        self._window_spin.valueChanged.connect(self._on_window_changed)
        # Seed the chart smoothing time constant from the current window.
        self._rate_chart.set_bandwidth_window(
            controller.current_config().bandwidth_window_seconds
        )
        win_row.addWidget(self._window_spin)
        win_row.addStretch(1)
        lay.addLayout(win_row)
        lay.addWidget(QLabel("<b>Received frames &amp; transport</b>"))
        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        header = self._table.horizontalHeader()
        # User-resizable columns (drag with the mouse).
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        self._table.setColumnWidth(0, 200)
        self._table.setColumnWidth(1, 90)
        self._table.setColumnWidth(2, 90)
        self._table.setColumnWidth(3, 90)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        lay.addWidget(self._table, 1)

        # Window-scoped reset.
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._reset_btn = QPushButton(CLEAR_BUTTON_LABEL)
        self._reset_btn.setToolTip(
            "Zero the per-category 'Since reset' counts in this window only and "
            "clear each pipeline stage's drop counter and red warning border; "
            "they start counting up again from now"
        )
        self._reset_btn.clicked.connect(self._on_reset_counters)
        btn_row.addWidget(self._reset_btn)
        lay.addLayout(btn_row)

        controller.stats_updated.connect(self._on_stats)
        # Keep the window control in sync when changed from the session window.
        controller.bandwidth_window_changed.connect(self._on_external_window_changed)
        # Error & loss table + expected-rate line driven by the error log and
        # the session_info sampling rate.
        controller.error_log.summary_changed.connect(self._on_error_summary)
        controller.session_sampling_changed.connect(self._on_sampling_changed)

        # Seed the session-info fields and expected-rate line if already known.
        sampling = controller.latest_sampling_info()
        if sampling is not None:
            self._on_sampling_changed(sampling[0], sampling[1])
        self._on_error_summary(controller.error_log.summary())

        # Refresh the live rates a couple of times a second.
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._refresh_rates)
        self._timer.start()

        self._refresh_rates()
        self._refresh_table()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_window_changed(self, seconds: float) -> None:
        log_user_action("Changed the bandwidth window to %.1f s", float(seconds))
        self._rate_chart.set_bandwidth_window(float(seconds))
        self._ctrl.set_bandwidth_window_seconds(float(seconds))

    def _on_external_window_changed(self, seconds: float) -> None:
        self._rate_chart.set_bandwidth_window(float(seconds))
        if abs(self._window_spin.value() - seconds) < 1e-9:
            return
        blocked = self._window_spin.blockSignals(True)
        self._window_spin.setValue(seconds)
        self._window_spin.blockSignals(blocked)

    def _on_reset_counters(self) -> None:
        # Snapshot the current totals; subsequent rows show counts since now.
        # This re-anchors only the window-local baselines (frame + error/loss);
        # the shared error log and general counters are untouched. It
        # also clears the pipeline node warnings — each stage's drop counter and
        # red border — until new frames are dropped.
        log_user_action("Clicked Clear counters & node warnings in bandwidth details")
        self._reset_baseline = baseline_counts(self._stats)
        self._reset_bytes_baseline = byte_baseline(self._stats)
        if self._summary is not None:
            self._error_reset_baseline = dict(self._summary.counts)
        self._pipeline.clear_warnings()
        self._refresh_table()

    def _on_stats(self, stats: DecodeStats) -> None:
        self._stats = stats
        self._refresh_table()

    def _on_sampling_changed(self, hz: float, channels: int) -> None:
        """Reflect the session_info sampling rate / channel count and expected rate."""
        self._sampling_label.setText(f"Sampling rate: {hz:.0f} Hz")
        self._channels_label.setText(f"Channels: {channels}")
        # Expected messages/s equals the producer sampling frequency. The Device
        # pipeline stage reflects this as the producer's frame rate.
        self._device_rate = float(hz)
        self._rate_chart.set_expected(hz)

    def _on_error_summary(self, summary: ErrorSummary) -> None:
        self._summary = summary
        self._refresh_table()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        log_user_action("Closed the bandwidth details window")
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _refresh_rates(self) -> None:
        with slow_span("bw.refresh_rates"):
            s = self._ctrl.bandwidth_sample()
            self._rates.setText(
                f"{_fmt_bytes_per_s(s.bytes_per_second)} · "
                f"{s.messages_per_second:.0f} msg/s · "
                f"{s.channels_per_second:.0f} ch/s"
            )
            self._rate_chart.add_sample(s.messages_per_second)
            with slow_span("bw.pipeline_snapshot"):
                snap = self._build_pipeline_snapshot(s.messages_per_second)
            self._pipeline.update_snapshot(snap)
            # Surface the same per-stage drop totals in the Error & Loss table,
            # read back from the widget's (baseline-adjusted) snapshot so the
            # table and diagram always agree.
            self._refresh_table()

    def _build_pipeline_snapshot(self, msg_s: float) -> PipelineSnapshot:
        """Assemble a per-stage snapshot from the controller's public metrics.

        A leading Device stage reflects the sensor board's reported frame rate
        and producer-side drops; on a BLE session a Host BT stage then represents
        the host OS Bluetooth stack, showing the produced-vs-received rate and
        the confirmed transport (sequence-gap) loss attributed to it; rates then
        flow through every host stage at the measured message rate; the in-flight
        gate carries the overload drop count and its bounded capacity, and the
        model/recorder show their current backlog. The widget derives utilisation
        and the bottleneck from this pure structure.
        """
        recording = self._ctrl.is_recording()
        recorder_overflow = 0
        if self._summary is not None:
            recorder_overflow = self._summary.counts.get(
                ErrorCategory.RECORDER_OVERFLOW, 0
            )
        # Net the Host BT stage against the live producer-drop count (not the
        # 1 Hz error summary) so the Device stage and the Host BT subtraction
        # reconcile against the same up-to-date "dr" that gated the reconcile
        # release — a device-caused drop then nets onto the Device stage without
        # briefly showing under Host BT while the coalesced summary catches up.
        producer_drops = self._ctrl.producer_drop_count()
        # Use the reconcile-held transport total so a device-caused drop is not
        # briefly attributed to the Host BT stage before its producer "dr" lands.
        transport_loss = self._ctrl.host_attributed_transport_loss()
        return build_host_pipeline_snapshot(
            message_rate=msg_s,
            dropped_frames=self._ctrl.dropped_frames(),
            gate_capacity=ForwardingSession.MAX_IN_FLIGHT,
            model_queue_depth=self._ctrl.data_model.max_buffer_size,
            recording=recording,
            recorder_queue_depth=self._ctrl.recorder.buffered_rows if recording else 0,
            device_rate=self._device_rate,
            producer_drops=producer_drops,
            transport_kind=self._ctrl.config.source.kind,
            transport_loss=transport_loss,
            recorder_overflow=recorder_overflow,
            wall_time_s=monotonic(),
        )

    @slow_span_fn("bw.refresh_table")
    def _refresh_table(self) -> None:
        rows = breakdown_since(self._stats, self._reset_baseline)
        byte_rows = bytes_breakdown_since(self._stats, self._reset_bytes_baseline)
        error_rows = (
            error_rows_since(
                self._summary.counts, self._error_reset_baseline, _ERROR_CATEGORIES
            )
            if self._summary is not None
            else []
        )
        # One unified table: received-frame categories, a separator, then the
        # error/loss categories appended as further rows so the whole
        # producer→decoded-frame path is traced in a single form. Finally a
        # "Dropped frames by stage" section attributes every dropped frame to the
        # stage that shed it, read from the same pipeline snapshot the diagram
        # uses so the two views always agree.
        drop_rows = self._stage_drop_rows()
        drops_block = (1 + len(drop_rows)) if drop_rows else 0
        total_rows = (
            len(rows) + (1 + len(error_rows) if error_rows else 0) + drops_block
        )
        self._table.setRowCount(total_rows)
        for r, ((label, total, since), (_lbl, tbytes, sbytes)) in enumerate(
            zip(rows, byte_rows)
        ):
            self._table.setItem(r, 0, QTableWidgetItem(label))
            self._table.setItem(r, 1, QTableWidgetItem(str(total)))
            self._table.setItem(r, 2, QTableWidgetItem(str(since)))
            self._table.setItem(r, 3, QTableWidgetItem(_fmt_size(tbytes)))
            self._table.setItem(r, 4, QTableWidgetItem(_fmt_size(sbytes)))
        next_row = len(rows)
        if error_rows:
            sep = next_row
            header = QTableWidgetItem("— Errors & loss —")
            self._table.setItem(sep, 0, header)
            for col in range(1, len(_COLUMNS)):
                self._table.setItem(sep, col, QTableWidgetItem(""))
            for i, (label, total, since) in enumerate(error_rows):
                r = sep + 1 + i
                self._table.setItem(r, 0, QTableWidgetItem(label))
                self._table.setItem(r, 1, QTableWidgetItem(str(total)))
                self._table.setItem(r, 2, QTableWidgetItem(str(since)))
                # Byte columns are not meaningful for error categories.
                self._table.setItem(r, 3, QTableWidgetItem("—"))
                self._table.setItem(r, 4, QTableWidgetItem("—"))
            next_row = sep + 1 + len(error_rows)
        if drop_rows:
            sep = next_row
            self._table.setItem(sep, 0, QTableWidgetItem("— Dropped frames by stage —"))
            for col in range(1, len(_COLUMNS)):
                self._table.setItem(sep, col, QTableWidgetItem(""))
            for i, (stage_name, drops) in enumerate(drop_rows):
                r = sep + 1 + i
                self._table.setItem(r, 0, QTableWidgetItem(f"{stage_name} drops"))
                self._table.setItem(r, 1, QTableWidgetItem(str(drops)))
                self._table.setItem(r, 2, QTableWidgetItem("—"))
                self._table.setItem(r, 3, QTableWidgetItem("—"))
                self._table.setItem(r, 4, QTableWidgetItem("—"))

    def _stage_drop_rows(self) -> list[tuple[str, int]]:
        """Per-stage dropped-frame rows for the table, newest snapshot, drops>0.

        Reads :func:`per_stage_drop_totals` from the pipeline widget's current
        (baseline-adjusted) snapshot so the Error & Loss table and the pipeline
        diagram report identical per-stage drop counts, and both clear together
        when node warnings are cleared.
        """
        snap = self._pipeline.snapshot
        if snap is None:
            return []
        return [
            (name, drops)
            for name, drops in per_stage_drop_totals(snap)
            if drops > 0
        ]
