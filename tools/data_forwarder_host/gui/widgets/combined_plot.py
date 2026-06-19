# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Combined rolling plot — all channels overlaid in a QtCharts ``QChartView``."""

from __future__ import annotations

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QPointF, Qt
from PySide6.QtCore import QMargins
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy

from data_forwarder_host.core.data_model import DataModel
from data_forwarder_host.core.decimation import decimate_minmax
from data_forwarder_host.utils.slow_span import slow_span
from data_forwarder_host.gui.theme import color_for_channel


class CombinedPlot(QChartView):
    """All channels overlaid on a single time axis (rolling window)."""

    def __init__(self, data_model: DataModel, parent=None) -> None:
        chart = QChart()
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        # Reduce internal chart padding so the plot area fills the view.
        chart.setMargins(QMargins(0, 0, 0, 0))
        chart.layout().setContentsMargins(0, 0, 0, 0)
        super().__init__(chart, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(280)

        self._model = data_model
        self._series: list[QLineSeries] = []
        self._hidden: set[int] = set()

        self._axis_x = QValueAxis()
        self._axis_x.setLabelFormat("%.1f")
        self._axis_x.setTitleText("Time (s)")
        self._axis_y = QValueAxis()
        self._axis_y.setLabelFormat("%.3g")
        chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)

        # Recording markers: dashed vertical lines drawn where the current
        # recording began (red) and ended (blue). Both are kept out of the
        # legend and the auto-range computation.
        self._marker = QLineSeries()
        self._marker.setName("recording start")
        marker_pen = QPen(QColor("#d62728"))
        marker_pen.setWidth(2)
        marker_pen.setStyle(Qt.PenStyle.DashLine)
        self._marker.setPen(marker_pen)
        chart.addSeries(self._marker)
        self._marker.attachAxis(self._axis_x)
        self._marker.attachAxis(self._axis_y)
        chart.legend().markers(self._marker)[0].setVisible(False)

        self._stop_marker = QLineSeries()
        self._stop_marker.setName("recording stop")
        stop_pen = QPen(QColor("#1f77b4"))
        stop_pen.setWidth(2)
        stop_pen.setStyle(Qt.PenStyle.DashLine)
        self._stop_marker.setPen(stop_pen)
        chart.addSeries(self._stop_marker)
        self._stop_marker.attachAxis(self._axis_x)
        self._stop_marker.attachAxis(self._axis_y)
        chart.legend().markers(self._stop_marker)[0].setVisible(False)

        data_model.channels_changed.connect(self._rebuild_series)
        data_model.data_appended.connect(self._on_data_changed)
        data_model.cleared.connect(self._on_data_changed)
        data_model.recording_marker_changed.connect(self._on_data_changed)

        self._rebuild_series()

    def _rebuild_series(self) -> None:
        chart = self.chart()
        for s in self._series:
            chart.removeSeries(s)
        self._series = []
        # Channel identity may change on rebuild, so visible-state is reset.
        self._hidden = set()
        for i, name in enumerate(self._model.channel_names):
            s = QLineSeries()
            s.setName(name)
            pen = QPen()
            pen.setColor(Qt.GlobalColor.black)
            pen.setWidth(1)
            s.setPen(pen)
            s.setColor(_to_qcolor(color_for_channel(i)))
            chart.addSeries(s)
            s.attachAxis(self._axis_x)
            s.attachAxis(self._axis_y)
            self._series.append(s)
            # Clicking a channel's legend marker toggles its visibility.
            for marker in chart.legend().markers(s):
                marker.clicked.connect(
                    lambda idx=i: self.toggle_channel(idx)
                )
        self._refresh()

    def toggle_channel(self, index: int) -> None:
        """Show/hide channel *index* in the combined overlay.

        A hidden channel's trace is removed from the chart and excluded from the
        x/y auto-range; its legend marker stays in the legend but is greyed.
        Toggling again restores the trace and its range participation.
        """
        if not (0 <= index < len(self._series)):
            return
        if index in self._hidden:
            self._hidden.discard(index)
        else:
            self._hidden.add(index)
        self._apply_marker_style(index)
        self._refresh()

    def _apply_marker_style(self, index: int) -> None:
        hidden = index in self._hidden
        for marker in self.chart().legend().markers(self._series[index]):
            brush = QBrush(marker.labelBrush())
            color = QColor(brush.color())
            color.setAlphaF(0.35 if hidden else 1.0)
            brush.setColor(color)
            marker.setLabelBrush(brush)

    def _on_data_changed(self, *_args) -> None:
        # Skip the (expensive) per-tick point rebuild whenever this chart is not
        # actually on screen — e.g. its collapsible section is collapsed or the
        # owning session tab is in the background. ``showEvent`` repaints it the
        # moment it becomes visible again, so no data is lost.
        if not self.isVisible():
            return
        self._refresh()

    def _refresh(self, *_args) -> None:
        window_ms = int(self._model.plot_window_seconds * 1000)
        y_min = float("inf")
        y_max = float("-inf")
        x_min = 0
        x_max = 0
        with slow_span("combined.refresh", extra=f"ch={len(self._series)}") as span:
            for i, s in enumerate(self._series):
                # Hidden channels are excluded from both the trace and
                # auto-range.
                if i in self._hidden:
                    s.replace([])
                    continue
                with slow_span("combined.latest_window"):
                    ts, val = self._model.buffer(i).latest_window(window_ms)
                if ts.size == 0:
                    s.replace([])
                    continue
                # Decimate at render time so the number of points pushed to
                # the series is bounded regardless of buffer size — redraw
                # becomes O(points_on_screen), not O(buffer_size).
                # min/max bucketing preserves peaks, so the
                # auto-range below is unchanged.
                with slow_span("combined.decimate", extra=f"in={ts.size}"):
                    ts, val = decimate_minmax(ts, val)
                with slow_span("combined.series_replace", extra=f"pts={ts.size}"):
                    pts = [
                        QPointF(float(t) / 1000.0, float(v))
                        for t, v in zip(ts.tolist(), val.tolist())
                    ]
                    s.replace(pts)
                x_min = ts[0] / 1000.0
                x_max = ts[-1] / 1000.0
                y_min = min(y_min, float(val.min()))
                y_max = max(y_max, float(val.max()))
            span.note(f"hidden={len(self._hidden)}")

        if x_max > x_min:
            # The right edge follows the smoothed playout position so
            # the view scrolls continuously between arrivals and late/batched
            # samples fill in as the edge passes their device timestamp. Falls
            # back to the newest sample when the clock is idle or behind the
            # oldest visible point.
            right_ms = self._model.playout_position_ms()
            x_hi = x_max
            if right_ms is not None and right_ms / 1000.0 > x_min:
                x_hi = right_ms / 1000.0
            self._axis_x.setRange(x_min, x_hi)
        else:
            x_hi = x_max

        # Override the X range with the model's shared rolling window so this
        # chart and every per-channel chart line up exactly: the same horizontal
        # position is the same time across all of them. The data-driven
        # x_min/x_max above remain the fallback for marker visibility
        # before any data has been seen.
        win = self._model.visible_x_window_ms()
        if win is not None:
            x_min = win[0] / 1000.0
            x_hi = win[1] / 1000.0
            self._axis_x.setRange(x_min, x_hi)

        if y_min < y_max:
            pad = 0.05 * (y_max - y_min) if (y_max - y_min) else 1.0
            self._axis_y.setRange(y_min - pad, y_max + pad)

        self._update_marker(x_min, x_hi)

    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        # Catch up immediately when revealed (e.g. section expanded or tab
        # re-activated) since refreshes were skipped while hidden.
        super().showEvent(event)
        self._refresh()

    def _update_marker(self, x_min: float, x_max: float) -> None:
        self._draw_marker(self._marker, self._model.recording_start_marker_ms,
                          x_min, x_max)
        self._draw_marker(self._stop_marker, self._model.recording_stop_marker_ms,
                          x_min, x_max)

    def _draw_marker(self, series: QLineSeries, marker_ms, x_min: float,
                     x_max: float) -> None:
        if marker_ms is None:
            series.replace([])
            return
        marker_s = marker_ms / 1000.0
        # Only draw the marker while it is within the visible rolling window.
        if x_max > x_min and not (x_min <= marker_s <= x_max):
            series.replace([])
            return
        lo, hi = self._axis_y.min(), self._axis_y.max()
        series.replace([QPointF(marker_s, lo), QPointF(marker_s, hi)])


def _to_qcolor(name: str):
    from PySide6.QtGui import QColor
    return QColor(name)
