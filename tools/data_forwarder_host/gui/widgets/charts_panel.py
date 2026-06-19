# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""ChartsPanel — combined overview + collapsible per-channel sub-charts.

Layout (inside one QScrollArea)::

    ▾ Combined View                     ← expanded by default
        <CombinedPlot — all channels overlaid>
    ▸ Individual Channels               ← collapsed by default; when expanded:
        ▾ ch0                             all child sections open together
        ▾ ch1
        ▾ ch2  ...

A single ``QScrollArea`` provides a common scroll bar for the whole view.
Per-channel sections are rebuilt automatically when ``channels_changed`` fires.
"""

from __future__ import annotations

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import Qt
from PySide6.QtCore import QMargins
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from data_forwarder_host.core.data_model import DataModel
from data_forwarder_host.core.decimation import decimate_minmax
from data_forwarder_host.utils.slow_span import slow_span
from data_forwarder_host.gui.theme import color_for_channel
from data_forwarder_host.gui.widgets.collapsible_section import CollapsibleSection
from data_forwarder_host.gui.widgets.combined_plot import CombinedPlot


# ---------------------------------------------------------------------------
# Internal single-channel chart
# ---------------------------------------------------------------------------


class _ChannelChart(QChartView):
    """Single-channel rolling time-series chart.

    The channel name is shown in the parent ``CollapsibleSection`` header, so
    the chart title is omitted to maximise the plot area.  X axis shows time
    in seconds (same unit as ``CombinedPlot``) so the user can correlate
    events across the two views.
    """

    def __init__(self, name: str, color: str, parent: QWidget | None = None) -> None:
        chart = QChart()
        # Title omitted — shown by the CollapsibleSection header above.
        chart.legend().setVisible(False)
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        # Minimise padding so the chart fills the available width.
        chart.setMargins(QMargins(0, 0, 0, 0))
        chart.layout().setContentsMargins(0, 0, 0, 0)
        super().__init__(chart, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(120)

        self._series = QLineSeries()
        self._series.setColor(QColor(color))
        chart.addSeries(self._series)

        # X axis in seconds — same unit as CombinedPlot so tick positions align.
        self._axis_x = QValueAxis()
        self._axis_x.setLabelFormat("%.1f")
        self._axis_y = QValueAxis()
        self._axis_y.setLabelFormat("%.3g")
        chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)
        self._series.attachAxis(self._axis_x)
        self._series.attachAxis(self._axis_y)

        # Dashed vertical recording markers: start (red) and stop (blue).
        self._marker = QLineSeries()
        marker_pen = QPen(QColor("#d62728"))
        marker_pen.setWidth(2)
        marker_pen.setStyle(Qt.PenStyle.DashLine)
        self._marker.setPen(marker_pen)
        chart.addSeries(self._marker)
        self._marker.attachAxis(self._axis_x)
        self._marker.attachAxis(self._axis_y)

        self._stop_marker = QLineSeries()
        stop_pen = QPen(QColor("#1f77b4"))
        stop_pen.setWidth(2)
        stop_pen.setStyle(Qt.PenStyle.DashLine)
        self._stop_marker.setPen(stop_pen)
        chart.addSeries(self._stop_marker)
        self._stop_marker.attachAxis(self._axis_x)
        self._stop_marker.attachAxis(self._axis_y)

    def update_data(self, ts, val, marker_ms=None, right_ms=None, stop_marker_ms=None, x_window=None) -> None:  # type: ignore[type-arg]
        from PySide6.QtCore import QPointF

        if ts.size == 0:
            self._series.replace([])
            self._marker.replace([])
            self._stop_marker.replace([])
            # Even with no points this chart must keep the shared X window so its
            # (empty) axis still lines up with the other charts.
            if x_window is not None:
                self._axis_x.setRange(x_window[0], x_window[1])
            return
        # Render-time min/max decimation bounds the on-screen point count so the
        # redraw is O(points_on_screen), not O(buffer_size). Peaks
        # are preserved, so the y auto-range below is unaffected.
        with slow_span("charts.decimate", extra=f"in={ts.size}"):
            ts, val = decimate_minmax(ts, val)
        # Convert ms → s so the X axis matches the CombinedPlot.
        with slow_span("charts.series_replace", extra=f"pts={ts.size}"):
            self._series.replace(
                [QPointF(float(t) / 1000.0, float(v)) for t, v in zip(ts.tolist(), val.tolist())]
            )
        x_lo, x_hi = ts[0] / 1000.0, ts[-1] / 1000.0
        if x_window is not None:
            # Use the model's shared rolling window so all charts line up exactly.
            # The data-derived edges remain the fallback when no shared
            # window is available yet.
            x_lo, x_hi = x_window[0], x_window[1]
        elif right_ms is not None and right_ms / 1000.0 > x_lo:
            # Right edge follows the smoothed playout position, matching
            # the CombinedPlot; fall back to the newest sample when idle/behind.
            x_hi = right_ms / 1000.0
        self._axis_x.setRange(x_lo, x_hi)
        vmin, vmax = float(val.min()), float(val.max())
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
        pad = 0.05 * (vmax - vmin)
        self._axis_y.setRange(vmin - pad, vmax + pad)
        self._update_marker(self._marker, marker_ms, x_lo, x_hi)
        self._update_marker(self._stop_marker, stop_marker_ms, x_lo, x_hi)

    def _update_marker(self, series, marker_ms, x_lo: float, x_hi: float) -> None:
        from PySide6.QtCore import QPointF

        if marker_ms is None:
            series.replace([])
            return
        marker_s = marker_ms / 1000.0
        if not (x_lo <= marker_s <= x_hi):
            series.replace([])
            return
        series.replace(
            [QPointF(marker_s, self._axis_y.min()), QPointF(marker_s, self._axis_y.max())]
        )


# ---------------------------------------------------------------------------
# ChartsPanel
# ---------------------------------------------------------------------------


class ChartsPanel(QWidget):
    """Charts area for one session tab.

    No internal scroll area — the parent ``session_tab`` provides a single
    common scroll bar that covers the charts and the log consoles together.

    * ``▾ Combined View`` — all channels overlaid, expanded by default.
    * ``▸ Individual Channels`` — collapsed by default; expanding it opens
      all per-channel sub-sections simultaneously.
    """

    def __init__(self, data_model: DataModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = data_model

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._layout = outer  # alias used by _rebuild_channels

        # ── Combined overview (expanded by default) ─────────────────────
        combined_chart = CombinedPlot(data_model)
        combined_chart.setMinimumHeight(320)
        self._combined_section = CollapsibleSection(
            "Combined View", combined_chart, expanded=True
        )
        outer.addWidget(self._combined_section)

        # ── Individual channels parent section (collapsed by default) ───
        self._channels_body = QWidget()
        self._channels_layout = QVBoxLayout(self._channels_body)
        self._channels_layout.setContentsMargins(0, 0, 0, 0)
        self._channels_layout.setSpacing(0)

        self._channels_section = CollapsibleSection(
            "Individual Channels", self._channels_body, expanded=False
        )
        # When the parent section is toggled, propagate to all children.
        self._channels_section.toggled.connect(self._on_channels_section_toggled)
        self._layout.addWidget(self._channels_section)

        self._layout.addStretch(1)

        # ── Per-channel state ───────────────────────────────────────────
        self._channel_charts: list[_ChannelChart] = []
        self._channel_sections: list[CollapsibleSection] = []

        data_model.channels_changed.connect(self._rebuild_channels)
        data_model.data_appended.connect(self._refresh_channels)
        data_model.cleared.connect(self._refresh_channels)
        data_model.recording_marker_changed.connect(self._refresh_channels)

        self._rebuild_channels()

    # ------------------------------------------------------------------

    def set_combined_visible(self, on: bool) -> None:
        """Show/hide the Combined View section (View ▸ Panels)."""
        self._combined_section.setVisible(on)

    def set_individual_visible(self, on: bool) -> None:
        """Show/hide the Individual Channels section (View ▸ Panels)."""
        self._channels_section.setVisible(on)

    # ------------------------------------------------------------------

    def _on_channels_section_toggled(self, expanded: bool) -> None:
        """Expand or collapse all per-channel sections together."""
        for sec in self._channel_sections:
            sec.set_expanded(expanded)
        # Catch up the per-channel charts now that they are visible again;
        # their refreshes are skipped while the section is collapsed.
        if expanded:
            self._refresh_channels()

    def _rebuild_channels(self) -> None:
        while self._channels_layout.count():
            item = self._channels_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        self._channel_charts = []
        self._channel_sections = []

        parent_expanded = self._channels_section.is_expanded()
        for i, name in enumerate(self._model.channel_names):
            chart = _ChannelChart(name, color_for_channel(i))
            sec = CollapsibleSection(name, chart, expanded=parent_expanded)
            self._channel_charts.append(chart)
            self._channel_sections.append(sec)
            self._channels_layout.addWidget(sec)

        self._channels_layout.addStretch(1)

    def _refresh_channels(self, *_args: object) -> None:
        # The per-channel charts only exist on screen while the "Individual
        # Channels" section is expanded (collapsed by default). Skip the costly
        # per-tick rebuild otherwise; expanding the section refreshes them
        # immediately via _on_channels_section_toggled.
        if not self._channels_section.is_expanded():
            return
        window_ms = int(self._model.plot_window_seconds * 1000)
        marker_ms = self._model.recording_start_marker_ms
        stop_marker_ms = self._model.recording_stop_marker_ms
        right_ms = self._model.playout_position_ms()
        # One shared rolling X window for every chart so all axes line up exactly.
        # Converted ms → s to match the per-chart axis unit.
        win = self._model.visible_x_window_ms()
        x_window = (win[0] / 1000.0, win[1] / 1000.0) if win is not None else None
        with slow_span("charts.refresh_channels", extra=f"n={self._model.channel_count}"):
            for i, chart in enumerate(self._channel_charts):
                if i < self._model.channel_count:
                    with slow_span("charts.latest_window"):
                        ts, val = self._model.buffer(i).latest_window(window_ms)
                    chart.update_data(ts, val, marker_ms, right_ms, stop_marker_ms, x_window)
