# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Grid of per-channel mini-plots."""

from __future__ import annotations

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QGridLayout, QSizePolicy, QWidget

from data_forwarder_host.core.data_model import DataModel
from data_forwarder_host.gui.theme import color_for_channel


class _MiniPlot(QChartView):
    def __init__(self, name: str, color: str, parent=None) -> None:
        chart = QChart()
        chart.setTitle(name)
        chart.legend().setVisible(False)
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        super().__init__(chart, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._series = QLineSeries()
        chart.addSeries(self._series)
        from PySide6.QtGui import QColor
        self._series.setColor(QColor(color))
        self._axis_x = QValueAxis()
        self._axis_y = QValueAxis()
        self._axis_x.setLabelFormat("%d")
        self._axis_y.setLabelFormat("%.3g")
        chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)
        self._series.attachAxis(self._axis_x)
        self._series.attachAxis(self._axis_y)

    def update_data(self, ts, val) -> None:
        if ts.size == 0:
            self._series.replace([])
            return
        self._series.replace([QPointF(float(t), float(v)) for t, v in zip(ts.tolist(), val.tolist())])
        self._axis_x.setRange(int(ts[0]), int(ts[-1]))
        vmin, vmax = float(val.min()), float(val.max())
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
        pad = 0.05 * (vmax - vmin)
        self._axis_y.setRange(vmin - pad, vmax + pad)


class ChannelGrid(QWidget):
    """Compact grid of per-channel mini-plots, recomputed at the model's redraw rate."""

    COLS = 4

    def __init__(self, data_model: DataModel, parent=None) -> None:
        super().__init__(parent)
        self._model = data_model
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(6)
        self._mini: list[_MiniPlot] = []

        data_model.channels_changed.connect(self._rebuild)
        data_model.data_appended.connect(self._refresh)
        data_model.cleared.connect(self._refresh)

        self._rebuild()

    def _rebuild(self) -> None:
        for w in self._mini:
            self._layout.removeWidget(w)
            w.deleteLater()
        self._mini = []
        for i, name in enumerate(self._model.channel_names):
            mp = _MiniPlot(name, color_for_channel(i))
            self._mini.append(mp)
            self._layout.addWidget(mp, i // self.COLS, i % self.COLS)

    def _refresh(self, *_args) -> None:
        window_ms = int(self._model.plot_window_seconds * 1000)
        for i, mp in enumerate(self._mini):
            ts, val = self._model.buffer(i).latest_window(window_ms)
            mp.update_data(ts, val)
