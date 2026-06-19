# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Pipeline flow visualisation for the bandwidth-details window.

:class:`PipelineFlowWidget` paints a left-to-right row of pipeline stages from a
:class:`~data_forwarder_host.core.pipeline_metrics.PipelineSnapshot`:
each stage is coloured by its queue utilisation (green → amber → red), shows a
queue-fill gauge and a drop badge when it has discarded frames, and the detected
bottleneck stage is emphasised. Animated ">"-style chevrons flow between stages
at a speed that tracks each stage's output rate.

The widget contains **no measurement logic** — it only renders the immutable
pure structures it is handed, and :meth:`update_snapshot` repaints in place
without rebuilding any child widgets.
"""

from __future__ import annotations

from dataclasses import replace
from time import monotonic

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QHelpEvent,
    QPainter,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from data_forwarder_host.core.pipeline_metrics import (
    PipelineSnapshot,
    StageMetrics,
    detect_bottleneck,
    group_stages_by_process,
)

_GREEN = QColor(0x2E, 0x7D, 0x32)
_AMBER = QColor(0xF9, 0xA8, 0x25)
_RED = QColor(0xC6, 0x28, 0x28)

# Human-readable explanations shown when the user hovers a pipeline stage.
# Keyed by the stage's display name. A trailing default covers any
# stage name without a specific entry.
STAGE_DESCRIPTIONS: dict[str, str] = {
    "Device": (
        "Device: the sensor board producing samples. The rate shown is how fast "
        "the device emits frames; drops here are samples the device discarded "
        "before they ever reached the host."
    ),
    "Wire": (
        "Wire: the transport link (USB/UART or BLE) carrying raw frames from the "
        "device to this host."
    ),
    "Decode": (
        "Decode: unframes and parses each transport packet into a structured "
        "message (COBS de-framing and CBOR decode)."
    ),
    "Gate": (
        "Gate: the back-pressure guard that limits how many sensor frames are in "
        "flight to the GUI at once. Drops here are sensor frames shed to keep the "
        "interface responsive; control frames are never dropped."
    ),
    "Plot Buffer": (
        "Plot Buffer: the in-memory sample buffer feeding the live plots. Its "
        "queue depth is the number of samples buffered for the next redraw."
    ),
    "Recorder": (
        "Recorder: writes samples to the CSV recording while a capture is active. "
        "Idle (zero rate) when not recording."
    ),
}

_DEFAULT_DESCRIPTION = (
    "Pipeline stage. Colour shows queue utilisation (green = idle, red = "
    "saturated); a red badge marks dropped frames."
)


def stage_description(name: str) -> str:
    """Return the hover description for a stage *name*.

    Pure and deterministic so the description text can be computed without a
    live widget or a synthesised tooltip event.
    """
    return STAGE_DESCRIPTIONS.get(name, _DEFAULT_DESCRIPTION)


def _lerp(a: int, b: int, f: float) -> int:
    return int(round(a + (b - a) * f))


def _blend(c0: QColor, c1: QColor, f: float) -> QColor:
    return QColor(
        _lerp(c0.red(), c1.red(), f),
        _lerp(c0.green(), c1.green(), f),
        _lerp(c0.blue(), c1.blue(), f),
    )


def utilization_color(utilization: float) -> QColor:
    """Map a queue utilisation in ``[0, 1]`` to a green→amber→red colour.

    Pure and deterministic so the colour mapping can be computed without
    inspecting painted pixels.
    """
    u = 0.0 if utilization < 0.0 else 1.0 if utilization > 1.0 else utilization
    if u <= 0.5:
        return _blend(_GREEN, _AMBER, u / 0.5)
    return _blend(_AMBER, _RED, (u - 0.5) / 0.5)


class PipelineFlowWidget(QWidget):
    """Renders a :class:`PipelineSnapshot` as a coloured, animated stage row."""

    _MIN_HEIGHT = 150
    _STAGE_GAP = 28.0
    #: Spacing between successive flow chevrons.
    _CHEVRON_SPACING_PX = 18.0
    #: Repaint cadence for smooth chevron motion (~33 FPS).
    _ANIM_INTERVAL_MS = 30

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumHeight(self._MIN_HEIGHT)
        self._snapshot: PipelineSnapshot | None = None
        self._bottleneck: StageMetrics | None = None
        # Per-stage drop totals captured at the last "Clear counters & node
        # warnings" press; subtracted from later snapshots so each stage's drop
        # badge and red border clear until *new* frames are dropped.
        self._raw_snapshot: PipelineSnapshot | None = None
        self._drop_baseline: dict[str, int] = {}
        # Per-connector accumulated chevron displacement (px), integrated as
        # ``speed * dt`` each repaint. Storing the travelled distance — rather
        # than recomputing ``absolute_time * speed`` — keeps the motion smooth
        # when a connector's speed changes between frames (a varying stage rate
        # would otherwise teleport the chevrons). Sized to the connector count.
        self._flow_distance: list[float] = []
        self._last_anim_t = monotonic()
        # Hit-rects for the most recently painted stages, used to map a hover
        # position to the stage under the cursor for tooltips.
        self._stage_rects: list[tuple[QRectF, str]] = []
        # Drive smooth chevron animation at a higher frame rate than the
        # snapshot refresh, independent of when new data arrives.
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(self._ANIM_INTERVAL_MS)
        self._anim_timer.timeout.connect(self.update)
        self._anim_timer.start()

    # -- public API --------------------------------------------------------
    @property
    def snapshot(self) -> PipelineSnapshot | None:
        """The most recently rendered snapshot (``None`` before first update)."""
        return self._snapshot

    @property
    def bottleneck(self) -> StageMetrics | None:
        """The bottleneck stage of the current snapshot, or ``None``."""
        return self._bottleneck

    def process_groups(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """Contiguous per-process stage groups of the current snapshot.

        Mirrors exactly what :meth:`paintEvent` boxes and labels; empty before
        the first snapshot.
        """
        if self._snapshot is None:
            return ()
        return group_stages_by_process(self._snapshot)

    def update_snapshot(self, snapshot: PipelineSnapshot) -> None:
        """Store *snapshot* and repaint in place (no widget-tree rebuild).

        Drop totals are shown net of the baseline captured by the last
        :meth:`clear_warnings`, so a stage that dropped before the reset is no
        longer flagged until it drops again.
        """
        self._raw_snapshot = snapshot
        adjusted = self._apply_drop_baseline(snapshot)
        self._snapshot = adjusted
        self._bottleneck = detect_bottleneck(adjusted)
        self.update()

    def clear_warnings(self) -> None:
        """Zero the per-stage drop counters and clear red-border emphasis.

        Captures the current per-stage drop totals as a baseline that is
        subtracted from subsequent snapshots, so every stage's drop badge and the
        bottleneck red border clear now and stay clear until *new* frames are
        dropped beyond this point. Live queue-utilisation bottlenecks reappear on
        the next snapshot if a stage is still saturated.
        """
        snap = self._raw_snapshot
        self._drop_baseline = (
            {st.name: st.drops_total for st in snap.stages} if snap is not None else {}
        )
        if snap is not None:
            adjusted = self._apply_drop_baseline(snap)
            self._snapshot = adjusted
            self._bottleneck = detect_bottleneck(adjusted)
        self.update()

    def _apply_drop_baseline(self, snapshot: PipelineSnapshot) -> PipelineSnapshot:
        """Return *snapshot* with per-stage drops reduced by the cleared baseline.

        When no baseline is set the original object is returned unchanged, so the
        common path allocates nothing and ``snapshot`` identity is preserved.
        """
        if not self._drop_baseline:
            return snapshot
        stages = tuple(
            replace(
                st,
                drops_total=max(0, st.drops_total - self._drop_baseline.get(st.name, 0)),
            )
            for st in snapshot.stages
        )
        return PipelineSnapshot(stages=stages, wall_time_s=snapshot.wall_time_s)


    def description_at(self, pos: QPoint | QPointF) -> str | None:
        """Return the stage description under *pos*, or ``None`` if outside any.

        Uses the hit-rects recorded by the last paint, so it reflects exactly
        what is on screen.
        """
        pt = QPointF(pos)
        for rect, name in self._stage_rects:
            if rect.contains(pt):
                return stage_description(name)
        return None

    def event(self, ev: QEvent) -> bool:  # noqa: N802 (Qt naming)
        # Custom-painted stages have no child widgets, so tooltips are resolved
        # by hit-testing the painted stage rects on demand.
        if ev.type() == QEvent.Type.ToolTip and isinstance(ev, QHelpEvent):
            text = self.description_at(ev.pos())
            if text:
                QToolTip.showText(ev.globalPos(), text, self)
            else:
                QToolTip.hideText()
                ev.ignore()
            return True
        return super().event(ev)

    # -- painting ----------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        snap = self._snapshot
        if snap is None or not snap.stages:
            painter.end()
            return

        n = len(snap.stages)
        w = float(self.width())
        h = float(self.height())
        total_gap = self._STAGE_GAP * (n - 1)
        box_w = max(40.0, (w - total_gap - 8.0) / n)
        # Reserve a strip at the bottom for the per-process group labels.
        box_h = min(72.0, h - 60.0)
        top = 8.0
        # Time elapsed since the previous repaint, used to integrate the chevron
        # displacement. Clamped so a large gap (widget hidden, first frame)
        # cannot make the chevrons leap.
        now = monotonic()
        dt = now - self._last_anim_t
        self._last_anim_t = now
        if dt < 0.0 or dt > 0.25:
            dt = 0.0

        centers: list[float] = []
        self._stage_rects = []
        for i, stage in enumerate(snap.stages):
            x = 4.0 + i * (box_w + self._STAGE_GAP)
            rect = QRectF(x, top, box_w, box_h)
            self._paint_stage(painter, rect, stage)
            self._stage_rects.append((rect, stage.name))
            centers.append(x + box_w)

        # Flow chevrons between consecutive stages; each connector advances by
        # its upstream stage's output-rate-derived speed, accumulated over time
        # so a varying rate changes the speed without jumping the chevrons.
        n_gaps = n - 1
        if len(self._flow_distance) != n_gaps:
            self._flow_distance = [0.0] * n_gaps
        for i in range(n_gaps):
            x0 = centers[i]
            x1 = 4.0 + (i + 1) * (box_w + self._STAGE_GAP)
            self._flow_distance[i] += flow_speed(snap.stages[i].out_rate) * dt
            self._paint_flow(
                painter, x0, x1, top + box_h / 2.0, self._flow_distance[i]
            )

        # Labelled boundary around each process's contiguous stages.
        self._paint_process_groups(painter, snap, top, box_h)
        painter.end()


    def _paint_stage(self, painter: QPainter, rect: QRectF, stage: StageMetrics) -> None:
        is_bottleneck = self._bottleneck is not None and stage.name == self._bottleneck.name
        fill = utilization_color(stage.utilization)
        painter.setBrush(QBrush(fill))
        border = QPen(_RED if is_bottleneck else QColor(0x37, 0x47, 0x4F))
        border.setWidth(3 if is_bottleneck else 1)
        painter.setPen(border)
        painter.drawRoundedRect(rect, 6.0, 6.0)

        # Stage name.
        painter.setPen(QPen(QColor(Qt.GlobalColor.white)))
        name_font = QFont(painter.font())
        name_font.setBold(True)
        painter.setFont(name_font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, stage.name)

        # Queue-fill gauge along the bottom edge.
        gauge = QRectF(rect.left() + 4.0, rect.bottom() - 8.0, rect.width() - 8.0, 4.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 60)))
        painter.drawRect(gauge)
        if stage.queue_capacity > 0:
            filled = QRectF(gauge)
            filled.setWidth(gauge.width() * stage.utilization)
            painter.setBrush(QBrush(QColor(Qt.GlobalColor.white)))
            painter.drawRect(filled)

        # Drop badge.
        if stage.is_dropping:
            badge = QRectF(rect.right() - 16.0, rect.top() - 6.0, 22.0, 16.0)
            painter.setBrush(QBrush(_RED))
            painter.setPen(QPen(QColor(Qt.GlobalColor.white)))
            painter.drawRoundedRect(badge, 8.0, 8.0)
            small = QFont(painter.font())
            small.setPointSizeF(max(6.0, small.pointSizeF() - 2.0))
            small.setBold(True)
            painter.setFont(small)
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, _fmt_drops(stage.drops_total))

    def _paint_flow(
        self, painter: QPainter, x0: float, x1: float, y: float,
        distance: float,
    ) -> None:
        painter.setPen(QPen(QColor(0x90, 0xA4, 0xAE), 1))
        painter.drawLine(QPointF(x0, y), QPointF(x1, y))
        span = x1 - x0
        if span <= 0:
            return
        # ">"-style chevrons march downstream; *distance* is the displacement
        # already accumulated for this connector (integrated from the upstream
        # output rate), so the motion stays smooth even when that rate varies.
        # Rendered as a stroked polyline at a higher repaint cadence.
        pen = QPen(QColor(0x29, 0xB6, 0xF6))
        pen.setWidth(2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        half_w = 4.0
        half_h = 4.0
        for d in chevron_offsets(span, distance, self._CHEVRON_SPACING_PX):
            cx = x0 + d
            chevron = QPolygonF([
                QPointF(cx - half_w, y - half_h),
                QPointF(cx, y),
                QPointF(cx - half_w, y + half_h),
            ])
            painter.drawPolyline(chevron)


    def _paint_process_groups(
        self, painter: QPainter, snap: PipelineSnapshot, top: float, box_h: float,
    ) -> None:
        """Draw a dashed, labelled boundary around each process's stages."""
        groups = group_stages_by_process(snap)
        # Nothing to convey if every stage is unattributed and ungrouped.
        if not groups or (len(groups) == 1 and groups[0][0] == "?"):
            return
        pad = 5.0
        label_h = 14.0
        box_pen = QPen(QColor(0xB0, 0xBE, 0xC5))
        box_pen.setStyle(Qt.PenStyle.DashLine)
        box_pen.setWidth(1)
        label_font = QFont(painter.font())
        label_font.setPointSizeF(max(7.0, label_font.pointSizeF() - 1.0))
        label_font.setBold(True)
        for proc, idxs in groups:
            rects = [self._stage_rects[i][0] for i in idxs]
            left = min(r.left() for r in rects) - pad
            right = max(r.right() for r in rects) + pad
            group_rect = QRectF(left, top - pad, right - left, box_h + 2 * pad)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(box_pen)
            painter.drawRoundedRect(group_rect, 6.0, 6.0)
            painter.setPen(QPen(QColor(0xCF, 0xD8, 0xDC)))
            painter.setFont(label_font)
            label_rect = QRectF(left, group_rect.bottom() + 1.0, right - left, label_h)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, proc)



def _fmt_drops(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.0f}k"
    return str(int(n))


def _soft_log(rate: float) -> float:
    # Maps 0→0, ~1→0.1, ~100→~0.46, ~1000→~0.69 for a gentle, bounded speed ramp.
    from math import log10

    if rate <= 0.0:
        return 0.0
    return min(1.0, log10(1.0 + rate) / 10.0)


def flow_speed(out_rate: float) -> float:
    """Chevron advance speed in px/s for an upstream stage's output rate.

    Deliberately gentle and bounded — a stalled stage (zero rate) holds its
    chevrons still, while a fast stage streams them only modestly quicker — so
    the motion reads as "flow" without becoming a distraction. Pure so the speed
    ramp can be computed without painting.
    """
    return 8.0 + 18.0 * _soft_log(max(0.0, out_rate))


def chevron_offsets(
    span: float, distance: float, spacing: float,
) -> tuple[float, ...]:
    """X offsets (in ``[0, span)``) of the chevrons at accumulated *distance*.

    The chevrons march from the upstream stage toward the downstream one, evenly
    spaced by ``spacing``; *distance* is the total displacement already travelled
    (the caller integrates ``speed * dt`` over time, rather than multiplying an
    absolute timestamp by the current speed) so a between-frame speed change only
    changes the *rate* of advance and never the instantaneous position — the
    flow stays continuous. Pure and deterministic so the animation can be
    computed without a live repaint.
    """
    if span <= 0.0 or spacing <= 0.0:
        return ()
    start = distance % spacing
    out: list[float] = []
    d = start
    while d < span:
        out.append(d)
        d += spacing
    return tuple(out)
