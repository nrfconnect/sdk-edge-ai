# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Per-session GUI-side data model — rolling buffers + redraw coalescer."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from data_forwarder_host.core.channels import ChannelBuffer
from data_forwarder_host.core.error_log import ErrorCategory, ErrorLog
from data_forwarder_host.core.playout import PlayoutClock
from data_forwarder_host.protocol.base import DecodedMessage
from data_forwarder_host.utils.slow_span import slow_span_fn

_REDRAW_HZ = 30
_REDRAW_INTERVAL_MS = int(1000 / _REDRAW_HZ)

# When no recording is in progress there is no need to keep accumulating live
# samples beyond what the plot shows; idle retention is capped at this many
# seconds (or the plot window, whichever is larger).
_IDLE_RETENTION_FLOOR_SECONDS = 100.0


class DataModel(QObject):
    """Per-session rolling buffers; lives on the GUI thread.

    Channels are discovered hybrid-style:
      * if a ``session_info`` message contains channel names, those win;
      * otherwise the count is locked from the first ``sensor_data`` message;
      * subsequent count mismatches are surfaced via :class:`ErrorLog`.
    """

    channels_changed = Signal()                 # ()
    data_appended = Signal(int)                 # (samples added since last tick)
    session_info = Signal(dict)                 # (raw session-info dict)
    cleared = Signal()
    recording_marker_changed = Signal()         # recording start/stop markers moved/cleared

    def __init__(
        self,
        *,
        plot_window_seconds: float,
        error_log: ErrorLog,
        parent: QObject | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(parent)
        self._plot_window_seconds = plot_window_seconds
        self._error_log = error_log
        # Smooth wall-clock-to-device-time playout for the chart right edge so
        # the view scrolls continuously even when samples arrive late/in bursts;
        # late points fill in as the playout position passes their device ts
        # (the time source is injectable for deterministic playout).
        self._time_fn: Callable[[], float] = time_fn or time.monotonic
        self._playout = PlayoutClock()
        self._channel_names: tuple[str, ...] = ()
        self._channels_locked: bool = False
        self._buffers: list[ChannelBuffer] = []
        self._pending_added: int = 0
        self._recording: bool = False

        # Recording markers: device timestamps for the vertical lines drawn on
        # the charts where recording began and ended. The start marker sits just
        # before the first recorded sample (after the last non-recorded one) and
        # PERSISTS after recording stops; the stop marker is fixed at the last
        # recorded sample when recording ends.
        self._last_appended_ts: int | None = None
        self._awaiting_first_recorded: bool = False
        self._recording_start_marker_ms: int | None = None
        self._recording_stop_marker_ms: int | None = None

        self._timer = QTimer(self)
        self._timer.setInterval(_REDRAW_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def plot_window_seconds(self) -> float:
        return self._plot_window_seconds

    @property
    def channel_names(self) -> tuple[str, ...]:
        return self._channel_names

    @property
    def channel_count(self) -> int:
        return len(self._buffers)

    @property
    def max_buffer_size(self) -> int:
        """Largest current per-channel sample count (diagnostics)."""
        return max((b.size for b in self._buffers), default=0)

    def buffer(self, index: int) -> ChannelBuffer:
        return self._buffers[index]

    @property
    def recording_start_marker_ms(self) -> int | None:
        """Device timestamp of the recording-start marker, or ``None``."""
        return self._recording_start_marker_ms

    @property
    def recording_stop_marker_ms(self) -> int | None:
        """Device timestamp of the recording-stop marker, or ``None``."""
        return self._recording_stop_marker_ms

    def begin_recording_marker(self) -> None:
        """Arm the start marker for a new recording, clearing prior markers.

        The next recorded sample fixes the new start moment. Any markers from a
        previous recording are cleared so only the current capture is shown.
        """
        self._awaiting_first_recorded = True
        changed = False
        if self._recording_start_marker_ms is not None:
            self._recording_start_marker_ms = None
            changed = True
        if self._recording_stop_marker_ms is not None:
            self._recording_stop_marker_ms = None
            changed = True
        if changed:
            self.recording_marker_changed.emit()

    def end_recording_marker(self) -> None:
        """Fix the stop marker at the last recorded sample on a normal stop.

        The start marker is left in place so both the begin and end of the
        recording stay visible on the charts.
        """
        self._awaiting_first_recorded = False
        stop = self._last_appended_ts
        if stop is not None and stop != self._recording_stop_marker_ms:
            self._recording_stop_marker_ms = stop
            self.recording_marker_changed.emit()

    def clear_recording_markers(self) -> None:
        """Remove both recording markers (e.g. when a recording is cancelled)."""
        self._awaiting_first_recorded = False
        changed = False
        if self._recording_start_marker_ms is not None:
            self._recording_start_marker_ms = None
            changed = True
        if self._recording_stop_marker_ms is not None:
            self._recording_stop_marker_ms = None
            changed = True
        if changed:
            self.recording_marker_changed.emit()

    def set_plot_window_seconds(self, seconds: float) -> None:
        if seconds <= 0 or seconds == self._plot_window_seconds:
            return
        self._plot_window_seconds = seconds
        for b in self._buffers:
            b.resize(plot_window_seconds=seconds)

    def set_recording(self, recording: bool) -> None:
        """Track recording state so idle retention can be capped."""
        self._recording = bool(recording)

    def playout_position_ms(self) -> float | None:
        """Smoothed device-time right edge for the charts, or ``None`` if idle.

        Returns the :class:`PlayoutClock` position — which advances
        continuously at the low-pass-filtered device rate and trails the newest
        received timestamp by a small de-jitter delay — so the chart right edge
        scrolls smoothly. Before any data has been seen it returns ``None`` and
        callers fall back to their data-derived range.
        """
        if not self._playout.is_initialized:
            return None
        return self._playout.position_ms(self._time_fn())

    def visible_x_window_ms(self) -> tuple[float, float] | None:
        """Shared rolling X window ``(lo_ms, hi_ms)`` for every chart.

        All charts — the combined overlay and each per-channel chart — must show
        the **same** X range so a given horizontal position maps to the same
        time across them. That range is computed once here: the right edge
        follows the smoothed playout clock, falling back to the newest
        appended device timestamp; the left edge is exactly one plot window
        behind it. Returns ``None`` before any data has been seen, so callers can
        keep their previous range until the stream starts.
        """
        right = self.playout_position_ms()
        if right is None:
            right = self._last_appended_ts
        if right is None:
            return None
        window_ms = self._plot_window_seconds * 1000.0
        return (float(right) - window_ms, float(right))

    def clear(self) -> None:
        for b in self._buffers:
            b.clear()
        self._pending_added = 0
        self._last_appended_ts = None
        self._playout.reset()
        self._awaiting_first_recorded = False
        self._recording_start_marker_ms = None
        self._recording_stop_marker_ms = None
        self.cleared.emit()
        self.recording_marker_changed.emit()

    def reset(self) -> None:
        self._buffers = []
        self._channel_names = ()
        self._channels_locked = False
        self._pending_added = 0
        self._last_appended_ts = None
        self._playout.reset()
        self._awaiting_first_recorded = False
        self._recording_start_marker_ms = None
        self._recording_stop_marker_ms = None
        self.channels_changed.emit()
        self.cleared.emit()
        self.recording_marker_changed.emit()

    # ------------------------------------------------------------------
    # Slot
    # ------------------------------------------------------------------

    @slow_span_fn("data_model.on_message")
    def on_message(self, msg: DecodedMessage) -> None:
        if msg.kind == "session_info":
            self._handle_session_info(msg.raw)
            return
        if msg.kind != "sensor_data" or msg.channels is None:
            return

        n = len(msg.channels)
        if not self._channels_locked:
            self._lock_channels(n)

        if n != len(self._buffers):
            self._error_log.add(
                ErrorCategory.CHANNEL_MISMATCH,
                f"expected {len(self._buffers)} channels, got {n}",
                seq=msg.seq,
            )
            return

        # X axis uses device timestamp when present; fall back to host time.
        t_ms = msg.t_device_ms if msg.t_device_ms is not None else msg.t_host_ms
        # Fix the recording-start marker on the first sample recorded after
        # recording began: place it just before this sample but after the last
        # non-recorded one (midpoint), so the vertical line sits between them.
        if self._awaiting_first_recorded:
            if self._last_appended_ts is not None and self._last_appended_ts < t_ms:
                self._recording_start_marker_ms = (self._last_appended_ts + t_ms) // 2
            else:
                self._recording_start_marker_ms = t_ms
            self._awaiting_first_recorded = False
            self.recording_marker_changed.emit()
        for i, v in enumerate(msg.channels):
            self._buffers[i].append(t_ms, v)
        self._last_appended_ts = t_ms
        # Feed the smoothed playout clock the newest device timestamp so the
        # chart right edge can advance continuously between arrivals.
        self._playout.observe(float(t_ms), self._time_fn())
        self._pending_added += 1

    def on_messages(self, messages) -> None:
        """Ingest a batch of decoded messages in one call.

        When the source runs out-of-process frames arrive batched,
        so the GUI thread pays a single hand-off per batch instead of per frame.
        This simply applies the per-message :meth:`on_message` logic to each item
        in order; the 30 Hz coalesced redraw still collapses the whole
        batch into one repaint.
        """
        for msg in messages:
            self.on_message(msg)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _handle_session_info(self, raw: dict[str, Any]) -> None:
        # Look for channel names in common spots.
        names = None
        for key in ("channels", "channel_names", "ch"):
            v = raw.get(key)
            if isinstance(v, (list, tuple)) and all(isinstance(s, str) for s in v):
                names = tuple(v)
                break
            if isinstance(v, dict):
                items = v.get("names") if isinstance(v.get("names"), (list, tuple)) else None
                if items and all(isinstance(s, str) for s in items):
                    names = tuple(items)
                    break

        # The device wire format (see samples/data_forwarder/cddl) nests the
        # channel names under the payload key "d" as "ch_n" (with channel count
        # in "ch"). Accept that, plus the generic "channels" spelling.
        d = raw.get("d") if isinstance(raw.get("d"), dict) else {}
        if names is None and isinstance(d, dict):
            for key in ("ch_n", "channel_names", "channels"):
                v = d.get(key)
                if isinstance(v, (list, tuple)) and v and all(isinstance(s, str) for s in v):
                    names = tuple(v)
                    break

        if names is not None:
            self._set_channel_names(names)
            self._channels_locked = True

        self.session_info.emit(raw)

    def _lock_channels(self, n: int) -> None:
        if self._channel_names and len(self._channel_names) == n:
            names = self._channel_names
        elif self._channel_names:
            # Pre-existing names had the wrong count — fall back to inferred.
            names = tuple(f"ch{i}" for i in range(n))
        else:
            names = tuple(f"ch{i}" for i in range(n))
        self._set_channel_names(names)
        self._channels_locked = True

    def _set_channel_names(self, names: tuple[str, ...]) -> None:
        if names == self._channel_names and len(self._buffers) == len(names):
            return
        self._channel_names = names
        self._buffers = [
            ChannelBuffer(plot_window_seconds=self._plot_window_seconds) for _ in names
        ]
        self.channels_changed.emit()

    @slow_span_fn("data_model._tick")
    def _tick(self) -> None:
        # While idle (not recording), cap how much history the live buffers keep
        # so nothing accumulates needlessly between recordings.
        if not self._recording and self._buffers:
            cap_ms = int(
                max(_IDLE_RETENTION_FLOOR_SECONDS, self._plot_window_seconds) * 1000
            )
            for b in self._buffers:
                b.retain_window(cap_ms)
        if self._pending_added == 0:
            return
        added, self._pending_added = self._pending_added, 0
        self.data_appended.emit(added)
