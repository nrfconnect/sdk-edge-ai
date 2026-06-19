# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Smoothed wall-clock-to-device-time playout clock for live charts.

:class:`PlayoutClock` is a pure, GUI-free state machine that decides *where the
right edge of a live chart should be* at any wall-clock moment. Instead of
snapping the view to the newest sample as it arrives — which makes the chart
stutter when data comes in bursts or late — the clock advances a **device-time
playout position** continuously, at a low-pass-filtered estimate of how fast the
device timeline is advancing (device-milliseconds per wall-second).

The position trails the newest received device timestamp by a small de-jitter
*delay* (so late samples have time to arrive before their slot scrolls into
view), is nudged toward that target with a gentle proportional *catch-up* (so
device-timer drift is absorbed smoothly rather than in steps), and is clamped so
it never runs ahead of the newest received timestamp (the chart never shows
future/unreceived data) nor falls more than *max_lag* behind it (after a long
stall it jumps forward to within the cap instead of crawling). The position is
monotonic non-decreasing.

The clock is deterministic: every method takes the current wall time as an
argument, so callers inject ``time.monotonic()`` (or a fake clock for determinism).
This module imports no Qt and holds no global state.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Target lag (device-ms) the playout position trails the newest timestamp by,
#: giving jittered/late samples time to arrive before their slot scrolls in.
DEFAULT_DELAY_MS: float = 150.0

#: Hard cap (device-ms) on how far the position may fall behind the newest
#: timestamp. After a long stall the position jumps forward to within this cap
#: rather than crawling. Must be >= ``DEFAULT_DELAY_MS``.
DEFAULT_MAX_LAG_MS: float = 1000.0

#: EWMA smoothing factor (0 < alpha <= 1) for the device-rate estimate. Smaller
#: means heavier low-pass filtering, so speed changes are applied more gradually.
DEFAULT_RATE_ALPHA: float = 0.1

#: Proportional gain (0 < k <= 1) for nudging the position toward its target lag.
DEFAULT_CATCHUP_GAIN: float = 0.1

#: Initial device-rate estimate (device-ms per wall-second) before any rate has
#: been measured: assume the device clock runs at real time (1000 ms/s).
DEFAULT_RATE_MS_PER_S: float = 1000.0

#: Outlier-rejection band: an instantaneous rate sample is folded into the EWMA
#: only if it lies within ``[1/BAND, BAND]`` of the current estimate, so a
#: burst of buffered samples delivered at once does not yank the speed estimate.
_RATE_OUTLIER_BAND: float = 10.0


@dataclass(frozen=True, slots=True)
class PlayoutConfig:
    """Tunable constants for a :class:`PlayoutClock` (all device-ms / unitless)."""

    delay_ms: float = DEFAULT_DELAY_MS
    max_lag_ms: float = DEFAULT_MAX_LAG_MS
    rate_alpha: float = DEFAULT_RATE_ALPHA
    catchup_gain: float = DEFAULT_CATCHUP_GAIN
    initial_rate_ms_per_s: float = DEFAULT_RATE_MS_PER_S

    def __post_init__(self) -> None:
        if not 0.0 < self.rate_alpha <= 1.0:
            raise ValueError("rate_alpha must be in (0, 1]")
        if not 0.0 < self.catchup_gain <= 1.0:
            raise ValueError("catchup_gain must be in (0, 1]")
        if self.delay_ms < 0.0:
            raise ValueError("delay_ms must be >= 0")
        if self.max_lag_ms < self.delay_ms:
            raise ValueError("max_lag_ms must be >= delay_ms")
        if self.initial_rate_ms_per_s <= 0.0:
            raise ValueError("initial_rate_ms_per_s must be > 0")


class PlayoutClock:
    """Maps wall-clock time to a smoothed device-time playout position.

    Feed the clock the newest device timestamp seen so far via :meth:`observe`
    (typically once per ingest batch), then read :meth:`position_ms` on every
    redraw tick. The returned position advances smoothly between observations at
    the filtered device rate and is clamped to ``[newest - max_lag, newest]``.
    """

    def __init__(self, config: PlayoutConfig | None = None) -> None:
        self._cfg = config or PlayoutConfig()
        self._reset_state()

    # -- lifecycle ---------------------------------------------------------
    def _reset_state(self) -> None:
        self._initialized = False
        self._position_ms = 0.0
        self._newest_ts = 0.0
        self._rate_ms_per_s = self._cfg.initial_rate_ms_per_s
        self._last_now_s = 0.0
        self._obs_ts = 0.0
        self._obs_now_s = 0.0

    def reset(self) -> None:
        """Return the clock to its uninitialised state (e.g. on new session)."""
        self._reset_state()

    @property
    def is_initialized(self) -> bool:
        """``True`` once the first :meth:`observe` has seeded the clock."""
        return self._initialized

    @property
    def rate_ms_per_s(self) -> float:
        """Current low-pass-filtered device rate (device-ms per wall-second)."""
        return self._rate_ms_per_s

    @property
    def newest_ts_ms(self) -> float:
        """The newest device timestamp observed so far (device-ms)."""
        return self._newest_ts

    # -- inputs ------------------------------------------------------------
    def observe(self, newest_device_ts_ms: float, now_s: float) -> None:
        """Record the newest device timestamp seen at wall time *now_s*.

        Updates the low-pass device-rate estimate from the progress made since
        the previous observation. Stalls (no new device progress) and outlier
        bursts do not perturb the rate estimate.
        """
        if not self._initialized:
            self._initialized = True
            self._newest_ts = float(newest_device_ts_ms)
            # Start already trailing by the target delay so we begin de-jittered.
            self._position_ms = self._newest_ts - self._cfg.delay_ms
            self._last_now_s = float(now_s)
            self._obs_ts = float(newest_device_ts_ms)
            self._obs_now_s = float(now_s)
            return

        dt = float(now_s) - self._obs_now_s
        d_ts = float(newest_device_ts_ms) - self._obs_ts
        if dt > 0.0 and d_ts > 0.0:
            inst_rate = d_ts / dt
            lo = self._rate_ms_per_s / _RATE_OUTLIER_BAND
            hi = self._rate_ms_per_s * _RATE_OUTLIER_BAND
            if lo <= inst_rate <= hi:
                a = self._cfg.rate_alpha
                self._rate_ms_per_s = (1.0 - a) * self._rate_ms_per_s + a * inst_rate
            self._obs_ts = float(newest_device_ts_ms)
            self._obs_now_s = float(now_s)
        elif d_ts > 0.0:
            # Progress with non-positive dt (same wall instant): advance anchor
            # without a rate sample.
            self._obs_ts = float(newest_device_ts_ms)
            self._obs_now_s = float(now_s)

        if newest_device_ts_ms > self._newest_ts:
            self._newest_ts = float(newest_device_ts_ms)

    # -- output ------------------------------------------------------------
    def position_ms(self, now_s: float) -> float:
        """Return the playout position (device-ms) at wall time *now_s*.

        Advances the position by the filtered rate over the elapsed wall time
        plus a gentle catch-up toward ``newest - delay``, then clamps it to
        never exceed ``newest`` nor trail it by more than ``max_lag``, and keeps
        it monotonic non-decreasing.
        """
        if not self._initialized:
            return 0.0

        dt = float(now_s) - self._last_now_s
        if dt <= 0.0:
            # No wall time has elapsed (e.g. a second widget reads the same tick,
            # or the clock went backwards): the position is unchanged. This keeps
            # repeated same-instant reads idempotent and the result monotonic.
            return self._position_ms

        base_advance = self._rate_ms_per_s * dt
        target = self._newest_ts - self._cfg.delay_ms
        error = target - (self._position_ms + base_advance)
        catchup = self._cfg.catchup_gain * error
        new_pos = self._position_ms + base_advance + catchup

        # Never show future/unreceived data.
        if new_pos > self._newest_ts:
            new_pos = self._newest_ts
        # Bound how far we trail: jump forward to the cap after a long stall.
        min_pos = self._newest_ts - self._cfg.max_lag_ms
        if new_pos < min_pos:
            new_pos = min_pos
        # Monotonic non-decreasing.
        if new_pos < self._position_ms:
            new_pos = self._position_ms

        self._position_ms = new_pos
        self._last_now_s = float(now_s)
        return self._position_ms

    def lag_ms(self, now_s: float) -> float:
        """Current gap (device-ms) between the newest timestamp and the position."""
        return self._newest_ts - self.position_ms(now_s)
