# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Headless flood reproduction for the Data Forwarder Host GUI freeze.

This drives synthetic, high-rate ``sensor_data`` frames through a **real**
:class:`SessionController` and the **real** live GUI widgets
(:class:`CombinedPlot`, :class:`ChartsPanel`, :class:`SessionLogPanel`,
:class:`BandwidthDetailsDialog`) running on the ``offscreen`` Qt platform — i.e.
the exact GUI-thread code path the app uses, minus the window manager.

The sequence numbers are deliberately given **large forward gaps** to mimic the
frames the GUI inbox drops when it is overloaded (see ``FrameInbox`` /
``_drain_inbox``): under real overload the GUI never sees the dropped frames, so
the frames it *does* see have gaps in ``seq`` — and those gaps are fed straight
into the transport-loss detector by ``SessionController._process_message``.

The always-on :mod:`slow_span` instrumentation prints ``slow-span ...`` lines to
stdout whenever any GUI-thread span exceeds its threshold, so the precise span
that blocks the UI thread is *proven*, not guessed. Peak RSS and the loss
detector's pending backlog are reported at the end to corroborate the OOM.

Usage::

    QT_QPA_PLATFORM=offscreen \\
    PYTHONPATH=/workspaces/edge-ai/edge-ai/tools \\
    /workspaces/edge-ai/.venv/bin/python \\
    edge-ai/tools/data_forwarder_host/scripts/repro_freeze.py [options]

Options:
    --seconds N       wall-clock duration to flood for (default 12)
    --rate N          frames delivered per drain tick (default 400)
    --interval-ms N   drain tick period in ms (default 8, matching MAX_DRAIN_MS)
    --gap N           seq numbers skipped between ticks = simulated inbox drops
                      (default 2000; use 0 for a clean, no-loss baseline)
    --channels N      number of channels per frame (default 3)
    --clean           shortcut for --gap 0 (no simulated drops)
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone


def _rss_mb() -> float:
    """Resident set size of this process in MiB (Linux ``/proc`` based)."""
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            resident_pages = int(fh.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        import resource

        # ru_maxrss is KiB on Linux.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--rate", type=int, default=400)
    parser.add_argument("--interval-ms", type=int, default=8)
    parser.add_argument("--gap", type=int, default=2000)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args(argv)
    if args.clean:
        args.gap = 0

    # Offscreen so no display is needed; must be set before any Qt import.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # Send slow-span (and everything on data_forwarder_host.debug) to stdout.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    import data_forwarder_host.utils.slow_span as ss

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from data_forwarder_host.gui.widgets.bandwidth_details import (
        BandwidthDetailsDialog,
    )
    from data_forwarder_host.gui.widgets.charts_panel import ChartsPanel
    from data_forwarder_host.gui.widgets.combined_plot import CombinedPlot
    from data_forwarder_host.gui.widgets.session_log_panel import SessionLogPanel
    from data_forwarder_host.protocol.base import DecodedMessage
    from data_forwarder_host.session.config import SessionConfig, SourceSpec
    from data_forwarder_host.session.controller import SessionController

    app = QApplication.instance() or QApplication([])

    n_ch = args.channels
    # loss_confirmation_window deliberately long-ish (2 s) so that, when we
    # generate losses faster than they expire, the detector's pending set
    # grows — exactly the condition that makes its per-frame sweep expensive.
    config = SessionConfig(
        tag="repro",
        source=SourceSpec(kind="ble"),
        plot_window_seconds=10.0,
        bandwidth_window_seconds=10.0,
        loss_confirmation_window_seconds=2.0,
    )
    # use_process=True => the controller builds no GUI-side source and we never
    # call start(), so no child process is spawned; we inject frames directly
    # into the same _process_message slot the live session would drive.
    ctrl = SessionController(config, use_process=True)

    # Real live widgets, wired exactly as the session window wires them.
    combined = CombinedPlot(ctrl.data_model)
    charts = ChartsPanel(ctrl.data_model)
    logs = SessionLogPanel(ctrl)
    bw = BandwidthDetailsDialog(ctrl)
    for w in (combined, charts, logs, bw):
        w.resize(900, 400)
        w.show()
    # Expand the per-channel section so those charts also refresh each tick.
    try:
        charts._channels_section.set_expanded(True)  # noqa: SLF001
    except Exception:
        pass
    app.processEvents()

    # Establish channel metadata so the model locks channel names/buffers and
    # the controller marks metadata ready (otherwise sensor_data is rejected by
    # _validate_sensor_data before it ever reaches the loss detector).
    now_utc = datetime.now(timezone.utc).isoformat()
    session_info = DecodedMessage(
        kind="session_info",
        t_host_ms=0,
        t_host_utc=now_utc,
        t_device_ms=0,
        seq=None,
        label=None,
        channels=None,
        raw={
            "t": "si",
            "d": {
                "ch_n": [f"ch{i}" for i in range(n_ch)],
                "ch": n_ch,
                "sid": 1,
                "hz": 200,
                "st": 0,
                "dr": 0,
                "name": "repro",
            },
        },
    )
    ctrl._process_message(session_info)  # noqa: SLF001
    app.processEvents()
    assert ctrl.metadata_ready, "session_info rejected — repro would not exercise the real path"

    state = {
        "seq": 0,
        "t_dev": 0,
        "sent": 0,
        "ticks": 0,
        "peak_rss": _rss_mb(),
        "t0": time.monotonic(),
    }
    sample_period_ms = 5  # 200 Hz

    def pump() -> None:
        # One drain-tick's worth of frames, delivered synchronously like the
        # real GUI drain loop. A forward gap is then opened in ``seq`` to mimic
        # the frames the inbox dropped to stay responsive.
        with ss.slow_span("repro.tick", extra=f"n={args.rate} gap={args.gap}"):
            for _ in range(args.rate):
                seq = state["seq"]
                t_dev = state["t_dev"]
                vals = tuple(
                    math.sin((t_dev / 1000.0) * (1.0 + i) * 2.0) + 0.01 * (seq % 7)
                    for i in range(n_ch)
                )
                msg = DecodedMessage(
                    kind="sensor_data",
                    t_host_ms=t_dev,
                    t_host_utc=now_utc,
                    t_device_ms=t_dev,
                    seq=seq,
                    label=None,
                    channels=vals,
                    raw={},
                )
                ctrl._process_message(msg)  # noqa: SLF001
                state["seq"] = seq + 1
                state["t_dev"] = t_dev + sample_period_ms
            # Simulate the inbox having dropped ``gap`` frames between ticks:
            # advance both the sequence cursor and the device clock past them.
            if args.gap:
                state["seq"] += args.gap
                state["t_dev"] += args.gap * sample_period_ms
        state["sent"] += args.rate
        state["ticks"] += 1
        rss = _rss_mb()
        if rss > state["peak_rss"]:
            state["peak_rss"] = rss

    timer = QTimer()
    timer.setInterval(args.interval_ms)
    timer.timeout.connect(pump)
    timer.start()

    def finish() -> None:
        timer.stop()
        app.quit()

    QTimer.singleShot(int(args.seconds * 1000), finish)

    print(
        f"[repro] flooding mode={'CLEAN' if args.gap == 0 else 'LOSSY'} "
        f"rate={args.rate}/tick interval={args.interval_ms}ms gap={args.gap} "
        f"channels={n_ch} for {args.seconds}s — watch for 'slow-span' lines…",
        flush=True,
    )
    app.exec()

    elapsed = time.monotonic() - state["t0"]
    pending = ctrl._seq_tracker.pending_count  # noqa: SLF001
    fps = state["sent"] / elapsed if elapsed else 0.0
    print(
        "\n[repro] done: "
        f"sent={state['sent']} frames in {elapsed:.1f}s "
        f"({fps:,.0f} frames/s) over {state['ticks']} ticks | "
        f"seq_tracker.pending={pending:,} | "
        f"peak_rss={state['peak_rss']:.0f} MiB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
