# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Pure per-stage host pipeline metrics and bottleneck detection.

The host data path is a chain of stages — source (wire) → decode → in-flight
gate → plot buffer → recorder — each with an input rate, an output rate, a
bounded queue, and a running drop count. This module models a snapshot of that
chain as plain, immutable data so the GUI can *visualise* where throughput is
lost without any stage having to know about the others.

Everything here is pure: callers feed in already-measured numbers (from the
existing :mod:`bandwidth`, :mod:`transfer_stats`, queue ``maxsize``/``qsize``
and drop counters) and read back derived utilisation and a bottleneck verdict.
No Qt, no I/O, no global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@dataclass(frozen=True, slots=True)
class StageMetrics:
    """Measured state of a single pipeline stage.

    Rates are in items per second; queue figures are item counts; ``drops_total``
    is a monotonic running total of items the stage has discarded.
    """

    name: str
    in_rate: float = 0.0
    out_rate: float = 0.0
    queue_depth: int = 0
    queue_capacity: int = 0
    drops_total: int = 0
    #: Name of the OS process this stage runs in (e.g. the acquisition child vs
    #: the GUI process), used to box stages by process in the flow view.
    #: Empty means "unattributed" and is treated as its own group.
    process: str = ""

    @property
    def utilization(self) -> float:
        """Queue fill fraction in ``[0, 1]`` (``0`` when the stage is unbounded)."""
        if self.queue_capacity <= 0:
            return 0.0
        return _clamp01(self.queue_depth / self.queue_capacity)

    @property
    def throughput_ratio(self) -> float:
        """``out_rate / in_rate`` in ``[0, 1]``; ``1`` when nothing is arriving.

        A value below 1 means the stage is emitting slower than it receives, i.e.
        it is shedding or backing up work.
        """
        if self.in_rate <= 0.0:
            return 1.0
        return _clamp01(self.out_rate / self.in_rate)

    @property
    def is_dropping(self) -> bool:
        """``True`` if this stage has discarded any items."""
        return self.drops_total > 0


@dataclass(frozen=True, slots=True)
class PipelineSnapshot:
    """An ordered, time-stamped snapshot of every stage in the pipeline."""

    stages: tuple[StageMetrics, ...] = field(default_factory=tuple)
    wall_time_s: float = 0.0

    def __post_init__(self) -> None:
        # Normalise any sequence to an immutable tuple, preserving order.
        object.__setattr__(self, "stages", tuple(self.stages))

    def stage(self, name: str) -> StageMetrics | None:
        """Return the stage with *name*, or ``None`` if absent."""
        for st in self.stages:
            if st.name == name:
                return st
        return None

    @property
    def total_drops(self) -> int:
        """Total frames dropped across every stage in the pipeline."""
        return sum(st.drops_total for st in self.stages)


def per_stage_drop_totals(snapshot: PipelineSnapshot) -> tuple[tuple[str, int], ...]:
    """Per-stage drop totals for the Error & Loss Analysis summary.

    Returns ``(stage_name, drops_total)`` for every stage in pipeline order, so
    the Error & Loss Analysis section can list dropped frames **by the same
    stage** the pipeline diagram shows them on. Because both the table and the
    pipeline widget read these totals from the identical snapshot, the figures
    are guaranteed to agree — dropping is acceptable, but it is always reported,
    and always attributed to the stage that did it.
    """
    return tuple((st.name, st.drops_total) for st in snapshot.stages)


def group_stages_by_process(
    snapshot: PipelineSnapshot,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Group the snapshot's stages into contiguous runs by owning process.

    Returns one ``(process_name, stage_indices)`` entry per maximal run of
    adjacent stages that share a ``process`` tag, in pipeline order. This lets
    the flow view draw a single labelled boundary around each process's stages
    without owning any of the grouping logic. Pure and deterministic so the
    mapping can be computed without the widget; an empty tag is reported as its own
    ``"?"`` group rather than being silently merged.
    """
    groups: list[tuple[str, list[int]]] = []
    for i, st in enumerate(snapshot.stages):
        proc = st.process or "?"
        if groups and groups[-1][0] == proc:
            groups[-1][1].append(i)
        else:
            groups.append((proc, [i]))
    return tuple((proc, tuple(idxs)) for proc, idxs in groups)




#: A stage is considered "pressured" once its queue is at least this full.
BOTTLENECK_UTILIZATION_THRESHOLD: float = 0.8


def build_host_pipeline_snapshot(
    *,
    message_rate: float,
    dropped_frames: int,
    gate_capacity: int,
    model_queue_depth: int,
    recording: bool,
    recorder_queue_depth: int = 0,
    device_rate: float = 0.0,
    producer_drops: int = 0,
    transport_kind: str = "",
    transport_loss: int = 0,
    recorder_overflow: int = 0,
    wall_time_s: float = 0.0,
) -> PipelineSnapshot:
    """Assemble the host data-path snapshot from already-measured numbers.

    The chain is Device → Wire → Decode → Gate → Plot Buffer → Recorder. The Device
    stage represents the sensor board producing samples: its rate is the device's
    reported frame rate and its drops are samples the device discarded before
    they reached the host. The remaining stages carry the measured host
    message rate; the gate holds the overload drop count and its bounded
    capacity, and the plot buffer/recorder show their current backlog. Pure: no Qt, no
    I/O — callers feed in the measurements and read back the structure.

    When *transport_kind* is ``"ble"`` an extra **Host BT** stage is inserted
    between Device and Wire to represent the host OS Bluetooth stack (controller
    → kernel HCI driver → BlueZ/WinRT/CoreBluetooth → bleak) — the part of the
    path that produces the frames the application first sees, but which is
    otherwise invisible. Its input rate is the device's produced rate and its
    output rate is the rate actually received, so a shortfall (host-stack
    coalescing/dropping notifications under load) is visible as a rate shrink.

    Its drop counter is the confirmed ``sensor_data`` sequence-gap loss
    (*transport_loss*) **net of the device's own producer drops**
    (*producer_drops*) **and the GUI inbox's consumer-side overflow drops**
    (*recorder_overflow*): ``max(0, transport_loss - producer_drops -
    recorder_overflow)``. The device advances the sequence number even for
    frames it drops internally, and a frame the GUI inbox dropped to stay
    responsive was already delivered by the host Bluetooth stack — so each such
    frame also appears to the host as a missing sequence number and would
    otherwise be counted twice — once on its own stage (Device producer drop or
    RECORDER_OVERFLOW) and again on Host BT (sequence gap). Subtracting both
    leaves only the loss that the link itself introduced (over-the-air plus
    host-stack loss), so the Host BT stage no longer mirrors the Device drop
    count or absorbs consumer-side drops. For non-BLE
    transports the stage is omitted, because there is no host Bluetooth layer in
    the path. The host cannot be queried for richer per-driver health portably,
    so these already-measured rates and the inferred loss are the cross-platform
    signal.
    """
    rec_rate = message_rate if recording else 0.0
    # Process attribution for the grouped flow view: the Device is the
    # external sensor board; the optional Host BT stage is the host OS Bluetooth
    # stack; the Wire + Decode stages run in the acquisition child process that
    # pulls and decodes frames; the Gate, Plot Buffer and Recorder live in the GUI
    # process that renders and records them.
    host_bt_stages: tuple[StageMetrics, ...] = ()
    if transport_kind == "ble":
        # Confirmed sequence-gap loss net of the device's own producer drops AND
        # the GUI inbox's consumer-side overflow drops: a frame the device
        # dropped internally still advances the sequence number, and a frame the
        # GUI inbox dropped to stay responsive was already delivered by the host
        # Bluetooth stack — both surface as host-side sequence gaps. Counting
        # either here as well as on its own stage (Device producer drop /
        # RECORDER_OVERFLOW) would double-report it and wrongly blame Host BT, so
        # both are subtracted.
        host_bt_loss = transport_loss - producer_drops - recorder_overflow
        if host_bt_loss < 0:
            host_bt_loss = 0
        host_bt_stages = (
            StageMetrics(
                name="Host BT",
                in_rate=device_rate,
                out_rate=message_rate,
                drops_total=host_bt_loss,
                process="Host",
            ),
        )
    stages = (
        StageMetrics(
            name="Device",
            in_rate=device_rate,
            out_rate=device_rate,
            drops_total=producer_drops,
            process="Device",
        ),
        *host_bt_stages,
        StageMetrics(
            name="Wire", in_rate=message_rate, out_rate=message_rate,
            process="Acquisition",
        ),
        StageMetrics(
            name="Decode", in_rate=message_rate, out_rate=message_rate,
            process="Acquisition",
        ),
        StageMetrics(
            name="Gate",
            in_rate=message_rate,
            out_rate=message_rate,
            queue_capacity=gate_capacity,
            drops_total=dropped_frames,
            process="GUI",
        ),
        StageMetrics(
            name="Plot Buffer",
            in_rate=message_rate,
            out_rate=message_rate,
            queue_depth=model_queue_depth,
            process="GUI",
        ),
        StageMetrics(
            name="Recorder",
            in_rate=rec_rate,
            out_rate=rec_rate,
            queue_depth=recorder_queue_depth if recording else 0,
            process="GUI",
        ),
    )
    return PipelineSnapshot(stages=stages, wall_time_s=wall_time_s)


def detect_bottleneck(snapshot: PipelineSnapshot) -> StageMetrics | None:
    """Return the limiting stage in *snapshot*, or ``None`` if all are healthy.

    A stage is a bottleneck candidate if it is actively dropping items or its
    queue utilisation is at or above :data:`BOTTLENECK_UTILIZATION_THRESHOLD`.
    Among candidates, the one with the highest utilisation wins; ties break
    toward the stage with more accumulated drops, then earliest in pipeline
    order (the upstream-most constraint). With no candidate, ``None`` is
    returned. Deterministic and side-effect free.
    """
    best: StageMetrics | None = None
    best_key: tuple[float, int] | None = None
    for st in snapshot.stages:
        if not (st.is_dropping or st.utilization >= BOTTLENECK_UTILIZATION_THRESHOLD):
            continue
        key = (st.utilization, st.drops_total)
        if best_key is None or key > best_key:
            best = st
            best_key = key
    return best
