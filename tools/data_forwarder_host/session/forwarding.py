# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Forwarding session — the 5-state machine + I/O worker.

``ForwardingSession`` is the **sole writer** of :class:`SessionState`. It owns
an :class:`IoWorker` that runs on a ``QThread``, pulls bytes from the data
source, feeds the single COBS/CBOR v1 decoder, and emits decoded messages.

State machine (5 states):

* ``CONFIGURED`` → ``RUNNING``   — opening the source is an entry action of RUNNING
* ``RUNNING``    → ``STOPPED``   — stopping the source is an entry action of STOPPED
* any state      → ``ERROR``
* ``ERROR``      → ``CONFIGURED`` — via :meth:`reset`
"""

from __future__ import annotations

import logging
import threading
from time import monotonic
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

from data_forwarder_host.core.error_log import ErrorCategory, ErrorLog
from data_forwarder_host.pipeline.inbox import DEFAULT_BUDGET, FrameInbox
from data_forwarder_host.pipeline.ipc import ChildSpec
from data_forwarder_host.pipeline.process_host import SourceProcessHost
from data_forwarder_host.protocol.base import (
    DecodedMessage,
    DecodeError,
    DecodeErrorKind,
    DecodeStats,
)
from data_forwarder_host.protocol.cobs_cbor_v1 import CobsCborV1
from data_forwarder_host.session.config import SessionConfig
from data_forwarder_host.session.process_worker import ProcessIoWorker
from data_forwarder_host.session.states import SessionState, can_transition
from data_forwarder_host.source.base import Source
from data_forwarder_host.utils.slow_span import slow_span

log = logging.getLogger(__name__)


_DECODE_ERR_TO_CATEGORY = {
    DecodeErrorKind.COBS: ErrorCategory.COBS,
    DecodeErrorKind.CRC: ErrorCategory.CRC,
    DecodeErrorKind.MALFORMED: ErrorCategory.MALFORMED,
    DecodeErrorKind.CBOR: ErrorCategory.CBOR,
}


class InFlightGate:
    """Bounded in-flight counter for host-side ingestion backpressure.

    The decoded ``sensor_data`` frames cross from the I/O worker thread to the
    GUI thread via a queued Qt signal. If the producer streams faster than the
    GUI thread can process, that queue grows without bound and the whole
    application freezes (an observed failure mode). This gate caps the number
    of frames *in flight* — emitted by the worker but not yet processed by the
    GUI thread. The worker calls :meth:`try_acquire` before emitting a frame;
    once the bound is reached it returns ``False`` and the frame is dropped
    instead of blocking. The GUI thread calls :meth:`release` once it has
    finished processing a frame. The class is pure (no Qt) and thread-safe.
    """

    def __init__(self, max_in_flight: int) -> None:
        self._max = max(1, int(max_in_flight))
        self._in_flight = 0
        self._dropped = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        """Reserve a slot for one frame; ``False`` (and count a drop) if full."""
        with self._lock:
            if self._in_flight >= self._max:
                self._dropped += 1
                return False
            self._in_flight += 1
            return True

    def release(self) -> None:
        """Mark one in-flight frame as processed, freeing its slot."""
        with self._lock:
            if self._in_flight > 0:
                self._in_flight -= 1

    @property
    def max_in_flight(self) -> int:
        return self._max

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def dropped(self) -> int:
        """Total frames dropped since this gate was created."""
        with self._lock:
            return self._dropped


class IoWorker(QObject):
    """Pulls bytes from the source, feeds the decoder, emits messages."""

    message = Signal(object)            # DecodedMessage
    decode_errors = Signal(list)        # list[DecodeError]
    stats = Signal(object)              # DecodeStats
    raw_bytes = Signal(bytes)           # raw chunk as received from the source
    overflow = Signal()                 # host ingestion overflow (frames dropped)
    failed = Signal(str)
    stopped = Signal()

    def __init__(
        self,
        source: Source,
        decoder: CobsCborV1,
        gate: InFlightGate | None = None,
    ) -> None:
        super().__init__()
        self._source = source
        self._decoder = decoder
        self._gate = gate
        self._running = True
        # Rising-edge latch so a sustained overflow surfaces a single event
        # rather than one per dropped frame.
        self._overflow_active = False

    @Slot()
    def run(self) -> None:
        # Opening the source is an entry action of the RUNNING state. A source
        # already opened in the New Session dialog (connect-in-dialog hand-off)
        # is adopted as-is: open() is idempotent for both
        # sources, but we skip it explicitly so a hand-off is unambiguous.
        try:
            if not self._source.is_open:
                self._source.open()
        except Exception as exc:
            self.failed.emit(str(exc))
            self.stopped.emit()   # MUST emit so _on_worker_stopped cleans up the thread
            return

        try:
            tick = 0
            for chunk in self._source.chunks():
                if not self._running:
                    break
                if not chunk:
                    continue
                self.raw_bytes.emit(bytes(chunk))
                for msg in self._decoder.feed(chunk):
                    self._emit_message(msg)
                errs = list(self._decoder.errors_drained())
                if errs:
                    self.decode_errors.emit(errs)
                tick += 1
                if (tick & 0x1F) == 0:
                    self.stats.emit(self._decoder.stats())
        except Exception as exc:
            log.exception("IoWorker crashed")
            self.failed.emit(str(exc))
        finally:
            try:
                self._source.close()
            except Exception:
                log.exception("source.close() failed in IoWorker")
            self.stopped.emit()

    def _emit_message(self, msg: DecodedMessage) -> None:
        """Emit a decoded message, applying host-side backpressure.

        High-rate ``sensor_data`` frames are gated through :class:`InFlightGate`
        so a flooding producer cannot grow the GUI thread's event queue without
        bound. When the bound is reached the frame is dropped and a single
        ``overflow`` event is surfaced on the rising edge. Control messages
        (e.g. ``session_info``) are never dropped — they are rare and carry
        metadata the rest of the pipeline depends on.
        """
        if self._gate is not None and msg.kind == "sensor_data":
            if not self._gate.try_acquire():
                if not self._overflow_active:
                    self._overflow_active = True
                    self.overflow.emit()
                return
            self._overflow_active = False
        self.message.emit(msg)

    @Slot()
    def request_stop(self) -> None:
        self._running = False
        # Also close the source here to unblock reads when no bytes are flowing.
        # Without this, generators that only yield on data can remain blocked
        # indefinitely and the UI appears stuck in RUNNING.
        try:
            self._source.close()
        except Exception:
            log.exception("source.close() failed in request_stop")


class ForwardingSession(QObject):
    """State-machine wrapper around a (source, decoder, I/O thread).

    Sole writer of :class:`SessionState` via :meth:`_set_state`.
    """

    state_changed = Signal(object)              # SessionState
    message_received = Signal(object)           # DecodedMessage
    session_info = Signal(object)               # DecodedMessage
    error_occurred = Signal(str)
    stats_updated = Signal(object)              # DecodeStats
    raw_bytes = Signal(bytes)                   # raw chunk as received
    raw_byte_count = Signal(int)                # raw byte count (process path)

    # Max decoded sensor_data frames allowed in flight from the I/O worker to
    # the GUI thread before excess frames are dropped (host-side backpressure).
    MAX_IN_FLIGHT = 2048

    # GUI-paced pull cadence for the out-of-process path: the GUI
    # thread drains at most ``PULL_BUDGET`` sensor frames from the inbox every
    # ``PULL_INTERVAL_MS`` so its per-tick work — and therefore CPU — is bounded
    # regardless of how fast the child produces. The event loop services paint
    # and input events between ticks, so a flood can never hang the UI.
    PULL_INTERVAL_MS = 20
    PULL_BUDGET = DEFAULT_BUDGET

    # Hard wall-clock cap (ms) on how long a single drain tick may run on the
    # GUI thread. Even if per-frame processing is unexpectedly slow, the tick
    # stops at this deadline and hands control back to the event loop, so the
    # UI can never freeze. Frames are drained in small chunks so the deadline
    # is checked frequently.
    MAX_DRAIN_MS = 8.0
    _DRAIN_CHUNK = 64

    def __init__(
        self,
        *,
        config: SessionConfig,
        source: Source | None,
        decoder: CobsCborV1,
        error_log: ErrorLog,
        parent: QObject | None = None,
        use_process: bool = False,
        process_host_factory: Callable[[ChildSpec], Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._source = source
        self._decoder = decoder
        self._error_log = error_log
        self._state: SessionState = SessionState.CONFIGURED
        self._thread: QThread | None = None
        self._worker: QObject | None = None
        self._gate: InFlightGate | None = None
        self._use_process = use_process
        self._process_host_factory = process_host_factory
        # Consumer-paced pull boundary (process path only).
        self._inbox: FrameInbox | None = None
        self._pull_timer: QTimer | None = None
        self._inbox_drops_reported = 0
        # Diagnostics: largest gap (ms) observed between drain ticks since the
        # last sample. The timer is meant to fire every ``PULL_INTERVAL_MS``; a
        # large gap proves the GUI event loop was starved (the freeze symptom).
        self._last_drain_ts: float | None = None
        self._drain_max_gap_ms = 0.0

        self._error_log.bind_state_provider(self.state)

    # ------------------------------------------------------------------
    # State (sole writer)
    # ------------------------------------------------------------------

    def state(self) -> SessionState:
        return self._state

    def _set_state(self, new_state: SessionState) -> None:
        if not can_transition(self._state, new_state):
            log.warning("rejected transition %s -> %s", self._state, new_state)
            return
        if new_state == self._state:
            return
        self._state = new_state
        self.state_changed.emit(new_state)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._state not in (SessionState.CONFIGURED, SessionState.STOPPED):
            return
        if self._use_process:
            self._start_process()
            return
        self._decoder.reset()
        self._gate = InFlightGate(self.MAX_IN_FLIGHT)
        self._worker = IoWorker(self._source, self._decoder, gate=self._gate)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.message.connect(self._on_message)
        self._worker.decode_errors.connect(self._on_decode_errors)
        self._worker.stats.connect(self.stats_updated)
        self._worker.raw_bytes.connect(self.raw_bytes)
        self._worker.overflow.connect(self._on_ingestion_overflow)
        self._worker.failed.connect(self._on_failed)
        self._worker.stopped.connect(self._on_worker_stopped)
        self._thread.start()
        # Opening the source is the entry action of RUNNING.
        self._set_state(SessionState.RUNNING)

    def _start_process(self) -> None:
        """Start acquisition in a separate OS process.

        The child rebuilds the source/decoder from the serializable config and
        ships decoded frames in batches, so a flood cannot grow the GUI thread's
        event queue (the in-flight gate is unnecessary here — back-pressure lives
        in the child's drop policy).
        """
        spec = ChildSpec(
            source_kind=self._config.source.kind,
            source_params=dict(self._config.source.params),
            expect_crc=self._config.expect_crc,
        )
        factory = self._process_host_factory or SourceProcessHost
        host = factory(spec)
        # The worker thread appends decoded frames straight into this inbox
        # (no per-batch Qt signal), and the GUI thread pulls from it on a timer.
        # This keeps the GUI event loop free of high-frequency
        # cross-thread traffic, so a flood cannot freeze the UI.
        self._inbox = FrameInbox()
        self._inbox_drops_reported = 0
        worker = ProcessIoWorker(host, inbox=self._inbox)
        self._worker = worker
        self._thread = QThread()
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        # NOTE: worker.messages is intentionally NOT connected — frames flow via
        # the inbox. Only low-frequency control/metadata signals use Qt.
        worker.decode_errors.connect(self._on_decode_errors)
        worker.stats.connect(self.stats_updated)
        worker.raw_byte_count.connect(self.raw_byte_count)
        worker.overflow.connect(self._on_ingestion_overflow)
        worker.failed.connect(self._on_failed)
        worker.stopped.connect(self._on_worker_stopped)
        # GUI-thread pull timer (owned by this session, runs on the GUI thread).
        self._pull_timer = QTimer(self)
        self._pull_timer.setInterval(self.PULL_INTERVAL_MS)
        self._pull_timer.timeout.connect(self._drain_inbox)
        self._pull_timer.start()
        self._thread.start()
        self._set_state(SessionState.RUNNING)

    def stop(self) -> None:
        if self._state != SessionState.RUNNING:
            return
        if self._worker is not None:
            # Stop request must run on the worker thread to release the read loop.
            self._worker.request_stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
        # Drain whatever the inbox still holds, then stop the pull timer so no
        # frames are stranded and the timer does not outlive the session.
        self._drain_inbox()
        self._stop_pull_timer()
        # Reflect user intent in the UI immediately even if worker teardown
        # completes slightly later.
        if self._state == SessionState.RUNNING:
            self._set_state(SessionState.STOPPED)

    def reset(self) -> None:
        """Allowed only from ERROR — return to CONFIGURED."""
        if self._state != SessionState.ERROR:
            return
        self._set_state(SessionState.CONFIGURED)

    def fail(self, message: str) -> None:
        """Force the session into ERROR and stop I/O.

        Used for controller-side fatal validation failures (for example,
        session metadata changing mid-session).
        """
        self._error_log.add(ErrorCategory.TRANSPORT, message)
        if self._worker is not None:
            try:
                self._worker.request_stop()
            except Exception:
                log.exception("failed to stop worker during fail()")
        self._set_state(SessionState.ERROR)
        self.error_occurred.emit(message)

    def shutdown(self) -> None:
        try:
            self.stop()
        finally:
            try:
                if self._source is not None:
                    self._source.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Worker signals
    # ------------------------------------------------------------------

    @Slot(object)
    def _on_message(self, msg: DecodedMessage) -> None:
        self._error_log.note_message()
        self.message_received.emit(msg)
        if msg.kind == "session_info":
            self.session_info.emit(msg)
        # Processing of a gated sensor_data frame is complete (all slots ran
        # synchronously on this thread); free its in-flight slot so the worker
        # can admit the next frame.
        if self._gate is not None and msg.kind == "sensor_data":
            self._gate.release()

    @Slot(list)
    def _on_messages(self, batch: list[DecodedMessage]) -> None:
        """Buffer a batch from the out-of-process worker.

        This runs on the GUI thread (queued from the worker) but does **no**
        per-frame pipeline work: it only appends the batch to the bounded
        :class:`FrameInbox`. The actual per-frame fan-out happens later, paced,
        in :meth:`_drain_inbox`. Appending is O(batch) and cheap, so even a
        flood of batches cannot starve the event loop here. If the inbox has
        not been created (defensive), fall back to direct emission.
        """
        if self._inbox is not None:
            self._inbox.append(batch)
            return
        for msg in batch:
            self._error_log.note_message()
            self.message_received.emit(msg)
            if msg.kind == "session_info":
                self.session_info.emit(msg)

    @Slot()
    def _drain_inbox(self) -> None:
        """GUI-paced pull: process frames under a strict per-tick time budget.

        The GUI thread is the *client* that pulls when it is ready; the inbox is
        the *server* that buffered the frames and counted any it had to drop.
        The drain is bounded **by wall-clock time**
        (``MAX_DRAIN_MS``) as well as by ``PULL_BUDGET`` frames, so no matter how
        expensive per-frame processing is, a single tick can never occupy the
        event loop long enough to freeze the UI — it always returns control so
        paint/input events run. Control frames are always returned (never
        dropped); surplus sensor frames are dropped at the inbox and reported.
        """
        inbox = self._inbox
        if inbox is None:
            return
        now = monotonic()
        if self._last_drain_ts is not None:
            gap_ms = (now - self._last_drain_ts) * 1000.0
            if gap_ms > self._drain_max_gap_ms:
                self._drain_max_gap_ms = gap_ms
        self._last_drain_ts = now
        dropped_total = inbox.dropped
        if dropped_total > self._inbox_drops_reported:
            delta = dropped_total - self._inbox_drops_reported
            self._inbox_drops_reported = dropped_total
            # Coalesced into a single event whose count reflects the number of
            # frames dropped (not the number of overflow episodes), so the
            # RECORDER_OVERFLOW total is frame-accurate and can be netted out of
            # the Host BT transport-loss attribution: a frame dropped at the GUI
            # inbox was already delivered by the host Bluetooth stack, so its
            # sequence gap must not be blamed on Host BT.
            self._error_log.add_bulk(
                ErrorCategory.RECORDER_OVERFLOW,
                delta,
                f"GUI inbox dropped {delta} sensor frame(s) to stay responsive",
            )
        deadline = monotonic() + (self.MAX_DRAIN_MS / 1000.0)
        processed = 0
        with slow_span("drain_inbox") as span:
            while processed < self.PULL_BUDGET:
                want = min(self._DRAIN_CHUNK, self.PULL_BUDGET - processed)
                chunk = inbox.drain(want)
                if not chunk:
                    break
                with slow_span("drain_inbox.emit_loop", extra=f"n={len(chunk)}"):
                    for msg in chunk:
                        self._error_log.note_message()
                        self.message_received.emit(msg)
                        if msg.kind == "session_info":
                            self.session_info.emit(msg)
                processed += len(chunk)
                if monotonic() >= deadline:
                    break
            span.note(f"processed={processed}")

    @Slot()
    def _on_ingestion_overflow(self) -> None:
        # Host could not keep up; frames are being dropped to keep the UI
        # responsive. Surfaced as a single RECORDER_OVERFLOW event per episode.
        self._error_log.add(
            ErrorCategory.RECORDER_OVERFLOW,
            "host ingestion overflow; dropping frames to keep the UI responsive",
        )

    @Slot(list)
    def _on_decode_errors(self, errs: list[DecodeError]) -> None:
        for e in errs:
            cat = _DECODE_ERR_TO_CATEGORY.get(e.kind, ErrorCategory.MALFORMED)
            self._error_log.add(cat, e.detail, **e.context)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._error_log.add(ErrorCategory.TRANSPORT, message)
        self._set_state(SessionState.ERROR)
        self.error_occurred.emit(message)

    @Slot()
    def _on_worker_stopped(self) -> None:
        # Stopping the source is the entry action of STOPPED.
        # Drain any frames still buffered before tearing the timer down.
        self._drain_inbox()
        self._stop_pull_timer()
        if self._state == SessionState.RUNNING:
            self._set_state(SessionState.STOPPED)
        self._worker = None
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread = None

    def _stop_pull_timer(self) -> None:
        """Stop and discard the GUI-paced pull timer if running."""
        if self._pull_timer is not None:
            self._pull_timer.stop()
            self._pull_timer = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def config(self) -> SessionConfig:
        return self._config

    def dropped_frames(self) -> int:
        """Total sensor_data frames dropped by host-side backpressure.

        Combines the in-process in-flight gate (legacy single-process path) and
        the GUI inbox (process path).
        """
        gate_drops = self._gate.dropped if self._gate is not None else 0
        inbox_drops = self._inbox.dropped if self._inbox is not None else 0
        return gate_drops + inbox_drops

    def inbox_pending(self) -> int:
        """Frames currently buffered in the GUI pull inbox (process path)."""
        return self._inbox.pending() if self._inbox is not None else 0

    def drain_gap_ms(self, *, reset: bool = True) -> float:
        """Largest gap (ms) between drain ticks since the last call.

        A value far above ``PULL_INTERVAL_MS`` means the GUI event loop was
        starved between ticks — the fingerprint of a freeze. Reading it
        optionally resets the high-water mark so each sample is fresh.
        """
        gap = self._drain_max_gap_ms
        if reset:
            self._drain_max_gap_ms = 0.0
        return gap

    def stats(self) -> DecodeStats:
        return self._decoder.stats()

    def describe_source(self) -> str:
        if self._config.source.kind == "uart":
            p = self._config.source.params
            return f"uart:{p.get('port', '?')}@{p.get('baudrate', 115200)}"
        if self._config.source.kind == "ble":
            return f"ble:{self._config.source.params.get('address', '?')}"
        return self._config.source.kind

    def describe_protocol(self) -> str:
        crc = self._config.expect_crc
        return f"cobs-cbor v1 (expect_crc={'true' if crc else 'false'})"

    def describe_session_info(self) -> dict[str, Any]:
        return {
            "tag": self._config.tag,
            "source": self.describe_source(),
            "protocol": self.describe_protocol(),
        }
