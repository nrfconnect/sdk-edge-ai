# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""SessionController — wires the session state machine, data model, recorder,
error log and CSV output for a single forwarding session.

There is no sink subsystem and no backend/frontend separation: the
:class:`~data_forwarder_host.session.forwarding.ForwardingSession` owns the I/O
worker directly and is the sole writer of ``SessionState``.
"""

from __future__ import annotations

import gc
import logging
import uuid
from collections import deque
from dataclasses import replace as dc_replace
from pathlib import Path
from time import monotonic

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from data_forwarder_host.core.csv_writer import RecordingCsvDump, write_recording_csv
from data_forwarder_host.core.bandwidth import BandwidthMeter, BandwidthSample
from data_forwarder_host.core.data_model import DataModel
from data_forwarder_host.core.error_log import ErrorCategory, ErrorLog
from data_forwarder_host.core.metadata import build_metadata_text
from data_forwarder_host.core.recorder import CsvDumpJob, Recorder, Recording
from data_forwarder_host.core.sequence_tracker import SequenceGapTracker
from data_forwarder_host.protocol.base import DecodedMessage
from data_forwarder_host.protocol.cobs_cbor_v1 import CobsCborV1
from data_forwarder_host.session.config import SessionConfig
from data_forwarder_host.session.forwarding import ForwardingSession
from data_forwarder_host.session.states import SessionState
from data_forwarder_host.utils.debug_stream import DebugStream, debug_stream_enabled
from data_forwarder_host.source import source_for_kind
from data_forwarder_host.utils.paths import default_recordings_dir
from data_forwarder_host.utils.slow_span import slow_span
log = logging.getLogger(__name__)

# A confirmed-loss batch with at most this many sequence numbers is journalled
# one entry per number (useful diagnostic detail); larger batches collapse into
# a single coalesced summary event so a transport flood cannot stall the GUI by
# emitting thousands of journal entries at once.
_TRANSPORT_LOSS_DETAIL_LIMIT = 64

# A confirmed sequence-gap is recorded as a TRANSPORT loss immediately (so the
# event journal stays truthful), but it is held out of the *Host BT* pipeline
# attribution for this long so the producer-side drop count ("dr", which arrives
# asynchronously in a later session_info) has time to catch up. Without the hold
# a device-caused drop briefly shows under Host BT and then jumps to the Device
# stage once "dr" updates; the hold lets the host-vs-device split settle first.
_LOSS_RECONCILE_WINDOW_S = 2.0


class SessionController(QObject):
    """Façade for one forwarding session, exposing Qt signals for the GUI."""

    # --- Public Qt signals (GUI widgets subscribe to these) ---
    state_changed = Signal(object)              # SessionState
    message_received = Signal(object)           # DecodedMessage
    error_occurred = Signal(str)
    recording_changed = Signal(bool)            # is_recording
    recording_saved = Signal(str)               # path to written CSV
    recording_save_started = Signal()           # CSV dump began (lock buttons)
    recording_save_progress = Signal(int, int)  # rows_written, total (-1 if unknown)
    recording_save_failed = Signal(str)         # CSV dump error message
    recording_empty = Signal()                  # recording had zero sensor rows
    session_info_received = Signal(object)      # raw session-info envelope
    session_info_mismatch = Signal(str, bool)   # detail, has_buffered_data
    session_phase_changed = Signal(str)         # created/awaiting/ready/recording/...
    stats_updated = Signal(object)              # DecodeStats
    raw_bytes_received = Signal(bytes)          # raw chunk as received
    bandwidth_window_changed = Signal(float)    # effective bandwidth window (s)
    session_sampling_changed = Signal(float, int)  # hz, channel count (session_info)

    def __init__(
        self,
        config: SessionConfig,
        parent: QObject | None = None,
        prepared_source: object | None = None,
        use_process: bool = False,
    ) -> None:
        super().__init__(parent)
        config.validate()
        self._id = uuid.uuid4().hex[:8]
        self._config = config

        # ErrorLog first — all other components bind to it.
        self.error_log = ErrorLog(self)

        # DataModel: rolling GUI-side buffers for live plots (GUI thread only).
        self.data_model = DataModel(
            plot_window_seconds=config.plot_window_seconds,
            error_log=self.error_log,
            parent=self,
        )

        # Recorder: RAM-first + spill-to-disk capture (gated on recording).
        self.recorder = Recorder(
            session_id=self._id,
            session_tag=config.tag,
            error_log=self.error_log,
            parent=self,
        )
        self.recorder.started.connect(lambda: self.recording_changed.emit(True))
        self.recorder.stopped.connect(lambda _r: self.recording_changed.emit(False))
        # Cap idle live-buffer retention while not recording.
        self.recorder.started.connect(lambda: self.data_model.set_recording(True))
        self.recorder.stopped.connect(
            lambda _r: self.data_model.set_recording(False)
        )

        # Source + decoder + forwarding session (single fixed protocol).
        # A *prepared_source* is a live source already opened in the New Session
        # dialog (connection / port reservation happens there, not at start);
        # the freshly created session adopts it. Reopened/saved sessions get a
        # fresh source rebuilt from the serializable config and (re)connect at
        # start.
        #
        # In out-of-process mode acquisition runs in a
        # separate OS process that rebuilds the source from the config, so no
        # GUI-side source is built. Any prepared_source must be released first
        # so the child can reopen the same port/device.
        if use_process:
            if prepared_source is not None:
                try:
                    prepared_source.close()
                except Exception:
                    log.exception("failed to release prepared source for process mode")
            source = None
        elif prepared_source is not None:
            source = prepared_source
        else:
            source_cls = source_for_kind(config.source.kind)
            source = source_cls(**config.source.params)
        decoder = CobsCborV1(expect_crc=config.expect_crc)
        self._session = ForwardingSession(
            config=config,
            source=source,
            decoder=decoder,
            error_log=self.error_log,
            parent=self,
            use_process=use_process,
        )
        # IoWorker runs in a QThread; these connections are delivered to the
        # GUI thread via QueuedConnection automatically.
        self._session.state_changed.connect(self._on_state_changed)
        self._session.message_received.connect(self._process_message)
        self._session.error_occurred.connect(self.error_occurred)
        self._session.stats_updated.connect(self.stats_updated)
        self._session.raw_bytes.connect(self.raw_bytes_received)
        # Out-of-process path reports raw bytes as a count, not the payload.
        self._session.raw_byte_count.connect(self._on_raw_byte_count_for_bandwidth)

        # Live throughput meter over a trailing window. Fed with raw
        # received byte counts and decoded sensor_data message events.
        self.bandwidth = BandwidthMeter(config.bandwidth_window_seconds)
        self.raw_bytes_received.connect(self._on_raw_bytes_for_bandwidth)

        # Transport message-loss detector: watches sensor_data
        # sequence numbers and confirms a gap as a TRANSPORT loss only if the
        # missing number has not arrived within the loss confirmation window.
        self._seq_tracker = SequenceGapTracker(
            window_seconds=config.loss_confirmation_window_seconds,
            on_loss=self._on_transport_loss,
        )
        # Sweep on a timer so pending losses are confirmed even when the stream
        # pauses; cadence is a fraction of the window (clamped to a sane range).
        self._seq_sweep_timer = QTimer(self)
        self._seq_sweep_timer.timeout.connect(self._seq_sweep_tick)
        self._apply_seq_sweep_interval()
        self._seq_sweep_timer.start()
        # Aggregate transport losses already surfaced (see _flush_untracked_losses).
        self._untracked_loss_reported: int = 0
        # Reconcile hold for Host-BT attribution: every confirmed TRANSPORT loss
        # is parked here with the time it was confirmed and the device-drop
        # observation epoch at that moment. A parked loss is only "released" into
        # the Host-BT-attributed total once a newer device-drop ("dr") reading
        # has arrived (the producer-drop information covering that gap's period
        # is now in hand), so a device-caused drop is netted onto the Device
        # stage without ever flashing under Host BT. When the device never
        # reports "dr" there is nothing to reconcile against, so the reconcile
        # window acts as a time-based fallback instead. ``_loss_clock`` is
        # injectable so the fallback can be driven deterministically.
        self._loss_clock = monotonic
        self._transport_loss_pending: deque[tuple[float, int, int]] = deque()
        self._transport_loss_released: int = 0
        # Monotonic count of valid device-drop readings observed via session_info;
        # gates the reconcile release above. Never reset, so it cannot run
        # backwards and strand a parked loss.
        self._dr_observation_epoch: int = 0

        # Recording-stop CSV dump runs incrementally off the event loop so a
        # large save never freezes the GUI. While a dump is in flight
        # the Record/Stop buttons stay locked.
        self._csv_dump_job: CsvDumpJob | None = None
        self._saving_csv: bool = False
        # Tracks whether automatic cyclic GC was paused for the current
        # recording so it is resumed exactly once, on whichever stop path runs.
        self._gc_suspended_for_recording: bool = False

        self._session_info_baseline: dict | None = None
        self._expected_channel_count: int | None = None
        # Latest sampling rate (Hz) and channel count from session_info. Updated
        # on every valid session_info so it tracks a varying producer frequency.
        self._latest_sampling: tuple[float, int] | None = None
        self._metadata_ready: bool = False
        self._record_on_metadata_pending: bool = False
        self._pending_mismatch_recording: Recording | None = None
        # Baseline of the producer-side drop count ("dr"); captured from the
        # first valid session_info so its real-time delta can be surfaced as a
        # PRODUCER_DROP statistic without treating it as a session mismatch.
        # ``_dr_baseline`` anchors the cumulative-since-startup figure
        # (PRODUCER_DROP_TOTAL); ``_dr_reset_baseline`` anchors the since-reset
        # figure (PRODUCER_DROP) and is re-anchored at each reset moment.
        self._dr_baseline: int | None = None
        self._dr_latest: int | None = None
        self._dr_reset_baseline: int | None = None
        self._phase: str = "created"
        self.session_phase_changed.emit(self._phase)

        # Opt-in, rate-limited diagnostics stream (off unless DFH_DEBUG_STREAM is
        # set). Surfaces the metrics that drive long-run slowdowns so the root
        # cause can be confirmed from logs without a heavy on-screen panel.
        self._debug_stream: DebugStream | None = None
        if debug_stream_enabled():
            self._debug_stream = DebugStream(self._debug_metrics, parent=self)
            self._debug_stream.start()

    def _debug_metrics(self) -> dict:
        """Snapshot the runtime metrics that drive long-run slowdowns."""
        gc0, gc1, gc2 = gc.get_count()
        return {
            "tag": self._config.tag,
            "rec": int(self.recorder.is_recording()),
            "spill": int(self.recorder.using_spill),
            "ram_rows": self.recorder.buffered_rows,
            "ram_mb": round(self.recorder.buffered_bytes / 1e6, 1),
            "buf_max": self.data_model.max_buffer_size,
            "msg_s": round(
                self.bandwidth.sample(
                    self.data_model.channel_count
                ).messages_per_second,
                1,
            ),
            "gc0": gc0,
            "gc1": gc1,
            "gc2": gc2,
            "dropped": self._session.dropped_frames(),
            "inbox": self._session.inbox_pending(),
            "tick_gap_ms": round(self._session.drain_gap_ms(), 1),
        }

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        return self._id

    @property
    def config(self) -> SessionConfig:
        return self._config

    def current_config(self) -> SessionConfig:
        """Return a config snapshot reflecting the current plot/bandwidth windows."""
        return dc_replace(
            self._config,
            plot_window_seconds=self.data_model.plot_window_seconds,
            bandwidth_window_seconds=self.bandwidth.window_seconds,
        )

    def set_label(self, label: str) -> None:
        """Update the recording label (gates the Record action)."""
        self._config = dc_replace(self._config, label=label)

    def set_output_dir(self, output_dir: str) -> None:
        self._config = dc_replace(self._config, output_dir=output_dir)

    def set_bandwidth_window_seconds(self, seconds: float) -> None:
        """Update the live bandwidth measurement window.

        The bandwidth window starts equal to the plot window at session
        creation and is adjustable from the session window or the bandwidth
        details sub-window; the change is broadcast via
        :attr:`bandwidth_window_changed` so both controls stay in sync.
        """
        seconds = float(seconds)
        if seconds == self.bandwidth.window_seconds:
            return
        self.bandwidth.set_window_seconds(seconds)
        self.bandwidth_window_changed.emit(self.bandwidth.window_seconds)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def state(self) -> SessionState:
        return self._session.state()

    def is_recording(self) -> bool:
        return self.recorder.is_recording()

    def is_saving(self) -> bool:
        """True while a recording-stop CSV dump is still being written."""
        return self._saving_csv

    def dropped_frames(self) -> int:
        """Frames dropped by the host in-flight gate under overload."""
        return self._session.dropped_frames()

    def can_record(self) -> bool:
        """Record action is enabled only when a valid label is defined (G4)."""
        return self._config.has_recording_label()

    def metadata_ready(self) -> bool:
        return self._metadata_ready

    def can_begin_recording(self) -> bool:
        return self.can_record() and self._metadata_ready

    def session_phase(self) -> str:
        return self._phase

    def describe_source(self) -> str:
        return self._session.describe_source()

    def describe_protocol(self) -> str:
        return self._session.describe_protocol()

    def bandwidth_sample(self) -> BandwidthSample:
        """Return the current throughput rates over the configured window."""
        return self.bandwidth.sample(self.data_model.channel_count)

    def latest_sampling_info(self) -> tuple[float, int] | None:
        """Latest ``(sampling_rate_hz, channel_count)`` from session_info, if any."""
        return self._latest_sampling

    def loss_confirmation_window_seconds(self) -> float:
        """Current transport loss confirmation window (seconds)."""
        return self._seq_tracker.window_seconds

    def set_loss_confirmation_window_seconds(self, seconds: float) -> None:
        """Change the transport loss confirmation window live."""
        if seconds <= 0:
            return
        self._config = dc_replace(
            self._config, loss_confirmation_window_seconds=float(seconds)
        )
        self._seq_tracker.set_window_seconds(float(seconds))
        self._apply_seq_sweep_interval()

    def _apply_seq_sweep_interval(self) -> None:
        """Sweep at a quarter of the window, clamped to [50, 1000] ms."""
        ms = int(self._seq_tracker.window_seconds * 1000 / 4)
        self._seq_sweep_timer.setInterval(max(50, min(1000, ms)))

    def _seq_sweep_tick(self) -> None:
        """Timer slot: confirm due losses, then flush any aggregate overflow.

        Under extreme sustained loss the detector caps how many missing numbers
        it tracks individually; the surplus is counted in aggregate. We surface
        that surplus here as coalesced TRANSPORT losses so the totals stay
        truthful without the per-frame cost or unbounded memory that tracking
        every number individually would impose.
        """
        self._seq_tracker.sweep()
        self._flush_untracked_losses()
        self._release_reconciled_losses()

    def _hold_transport_loss(self, count: int) -> None:
        """Park ``count`` confirmed losses for device-drop reconciliation.

        The losses are recorded as TRANSPORT events immediately elsewhere; this
        only defers when they become attributable to the Host BT stage. Each
        batch is tagged with the current device-drop observation epoch so it can
        be released once a newer ``dr`` reading has had a chance to claim it.
        """
        if count > 0:
            self._transport_loss_pending.append(
                (self._loss_clock(), count, self._dr_observation_epoch)
            )

    def _release_reconciled_losses(self) -> None:
        """Release parked losses the device-drop count has had a chance to claim.

        While the device reports ``dr`` a batch is released only once a newer
        ``dr`` observation has arrived (``epoch`` advanced past the batch's parked
        epoch), so a device-caused gap nets onto the Device stage instead of
        flashing under Host BT. When the device never reports ``dr`` there is
        nothing to reconcile against, so the fixed reconcile window is used as a
        fallback. Entries are FIFO and both the epoch and the timestamp are
        non-decreasing along the queue, so a stop at the head stops the sweep.
        """
        now = self._loss_clock()
        epoch = self._dr_observation_epoch
        reconciled_against_drops = self._dr_baseline is not None
        pending = self._transport_loss_pending
        while pending:
            confirmed_at, count, parked_epoch = pending[0]
            if reconciled_against_drops:
                if epoch <= parked_epoch:
                    break
            elif now - confirmed_at < _LOSS_RECONCILE_WINDOW_S:
                break
            self._transport_loss_released += count
            pending.popleft()

    def host_attributed_transport_loss(self) -> int:
        """Confirmed TRANSPORT losses the device-drop count has had a chance to claim.

        Recently confirmed losses are withheld from the Host BT pipeline figure
        until a newer producer-drop (``dr``) reading has arrived to claim any
        device-caused gaps (or, for sources without ``dr``, until the reconcile
        window elapses); the raw TRANSPORT event total is unaffected.
        """
        self._release_reconciled_losses()
        return self._transport_loss_released

    def producer_drop_count(self) -> int:
        """Live since-reset producer-drop count (the ``dr`` delta).

        Mirrors the ``PRODUCER_DROP`` statistic but without the error summary's
        1 Hz coalescing, so the bandwidth pipeline view nets Host-BT loss against
        the same up-to-date device-drop figure that gates the reconcile release —
        preventing a device drop from briefly showing under Host BT while the
        coalesced summary catches up.
        """
        if self._dr_latest is None or self._dr_baseline is None:
            return 0
        reset_base = (
            self._dr_reset_baseline
            if self._dr_reset_baseline is not None
            else self._dr_baseline
        )
        return self._dr_latest - reset_base

    def _flush_untracked_losses(self) -> None:
        total = self._seq_tracker.untracked_loss_count
        delta = total - self._untracked_loss_reported
        if delta <= 0:
            return
        self._untracked_loss_reported = total
        self.error_log.add_bulk(
            ErrorCategory.TRANSPORT,
            delta,
            f"{delta} sensor_data frame(s) confirmed lost in aggregate under "
            "heavy transport loss (exceeded the loss-tracker capacity)",
        )
        self._hold_transport_loss(delta)

    def _on_transport_loss(self, seqs: list[int]) -> None:
        """Record confirmed missing sequence numbers as TRANSPORT losses.

        Reporting is coalesced: a single confirmation can carry many sequence
        numbers (up to the loss tracker's bounded capacity), and emitting one
        journal event per number would itself stall the GUI. A small batch is
        logged number-by-number for diagnostic detail; a large batch collapses
        into one summary event while still advancing the loss count correctly.
        """
        n = len(seqs)
        if n == 0:
            return
        if n <= _TRANSPORT_LOSS_DETAIL_LIMIT:
            with slow_span("transport_loss.fanout", extra=f"n={n}"):
                for seq in seqs:
                    self.error_log.add(
                        ErrorCategory.TRANSPORT,
                        f"sensor_data sequence {seq} not received within the loss "
                        f"confirmation window ({self._seq_tracker.window_seconds:g} s)",
                        seq=seq,
                    )
            self._hold_transport_loss(n)
            return
        lo, hi = seqs[0], seqs[-1]
        self.error_log.add_bulk(
            ErrorCategory.TRANSPORT,
            n,
            f"{n} sensor_data frames not received within the loss confirmation "
            f"window ({self._seq_tracker.window_seconds:g} s); "
            f"sequence range {lo}..{hi}",
        )
        self._hold_transport_loss(n)

    # ------------------------------------------------------------------
    # User commands
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the device channel and start streaming/listening.

        This is invoked automatically when the session tab opens so the
        device channel is reserved and ``session_info`` is observed without
        waiting for the user to press Record. Recording (CSV capture) is a
        separate action started via :meth:`start_recording`.
        """
        self.data_model.clear()
        self._session_info_baseline = None
        self._expected_channel_count = None
        self._metadata_ready = False
        self._pending_mismatch_recording = None
        self._record_on_metadata_pending = False
        self._dr_baseline = None
        self._dr_latest = None
        self._dr_reset_baseline = None
        self._session.start()
        self._set_phase("awaiting device metadata")

    def stop(self) -> None:
        """Stop the I/O session (auto-stops an active recording first)."""
        self._set_phase("stopping")
        if self.is_recording():
            try:
                self.stop_recording()
            except Exception:
                log.exception("auto stop_recording on stop() failed")
        self._session.stop()
        self._record_on_metadata_pending = False

    def reset(self) -> None:
        """Return from ERROR to CONFIGURED."""
        self._session.reset()
        self._set_phase("created")

    @Slot(object)
    def _on_state_changed(self, state: SessionState) -> None:
        self.state_changed.emit(state)
        if state == SessionState.RUNNING:
            # A fresh stream: forget any stale sequence baseline so the first
            # message of the new run only sets the baseline (no spurious losses).
            self._seq_tracker.reset()
            if self.is_recording():
                self._set_phase("recording")
            elif self._metadata_ready:
                self._set_phase("ready to record")
            else:
                self._set_phase("awaiting device metadata")
        elif state == SessionState.STOPPED:
            self._set_phase("closed")
        elif state == SessionState.ERROR:
            self._set_phase("closed")
        elif state == SessionState.CONFIGURED:
            self._set_phase("created")
        # If the source halted on its own while recording, finalise.
        if state in (SessionState.STOPPED, SessionState.ERROR) and self.is_recording():
            log.warning("session halted (state=%s) while recording; auto-stopping", state)
            try:
                self.stop_recording()
            except Exception:
                log.exception("auto stop_recording after halt failed")

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        """Begin recording to RAM/disk. Requires a valid label."""
        if not self.can_record():
            raise RuntimeError("a recording label must be defined before recording")
        if not self._metadata_ready:
            raise RuntimeError("session metadata not received yet")
        # Clear stale error statistics so this recording's panel shows only
        # events that happen during the new capture.
        self.error_log.clear()
        # Recording start is a reset moment: the error log was just cleared, so
        # every category restarts from zero. The producer drop count is split in
        # two — PRODUCER_DROP_TOTAL preserves the cumulative-since-startup figure
        # while the since-reset PRODUCER_DROP restarts from zero at
        # this reset moment.
        self._mark_producer_drop_reset()
        self.recorder.start(channel_names=self.data_model.channel_names)
        # Pause automatic cyclic GC for the capture window so the growing
        # recorder heap is not repeatedly rescanned (see
        # _suspend_gc_for_recording).
        self._suspend_gc_for_recording()
        # Arm the chart marker so the next recorded sample fixes the start
        # moment on the live plots.
        self.data_model.begin_recording_marker()
        self._set_phase("recording")

    def stop_recording(self) -> Recording:
        """Finalise the current recording and write the CSV file.

        The CSV is written incrementally off the event loop: this
        returns as soon as capture has stopped, having *started* the dump; the
        ``recording_save_started`` / ``recording_save_progress`` /
        ``recording_saved`` / ``recording_save_failed`` signals report its
        lifecycle so the GUI can show a progress bar and keep the Record/Stop
        buttons locked until the save completes.
        """
        self._set_phase("stopping")
        rec = self.recorder.stop()
        # Capture has ended: re-enable automatic GC and reclaim any cycles
        # created during the window (idempotent across every stop path).
        self._resume_gc_after_recording()
        self.data_model.end_recording_marker()
        if not self._recording_has_rows(rec):
            self.recording_empty.emit()
            if self.state() == SessionState.RUNNING and self._metadata_ready:
                self._set_phase("ready to record")
            return rec
        self._start_csv_dump(rec)
        return rec

    def _suspend_gc_for_recording(self) -> None:
        """Pause automatic cyclic GC for the duration of a recording.

        The recorder accumulates a large, steadily growing heap of GC-tracked
        row tuples while capturing. With automatic generational collection left
        on, every generation-2 pass rescans that whole growing heap, producing
        latency spikes that get worse the longer the recording runs. Suspending
        automatic collection for the bounded recording window removes those
        rescans; it is re-enabled (with a manual collection to reclaim any
        cycles created in the window) when the recording ends. Idempotent, and
        only takes ownership when GC was actually enabled to begin with.

        This targets recording-time GC overhead specifically; it does not affect
        GUI-consumer or chart-render cost while no recording is in progress.
        """
        if self._gc_suspended_for_recording:
            return
        if gc.isenabled():
            gc.disable()
            self._gc_suspended_for_recording = True

    def _resume_gc_after_recording(self) -> None:
        """Re-enable automatic GC after a recording and reclaim any cycles.

        Safe to call on every recording-end path (normal stop, cancel, and the
        auto-stop on a session halt); a no-op when GC was not suspended, so GC
        is never left disabled once no recording is active.
        """
        if not self._gc_suspended_for_recording:
            return
        self._gc_suspended_for_recording = False
        gc.enable()
        gc.collect()

    def _start_csv_dump(self, rec: Recording) -> None:
        """Begin a non-blocking CSV dump of *rec*."""
        output_dir = self._config.output_dir or str(default_recordings_dir())
        try:
            dump = RecordingCsvDump(
                rec,
                label=self._config.label,
                output_dir=output_dir,
                metadata_text=self._build_metadata_text(rec),
            )
        except Exception as exc:
            log.exception("failed to start recording CSV dump")
            self.error_occurred.emit(f"failed to write CSV: {exc}")
            self._finish_save_phase()
            return
        job = CsvDumpJob(dump, parent=self)
        self._csv_dump_job = job
        self._saving_csv = True
        job.progress.connect(self.recording_save_progress)
        job.finished.connect(self._on_csv_dump_finished)
        job.failed.connect(self._on_csv_dump_failed)
        self.recording_save_started.emit()
        job.start()

    @Slot(str)
    def _on_csv_dump_finished(self, path: str) -> None:
        self._csv_dump_job = None
        self._saving_csv = False
        self.recording_saved.emit(path)
        self._finish_save_phase()

    @Slot(str)
    def _on_csv_dump_failed(self, message: str) -> None:
        self._csv_dump_job = None
        self._saving_csv = False
        log.error("recording CSV dump failed: %s", message)
        self.error_occurred.emit(f"failed to write CSV: {message}")
        self.recording_save_failed.emit(message)
        self._finish_save_phase()

    def _finish_save_phase(self) -> None:
        if self.state() == SessionState.RUNNING and self._metadata_ready:
            self._set_phase("ready to record")

    @staticmethod
    def _recording_has_rows(rec: Recording) -> bool:
        for _ in rec.storage.iter_rows():
            return True
        return False

    def _write_csv(self, rec: Recording) -> Path:
        output_dir = self._config.output_dir or str(default_recordings_dir())
        return write_recording_csv(
            rec,
            label=self._config.label,
            output_dir=output_dir,
            metadata_text=self._build_metadata_text(rec),
        )

    def _build_metadata_text(self, rec: Recording) -> str | None:
        """Render the ``.txt`` session-metadata sidecar for *rec* (best-effort).

        Bundles transport/device, host OS/timestamps, ``session_info`` and the
        channel/error layout so the CSV ships with a self-describing companion.
        A failure here must never block the CSV dump, so it degrades to *None*.
        """
        try:
            return build_metadata_text(
                rec,
                label=self._config.label,
                source_kind=self._config.source.kind,
                source_params=dict(self._config.source.params),
                source_description=self._session.describe_source(),
                protocol_description=self._session.describe_protocol(),
            )
        except Exception:
            log.exception("failed to build recording metadata sidecar")
            return None

    def cancel_recording(self) -> None:
        """Stop the current recording WITHOUT writing a CSV and drop its data."""
        if self.is_recording():
            try:
                self.recorder.stop()
            except Exception:
                log.exception("recorder.stop() during cancel failed")
        self._resume_gc_after_recording()
        # A cancelled recording produced no saved file, so remove both markers
        # rather than leaving a start/stop pair on the charts.
        self.data_model.clear_recording_markers()
        self.recorder.cleanup_spill()
        if self.state() == SessionState.RUNNING and self._metadata_ready:
            self._set_phase("ready to record")

    def save_pending_mismatch_recording(self) -> Path | None:
        rec = self._pending_mismatch_recording
        if rec is None:
            return None
        path = self._write_csv(rec)
        self.recording_saved.emit(str(path))
        self._pending_mismatch_recording = None
        return path

    def drop_pending_mismatch_recording(self) -> None:
        self._pending_mismatch_recording = None
        self.recorder.cleanup_spill()

    def shutdown(self) -> None:
        """Shut down the recorder and session cleanly."""
        self._set_phase("closing")
        try:
            if self.is_recording():
                self.stop_recording()
        finally:
            self._session.shutdown()

    # ------------------------------------------------------------------
    # GUI-thread message processing
    # ------------------------------------------------------------------

    @Slot(bytes)
    def _on_raw_bytes_for_bandwidth(self, chunk: bytes) -> None:
        self.bandwidth.add_bytes(len(chunk))

    @Slot(int)
    def _on_raw_byte_count_for_bandwidth(self, n_bytes: int) -> None:
        self.bandwidth.add_bytes(int(n_bytes))

    @Slot(object)
    def _process_message(self, msg: DecodedMessage) -> None:
        """Update DataModel and Recorder — always on the GUI thread."""
        if msg.kind == "session_info":
            # Establish/validate the metadata baseline FIRST so that any slot
            # connected to session_info_received observes the up-to-date
            # metadata_ready / phase state.
            self._process_session_info(msg)
            self.session_info_received.emit(msg.raw)
            # Propagate the validated metadata to the live plots so charts are
            # labelled with the device channel names (ch_n), not generic ch0/ch1.
            if self._metadata_ready:
                self.data_model.on_message(msg)
            self.recorder.set_session_info(msg.raw, self.data_model.channel_names)
            self.message_received.emit(msg)
            return

        if msg.kind == "sensor_data":
            if not self._validate_sensor_data(msg):
                self.message_received.emit(msg)
                return
            with slow_span("process_message.bandwidth"):
                with slow_span("process_message.bw.meter"):
                    self.bandwidth.add_message()
                # Feed the transport-loss detector with this message's sequence
                # number so missing numbers can be confirmed after the grace
                # window. The observe() call also sweeps due losses, which can
                # synchronously fan out error events — measured separately so a
                # slow span here is attributable, not assumed.
                with slow_span("process_message.bw.seq_observe"):
                    self._seq_tracker.observe(msg.seq)

        with slow_span("process_message.data_model"):
            self.data_model.on_message(msg)
        with slow_span("process_message.recorder"):
            self.recorder.on_message(msg)
        with slow_span("process_message.emit"):
            self.message_received.emit(msg)

    def _process_session_info(self, msg: DecodedMessage) -> None:
        normalized = self._normalize_session_info(msg.raw)
        if normalized is None:
            self.error_log.add(
                ErrorCategory.SESSION_INFO_INVALID,
                "invalid session_info payload (missing required fields or invalid ch_n)",
            )
            return

        # Track the producer-side drop count ("dr"). It legitimately increments
        # over the session, so it is excluded from the mismatch comparison and
        # instead surfaced as a real-time PRODUCER_DROP delta.
        self._track_drop_count(msg.raw)

        # Surface the sampling rate (Hz) and channel count to interested views
        # (e.g. the bandwidth details window). Re-emitted on every valid
        # session_info so a varying producer frequency is reflected live.
        hz = float(normalized["d"]["hz"])
        ch_count = len(normalized["d"]["ch_n"])
        sampling = (hz, ch_count)
        if sampling != self._latest_sampling:
            self._latest_sampling = sampling
            self.session_sampling_changed.emit(hz, ch_count)

        if self._session_info_baseline is None:
            self._session_info_baseline = normalized
            self._metadata_ready = True
            self._expected_channel_count = len(normalized["d"]["ch_n"])
            if self.state() == SessionState.RUNNING and not self.is_recording():
                self._set_phase("ready to record")
            if (
                self._record_on_metadata_pending
                and self.state() == SessionState.RUNNING
                and not self.is_recording()
                and self.can_begin_recording()
            ):
                try:
                    self.start_recording()
                    self._record_on_metadata_pending = False
                except Exception as exc:
                    log.exception("auto start_recording on session_info failed")
                    self.error_occurred.emit(f"could not start recording: {exc}")
            return

        if normalized != self._session_info_baseline:
            # Only a change to the session identity (sid) or the channel set
            # (ch_n) terminates the session; all other fields (hz, st, name, …)
            # may vary and merely update the read-only Session Info display.
            # dr is already excluded upstream.
            if self._critical_identity(normalized) != self._critical_identity(
                self._session_info_baseline
            ):
                self._handle_session_info_mismatch()
            else:
                # Non-critical drift: adopt the new values as the live baseline
                # so the displayed metadata reflects the latest device payload.
                self._session_info_baseline = normalized

    @staticmethod
    def _critical_identity(normalized: dict) -> tuple:
        """Return the (sid, ch_n) pair that must stay constant."""
        d = normalized.get("d", {}) if isinstance(normalized, dict) else {}
        return (d.get("sid"), tuple(d.get("ch_n", ())))

    def _track_drop_count(self, raw: dict) -> None:
        """Record the real-time delta of the producer ``dr`` drop count.

        The first valid value establishes the baseline; every subsequent value
        updates the absolute ``PRODUCER_DROP`` count to ``dr - baseline`` so the
        Error & Loss Analysis reflects drops accumulated since startup.
        """
        d = raw.get("d") if isinstance(raw, dict) else None
        if not isinstance(d, dict):
            return
        dr = d.get("dr")
        if not isinstance(dr, int) or isinstance(dr, bool):
            return
        if self._dr_baseline is None:
            self._dr_baseline = dr
        if self._dr_reset_baseline is None:
            self._dr_reset_baseline = dr
        self._dr_latest = dr
        # A fresh device-drop reading: any device-caused gap up to this point is
        # now reflected in ``dr``, so parked Host-BT losses may be reconciled.
        self._dr_observation_epoch += 1
        self._refresh_producer_drop_counts()

    def _mark_producer_drop_reset(self) -> None:
        """Re-anchor the since-reset producer-drop figure at a reset moment.

        The cumulative ``PRODUCER_DROP_TOTAL`` is unaffected; the since-reset
        ``PRODUCER_DROP`` restarts from zero at the latest observed drop count.
        """
        if self._dr_latest is not None:
            self._dr_reset_baseline = self._dr_latest
        self._refresh_producer_drop_counts()

    def _refresh_producer_drop_counts(self) -> None:
        """Update both producer drop-count statistics from the latest ``dr``.

        ``PRODUCER_DROP_TOTAL`` is ``dr - startup_baseline`` (cumulative since
        the session started); ``PRODUCER_DROP`` is
        ``dr - reset_baseline`` (since the last reset moment).
        """
        if self._dr_latest is None or self._dr_baseline is None:
            return
        reset_base = (
            self._dr_reset_baseline
            if self._dr_reset_baseline is not None
            else self._dr_baseline
        )
        self.error_log.set_count(
            ErrorCategory.PRODUCER_DROP_TOTAL, self._dr_latest - self._dr_baseline
        )
        self.error_log.set_count(
            ErrorCategory.PRODUCER_DROP, self._dr_latest - reset_base
        )

    def _handle_session_info_mismatch(self) -> None:
        detail = (
            "session_info changed after baseline was established; "
            "session is being terminated"
        )
        self.error_log.add(ErrorCategory.SESSION_INFO_MISMATCH, detail)
        self._record_on_metadata_pending = False

        has_buffered_data = False
        if self.is_recording():
            try:
                rec = self.recorder.stop()
                has_buffered_data = self._recording_has_rows(rec)
                if has_buffered_data:
                    self._pending_mismatch_recording = rec
            except Exception:
                log.exception("failed to stop recording after session_info mismatch")

        self.session_info_mismatch.emit(detail, has_buffered_data)
        self._session.fail(detail)

    def _validate_sensor_data(self, msg: DecodedMessage) -> bool:
        if msg.seq is None or msg.t_device_ms is None or msg.channels is None:
            self.error_log.add(
                ErrorCategory.SENSOR_DATA_INVALID,
                "sensor_data missing one or more required fields: seq, ts, val",
            )
            return False

        if self._expected_channel_count is None:
            # session_info not received yet — wait for metadata before ingesting.
            return False

        if len(msg.channels) != self._expected_channel_count:
            self.error_log.add(
                ErrorCategory.SENSOR_DATA_INVALID,
                "sensor_data val count does not match session_info ch_n length",
                expected=self._expected_channel_count,
                got=len(msg.channels),
            )
            return False
        return True

    @staticmethod
    def _normalize_session_info(raw: dict) -> dict | None:
        if not isinstance(raw, dict) or raw.get("t") != "si":
            return None
        d = raw.get("d") if isinstance(raw.get("d"), dict) else None
        if d is None:
            return None
        # "ch_n" carries the channel labels and must be a non-empty list of
        # non-empty strings (the chart/CSV column names depend on it).
        ch_n = d.get("ch_n")
        if not isinstance(ch_n, (list, tuple)) or not ch_n:
            return None
        if not all(isinstance(name, str) and name for name in ch_n):
            return None

        # "ch" is the channel count; when present it must agree with ch_n.
        ch = d.get("ch")
        if ch is not None and (not isinstance(ch, int) or isinstance(ch, bool) or ch != len(ch_n)):
            return None

        required = ("sid", "hz", "st", "dr", "name")
        if any(key not in d for key in required):
            return None

        nd = dict(d)
        nd.pop("ch", None)  # redundant field; rely on ch_n names only
        # "dr" (producer drop count) legitimately increments during a session,
        # so it is excluded from the constant-metadata mismatch check.
        nd.pop("dr", None)
        nd["ch_n"] = tuple(ch_n)
        return {"t": "si", "d": nd}

    def _set_phase(self, phase: str) -> None:
        if phase == self._phase:
            return
        self._phase = phase
        self.session_phase_changed.emit(phase)
