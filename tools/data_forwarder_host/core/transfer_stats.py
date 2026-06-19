# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Pure derivation of received-frame categories for the bandwidth details view.

The decoder reports raw counters (:class:`DecodeStats`); this module turns them
into the human-facing reception categories shown in the bandwidth details
sub-window: how many frames were received in total, how many decoded
without issues, and the per-reason breakdown of the rest (including frames that
arrived but were incomplete/corrupted). Kept GUI-free so it is isolated from the GUI.
"""

from __future__ import annotations

from data_forwarder_host.protocol.base import DecodeStats


def received_frame_breakdown(stats: DecodeStats) -> list[tuple[str, int]]:
    """Return ordered ``(label, count)`` reception categories from *stats*.

    * ``Received (all frames)`` — every frame seen, good or bad
      (``frames_ok + frames_bad``).
    * ``Decoded OK (no issues)`` — fully decoded frames (``frames_ok``).
    * ``Received with issues`` — frames that arrived but failed decoding
      (``frames_bad``); these are the incomplete/corrupted ones.
    * Per-reason rows for the failures: COBS, CRC, malformed, CBOR.
    """
    received_all = stats.frames_ok + stats.frames_bad
    return [
        ("Received (all frames)", received_all),
        ("Decoded OK (no issues)", stats.frames_ok),
        ("Received with issues", stats.frames_bad),
        ("  COBS framing errors", stats.cobs_errors),
        ("  CRC errors", stats.crc_errors),
        ("  Malformed frames", stats.malformed),
        ("  CBOR decode errors", stats.cbor_errors),
    ]


def received_bytes_breakdown(stats: DecodeStats) -> list[tuple[str, int]]:
    """Return ordered ``(label, bytes)`` reception categories from *stats*.

    Mirrors :func:`received_frame_breakdown` but reports the real number of
    bytes seen per category rather than the frame counts. For decoded
    frames this is the COBS-decoded frame length; for COBS failures it is the
    length of the raw on-wire chunk that could not be decoded.
    """
    return [
        ("Received (all frames)", stats.bytes_ok + stats.bytes_bad),
        ("Decoded OK (no issues)", stats.bytes_ok),
        ("Received with issues", stats.bytes_bad),
        ("  COBS framing errors", stats.bytes_cobs),
        ("  CRC errors", stats.bytes_crc),
        ("  Malformed frames", stats.bytes_malformed),
        ("  CBOR decode errors", stats.bytes_cbor),
    ]


def byte_baseline(stats: DecodeStats) -> dict[str, int]:
    """Capture per-category byte totals as a reset baseline.

    The byte analogue of :func:`baseline_counts`; subtract from a later
    :func:`received_bytes_breakdown` to obtain "since reset" byte deltas.
    """
    return {label: count for label, count in received_bytes_breakdown(stats)}


def bytes_breakdown_since(
    stats: DecodeStats, baseline: dict[str, int]
) -> list[tuple[str, int, int]]:
    """Return ordered ``(label, total_bytes, since_reset_bytes)`` rows.

    The byte analogue of :func:`breakdown_since`: ``since_reset_bytes`` is
    ``total_bytes - baseline[label]`` (never negative); an empty baseline yields
    ``since_reset_bytes == total_bytes``.
    """
    return [
        (label, total, max(0, total - baseline.get(label, 0)))
        for label, total in received_bytes_breakdown(stats)
    ]


def baseline_counts(stats: DecodeStats) -> dict[str, int]:
    """Capture the current per-category counts as a reset baseline.

    The returned mapping is keyed by the same labels as
    :func:`received_frame_breakdown`, so it can be subtracted from a later
    breakdown to obtain "since reset" deltas.
    """
    return {label: count for label, count in received_frame_breakdown(stats)}


def breakdown_since(
    stats: DecodeStats, baseline: dict[str, int]
) -> list[tuple[str, int, int]]:
    """Return ordered ``(label, total, since_reset)`` rows.

    ``total`` is the session-cumulative count; ``since_reset`` is
    ``total - baseline[label]`` (never negative), i.e. how many of each
    category were received since the window-scoped counter was last reset. A
    missing baseline key counts as zero, so an empty baseline yields
    ``since_reset == total``.
    """
    return [
        (label, total, max(0, total - baseline.get(label, 0)))
        for label, total in received_frame_breakdown(stats)
    ]
