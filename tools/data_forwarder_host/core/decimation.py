# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Pure, Qt-free render-time decimation for live charts.

The live charts must redraw in time proportional to the number of points they
put *on screen*, not the number of samples buffered. This module provides a
single pure function, :func:`decimate_minmax`, that reduces a channel's
``(ts, val)`` samples to a bounded number of points using **min/max bucketing**.

Why min/max (and not stride / every-Nth sampling): stride sampling can step over
a transient spike entirely, hiding it. Min/max bucketing splits the input into a
fixed number of contiguous, x-ordered buckets and, for each bucket, emits *both*
the minimum and the maximum sample (in x order). Peaks and dips therefore always
survive, while the on-screen point count stays bounded.

Contract
--------
``decimate_minmax(ts, val, target_buckets=DEFAULT_TARGET_BUCKETS) -> (ts, val)``

* Inputs: two equal-length sequences (lists or NumPy arrays) — ``ts`` (x, must be
  non-decreasing) and ``val`` (y).
* Output length is bounded: ``len(out) <= 2 * target_buckets`` for any input.
* If ``len(ts) <= 2 * target_buckets`` the input is returned unchanged (passthrough).
* Output x order is preserved (non-decreasing); the function is deterministic.
* No Qt import — the GUI widgets call this; they do not implement downsampling.
"""

from __future__ import annotations

import numpy as np

# Default number of horizontal buckets. Each bucket can emit up to two points
# (its min and its max), so the on-screen point count per channel is capped at
# ``2 * DEFAULT_TARGET_BUCKETS`` (≈2000) regardless of how many samples are
# buffered. Tuned for smooth interaction on a typical desktop display.
DEFAULT_TARGET_BUCKETS = 1000


def decimate_minmax(
    ts: np.ndarray,
    val: np.ndarray,
    target_buckets: int = DEFAULT_TARGET_BUCKETS,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce ``(ts, val)`` to a bounded set of points via min/max bucketing.

    Args:
        ts: x values (timestamps), non-decreasing, length ``n``.
        val: y values, length ``n``.
        target_buckets: number of horizontal buckets; output is capped at
            ``2 * target_buckets`` points. Must be >= 1.

    Returns:
        ``(ts_out, val_out)`` NumPy arrays. When ``n <= 2 * target_buckets`` the
        input values are returned unchanged (as arrays).

    Raises:
        ValueError: if ``target_buckets < 1`` or the inputs differ in length.
    """
    if target_buckets < 1:
        raise ValueError("target_buckets must be >= 1")

    ts_arr = np.asarray(ts)
    val_arr = np.asarray(val)
    if ts_arr.shape[0] != val_arr.shape[0]:
        raise ValueError("ts and val must have the same length")

    n = ts_arr.shape[0]
    cap = 2 * target_buckets
    # Passthrough: already small enough that decimation would not reduce it.
    if n <= cap:
        return ts_arr.copy(), val_arr.copy()

    # Contiguous, x-ordered bucket boundaries over the sample index range. Using
    # index boundaries (rather than time boundaries) keeps each bucket non-empty
    # and the work O(n) with no dependence on the time distribution.
    edges = np.linspace(0, n, target_buckets + 1).astype(np.int64)

    # Keep only non-empty buckets. Because the buckets partition ``[0, n)`` into
    # contiguous index ranges, dropping the zero-width ones leaves a set of
    # strictly increasing segment starts that still cover every sample — exactly
    # what ``np.*.reduceat`` needs.
    starts = edges[:-1]
    ends = edges[1:]
    starts = starts[starts < ends]
    sizes = np.diff(np.append(starts, n))

    # Per-bucket min/max VALUES via segmented reductions (fully vectorised; no
    # Python-level loop over buckets — this is the live-redraw hot path).
    seg_min = np.minimum.reduceat(val_arr, starts)
    seg_max = np.maximum.reduceat(val_arr, starts)

    # Per-bucket FIRST index achieving that min/max, matching ``np.argmin`` /
    # ``np.argmax`` semantics. For each sample, take its own index where it ties
    # the bucket extreme and ``n`` otherwise, then reduce with ``minimum`` to get
    # the earliest qualifying index in the bucket.
    idx = np.arange(n, dtype=np.int64)
    min_exp = np.repeat(seg_min, sizes)
    max_exp = np.repeat(seg_max, sizes)
    cand_min = np.where(val_arr == min_exp, idx, n)
    cand_max = np.where(val_arr == max_exp, idx, n)
    i_min = np.minimum.reduceat(cand_min, starts)
    i_max = np.minimum.reduceat(cand_max, starts)

    # Emit the earlier-indexed extreme first so the x-order stays non-decreasing;
    # collapse to a single point where the bucket's min and max coincide.
    lo = np.minimum(i_min, i_max)
    hi = np.maximum(i_min, i_max)
    same = i_min == i_max
    keep = np.empty(2 * lo.shape[0], dtype=np.int64)
    keep[0::2] = lo
    keep[1::2] = hi
    mask = np.ones(keep.shape[0], dtype=bool)
    mask[1::2] = ~same
    keep = keep[mask]
    return ts_arr[keep], val_arr[keep]
