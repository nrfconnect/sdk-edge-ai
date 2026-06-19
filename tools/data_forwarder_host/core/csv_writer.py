# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Fixed CSV output for a finished recording.

A single CSV file in one predefined format:

* Filename: ``{label}_{session}_{utc}.csv`` where ``{label}`` is the
  user-defined recording label, ``{session}`` is the session tag, and
  ``{utc}`` is the host UTC timestamp of the recording-start moment with
  millisecond resolution.
* Row 1 (header): ``device_time_ms`` followed by the N channel names and a
  trailing ``label`` column.
* Rows 2..M (data): the device timestamp in milliseconds followed by the N
  channel values, in the same column order as the header, and finally the
  user-defined recording label (constant for every row).

A companion ``{stem}.txt`` sidecar holding human-readable session metadata
(transport, device, host OS/timestamps, ``session_info``, channel layout and an
error summary) is written next to the CSV when *metadata_text* is supplied.

There is no append mode and no configurable filename pattern.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from data_forwarder_host.core.recorder import Recording

DEVICE_TIME_COLUMN = "device_time_ms"
LABEL_COLUMN = "label"

# --- Recording-start timestamp appended to the CSV filename ------------------
# Edit these two expressions to change the trailing timestamp in CSV filenames.
# ``CSV_FILENAME_TIMESTAMP_FORMAT`` is a ``strftime`` pattern applied to the
# host UTC start time; milliseconds and the ``Z`` suffix are appended by
# ``CSV_FILENAME_TIMESTAMP_SUFFIX`` to keep millisecond resolution. The rendered
# stamp is human-readable and uses ONLY digits, dashes and underscores (plus a
# trailing ``Z`` to mark UTC) — no dots, spaces, colons or other special signs,
# so it stays filesystem-safe across platforms.
CSV_FILENAME_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"
CSV_FILENAME_TIMESTAMP_SUFFIX = "_{ms:03d}Z"


def format_filename_timestamp(started_utc: str) -> str:
    """Render *started_utc* (ISO-8601 ``...Z``) as a filesystem-safe stamp.

    Falls back to a sanitised copy of the input if it cannot be parsed, so a
    filename is always produced.
    """
    try:
        dt = datetime.strptime(started_utc, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
    except (ValueError, TypeError):
        return "".join(c if c.isalnum() else "-" for c in str(started_utc))
    ms = dt.microsecond // 1000
    return dt.strftime(CSV_FILENAME_TIMESTAMP_FORMAT) + CSV_FILENAME_TIMESTAMP_SUFFIX.format(ms=ms)


def csv_filename(label: str, session_tag: str, started_utc: str) -> str:
    """Return ``{label}_{session}_{utc}.csv``."""
    return f"{csv_stem(label, session_tag, started_utc)}.csv"


def csv_stem(label: str, session_tag: str, started_utc: str) -> str:
    """Return the CSV filename stem (no extension): ``{label}_{session}_{utc}``."""
    return f"{label}_{session_tag}_{format_filename_timestamp(started_utc)}"


def unique_csv_path(directory: Path, stem: str) -> Path:
    """Return a non-colliding ``directory/<stem>.csv`` path.

    If ``<stem>.csv`` does not exist it is returned unchanged; otherwise an
    incrementing numeric suffix is appended to the stem until a free path is
    found: ``<stem>_1.csv``, ``<stem>_2.csv``, … An existing file is never
    overwritten and the user is never prompted.
    """
    base = directory / f"{stem}.csv"
    if not base.exists():
        return base
    n = 1
    while True:
        candidate = directory / f"{stem}_{n}.csv"
        if not candidate.exists():
            return candidate
        n += 1


def write_recording_csv(recording: Recording, *, label: str, output_dir: str | Path,
                        metadata_text: str | None = None) -> Path:
    """Write *recording* to ``{label}_{session}_{utc}.csv`` under *output_dir*.

    Returns the path of the written file. Raises ``ValueError`` if *label* is
    empty (the Record action must be gated on a defined label). When
    *metadata_text* is given, a ``{stem}.txt`` metadata sidecar is written next
    to the CSV.

    This is the blocking convenience wrapper; the GUI uses :class:`RecordingCsvDump`
    directly so the dump can be driven incrementally off the recording-stop event
    without freezing the event loop.
    """
    return RecordingCsvDump(
        recording, label=label, output_dir=output_dir, metadata_text=metadata_text
    ).run_to_completion()


class RecordingCsvDump:
    """Incremental writer for a finished recording's CSV.

    The whole capture is RAM/spill-buffered while recording; at *stop* it must
    be flushed to a single CSV. For large recordings that flush can take long
    enough to freeze the GUI if done in one blocking call, so this class writes
    the file in bounded **chunks**: each :meth:`step` writes at most
    ``chunk_rows`` rows and returns whether more remain, letting a caller yield
    to the Qt event loop (or another thread) between chunks while a progress bar
    tracks :attr:`rows_written` against :attr:`total_rows`.

    The produced file is byte-for-byte identical to the one-shot writer.

    :raises ValueError: if *label* is empty (Record is gated on a label).
    """

    def __init__(
        self,
        recording: Recording,
        *,
        label: str,
        output_dir: str | Path,
        chunk_rows: int = 4096,
        metadata_text: str | None = None,
    ) -> None:
        if not label or not label.strip():
            raise ValueError("a recording label is required to write the CSV file")
        self._row_label = label.strip()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = csv_stem(self._row_label, recording.session_tag, recording.started_utc)
        self.path: Path = unique_csv_path(out_dir, stem)
        self._metadata_text = metadata_text
        self._channel_names = list(recording.channel_names)
        self._rows = recording.storage.iter_rows()
        self._chunk_rows = max(1, int(chunk_rows))
        # Best-effort total for the progress bar: known for RAM storage, None
        # (indeterminate) once capture has spilled to disk shards.
        count = getattr(recording.storage, "row_count", None)
        self.total_rows: int | None = int(count) if count is not None else None
        self.rows_written: int = 0
        self._fh = None
        self._writer = None
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    def step(self) -> bool:
        """Write at most one chunk of rows; return *True* while rows remain.

        Opens the file and writes the header on the first call. On the final
        call (or on error) the file handle is closed. Returns *False* once the
        whole recording has been written.
        """
        if self._done:
            return False
        if self._fh is None:
            self._fh = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._fh)
            self._writer.writerow([DEVICE_TIME_COLUMN, *self._channel_names, LABEL_COLUMN])
        try:
            for _ in range(self._chunk_rows):
                _t_host_ms, t_device_ms, _seq, _lbl, channels = next(self._rows)
                device_time = "" if t_device_ms is None else t_device_ms
                # The trailing label column is the user-defined recording label,
                # identical on every row (the device's per-row ``lbl`` is ignored).
                self._writer.writerow([device_time, *channels, self._row_label])
                self.rows_written += 1
            return True
        except StopIteration:
            self._write_metadata_sidecar()
            self._close()
            return False
        except Exception:
            self._close()
            raise

    def run_to_completion(self) -> Path:
        """Drive :meth:`step` until the whole recording is written; return the path."""
        while self.step():
            pass
        return self.path

    @property
    def metadata_path(self) -> Path:
        """Path of the ``{stem}.txt`` metadata sidecar paired with the CSV."""
        return self.path.with_suffix(".txt")

    def _write_metadata_sidecar(self) -> None:
        """Write the metadata sidecar next to the CSV (best-effort).

        The sidecar shares the CSV's base name with a ``.txt`` extension. A
        failure to write it must not abort an otherwise successful CSV dump, so
        any error is swallowed (the CSV is the primary artefact).
        """
        if not self._metadata_text:
            return
        try:
            self.metadata_path.write_text(self._metadata_text, encoding="utf-8")
        except Exception:
            pass

    def _close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None
        self._done = True
