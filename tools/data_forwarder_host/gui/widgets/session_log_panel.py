# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""SessionLogPanel — two collapsible log consoles for one session.

Consoles (all collapsed by default, each capped at 1000 lines):

``Channel ASCII``
    One line per ``sensor_data`` message:
    ``{t_host_ms}, {ch0}, {ch1}, …``

``Decoded Frames``
    Every decoded message's kind + raw dict, e.g.
    ``[1234567] sensor_data  {'ts': 42, 'val': [0.1, 0.2], …}``
"""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget

from data_forwarder_host.gui.widgets.collapsible_section import CollapsibleSection
from data_forwarder_host.session.controller import SessionController

_MAX_LINES = 1000
_FLUSH_MS = 100              # flush text consoles at most 10 × / s
_TEXT_BUF_CAP = 2000         # max formatted text lines held between flushes


# ---------------------------------------------------------------------------
# Thin log console
# ---------------------------------------------------------------------------


class _LogConsole(QPlainTextEdit):
    """Read-only, fixed-height plain-text console with its own scrollbar."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(_MAX_LINES)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        # Fixed height: tall enough to show ~6 lines; internal scrollbar handles
        # overflow.  The outer (session-tab) scroll area is shared with charts.
        self.setMinimumHeight(100)
        self.setMaximumHeight(220)


# ---------------------------------------------------------------------------
# SessionLogPanel
# ---------------------------------------------------------------------------


class SessionLogPanel(QWidget):
    """Two collapsible log consoles wired to a ``SessionController``."""

    def __init__(
        self, controller: SessionController, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Channel ASCII ───────────────────────────────────────────────
        self._ascii = _LogConsole()
        self._ascii.setPlaceholderText("No sensor_data messages yet — waiting for data…")
        self._sec_ascii = CollapsibleSection("Channel ASCII", self._ascii, expanded=False)
        layout.addWidget(self._sec_ascii)

        # ── Decoded Frames ──────────────────────────────────────────────
        self._frames = _LogConsole()
        self._frames.setPlaceholderText("No decoded frames yet…")
        self._sec_frames = CollapsibleSection("Decoded Frames", self._frames, expanded=False)
        layout.addWidget(self._sec_frames)

        # ── Wire signals ────────────────────────────────────────────────
        controller.message_received.connect(self._on_message)
        controller.state_changed.connect(self._on_state_changed)
        controller.recording_changed.connect(self._on_recording_changed)
        # ── Rate-limited log buffers ─────────────────────────────────────
        # Decoded text arrives at transport speed (can be thousands of
        # messages/s). Buffering it and flushing on a single timer turns a
        # per-message append storm — the main long-run FPS sink — into at most
        # one (multi-line) append per console every _FLUSH_MS, while still
        # preserving the last _MAX_LINES of history.
        self._ascii_buf: list[str] = []
        self._frames_buf: list[str] = []
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(_FLUSH_MS)
        self._flush_timer.timeout.connect(self._flush_log_buffers)
        self._flush_timer.start()

    # ------------------------------------------------------------------
    # Panel visibility (View ▸ Panels)
    # ------------------------------------------------------------------

    def set_channel_ascii_visible(self, on: bool) -> None:
        self._sec_ascii.setVisible(on)

    def set_decoded_frames_visible(self, on: bool) -> None:
        self._sec_frames.setVisible(on)

    # ------------------------------------------------------------------
    # Layout state persistence
    # ------------------------------------------------------------------

    def get_layout_state(self) -> dict:
        """Return expand/collapse states of all sections as a serialisable dict."""
        return {
            "ascii_expanded": self._sec_ascii.is_expanded(),
            "frames_expanded": self._sec_frames.is_expanded(),
        }

    def apply_layout_state(self, state: dict) -> None:
        """Restore expand/collapse states from a previously saved dict."""
        if "ascii_expanded" in state:
            self._sec_ascii.set_expanded(bool(state["ascii_expanded"]))
        if "frames_expanded" in state:
            self._sec_frames.set_expanded(bool(state["frames_expanded"]))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_state_changed(self, _state: object) -> None:
        """Clear all consoles when a new stream starts."""
        from data_forwarder_host.session.states import SessionState
        if _state == SessionState.RUNNING:
            self._ascii.clear()
            self._frames.clear()
            self._ascii_buf.clear()
            self._frames_buf.clear()

    def _on_recording_changed(self, recording: bool) -> None:
        """Emit a one-line banner into the log consoles when recording toggles."""
        banner = (
            "───── RECORDING STARTED ─────"
            if recording
            else "───── RECORDING STOPPED ─────"
        )
        # Flush first so the banner lands after every message buffered so far,
        # preserving chronological order with the coalesced text consoles.
        self._flush_log_buffers()
        self._ascii.appendPlainText(banner)
        self._frames.appendPlainText(banner)

    def _on_message(self, msg: object) -> None:
        # msg is a DecodedMessage (avoid hard type import to keep the module
        # importable in headless contexts where Qt may not be initialised yet)
        kind = getattr(msg, "kind", None)
        t_ms = getattr(msg, "t_host_ms", 0)
        raw = getattr(msg, "raw", {})
        channels = getattr(msg, "channels", None)

        # ── Channel ASCII: only sensor_data rows ────────────────────────
        if kind == "sensor_data" and channels is not None:
            fields = [str(t_ms)] + [f"{v:.6g}" for v in channels]
            self._ascii_buf.append(",".join(fields))
            if len(self._ascii_buf) > _TEXT_BUF_CAP:
                del self._ascii_buf[:-_TEXT_BUF_CAP]

        # ── Decoded Frames: every message ───────────────────────────────
        self._frames_buf.append(f"[{t_ms:>10}] {kind}  {raw!r}")
        if len(self._frames_buf) > _TEXT_BUF_CAP:
            del self._frames_buf[:-_TEXT_BUF_CAP]

    def _flush_log_buffers(self) -> None:
        """Drain the coalescing buffers into their consoles (timer-driven).

        Each console gets at most one append per tick: the buffered text lines
        are joined and appended in a single call so QPlainTextEdit relayouts
        once instead of once per message.
        """
        if self._ascii_buf:
            lines, self._ascii_buf = self._ascii_buf, []
            self._ascii.appendPlainText("\n".join(lines))
        if self._frames_buf:
            lines, self._frames_buf = self._frames_buf, []
            self._frames.appendPlainText("\n".join(lines))
