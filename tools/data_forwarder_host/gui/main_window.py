# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Top-level application window."""

from __future__ import annotations

import json
import logging
import threading
from base64 import b64decode, b64encode
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QByteArray, QTimer, Signal
from PySide6.QtGui import QAction, QCloseEvent, QColor
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from data_forwarder_host.gui.dialogs.about_dialog import AboutDialog
from data_forwarder_host.gui.dialogs.new_session_dialog import (
    NewSessionDialog,
    _BusyDialog,
)
from data_forwarder_host.gui.menus import build_menus
from data_forwarder_host.gui.session_tab import SessionTab
from data_forwarder_host.gui.theme import Theme, apply_theme
from data_forwarder_host.gui.widgets.host_metrics_label import HostMetricsLabel
from data_forwarder_host.gui.widgets.log_panel import LogPanel
from data_forwarder_host.gui.widgets.session_tab_widget import SessionTabWidget
from data_forwarder_host.session.states import SessionState
from data_forwarder_host.utils.logging import configure_logging, log_user_action
from data_forwarder_host.utils.version import get_version

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from data_forwarder_host.app import AppContext
    from data_forwarder_host.session.config import SessionConfig

# Tab title prefix per session state.
_STATE_ICON: dict[SessionState, str] = {
    SessionState.CONFIGURED: "○",
    SessionState.RUNNING:    "▶",
    SessionState.STOPPED:    "◼",
    SessionState.ERROR:      "✕",
}
_ICON_RECORDING = "⏺"

# Tab text colours per state (empty string → invalid QColor → reset to palette default).
_STATE_COLOR: dict[SessionState, str] = {
    SessionState.RUNNING:    "",         # default colour — streaming but not recording
    SessionState.STOPPED:    "#888888",  # gray
    SessionState.CONFIGURED: "#888888",  # gray — not yet started
    SessionState.ERROR:      "#FF4444",  # red
}


def _tab_color(state: SessionState, recording: bool, rec_tick: int = 0) -> QColor:
    if recording:
        # Blink: bright red on even ticks, dim red on odd ticks.
        return QColor("#FF2222") if rec_tick else QColor("#993333")
    c = _STATE_COLOR.get(state)
    return QColor(c) if c else QColor()  # invalid QColor resets to palette default


class _WelcomeWidget(QWidget):
    """Shown when no session tab is open.

    Reorganised around a single focal call-to-action: the branding is demoted
    to a small muted header in the top-left corner, while a centred hero
    promotes **Get started** with a large **New Session** button. Supporting
    material (quick-start steps, keyboard shortcuts) sits in a muted footer.
    """

    def __init__(
        self,
        on_new_session: "Callable[[], None]",
        on_open_session: "Callable[[], None]",
        parent: QWidget | None = None,
    ) -> None:
        from typing import Callable  # noqa: PLC0415
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 28, 40, 28)
        root.setSpacing(0)

        # ── Branding pushed to the top-left corner, de-emphasised ─────
        brand_row = QHBoxLayout()
        brand = QLabel("Data Forwarder Host")
        bf = brand.font()
        bf.setPointSize(12)
        bf.setBold(True)
        brand.setFont(bf)
        brand.setStyleSheet("color: palette(mid);")
        brand_row.addWidget(brand)
        brand_row.addStretch(1)
        root.addLayout(brand_row)

        # ── Centred hero: the promoted call-to-action ─────────────────
        root.addStretch(3)

        hero = QVBoxLayout()
        hero.setSpacing(0)

        get_started = QLabel("Get started")
        gf = get_started.font()
        gf.setPointSize(34)
        gf.setBold(True)
        get_started.setFont(gf)
        get_started.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        hero.addWidget(get_started)

        hero.addSpacing(10)

        tagline = QLabel(
            "Receive, record and forward sensor data from nRF devices "
            "over UART or Bluetooth LE."
        )
        tagline.setStyleSheet("color: palette(mid); font-size: 14px;")
        tagline.setWordWrap(True)
        tagline.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        tagline.setMaximumWidth(520)
        hero.addWidget(tagline, alignment=Qt.AlignmentFlag.AlignHCenter)

        hero.addSpacing(36)

        btn_new = QPushButton("New Session")
        btn_new.setMinimumHeight(56)
        btn_new.setMinimumWidth(300)
        btn_new.setMaximumWidth(420)
        btn_new.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn_new.setDefault(True)
        btn_new.setAutoDefault(True)
        btn_new.setCursor(Qt.CursorShape.PointingHandCursor)
        # Explicit accent styling (fixed colours, not palette-derived) so the
        # primary call-to-action reads identically and with strong contrast in
        # every theme — System / Light / Dark. The default QPushButton inherited
        # the window palette, which left it near-invisible (dark-on-dark) on a
        # dark desktop and never stood out as the obvious place to click.
        btn_new.setStyleSheet(
            "QPushButton {"
            "  background-color: #2E7CF6;"
            "  color: #FFFFFF;"
            "  padding: 12px 32px;"
            "  font-size: 16px;"
            "  font-weight: bold;"
            "  border: none;"
            "  border-radius: 10px;"
            "}"
            "QPushButton:hover { background-color: #4A90FF; }"
            "QPushButton:pressed { background-color: #1F6AE0; }"
        )
        btn_new.clicked.connect(on_new_session)
        hero.addWidget(btn_new, alignment=Qt.AlignmentFlag.AlignHCenter)

        hero.addSpacing(12)

        btn_open = QPushButton("Open saved session config…")
        btn_open.setMinimumHeight(36)
        btn_open.setMaximumWidth(420)
        btn_open.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        btn_open.setFlat(True)
        btn_open.setStyleSheet(
            "QPushButton { font-size: 12px; color: palette(mid); border: none; }"
        )
        btn_open.clicked.connect(on_open_session)
        hero.addWidget(btn_open, alignment=Qt.AlignmentFlag.AlignHCenter)

        root.addLayout(hero)

        root.addStretch(3)

        # ── Muted footer: quick-start steps + keyboard shortcuts ──────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(sep)
        root.addSpacing(16)

        footer = QHBoxLayout()
        footer.setSpacing(48)

        # Quick start steps
        qs = QVBoxLayout()
        qs.setSpacing(0)
        qs_hdr = QLabel("Quick start")
        qf = qs_hdr.font()
        qf.setBold(True)
        qs_hdr.setFont(qf)
        qs_hdr.setStyleSheet("color: palette(mid); font-size: 11px;")
        qs.addWidget(qs_hdr)
        qs.addSpacing(8)
        for i, step in enumerate(
            [
                "Connect your nRF device via USB, or power it on within "
                "Bluetooth LE range.",
                "Click <b>New Session</b> and choose the data source "
                "(UART or Bluetooth LE).",
                "For Bluetooth LE, select the device and press <b>Connect</b>.",
                "Set any other session details, then click <b>Create session</b>.",
                "Enter a recording <b>label</b> and choose an output directory.",
                "Press <b>Record</b> to start capturing data.",
                "Press <b>Stop</b> to finish; a single CSV file is written to your "
                "output directory.",
            ],
            1,
        ):
            row = QHBoxLayout()
            num = QLabel(f"<b>{i}</b>")
            num.setStyleSheet(
                "min-width: 18px; max-width: 18px; color: palette(mid); font-size: 11px;"
            )
            num.setAlignment(Qt.AlignmentFlag.AlignTop)
            lbl = QLabel(step)
            lbl.setStyleSheet("color: palette(mid); font-size: 11px;")
            lbl.setWordWrap(True)
            row.addWidget(num)
            row.addWidget(lbl)
            row.addStretch()
            qs.addLayout(row)
            qs.addSpacing(4)
        qs.addStretch(1)
        footer.addLayout(qs, stretch=3)

        # Vertical separator between footer columns
        vsep = QFrame()
        vsep.setFrameShape(QFrame.Shape.VLine)
        vsep.setFrameShadow(QFrame.Shadow.Sunken)
        footer.addWidget(vsep)

        # Keyboard shortcuts
        kb = QVBoxLayout()
        kb.setSpacing(0)
        kbd_hdr = QLabel("Keyboard shortcuts")
        kf = kbd_hdr.font()
        kf.setBold(True)
        kbd_hdr.setFont(kf)
        kbd_hdr.setStyleSheet("color: palette(mid); font-size: 11px;")
        kb.addWidget(kbd_hdr)
        kb.addSpacing(8)
        for key, action in (
            ("Ctrl+N", "New session"),
            ("Ctrl+O", "Open saved session config"),
        ):
            krow = QHBoxLayout()
            kbd = QLabel(key)
            kbd.setStyleSheet(
                "font-family: monospace; background: palette(mid);"
                " padding: 2px 5px; border-radius: 3px; font-size: 10px;"
            )
            kbd.setFixedWidth(96)
            desc = QLabel(action)
            desc.setStyleSheet("color: palette(mid); font-size: 11px;")
            krow.addWidget(kbd)
            krow.addSpacing(8)
            krow.addWidget(desc)
            krow.addStretch()
            kb.addLayout(krow)
            kb.addSpacing(5)
        kb.addStretch(1)
        footer.addLayout(kb, stretch=2)

        root.addLayout(footer)


class MainWindow(QMainWindow):
    # Emitted from the background shutdown thread once all sessions have been
    # torn down; routed (queued) back to the GUI thread to quit the app.
    _teardown_finished = Signal()

    def __init__(self, ctx: "AppContext") -> None:
        super().__init__()
        # Explicitly request all three title-bar buttons; required on some
        # Linux/Wayland compositors that don't add them by default.
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self._ctx = ctx
        self.setWindowTitle(f"Data Forwarder Host  v{get_version()}")
        self.resize(1280, 800)

        # Tabs. A browser-style "+" button sits to the right of the last tab;
        # it is a real button, not a tab, and opens New Session.
        self._tabs = SessionTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.setMovable(True)
        self._tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        # Open New Session on the next event-loop turn rather than synchronously
        # inside the "+" button's clicked handler: entering the dialog's modal
        # loop while the button still holds the mouse grab disrupts the BLE
        # scanner thread/timers started in the dialog's showEvent and makes the
        # scan spuriously report "Bluetooth off". Deferring matches the previous
        # pseudo-tab behaviour, which opened the dialog via a queued callback.
        self._tabs.new_session_requested.connect(
            lambda: QTimer.singleShot(0, self._on_new_session)
        )

        # Stack: page 0 = welcome, page 1 = tab widget.
        self._welcome = _WelcomeWidget(
            on_new_session=self._on_new_session,
            on_open_session=self._on_open_session,
        )
        self._stack = QStackedWidget()
        self._stack.addWidget(self._welcome)
        self._stack.addWidget(self._tabs)
        self._stack.setCurrentIndex(0)
        self.setCentralWidget(self._stack)

        # Status bar
        status = QStatusBar()
        self._host_metrics = HostMetricsLabel()
        self._host_metrics.setToolTip("Host and application resource usage")
        status.addPermanentWidget(self._host_metrics)
        status.addPermanentWidget(QLabel(f"v{get_version()}"))
        self.setStatusBar(status)

        # Log panel as a dock (hidden by default)
        self._log_dock = QDockWidget("Application log", self)
        self._log_dock.setWidget(LogPanel())
        self._log_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._log_dock)
        self._log_dock.hide()

        # Menus
        self._actions: dict[str, QAction] = build_menus(
            self,
            on_new_session=self._on_new_session,
            on_close_session=self._on_close_current_session,
            on_open_session=self._on_open_session,
            on_save_session=self._on_save_session,
            on_save_session_as=self._on_save_session_as,
            on_quit=self.close,
            on_toggle_error_panel=self._on_toggle_error_panel,
            on_toggle_log_panel=self._on_toggle_log_panel,
            on_theme=self._on_theme,
            on_log_level=self._on_log_level,
            current_log_level=ctx.settings.log_level,
            on_about=self._on_about,
            on_toggle_panel=self._on_toggle_panel,
        )

        # SessionManager wiring
        ctx.session_manager.session_closed.connect(self._on_session_closed)

        # Background-shutdown bookkeeping (responsive close).
        self._closing = False
        self._teardown_finished.connect(self._finish_quit)

        # Recording-indicator timer: 1-second tick drives blink and elapsed display.
        self._rec_tick: int = 0
        self._rec_timer = QTimer(self)
        self._rec_timer.setInterval(1000)
        self._rec_timer.timeout.connect(self._on_rec_tick)
        self._rec_timer.start()

        # Restore previously saved window geometry.
        self._restore_geometry()

    def _restore_geometry(self) -> None:
        b64 = getattr(self._ctx.settings, "window_geometry_b64", "")
        if not b64:
            return
        try:
            self.restoreGeometry(QByteArray(b64decode(b64.encode("ascii"))))
        except Exception:
            log.exception("failed to restore window geometry")

    def _save_geometry(self) -> None:
        try:
            data = bytes(self.saveGeometry())
            self._ctx.settings.window_geometry_b64 = b64encode(data).decode("ascii")
        except Exception:
            log.exception("failed to save window geometry")

    # ------------------------------------------------------------------
    # Menu handlers
    # ------------------------------------------------------------------

    def _on_new_session(self) -> None:
        log_user_action("Requested a new session")
        dlg = NewSessionDialog(self._ctx.platform, parent=self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            # The dialog logs its own cancel/close action.
            return
        cfg = dlg.result_config()
        if cfg is None:
            return
        prepared_source = dlg.prepared_source()
        # Bridge the visible gap between confirming the dialog and the session
        # tab appearing (charts, log consoles and controller wiring take a
        # moment to build). Mirror the "Connecting to <device>…" feedback with
        # a small "Opening session…" popup. The actual construction is kicked
        # off from inside the popup's own modal exec() loop so the popup is
        # mapped/painted before the GUI thread blocks on building the widgets.
        popup = _BusyDialog(self, "Opening session…")
        popup.setWindowTitle("Opening session")

        def _do_open() -> None:
            try:
                self._open_session_tab(cfg, prepared_source=prepared_source)
            finally:
                popup.accept()

        QTimer.singleShot(80, _do_open)
        popup.exec()
        popup.deleteLater()

    def _open_session_tab(self, cfg: "SessionConfig", prepared_source=None) -> None:
        from data_forwarder_host.session.config import SessionConfig  # noqa: PLC0415
        from data_forwarder_host.pipeline.process_host import (  # noqa: PLC0415
            process_mode_enabled,
        )
        try:
            controller = self._ctx.session_manager.create(
                cfg, prepared_source, use_process=process_mode_enabled()
            )
        except Exception as exc:
            QMessageBox.critical(self, "Session", f"Failed to create session: {exc}")
            return
        tab = SessionTab(controller)
        # Restore previously saved layout (expanded sections, splitter sizes).
        if cfg.layout_state:
            tab.apply_layout_state(cfg.layout_state)
        # Honour the global "Show error panel" preference for new sessions.
        self._set_error_panel_visible(tab, self._actions["show_error"].isChecked())
        # Honour the global View ▸ Panels preferences for new sessions.
        self._apply_panel_states(tab)
        idx = self._tabs.addTab(tab, self._tab_title(cfg.tag, SessionState.CONFIGURED, False))
        self._tabs.setCurrentIndex(idx)
        self._stack.setCurrentIndex(1)
        self._refresh_file_actions()

        # Place the cursor in the recording-label field so the user can type a
        # label immediately. Deferred so it sticks after the tab is
        # shown and activated.
        QTimer.singleShot(0, tab.focus_recording_label)

        # Keep the tab title in sync with session state and recording flag.
        def _on_state(state: SessionState, _tab: SessionTab = tab) -> None:
            self._update_tab_title(_tab)

        def _on_rec(_recording: bool, _tab: SessionTab = tab) -> None:
            self._update_tab_title(_tab)

        controller.state_changed.connect(_on_state)
        controller.recording_changed.connect(_on_rec)

        # Reserve and listen on the device channel immediately so session_info
        # is observed without waiting for the user to press Record.
        try:
            controller.start()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Session",
                f"Could not open device channel: {exc}",
            )

    # ------------------------------------------------------------------
    # Tab title helpers
    # ------------------------------------------------------------------

    def _tab_title(self, tag: str, state: SessionState, recording: bool, ctrl=None) -> str:
        if recording and ctrl is not None:
            elapsed = ctrl.recorder.elapsed_seconds()
            dot = "⏺" if self._rec_tick else "●"
            if elapsed is not None:
                m = int(elapsed) // 60
                s = int(elapsed) % 60
                return f"{dot} {tag}  {m}:{s:02d}"
            return f"{dot} {tag}"
        icon = _ICON_RECORDING if recording else _STATE_ICON.get(state, "○")
        return f"{icon} {tag}"

    def _update_tab_title(self, tab: "SessionTab") -> None:
        idx = self._tabs.indexOf(tab)
        if idx < 0:
            return
        ctrl = tab.controller
        state = ctrl.state()
        recording = ctrl.is_recording()
        title = self._tab_title(ctrl.config.tag, state, recording, ctrl)
        self._tabs.setTabText(idx, title)
        self._tabs.tabBar().setTabTextColor(idx, _tab_color(state, recording, self._rec_tick))

    def _on_rec_tick(self) -> None:
        """Timer callback: toggle blink state and refresh all recording tab titles."""
        self._rec_tick ^= 1
        for i in range(self._tabs.count()):
            w = self._tabs.widget(i)
            if isinstance(w, SessionTab) and w.controller.is_recording():
                self._update_tab_title(w)

    def _on_close_current_session(self) -> None:
        log_user_action("Requested closing the current session")
        idx = self._tabs.currentIndex()
        if idx < 0:
            return
        self._close_tab(idx)

    def _on_tab_close_requested(self, idx: int) -> None:
        log_user_action("Clicked the close button on a session tab")
        self._close_tab(idx)

    def _close_tab(self, idx: int) -> None:
        widget = self._tabs.widget(idx)
        if not isinstance(widget, SessionTab):
            self._tabs.removeTab(idx)
            return
        ctrl = widget.controller
        if not self._confirm_and_finalise(ctrl):
            return
        self._tabs.removeTab(idx)
        self._ctx.session_manager.close(ctrl.id)
        # Switch back to the welcome screen once the last session tab is gone.
        if self._tabs.count() == 0:
            self._stack.setCurrentIndex(0)
        self._refresh_file_actions()

    def _confirm_and_finalise(self, ctrl) -> bool:
        """Pop the 3-option close dialog if needed and finalise the session.

        The data-loss decision dialog is shown **only while a recording is
        actively in progress**. A session that is merely streaming or
        already stopped has nothing unsaved to lose, so it closes silently
        (stopping any live stream first).

        Returns ``True`` if the caller may proceed to close the session,
        ``False`` if the user cancelled.
        """
        recording = ctrl.is_recording()
        if not recording:
            # Nothing unsaved — stop any live stream and allow the close.
            if ctrl.state() == SessionState.RUNNING:
                try:
                    ctrl.stop()
                except Exception:
                    log.exception("error stopping stream on close")
            return True

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Close session")
        box.setText(
            f"Session \u2018{ctrl.config.tag}\u2019 is recording.\n\n"
            "Stop and save the recorded data to its CSV file, discard the "
            "recorded data, or cancel and keep the session open?"
        )
        save_btn = box.addButton("Stop && Save", QMessageBox.ButtonRole.AcceptRole)
        discard_btn = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel_btn)
        box.exec()
        clicked = box.clickedButton()

        if clicked is cancel_btn:
            return False
        try:
            if clicked is save_btn:
                ctrl.stop()                 # auto-writes CSV if recording
            elif clicked is discard_btn:
                ctrl.cancel_recording()
                ctrl.stop()
        except Exception:
            log.exception("error finalising session on close")
        return True

    def _on_session_closed(self, _session_id: str) -> None:
        # Tab is removed in _close_tab already.
        pass

    # ------------------------------------------------------------------
    # File menu — save / open session config
    # ------------------------------------------------------------------

    def _refresh_file_actions(self) -> None:
        has = self._tabs.count() > 0
        self._actions["save_session"].setEnabled(has)
        self._actions["save_session_as"].setEnabled(has)

    def _on_open_session(self) -> None:
        log_user_action("Requested opening a saved session config")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open session configuration", "", "Session config (*.json)"
        )
        if not path:
            log_user_action("Cancelled the open-session dialog")
            return
        from data_forwarder_host.session.config import config_from_dict, ConfigError  # noqa: PLC0415
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            cfg = config_from_dict(data)
        except Exception as exc:
            QMessageBox.critical(self, "Open", f"Failed to load configuration:\n{exc}")
            return
        self._open_session_tab(cfg)

    def _on_save_session(self) -> None:
        log_user_action("Requested saving the session config")
        widget = self._tabs.currentWidget()
        if not isinstance(widget, SessionTab):
            return
        if not hasattr(self, "_last_save_path"):
            self._last_save_path: str | None = None
        if self._last_save_path:
            self._write_session_config(widget, self._last_save_path)
        else:
            self._on_save_session_as()

    def _on_save_session_as(self) -> None:
        log_user_action("Requested saving the session config as a new file")
        widget = self._tabs.currentWidget()
        if not isinstance(widget, SessionTab):
            return
        default = f"{widget.controller.config.tag}.json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save session configuration", default, "Session config (*.json)"
        )
        if path:
            self._last_save_path = path
            self._write_session_config(widget, path)

    def _write_session_config(self, tab: SessionTab, path: str) -> None:
        from dataclasses import replace as dc_replace  # noqa: PLC0415
        from data_forwarder_host.session.config import config_to_dict  # noqa: PLC0415
        try:
            cfg = tab.controller.current_config()
            layout_state = tab.get_layout_state()
            if layout_state:
                cfg = dc_replace(cfg, layout_state=layout_state)
            Path(path).write_text(
                json.dumps(config_to_dict(cfg), indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save", f"Failed to save:\n{exc}")

    def _on_toggle_error_panel(self, on: bool) -> None:
        log_user_action("Toggled the error panel %s", "on" if on else "off")
        # Apply to every open session, not just the current tab, so the View
        # menu acts as a global preference across all sessions.
        for i in range(self._tabs.count()):
            w = self._tabs.widget(i)
            if isinstance(w, SessionTab):
                self._set_error_panel_visible(w, on)

    @staticmethod
    def _set_error_panel_visible(tab: SessionTab, on: bool) -> None:
        # The error panel is the last widget in the tab's outer layout.
        for i in range(tab.layout().count()):
            w = tab.layout().itemAt(i).widget()
            if w is not None and w.__class__.__name__ == "ErrorPanel":
                w.setVisible(on)

    # ------------------------------------------------------------------
    # View ▸ Panels (global per-panel visibility preference)
    # ------------------------------------------------------------------

    _PANEL_KEYS = ("combined", "individual", "channel_ascii", "decoded_frames")

    def _apply_panel_states(self, tab: SessionTab) -> None:
        """Initialise a tab's panel visibility from the current menu states."""
        for key in self._PANEL_KEYS:
            act = self._actions.get(f"panel_{key}")
            if act is not None:
                tab.set_panel_visible(key, act.isChecked())

    def _on_toggle_panel(self, key: str, on: bool) -> None:
        log_user_action("Toggled the %s panel %s", key, "on" if on else "off")
        # Global preference: apply to every open session tab.
        for i in range(self._tabs.count()):
            w = self._tabs.widget(i)
            if isinstance(w, SessionTab):
                w.set_panel_visible(key, on)

    def _on_toggle_log_panel(self, on: bool) -> None:
        log_user_action("Toggled the application log panel %s", "on" if on else "off")
        self._log_dock.setVisible(on)

    def _on_log_level(self, level: str) -> None:
        log_user_action("Changed the log level to %s", level)
        configure_logging(level)
        self._ctx.settings.log_level = level
        log.info("log level changed to %s", level)

    def _on_about(self) -> None:
        log_user_action("Opened the About dialog")
        AboutDialog(self).exec()

    def _on_theme(self, theme: "Theme") -> None:
        log_user_action("Changed the theme to %s", getattr(theme, "name", theme))
        apply_theme(theme)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        log_user_action("Clicked the application close button")

        # Re-entrant close (e.g. the implicit close after the background
        # teardown quits the app): just let it through.
        if self._closing:
            event.accept()
            return

        # Ask per session that still has an active stream or recording. This
        # (and the decision dialog) must stay on the GUI thread.
        for i in range(self._tabs.count()):
            w = self._tabs.widget(i)
            if not isinstance(w, SessionTab):
                continue
            ctrl = w.controller
            if ctrl.state() == SessionState.RUNNING or ctrl.is_recording():
                if not self._confirm_and_finalise(ctrl):
                    log_user_action("Cancelled application close")
                    event.ignore()
                    return

        # Make the GUI disappear immediately for a responsive feel, then finish
        # tearing the sessions down on a background thread (closing a device —
        # especially Bluetooth LE — can block for seconds). The app quits once
        # teardown completes.
        self._closing = True
        log_user_action("Closing application; finalising sessions in background")
        self._save_geometry()
        self.hide()
        QApplication.processEvents()

        threading.Thread(
            target=self._background_teardown, name="dfh-shutdown", daemon=True
        ).start()
        event.ignore()  # stay alive until the background teardown asks us to quit

    def _background_teardown(self) -> None:
        """Tear down all sessions off the GUI thread, then request quit."""
        try:
            self._ctx.session_manager.shutdown_all()
        except Exception:
            log.exception("error during background shutdown")
        finally:
            self._teardown_finished.emit()

    def _finish_quit(self) -> None:
        """Runs on the GUI thread once background teardown is complete."""
        log_user_action("Application teardown complete; exiting")
        QApplication.instance().quit()
