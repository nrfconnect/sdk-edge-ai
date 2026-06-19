# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Menu construction for the main window."""

from __future__ import annotations

from typing import Callable

from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMainWindow, QMenu

from data_forwarder_host.gui.theme import Theme


def build_menus(
    window: QMainWindow,
    *,
    on_new_session: Callable[[], None],
    on_close_session: Callable[[], None],
    on_open_session: Callable[[], None],
    on_save_session: Callable[[], None],
    on_save_session_as: Callable[[], None],
    on_quit: Callable[[], None],
    on_toggle_error_panel: Callable[[bool], None],
    on_toggle_log_panel: Callable[[bool], None],
    on_theme: Callable[[Theme], None],
    on_log_level: Callable[[str], None],
    current_log_level: str = "INFO",
    on_about: Callable[[], None],
    on_toggle_panel: Callable[[str, bool], None] | None = None,
) -> dict[str, QAction]:
    """Build the menu bar; return a dict of actions for state toggling."""
    bar = window.menuBar()

    # ── File ──────────────────────────────────────────────────────────
    m_file: QMenu = bar.addMenu("&File")

    act_quit = QAction("&Quit", window)
    act_quit.triggered.connect(on_quit)
    m_file.addAction(act_quit)

    # ── Session ───────────────────────────────────────────────────────
    m_session: QMenu = bar.addMenu("&Session")

    act_new = QAction("&New session…", window)
    act_new.setShortcut("Ctrl+N")
    act_new.triggered.connect(on_new_session)
    m_session.addAction(act_new)

    act_close = QAction("&Close session", window)
    act_close.triggered.connect(on_close_session)
    m_session.addAction(act_close)

    m_session.addSeparator()

    act_open = QAction("&Open session config…", window)
    act_open.setShortcut("Ctrl+O")
    act_open.triggered.connect(on_open_session)
    m_session.addAction(act_open)

    act_save = QAction("&Save session config", window)
    act_save.setShortcut("Ctrl+S")
    act_save.setEnabled(False)          # enabled once a session exists
    act_save.triggered.connect(on_save_session)
    m_session.addAction(act_save)

    act_save_as = QAction("Save session config &as…", window)
    act_save_as.setShortcut("Ctrl+Shift+S")
    act_save_as.setEnabled(False)
    act_save_as.triggered.connect(on_save_session_as)
    m_session.addAction(act_save_as)

    # View
    m_view: QMenu = bar.addMenu("&View")
    act_err = QAction("Show error panel", window, checkable=True)
    act_err.setChecked(False)
    act_err.toggled.connect(on_toggle_error_panel)
    m_view.addAction(act_err)

    act_log = QAction("Show log panel", window, checkable=True)
    act_log.setChecked(False)
    act_log.toggled.connect(on_toggle_log_panel)
    m_view.addAction(act_log)

    # Panels submenu — per-panel visibility toggles applied to every open
    # session tab as a global preference. Combined View and Individual Channels
    # are on by default; Channel ASCII and Decoded Frames are off by default.
    m_panels = m_view.addMenu("Panels")
    panel_actions: dict[str, QAction] = {}
    _PANELS = (
        ("combined", "Combined View", True),
        ("individual", "Individual Channels", True),
        ("channel_ascii", "Channel ASCII", False),
        ("decoded_frames", "Decoded Frames", False),
    )
    for key, label, default_on in _PANELS:
        act = QAction(label, window, checkable=True)
        act.setChecked(default_on)
        if on_toggle_panel is not None:
            act.toggled.connect(lambda on, k=key: on_toggle_panel(k, on))
        m_panels.addAction(act)
        panel_actions[f"panel_{key}"] = act

    m_theme = m_view.addMenu("Theme")
    group = QActionGroup(window)
    for label, theme in (("System", Theme.SYSTEM), ("Light", Theme.LIGHT), ("Dark", Theme.DARK)):
        act = QAction(label, window, checkable=True)
        group.addAction(act)
        m_theme.addAction(act)
        if theme == Theme.SYSTEM:
            act.setChecked(True)
        act.triggered.connect(lambda _checked=False, t=theme: on_theme(t))

    m_log_level = m_view.addMenu("Log level")
    log_group = QActionGroup(window)
    log_group.setExclusive(True)
    _LOG_LEVELS = (("DEBUG", "DEBUG"), ("INFO", "INFO"), ("WARNING", "WARNING"), ("ERROR", "ERROR"))
    for label, lvl in _LOG_LEVELS:
        act = QAction(label, window, checkable=True)
        act.setChecked(lvl == current_log_level.upper())
        log_group.addAction(act)
        m_log_level.addAction(act)
        act.triggered.connect(lambda _checked=False, l=lvl: on_log_level(l))

    # Help
    m_help: QMenu = bar.addMenu("&Help")
    act_about = QAction("&About…", window)
    act_about.triggered.connect(on_about)
    m_help.addAction(act_about)

    return {
        "new_session": act_new,
        "close_session": act_close,
        "open_session": act_open,
        "save_session": act_save,
        "save_session_as": act_save_as,
        "quit": act_quit,
        "show_error": act_err,
        "show_log": act_log,
        "about": act_about,
        **panel_actions,
    }
