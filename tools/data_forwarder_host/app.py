# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Application bootstrap — Settings, AppContext, Application, main().

There is no command-line interface and no headless mode: the application is a
GUI launched via ``python3 -m data_forwarder_host``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field

from data_forwarder_host.platform.base import PlatformAdapter
from data_forwarder_host.platform.factory import detect_platform
from data_forwarder_host.session.manager import SessionManager
from data_forwarder_host.utils.logging import configure_logging
from data_forwarder_host.utils.paths import settings_file
from data_forwarder_host.utils.version import get_version

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Linux display helpers
# ---------------------------------------------------------------------------


def _find_xauthority() -> str | None:
    """Return the X authority file path by scanning /proc for a user process
    that already has XAUTHORITY set (e.g. gnome-shell, gnome-session).

    This is needed in terminals (such as the nRF Connect SDK terminal in VS
    Code) that inherit DISPLAY from the compositor session but not the
    per-session XAUTHORITY path.
    """
    import glob

    uid = os.getuid()
    for env_path in glob.iglob("/proc/*/environ"):
        try:
            if os.stat(env_path).st_uid != uid:
                continue
            with open(env_path, "rb") as fh:
                for token in fh.read().split(b"\x00"):
                    if token.startswith(b"XAUTHORITY="):
                        path = token[11:].decode(errors="replace")
                        if os.path.isfile(path):
                            return path
        except OSError:
            continue
    return None


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    """Persistent application settings.

    Spill thresholds are internal recorder constants and are intentionally NOT
    represented here.
    """

    window_geometry_b64: str = ""
    theme: str = "SYSTEM"
    plot_window_seconds_default: float = 10.0
    log_level: str = "INFO"

    def save(self) -> None:
        try:
            settings_file().write_text(
                json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8"
            )
        except OSError:
            log.exception("failed to save settings")

    @classmethod
    def load(cls) -> "Settings":
        path = settings_file()
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.exception("failed to load settings; using defaults")
            return cls()
        s = cls()
        for k, v in data.items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s


# ---------------------------------------------------------------------------
# AppContext
# ---------------------------------------------------------------------------


@dataclass
class AppContext:
    version: str
    platform: PlatformAdapter
    settings: Settings
    session_manager: SessionManager = field(default_factory=SessionManager)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class Application:
    """Owns the AppContext and the Qt QApplication."""

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx

    def run(self) -> int:
        # ── Linux display / platform bootstrap ────────────────────────────
        # Must happen BEFORE Qt is imported so that dlopen() calls made by the
        # Qt platform plugins pick up the correct libraries.
        if sys.platform.startswith("linux"):
            _sys_lib = "/usr/lib/x86_64-linux-gnu"
            if os.path.isdir(_sys_lib):
                _ld = os.environ.get("LD_LIBRARY_PATH", "")
                if _sys_lib not in _ld.split(":"):
                    os.environ["LD_LIBRARY_PATH"] = (
                        f"{_sys_lib}:{_ld}" if _ld else _sys_lib
                    )

            if os.environ.get("WAYLAND_DISPLAY"):
                os.environ.setdefault("QT_QPA_PLATFORM", "wayland")
                _decor_dir = f"{_sys_lib}/libdecor/plugins-1"
                if os.path.isdir(_decor_dir):
                    os.environ.setdefault("LIBDECOR_PLUGIN_DIR", _decor_dir)
            elif os.environ.get("DISPLAY"):
                if not os.environ.get("XAUTHORITY"):
                    _xauth = _find_xauthority()
                    if _xauth:
                        os.environ["XAUTHORITY"] = _xauth
                os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

        from PySide6.QtWidgets import QApplication

        from data_forwarder_host.gui.main_window import MainWindow
        from data_forwarder_host.gui.theme import Theme, apply_theme

        qt_app = QApplication.instance() or QApplication(sys.argv)
        qt_app.setApplicationName("Data Forwarder")
        qt_app.setOrganizationName("Nordic Semiconductor")
        # Force the Fusion style so our explicit light/dark palettes are honoured
        # consistently across platforms. The native styles (e.g. GTK/Adwaita on
        # Linux, the Windows style) only partially apply a custom QPalette, which
        # is what produced the unreadable dark-background / dark-text mix.
        qt_app.setStyle("Fusion")

        try:
            theme = Theme[self.ctx.settings.theme]
        except KeyError:
            theme = Theme.SYSTEM
        apply_theme(theme)

        window = MainWindow(self.ctx)
        window.show()
        exit_code = qt_app.exec()
        self.ctx.settings.save()
        return int(exit_code)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """GUI entry point. Returns a process exit code."""
    settings = Settings.load()
    configure_logging(settings.log_level)
    ctx = AppContext(
        version=get_version(),
        platform=detect_platform(),
        settings=settings,
    )
    app = Application(ctx)
    try:
        return app.run()
    except Exception as exc:  # pragma: no cover
        log.exception("fatal: %s", exc)
        return 1
