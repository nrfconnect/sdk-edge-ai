# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Simple light/dark theme palette helpers."""

from __future__ import annotations

from enum import Enum, auto

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


class Theme(Enum):
    LIGHT = auto()
    DARK = auto()
    SYSTEM = auto()


def _detect_system_theme(app: QApplication) -> Theme:
    """Resolve ``SYSTEM`` to ``LIGHT``/``DARK`` from the OS color scheme.

    Relies on ``QStyleHints.colorScheme()`` (Qt 6.5+). When the platform cannot
    report a scheme (``Qt.ColorScheme.Unknown``) we fall back to ``LIGHT`` so the
    app is always readable rather than inheriting a half-applied dark desktop
    palette (dark window background with dark default text).
    """
    try:
        scheme = app.styleHints().colorScheme()
    except (AttributeError, RuntimeError):
        return Theme.LIGHT
    return Theme.DARK if scheme == Qt.ColorScheme.Dark else Theme.LIGHT


def _build_palette(roles: dict[QPalette.ColorRole, QColor], disabled_text: QColor) -> QPalette:
    """Assemble a full :class:`QPalette` and dim its disabled text group.

    Every shade role (``Mid``, ``Midlight``, ``Light``, ``Dark``, ``Shadow``,
    ``PlaceholderText`` …) is set explicitly so that ``palette(...)`` references
    in widget stylesheets resolve to theme-correct colours. Leaving them unset
    is the classic dark-theme bug: Qt falls back to the *default* (light) shades,
    so ``palette(mid)`` borders/labels stay light-grey on a dark window.
    """
    palette = QPalette()
    for role, color in roles.items():
        palette.setColor(role, color)
    for group in (
        QPalette.ColorGroup.Disabled,
    ):
        palette.setColor(group, QPalette.ColorRole.WindowText, disabled_text)
        palette.setColor(group, QPalette.ColorRole.Text, disabled_text)
        palette.setColor(group, QPalette.ColorRole.ButtonText, disabled_text)
    return palette


def _light_palette() -> QPalette:
    return _build_palette(
        {
            QPalette.ColorRole.Window: QColor(245, 245, 245),
            QPalette.ColorRole.WindowText: QColor(20, 20, 20),
            QPalette.ColorRole.Base: QColor(255, 255, 255),
            QPalette.ColorRole.AlternateBase: QColor(235, 235, 235),
            QPalette.ColorRole.ToolTipBase: QColor(255, 255, 225),
            QPalette.ColorRole.ToolTipText: QColor(20, 20, 20),
            QPalette.ColorRole.PlaceholderText: QColor(120, 120, 120),
            QPalette.ColorRole.Text: QColor(20, 20, 20),
            QPalette.ColorRole.Button: QColor(240, 240, 240),
            QPalette.ColorRole.ButtonText: QColor(20, 20, 20),
            QPalette.ColorRole.BrightText: QColor(200, 0, 0),
            QPalette.ColorRole.Link: QColor(0, 90, 200),
            QPalette.ColorRole.LinkVisited: QColor(110, 60, 180),
            QPalette.ColorRole.Highlight: QColor(45, 110, 200),
            QPalette.ColorRole.HighlightedText: QColor(255, 255, 255),
            QPalette.ColorRole.Light: QColor(255, 255, 255),
            QPalette.ColorRole.Midlight: QColor(248, 248, 248),
            QPalette.ColorRole.Mid: QColor(150, 150, 150),
            QPalette.ColorRole.Dark: QColor(120, 120, 120),
            QPalette.ColorRole.Shadow: QColor(90, 90, 90),
        },
        disabled_text=QColor(160, 160, 160),
    )


def _dark_palette() -> QPalette:
    return _build_palette(
        {
            QPalette.ColorRole.Window: QColor(45, 45, 45),
            QPalette.ColorRole.WindowText: QColor(220, 220, 220),
            QPalette.ColorRole.Base: QColor(30, 30, 30),
            QPalette.ColorRole.AlternateBase: QColor(55, 55, 55),
            QPalette.ColorRole.ToolTipBase: QColor(60, 60, 60),
            QPalette.ColorRole.ToolTipText: QColor(220, 220, 220),
            QPalette.ColorRole.PlaceholderText: QColor(140, 140, 140),
            QPalette.ColorRole.Text: QColor(220, 220, 220),
            QPalette.ColorRole.Button: QColor(60, 60, 60),
            QPalette.ColorRole.ButtonText: QColor(220, 220, 220),
            QPalette.ColorRole.BrightText: QColor(255, 90, 90),
            QPalette.ColorRole.Link: QColor(94, 160, 255),
            QPalette.ColorRole.LinkVisited: QColor(170, 130, 255),
            QPalette.ColorRole.Highlight: QColor(53, 120, 224),
            QPalette.ColorRole.HighlightedText: QColor(255, 255, 255),
            QPalette.ColorRole.Light: QColor(75, 75, 75),
            QPalette.ColorRole.Midlight: QColor(62, 62, 62),
            QPalette.ColorRole.Mid: QColor(120, 120, 120),
            QPalette.ColorRole.Dark: QColor(28, 28, 28),
            QPalette.ColorRole.Shadow: QColor(15, 15, 15),
        },
        disabled_text=QColor(120, 120, 120),
    )


def apply_theme(theme: Theme) -> None:
    app = QApplication.instance()
    if app is None:
        return
    # Always apply an explicit, self-consistent palette — including for SYSTEM,
    # where ``style().standardPalette()`` returns a fixed light palette that does
    # not follow the OS scheme, producing an unreadable dark-background /
    # dark-text mix on a dark Ubuntu desktop.
    resolved = _detect_system_theme(app) if theme == Theme.SYSTEM else theme
    palette = _dark_palette() if resolved == Theme.DARK else _light_palette()
    app.setPalette(palette)


# Distinct, colour-blind-aware palette for channel traces.
CHANNEL_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def color_for_channel(idx: int) -> str:
    return CHANNEL_COLORS[idx % len(CHANNEL_COLORS)]
