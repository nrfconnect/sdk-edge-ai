# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""CollapsibleSection — a clickable header that shows or hides a content widget.

Usage::

    chart = CombinedPlot(model)
    section = CollapsibleSection("Combined View", chart, expanded=True)
    layout.addWidget(section)

The header row shows a disclosure chevron (▸ collapsed / ▾ expanded).
Clicking anywhere on the header row toggles the content.

When collapsed the widget shrinks to the header height so that the parent
layout (splitter, VBox …) reclaims the freed vertical space immediately.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPushButton, QSizePolicy, QVBoxLayout, QWidget

from data_forwarder_host.utils.logging import log_user_action


class CollapsibleSection(QWidget):
    """Header-toggle widget wrapping any QWidget as collapsible content.

    Parameters
    ----------
    title:
        Text shown in the header row next to the chevron.
    content:
        The widget to show/hide.  It is reparented to this section.
    expanded:
        Initial state.  ``True`` = content visible.
    """

    #: Emitted after every toggle with the new expanded state.
    toggled = Signal(bool)

    _HDR_H: int = 26   # fixed header row height (pixels)

    def __init__(
        self,
        title: str,
        content: QWidget,
        *,
        expanded: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header button ───────────────────────────────────────────────
        self._btn = QPushButton()
        self._btn.setCheckable(True)
        self._btn.setChecked(expanded)
        self._btn.setFlat(True)
        self._btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._btn.setFixedHeight(self._HDR_H)
        self._btn.setStyleSheet(
            "QPushButton {"
            "  text-align: left;"
            "  padding: 2px 8px;"
            "  font-weight: bold;"
            "  border: none;"
            "  border-bottom: 1px solid palette(mid);"
            "  background: palette(button);"
            "}"
            "QPushButton:hover { background: palette(midlight); }"
        )
        self._btn.clicked.connect(self._on_toggle)
        outer.addWidget(self._btn)

        # ── Content ─────────────────────────────────────────────────────
        self._content = content
        outer.addWidget(content)

        # ── Apply initial state ─────────────────────────────────────────
        self._title = title
        self._apply(expanded, emit=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_expanded(self, on: bool) -> None:
        """Programmatically expand or collapse the section."""
        self._btn.setChecked(on)
        self._apply(on)

    def is_expanded(self) -> bool:
        return self._btn.isChecked()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_toggle(self, checked: bool) -> None:
        log_user_action(
            "%s section %r", "Expanded" if checked else "Collapsed", self._title
        )
        self._apply(checked)

    def _apply(self, expanded: bool, *, emit: bool = True) -> None:
        chevron = "▾" if expanded else "▸"
        self._btn.setText(f"  {chevron}  {self._title}")
        self._content.setVisible(expanded)

        # Constrain the section's maximum height so the parent layout
        # reclaims vertical space while the content is hidden.
        if expanded:
            self.setMaximumHeight(16_777_215)   # QWIDGETSIZE_MAX — unconstrained
        else:
            self.setMaximumHeight(self._HDR_H + 2)

        if emit:
            self.toggled.emit(expanded)
