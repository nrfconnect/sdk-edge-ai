# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Session tab strip with a browser-style "new tab" (+) button.

:class:`SessionTabWidget` is a :class:`~PySide6.QtWidgets.QTabWidget` that hosts
one tab per recording session and, like Google Chrome / Firefox, shows a small
``+`` button immediately to the right of the last tab. The button is **not** a
tab: it never holds page content, cannot be dragged, reordered or closed, and is
shown **only while at least one session tab exists**. Clicking it emits
:attr:`new_session_requested`; the owner connects that to its New-Session flow.

The widget keeps the button glued to the right edge of the last tab as tabs are
added, removed, renamed or the strip is resized, by repositioning it whenever
the underlying tab bar re-lays-out.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import QTabBar, QTabWidget, QToolButton


class _RelayoutTabBar(QTabBar):
    """A :class:`QTabBar` that signals whenever its tab layout changes.

    The ``+`` button position depends on the last tab's rectangle, which moves
    on insert/remove/rename/resize. Qt recomputes that layout in
    :meth:`tabLayoutChange` and :meth:`resizeEvent`; we surface both as a single
    :attr:`relaid_out` signal so the owning :class:`SessionTabWidget` can keep
    the button glued to the last tab.
    """

    relaid_out = Signal()

    def tabLayoutChange(self) -> None:  # noqa: N802 (Qt override)
        super().tabLayoutChange()
        self.relaid_out.emit()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self.relaid_out.emit()


class SessionTabWidget(QTabWidget):
    """Tab widget whose trailing ``+`` button requests a new session.

    Public interface:
      * :attr:`new_session_requested` — emitted on a genuine ``+`` click.
      * standard :class:`QTabWidget` API for adding/removing session tabs.

    The ``+`` button is a child of the tab widget (not of the bar) so it stays
    visible even when the bar is only as wide as its tabs, and it is hidden
    whenever no tabs remain.
    """

    new_session_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        bar = _RelayoutTabBar(self)
        self.setTabBar(bar)
        bar.relaid_out.connect(self._reposition_plus)

        self._plus = QToolButton(self)
        self._plus.setText("+")
        self._plus.setAutoRaise(True)
        self._plus.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._plus.setCursor(Qt.CursorShape.PointingHandCursor)
        self._plus.setToolTip("New session  (Ctrl+N)")
        self._plus.clicked.connect(self.new_session_requested)
        self._plus.hide()

    # ------------------------------------------------------------------
    # Plus-button placement
    # ------------------------------------------------------------------
    def _reposition_plus(self) -> None:
        bar = self.tabBar()
        n = bar.count()
        if n == 0:
            self._plus.hide()
            return
        last = bar.tabRect(n - 1)
        if last.isNull() or last.isEmpty():
            self._plus.hide()
            return
        side = max(16, last.height() - 6)
        self._plus.setFixedSize(side, side)
        top_left = QPoint(last.right() + 4, last.top() + (last.height() - side) // 2)
        self._plus.move(bar.mapTo(self, top_left))
        self._plus.show()
        self._plus.raise_()

    # Qt overrides that change the tab layout without going through the bar.
    def tabInserted(self, index: int) -> None:  # noqa: N802 (Qt override)
        super().tabInserted(index)
        self._reposition_plus()

    def tabRemoved(self, index: int) -> None:  # noqa: N802 (Qt override)
        super().tabRemoved(index)
        self._reposition_plus()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._reposition_plus()
