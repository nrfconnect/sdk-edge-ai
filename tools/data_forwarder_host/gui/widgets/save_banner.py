# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Always-on "last saved CSV" panel.

A small permanent sub-window at the bottom of the session tab that reports the
most recently saved CSV file. It is shown at all times (even before the first
save, where it states that nothing has been saved yet). The saved path is
selectable text so it can be copied with the cursor. Two actions are offered:
**Open File** and **Open Containing Folder**. There is no dismiss control.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStyle,
    QWidget,
)

from data_forwarder_host.utils.open_path import open_containing_folder, open_file
from data_forwarder_host.utils.logging import log_user_action

_PLACEHOLDER = "No recording saved yet."


class SaveBanner(QWidget):
    """A permanent panel reporting the most recently saved CSV file."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._path: str | None = None

        self.setObjectName("saveBanner")
        self.setStyleSheet(
            "#saveBanner { background:palette(alternate-base); "
            "border:1px solid palette(mid); border-radius:6px; }"
        )

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(8)

        self._label = QLabel(_PLACEHOLDER)
        self._label.setStyleSheet("color:#2EA043;")
        # The saved path must be selectable so the user can copy it with the
        # cursor.
        self._label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self._label.setCursor(Qt.CursorShape.IBeamCursor)
        lay.addWidget(self._label, 1)

        self._btn_open = QPushButton("Open File")
        self._btn_open.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogYesButton)
        )
        self._btn_folder = QPushButton("Open Containing Folder")
        self._btn_folder.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        )
        lay.addWidget(self._btn_open)
        lay.addWidget(self._btn_folder)

        self._btn_open.clicked.connect(self._on_open_file)
        self._btn_folder.clicked.connect(self._on_open_folder)

        # No file saved yet: actions are disabled until the first save.
        self._set_actions_enabled(False)
        # Always visible — there is no dismiss control.
        self.setVisible(True)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def show_saved(self, path: str) -> None:
        """Record *path* as the last saved file and refresh the panel."""
        self._path = path
        self._label.setText(f"Saved: {path}")
        self._set_actions_enabled(True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _set_actions_enabled(self, enabled: bool) -> None:
        self._btn_open.setEnabled(enabled)
        self._btn_folder.setEnabled(enabled)

    def _on_open_file(self) -> None:
        log_user_action("Clicked Open File in the save panel")
        path = self._path
        if not path:
            return
        if QDesktopServices.openUrl(QUrl.fromLocalFile(path)):
            return
        if open_file(path):
            return
        QMessageBox.information(
            self,
            "Open File",
            f"Could not launch a viewer automatically.\n\nThe file is saved at:\n{path}",
        )

    def _on_open_folder(self) -> None:
        log_user_action("Clicked Open Containing Folder in the save panel")
        path = self._path
        if not path:
            return
        if open_containing_folder(path):
            return
        QMessageBox.information(
            self,
            "Open Containing Folder",
            f"Could not open the folder automatically.\n\nThe file is saved at:\n{path}",
        )
