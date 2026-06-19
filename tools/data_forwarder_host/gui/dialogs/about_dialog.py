# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""About dialog."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from data_forwarder_host.utils.logging import log_user_action
from data_forwarder_host.utils.version import get_version


class AboutDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About Data Forwarder")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<b>Data Forwarder Host</b><br>v{get_version()}"))
        layout.addWidget(QLabel(
            "Copyright © 2026 Nordic Semiconductor ASA<br>"
            "License: LicenseRef-Nordic-5-Clause"
        ))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        bb.accepted.connect(self._on_ok)
        layout.addWidget(bb)

    def _on_ok(self) -> None:
        log_user_action("Clicked OK in the About dialog")
        self.accept()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        log_user_action("Closed the About dialog")
        super().closeEvent(event)
