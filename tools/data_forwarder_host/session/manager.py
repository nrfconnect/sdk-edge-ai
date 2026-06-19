# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""App-level SessionController registry."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from data_forwarder_host.session.config import SessionConfig
from data_forwarder_host.session.controller import SessionController


class SessionManager(QObject):
    """Owns ``SessionController`` instances by id."""

    session_created = Signal(str)               # session_id
    session_closed = Signal(str)                # session_id

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._sessions: dict[str, SessionController] = {}

    def create(
        self,
        config: SessionConfig,
        prepared_source: object | None = None,
        use_process: bool = False,
    ) -> SessionController:
        ctrl = SessionController(
            config,
            parent=self,
            prepared_source=prepared_source,
            use_process=use_process,
        )
        self._sessions[ctrl.id] = ctrl
        self.session_created.emit(ctrl.id)
        return ctrl

    def close(self, session_id: str) -> None:
        ctrl = self._sessions.pop(session_id, None)
        if ctrl is None:
            return
        try:
            ctrl.shutdown()
        finally:
            self.session_closed.emit(session_id)
            ctrl.deleteLater()

    def get(self, session_id: str) -> SessionController:
        return self._sessions[session_id]

    def all(self) -> list[SessionController]:
        return list(self._sessions.values())

    def shutdown_all(self) -> None:
        for sid in list(self._sessions.keys()):
            self.close(sid)
