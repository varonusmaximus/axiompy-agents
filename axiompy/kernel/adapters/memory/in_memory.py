"""In-memory session store."""

from __future__ import annotations

from typing import Dict, List, Optional

from axiompy.kernel.domain.models import Message
from axiompy.kernel.domain.ports import MemoryStore


class InMemoryMemoryStore(MemoryStore):
    """Memory store backed by dicts."""

    def __init__(self) -> None:
        self._kv: Dict[str, str] = {}
        self._sessions: Dict[str, List[Message]] = {}

    def get(self, key: str, session_id: Optional[str] = None) -> Optional[str]:
        full_key = f"{session_id}:{key}" if session_id else key
        return self._kv.get(full_key)

    def put(self, key: str, value: str, session_id: Optional[str] = None) -> None:
        full_key = f"{session_id}:{key}" if session_id else key
        self._kv[full_key] = value

    def append_message(self, session_id: str, message: Message) -> None:
        self._sessions.setdefault(session_id, []).append(message)

    def get_messages(self, session_id: str) -> List[Message]:
        return list(self._sessions.get(session_id, []))
