"""Redis-backed memory store (optional extra)."""

from __future__ import annotations

import json
from typing import List, Optional

from axiompy.kernel.domain.models import Message
from axiompy.kernel.domain.ports import MemoryStore


class RedisMemoryStore(MemoryStore):
    """Session memory using Redis."""

    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "redis is required. Install with: pip install axiompy-agents[memory-redis]"
            ) from exc
        self._client = redis.from_url(url)

    def _key(self, key: str, session_id: Optional[str]) -> str:
        return f"{session_id}:{key}" if session_id else key

    def get(self, key: str, session_id: Optional[str] = None) -> Optional[str]:
        raw = self._client.get(self._key(key, session_id))
        return raw.decode() if raw else None

    def put(self, key: str, value: str, session_id: Optional[str] = None) -> None:
        self._client.set(self._key(key, session_id), value)

    def append_message(self, session_id: str, message: Message) -> None:
        list_key = f"session:{session_id}:messages"
        payload = json.dumps(
            {
                "role": message.role,
                "content": message.content,
                "tool_call_id": message.tool_call_id,
            }
        )
        self._client.rpush(list_key, payload)

    def get_messages(self, session_id: str) -> List[Message]:
        list_key = f"session:{session_id}:messages"
        raw_items = self._client.lrange(list_key, 0, -1)
        messages: List[Message] = []
        for raw in raw_items:
            data = json.loads(raw)
            messages.append(
                Message(
                    role=data["role"],
                    content=data["content"],
                    tool_call_id=data.get("tool_call_id"),
                )
            )
        return messages
