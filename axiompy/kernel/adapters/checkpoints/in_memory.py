"""In-memory checkpoint store."""

from __future__ import annotations

from typing import Dict, Optional

from axiompy.kernel.domain.models import CheckpointState
from axiompy.kernel.domain.ports import CheckpointStore


class InMemoryCheckpointStore(CheckpointStore):
    """Store checkpoints in memory."""

    def __init__(self) -> None:
        self._states: Dict[str, CheckpointState] = {}

    def save(self, state: CheckpointState) -> None:
        self._states[state.run_id] = state

    def load(self, run_id: str) -> Optional[CheckpointState]:
        return self._states.get(run_id)
