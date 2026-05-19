"""Arrow-backed dataset service using axiompy-data consuming clients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from axiompy.reasoning.base import BaseDatasetService
from axiompy.reasoning.metadata import DatasetMetadata

if TYPE_CHECKING:
    from axiompy.data.consuming.base import Client as ConsumingClient


class ArrowDatasetService(BaseDatasetService):
    """
    :class:`BaseDatasetService` implementation over :mod:`axiompy.data.consuming`.

    Converts columnar :class:`~axiompy.data.consuming.results.QueryResult` rows
    to ``list[dict]`` for :class:`~axiompy.reasoning.agents.query.QueryAgent`.

    Requires optional extra: ``pip install axiompy-agents[reasoning-data]``.
    """

    def __init__(
        self,
        client: ConsumingClient,
        metadata: DatasetMetadata,
        capabilities: Optional[list[str]] = None,
        default_limit: int = 100,
    ) -> None:
        """
        Args:
            client: Analytical SQL client from ``axiompy.data.consuming.Factory``
            metadata: Dataset metadata for SQL generation and routing
            capabilities: Capability strings for agent routing
            default_limit: Default row cap when ``limit`` is not passed to :meth:`query`
        """
        self._client = client
        self._metadata = metadata
        self._capabilities = capabilities or []
        self._default_limit = default_limit
        self.dataset_name = metadata.dataset
        self.description = metadata.description

    def query(self, sql: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Execute SQL and return rows as dictionaries."""
        row_limit = limit if limit is not None else self._default_limit
        sql_stripped = sql.strip().rstrip(";")
        if row_limit > 0 and " limit " not in sql_stripped.lower():
            sql_stripped = f"{sql_stripped} LIMIT {row_limit}"

        result = self._client.query(sql_stripped)
        frame = result.to_pandas()
        return frame.to_dict(orient="records")

    def get_capabilities(self) -> list[str]:
        return list(self._capabilities)

    def get_metadata(self) -> DatasetMetadata:
        return self._metadata
