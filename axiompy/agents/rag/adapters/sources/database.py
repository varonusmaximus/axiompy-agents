"""Database Document Source.

Load documents from database tables/queries using axiompy DatabaseFactory.

Supports:
- PostgreSQL, MySQL, SQLite, DynamoDB
- Custom SQL queries as document sources
- Table scanning with configurable columns

Example:
    from axiompy.agents.rag.adapters.sources import DatabaseSource
    from axiompy.io.database import DatabaseSettings, DatabaseType

    settings = DatabaseSettings(
        host="localhost",
        port=5432,
        database="mydb",
        username="user",
        password="pass",
    )
    source = DatabaseSource(
        database_type=DatabaseType.POSTGRES,
        settings=settings,
    )

    # Load from a table
    docs = source.load_from_table(
        table="articles",
        content_column="body",
        id_column="id",
        title_column="title",
    )

    # Load from a custom query
    docs = source.load_from_query(
        query="SELECT id, title, content FROM posts WHERE status = 'published'",
        content_column="content",
        id_column="id",
        title_column="title",
    )
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from axiompy.agents.rag.domain.models import Document, DocumentMetadata
from axiompy.agents.rag.errors import RAGIngestionError
from axiompy.io.database import Database, DatabaseFactory, DatabaseSettings, DatabaseType
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)


class DatabaseSource:
    """
    Document source for loading content from database tables.

    Uses axiompy DatabaseFactory for database operations.

    Attributes:
        database_type: Type of database (POSTGRES, MYSQL, SQLITE, DYNAMODB)

    Example:
        settings = DatabaseSettings(host="localhost", database="mydb")
        source = DatabaseSource(DatabaseType.POSTGRES, settings)
        docs = source.load_from_table("articles", "body", "id", "title")
    """

    def __init__(
        self,
        database_type: DatabaseType,
        settings: DatabaseSettings,
    ) -> None:
        """
        Initialize database source.

        Args:
            database_type: Type of database to connect to
            settings: Database connection settings
        """
        self._database_type = database_type
        self._settings = settings
        self._db: Database = DatabaseFactory.create(database_type, settings)

        logger.debug(
            f"DatabaseSource initialized: {database_type.value} @ "
            f"{settings.host or settings.database}"
        )

    def load_from_table(
        self,
        table: str,
        content_column: str,
        id_column: str = "id",
        title_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Document]:
        """
        Load documents from a database table.

        Args:
            table: Table name
            content_column: Column containing document content
            id_column: Column containing unique identifier (default: "id")
            title_column: Column containing document title (optional)
            metadata_columns: Additional columns to include in metadata
            where_clause: SQL WHERE clause (without WHERE keyword)
            limit: Maximum number of rows to load

        Returns:
            List of Document objects

        Raises:
            RAGIngestionError: If query fails
        """
        # Build column list
        columns = [id_column, content_column]
        if title_column:
            columns.append(title_column)
        if metadata_columns:
            columns.extend(metadata_columns)

        # Build query
        column_str = ", ".join(columns)
        query = f"SELECT {column_str} FROM {table}"

        if where_clause:
            query += f" WHERE {where_clause}"
        if limit:
            query += f" LIMIT {limit}"

        return self.load_from_query(
            query=query,
            content_column=content_column,
            id_column=id_column,
            title_column=title_column,
            source_name=f"table:{table}",
        )

    def load_from_query(
        self,
        query: str,
        content_column: str,
        id_column: str = "id",
        title_column: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> List[Document]:
        """
        Load documents from a custom SQL query.

        Args:
            query: SQL query to execute
            content_column: Column name containing document content
            id_column: Column name containing unique identifier
            title_column: Column name containing document title (optional)
            source_name: Name to use in metadata source field

        Returns:
            List of Document objects

        Raises:
            RAGIngestionError: If query fails
        """
        try:
            # Execute query
            results = self._db.execute(query)

            if not results:
                logger.info(f"Query returned no results: {query[:100]}...")
                return []

            documents = []

            for row in results:
                try:
                    doc = self._row_to_document(
                        row=row,
                        content_column=content_column,
                        id_column=id_column,
                        title_column=title_column,
                        source_name=source_name or f"query:{query[:50]}",
                    )
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.warning(f"Skipping row due to error: {e}")
                    continue

            logger.info(f"Loaded {len(documents)} documents from database")
            return documents

        except Exception as e:
            raise RAGIngestionError(f"Failed to execute query: {e}") from e

    def _row_to_document(
        self,
        row: Dict[str, Any],
        content_column: str,
        id_column: str,
        title_column: Optional[str],
        source_name: str,
    ) -> Optional[Document]:
        """Convert a database row to a Document."""
        # Get required fields
        content = row.get(content_column)
        row_id = row.get(id_column)

        if not content or not row_id:
            return None

        # Convert content to string if needed
        if not isinstance(content, str):
            content = str(content)

        # Generate document ID
        doc_id = f"db:{self._database_type.value}:{row_id}"

        # Get optional title
        title = None
        if title_column:
            title = row.get(title_column)
            if title and not isinstance(title, str):
                title = str(title)

        # Build extra metadata from remaining columns
        extra: Dict[str, Any] = {
            "database_type": self._database_type.value,
            "row_id": row_id,
        }
        for key, value in row.items():
            if key not in {content_column, id_column, title_column}:
                extra[key] = value

        metadata = DocumentMetadata(
            source=source_name,
            title=title,
            content_type="text/plain",
            created_at=datetime.now(),
            extra=extra,
        )

        return Document(id=doc_id, content=content, metadata=metadata)

    def load_document(self, path: str) -> Document:
        """
        Load a single document (required by DocumentSource protocol).

        For DatabaseSource, path is interpreted as "table:id" format.

        Args:
            path: Document path in format "table:id" or "table:id_column:value"

        Returns:
            Document object

        Raises:
            RAGIngestionError: If document not found or query fails
        """
        parts = path.split(":")
        if len(parts) < 2:
            raise RAGIngestionError(
                f"Invalid path format: {path}. Use 'table:id' or 'table:column:value'"
            )

        table = parts[0]
        if len(parts) == 2:
            # table:id format
            id_value = parts[1]
            id_column = "id"
        else:
            # table:column:value format
            id_column = parts[1]
            id_value = parts[2]

        # Query for the row
        query = f"SELECT * FROM {table} WHERE {id_column} = %s"
        try:
            results = self._db.execute(query, (id_value,))
            if not results:
                raise RAGIngestionError(f"Document not found: {path}")

            row = results[0]

            # Try to find content column
            content_column = self._guess_content_column(row)
            if not content_column:
                raise RAGIngestionError(f"Could not determine content column for table {table}")

            doc = self._row_to_document(
                row=row,
                content_column=content_column,
                id_column=id_column,
                title_column=self._guess_title_column(row),
                source_name=f"table:{table}",
            )

            if not doc:
                raise RAGIngestionError(f"Failed to convert row to document: {path}")

            return doc

        except RAGIngestionError:
            raise
        except Exception as e:
            raise RAGIngestionError(f"Failed to load document {path}: {e}") from e

    def load_documents(self, paths: List[str]) -> List[Document]:
        """
        Load multiple documents.

        For DatabaseSource, paths can be:
        - "table:id" - Load specific row
        - "table:*" - Load all rows from table

        Args:
            paths: List of paths to load

        Returns:
            List of Document objects
        """
        documents = []

        for path in paths:
            parts = path.split(":")
            if len(parts) >= 2 and parts[1] == "*":
                # Load all from table
                table = parts[0]
                content_col = parts[2] if len(parts) > 2 else None
                if content_col:
                    docs = self.load_from_table(table, content_col)
                else:
                    # Try to guess content column
                    try:
                        sample = self._db.execute(f"SELECT * FROM {table} LIMIT 1")
                        if sample:
                            content_col = self._guess_content_column(sample[0])
                            if content_col:
                                docs = self.load_from_table(table, content_col)
                            else:
                                logger.warning(f"Could not determine content column for {table}")
                                continue
                        else:
                            continue
                    except Exception as e:
                        logger.warning(f"Failed to load table {table}: {e}")
                        continue
                documents.extend(docs)
            else:
                # Load single document
                try:
                    doc = self.load_document(path)
                    documents.append(doc)
                except RAGIngestionError as e:
                    logger.warning(f"Skipping {path}: {e}")
                    continue

        return documents

    def _guess_content_column(self, row: Dict[str, Any]) -> Optional[str]:
        """Guess which column contains the main content."""
        # Priority order for content column names
        candidates = [
            "content",
            "body",
            "text",
            "description",
            "message",
            "article",
            "post",
            "data",
        ]
        for name in candidates:
            if name in row:
                return name

        # Fall back to first text-like column
        for key, value in row.items():
            if isinstance(value, str) and len(value) > 50:
                return key

        return None

    def _guess_title_column(self, row: Dict[str, Any]) -> Optional[str]:
        """Guess which column contains the title."""
        candidates = ["title", "name", "subject", "heading", "label"]
        for name in candidates:
            if name in row:
                return name
        return None

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._db, "close"):
            self._db.close()

    def __repr__(self) -> str:
        return f"DatabaseSource({self._database_type.value})"
