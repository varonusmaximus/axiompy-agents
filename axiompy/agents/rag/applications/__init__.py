"""RAG Applications - Entry points for the RAG agent.

Provides:
- cli.py: Command-line interface (axiompy rag)
- api.py: REST API service

Usage:
    # CLI
    axiompy rag ingest ./docs/
    axiompy rag query "What is the main feature?"
    axiompy rag chat

    # CLI with persistent storage
    axiompy rag ingest ./docs/ --store chroma --persist-path ./rag_data

    # API
    from axiompy.agents.rag.applications import RAGAPIService, APISettings

    service = RAGAPIService.create()
    service.run()
"""

from axiompy.agents.rag.applications.api import APISettings, RAGAPIService
from axiompy.agents.rag.applications.cli import main as cli_main

__all__ = [
    "cli_main",
    "APISettings",
    "RAGAPIService",
]
