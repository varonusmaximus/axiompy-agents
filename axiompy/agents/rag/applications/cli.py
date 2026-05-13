"""CLI Adapter - Command-line interface for RAG.

Provides the `axiompy rag` command for document ingestion and querying.

Usage:
    axiompy rag ingest ./docs/              # Ingest documents
    axiompy rag ingest ./docs/ --watch      # Watch for changes (planned)
    axiompy rag query "What is X?"          # Single query
    axiompy rag chat                        # Interactive chat mode
    axiompy rag stats                       # Show index statistics

Examples:
    # Ingest markdown files
    axiompy rag ingest ./docs/*.md

    # Query with local embeddings and Ollama
    axiompy rag query "How does authentication work?"

    # Use fastembed instead of sentence-transformers
    axiompy rag query "Explain the API" --embedder fastembed

    # Interactive chat
    axiompy rag chat --model mistral

    # Use persistent Chroma store
    axiompy rag ingest ./docs/ --store chroma --persist-path ./rag_data
    axiompy rag query "What is X?" --store chroma --persist-path ./rag_data
"""

import argparse
import sys
from typing import List, Optional

from axiompy.agents.rag.defaults import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)
from axiompy.agents.rag.factory import (
    ChunkerSettings,
    EmbedderSettings,
    EmbedderType,
    LLMSettings,
    RAGServiceFactory,
    VectorStoreSettings,
    VectorStoreType,
)
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)

# Default configuration
DEFAULT_EMBEDDER = "sentence_transformers"
DEFAULT_VECTOR_STORE = "memory"
DEFAULT_LLM = "ollama"
DEFAULT_MODEL = "mistral"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_PERSIST_PATH = "./rag_data"


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="axiompy rag",
        description="RAG (Retrieval-Augmented Generation) for document Q&A.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  axiompy rag ingest ./docs/              Ingest documents from directory
  axiompy rag ingest *.md *.txt           Ingest specific files
  axiompy rag query "What is X?"          Ask a question
  axiompy rag chat                        Interactive chat mode
  axiompy rag stats                       Show index statistics

Storage options (use chroma for persistence):
  --store memory                         In-memory (default, not persistent)
  --store chroma                         ChromaDB (persistent)
  --persist-path ./rag_data              Path for Chroma persistence

Embedder options:
  --embedder sentence_transformers       PyTorch-based (default, best quality)
  --embedder fastembed                   ONNX-based (faster, lighter)
  --embedder mock                        For testing

LLM options:
  --llm ollama                           Local Ollama (default)
  --llm openai                           OpenAI API
  --llm mock                             For testing
        """,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- ingest command ---
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest documents into the vector store",
    )
    ingest_parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to ingest",
    )
    ingest_parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size in characters (default: {DEFAULT_CHUNK_SIZE})",
    )
    ingest_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk overlap in characters (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    _add_common_args(ingest_parser)

    # --- query command ---
    query_parser = subparsers.add_parser(
        "query",
        help="Query the RAG system",
    )
    query_parser.add_argument(
        "question",
        help="Question to ask",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    query_parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Show source chunks used for the answer",
    )
    _add_common_args(query_parser)

    # --- chat command ---
    chat_parser = subparsers.add_parser(
        "chat",
        help="Interactive chat mode",
    )
    chat_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    _add_common_args(chat_parser)

    # --- stats command ---
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show index statistics",
    )
    _add_common_args(stats_parser)

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a subparser."""
    # Embedder configuration
    parser.add_argument(
        "--embedder",
        choices=["sentence_transformers", "fastembed", "mock"],
        default=DEFAULT_EMBEDDER,
        help=f"Embedding provider (default: {DEFAULT_EMBEDDER})",
    )
    parser.add_argument(
        "--embed-model",
        help="Embedding model name (uses provider default if not set)",
    )
    parser.add_argument(
        "--model-cache-dir",
        help="Directory for cached embedding models",
    )

    # Vector store configuration
    parser.add_argument(
        "--store",
        choices=["memory", "chroma", "mock"],
        default=DEFAULT_VECTOR_STORE,
        help=f"Vector store backend (default: {DEFAULT_VECTOR_STORE})",
    )
    parser.add_argument(
        "--persist-path",
        help=f"Persist path for chroma store (default: {DEFAULT_PERSIST_PATH})",
    )
    parser.add_argument(
        "--collection",
        default="rag_documents",
        help="Collection name for vector store (default: rag_documents)",
    )

    # LLM configuration
    parser.add_argument(
        "--llm",
        choices=["ollama", "openai", "anthropic", "mock"],
        default=DEFAULT_LLM,
        help=f"LLM provider (default: {DEFAULT_LLM})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help=f"Ollama host URL (default: {DEFAULT_OLLAMA_HOST})",
    )

    # Output configuration
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )


def main(args: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point.

    Args:
        args: Command-line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 for success)
    """
    parser = create_parser()
    opts = parser.parse_args(args)

    if not opts.command:
        parser.print_help()
        return 1

    try:
        match opts.command:
            case "ingest":
                return run_ingest(opts)
            case "query":
                return run_query(opts)
            case "chat":
                return run_chat(opts)
            case "stats":
                return run_stats(opts)
            case _:
                parser.print_help()
                return 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if opts.verbose:
            import traceback

            traceback.print_exc()
        return 1


def create_rag_service(opts):
    """Create RAGService from CLI options."""
    # Map CLI options to enum types
    embedder_type_map = {
        "sentence_transformers": EmbedderType.SENTENCE_TRANSFORMERS,
        "fastembed": EmbedderType.FASTEMBED,
        "mock": EmbedderType.MOCK,
    }

    store_type_map = {
        "memory": VectorStoreType.MEMORY,
        "chroma": VectorStoreType.CHROMA,
        "mock": VectorStoreType.MOCK,
    }

    embedder_type = embedder_type_map[opts.embedder]
    store_type = store_type_map[opts.store]

    # Build settings
    embedder_settings = EmbedderSettings(
        model=opts.embed_model,
        cache_dir=getattr(opts, "model_cache_dir", None),
    )

    # Vector store settings with persistence support
    persist_path = getattr(opts, "persist_path", None)
    collection = getattr(opts, "collection", "rag_documents")

    # For chroma, require persist_path or use default
    if store_type == VectorStoreType.CHROMA and not persist_path:
        persist_path = DEFAULT_PERSIST_PATH

    vector_store_settings = VectorStoreSettings(
        collection_name=collection,
        persist_path=persist_path,
    )

    llm_settings = LLMSettings(
        model=opts.model,
        endpoint=f"{opts.ollama_host}/api/generate" if opts.llm == "ollama" else None,
    )

    chunker_settings = None
    if hasattr(opts, "chunk_size"):
        chunker_settings = ChunkerSettings(
            chunk_size=opts.chunk_size,
            chunk_overlap=opts.chunk_overlap,
        )

    return RAGServiceFactory.create(
        embedder_type=embedder_type,
        vector_store_type=store_type,
        llm_provider=opts.llm,
        embedder_settings=embedder_settings,
        vector_store_settings=vector_store_settings,
        llm_settings=llm_settings,
        chunker_settings=chunker_settings,
    )


def run_ingest(opts) -> int:
    """Ingest documents into the vector store."""
    print(f"🔍 Initializing RAG with {opts.embedder} embeddings...")

    # Show storage info
    if opts.store == "chroma":
        persist_path = opts.persist_path or DEFAULT_PERSIST_PATH
        print(f"💾 Using persistent Chroma store at: {persist_path}")
    else:
        print("⚠️  Using in-memory store (data will not persist between runs)")

    rag = create_rag_service(opts)

    print(f"📄 Ingesting documents from: {', '.join(opts.paths)}")

    # Use FileSystemSource for proper document loading
    from axiompy.agents.rag.adapters.sources import FileSystemSource

    source = FileSystemSource()
    documents = source.load_documents(opts.paths)

    if opts.verbose:
        for doc in documents:
            print(f"  📁 {doc.metadata.source if doc.metadata else doc.id}")

    if not documents:
        print("❌ No documents found to ingest.")
        return 1

    print(f"📊 Found {len(documents)} documents")

    # Ingest
    stats = rag.ingest(documents)

    print(
        f"✅ Ingested {stats['chunks_created']} chunks from {stats['documents_processed']} documents"
    )
    return 0


def run_query(opts) -> int:
    """Run a single query."""
    print(f"🔍 Initializing RAG with {opts.embedder} embeddings...")

    rag = create_rag_service(opts)

    print(f"❓ Query: {opts.question}\n")

    response = rag.query(opts.question, top_k=opts.top_k)

    print("💬 Answer:")
    print(response.answer)
    print()

    if opts.show_sources and response.sources:
        print("📚 Sources:")
        for i, source in enumerate(response.sources, 1):
            print(f"\n--- Source {i} (score: {source.score:.3f}) ---")
            print(f"Document: {source.chunk.document_id}")
            print(
                source.chunk.content[:200] + "..."
                if len(source.chunk.content) > 200
                else source.chunk.content
            )

    return 0


def run_chat(opts) -> int:
    """Run interactive chat mode."""
    print(f"🔍 Initializing RAG with {opts.embedder} embeddings...")
    print("   (This may take a moment to load the model...)\n")

    rag = create_rag_service(opts)

    print("💬 RAG Chat Mode")
    print("   Type your questions, or 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()

            if not question:
                continue

            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            response = rag.query(question, top_k=opts.top_k)

            print(f"\nAssistant: {response.answer}\n")

        except EOFError:
            print("\nGoodbye!")
            break

    return 0


def run_stats(opts) -> int:
    """Show index statistics."""
    print(f"🔍 Initializing RAG with {opts.embedder} embeddings...")

    rag = create_rag_service(opts)

    stats = rag.get_stats()

    print("\n📊 RAG Index Statistics")
    print(f"   Chunks indexed: {stats.get('chunk_count', 0)}")
    print(f"   Documents: {stats.get('document_count', 'N/A')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
