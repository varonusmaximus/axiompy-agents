"""agents.io default configuration.

Default values and constants for embeddings, chunking, and retrieval prompts.
"""

# Chunking defaults
DEFAULT_CHUNK_SIZE = 500  # Characters
DEFAULT_CHUNK_OVERLAP = 50  # Characters

# Query defaults
DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.0

# LLM defaults
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000

# Embedding defaults
DEFAULT_EMBEDDING_BATCH_SIZE = 100

# Default RAG prompt template
DEFAULT_RETRIEVAL_PROMPT = """Use the following context to answer the question. If you cannot answer \
the question based on the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
