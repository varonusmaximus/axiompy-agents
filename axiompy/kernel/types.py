"""Kernel type enums."""

from enum import Enum


class RuntimeType(str, Enum):
    """Agent runtime implementations."""

    NATIVE = "native"
    LANGGRAPH = "langgraph"
    LANGCHAIN = "langchain"
