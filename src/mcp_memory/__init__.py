"""
MCP Memory Server — Persistent memory for IDE code assistants.

Provides semantic search, local LLM reasoning (Gemma 2B),
and TurboQuant-compressed vector storage.
"""

__version__ = "0.2.0"

from .memory import Memory
from .embeddings import embed, cosine_similarity
from .quantization import TurboQuantCompressor, turbo_compress, turbo_decompress

__all__ = [
    "Memory",
    "embed",
    "cosine_similarity",
    "TurboQuantCompressor",
    "turbo_compress",
    "turbo_decompress",
]
