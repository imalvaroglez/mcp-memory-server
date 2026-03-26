"""
High-level memory interface.

Combines embedding generation with storage for a simple API.
Now includes local LLM reasoning and TurboQuant compression.
"""

import uuid
import logging
import time
from datetime import datetime
from typing import Optional
from pathlib import Path

import numpy as np

from .embeddings import embed, cosine_similarity
from .storage import MemoryStorage

logger = logging.getLogger(__name__)


class Memory:
    """High-level memory management with LLM reasoning and TurboQuant compression."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize memory system.

        Args:
            db_path: Path to storage database
        """
        self.storage = MemoryStorage(db_path)
        self._llm_engine = None
        self._compressor = None

    @property
    def compressor(self):
        """Lazy-load TurboQuant compressor."""
        if self._compressor is None:
            from .quantization import TurboQuantCompressor

            self._compressor = TurboQuantCompressor(dimension=384, bits=4)
        return self._compressor

    @property
    def llm(self):
        """Lazy-load LLM engine."""
        if self._llm_engine is None:
            from .llm import LLMEngine

            self._llm_engine = LLMEngine()
        return self._llm_engine

    def store(
        self,
        text: str,
        project: Optional[str] = None,
        type: Optional[str] = None,
        importance: Optional[int] = None,
        file: Optional[str] = None,
        metadata: Optional[dict] = None,
        auto_compress: bool = True,
    ) -> dict:
        """
        Store a memory.

        Args:
            text: Content to remember
            project: Project name
            type: Memory type (decision, pattern, snippet, note, gotcha)
            importance: Importance score (1-10)
            file: Related file path
            metadata: Additional metadata
            auto_compress: Whether to auto-compress embedding with TurboQuant

        Returns:
            Stored memory with id
        """
        # Generate embedding
        embedding_result = embed(text)

        # Build metadata
        full_metadata = metadata or {}
        if project:
            full_metadata["project"] = project
        if type:
            full_metadata["type"] = type
        if importance is not None:
            full_metadata["importance"] = max(1, min(10, importance))
        if file:
            full_metadata["file"] = file

        # Generate ID
        id = str(uuid.uuid4())

        # TurboQuant compression (optional, adds ~2ms)
        compressed_embedding = None
        compression_ms = 0.0
        if auto_compress:
            try:
                start = time.time()
                vec = np.array(embedding_result.embedding, dtype=np.float32)
                compressed_embedding = self.compressor.compress(vec)
                compression_ms = (time.time() - start) * 1000
            except Exception as e:
                logger.warning(f"TurboQuant compression failed: {e}")

        # Store
        self.storage.store(
            id=id,
            text=text,
            embedding=embedding_result.embedding,
            metadata=full_metadata,
            compressed_embedding=compressed_embedding,
        )

        logger.info(
            f"Stored memory {id} "
            f"({embedding_result.inference_ms:.1f}ms embedding"
            f"{f', {compression_ms:.1f}ms compression' if compressed_embedding else ''})"
        )

        return {
            "id": id,
            "text": text,
            "metadata": full_metadata,
            "embedding_time_ms": embedding_result.inference_ms,
            "compressed": compressed_embedding is not None,
            "compression_time_ms": round(compression_ms, 2),
        }

    def recall(
        self,
        query: str,
        top_k: int = 5,
        project: Optional[str] = None,
        type: Optional[str] = None,
        min_similarity: float = 0.3,
    ) -> list[dict]:
        """
        Recall relevant memories.

        Args:
            query: Search query
            top_k: Maximum results
            project: Filter by project
            type: Filter by type
            min_similarity: Minimum similarity threshold

        Returns:
            List of relevant memories with similarity scores
        """
        # Embed query
        query_result = embed(query)

        # Search
        filter_dict = {}
        if project:
            filter_dict["project"] = project
        if type:
            filter_dict["type"] = type

        results = self.storage.search(
            query_embedding=query_result.embedding,
            top_k=top_k * 2,  # Fetch more to filter by threshold
            filter=filter_dict if filter_dict else None,
        )

        # Filter by threshold
        filtered = [r for r in results if r.get("similarity", 0) >= min_similarity]

        # Format response
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "similarity": round(r["similarity"], 3),
                "metadata": r["metadata"],
                "created_at": r["created_at"],
            }
            for r in filtered[:top_k]
        ]

    def list(
        self,
        project: Optional[str] = None,
        type: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        List memories by criteria.

        Args:
            project: Filter by project
            type: Filter by type
            since: Filter by date (ISO format)
            limit: Maximum results

        Returns:
            List of memories
        """
        results = self.storage.list(
            project=project,
            type=type,
            since=since,
            limit=limit,
        )

        return [
            {
                "id": r["id"],
                "text": r["text"],
                "metadata": r["metadata"],
                "created_at": r["created_at"],
            }
            for r in results
        ]

    def get(self, id: str) -> Optional[dict]:
        """Get a specific memory by ID."""
        result = self.storage.get(id)
        if result is None:
            return None

        return {
            "id": result["id"],
            "text": result["text"],
            "metadata": result["metadata"],
            "created_at": result["created_at"],
            "updated_at": result["updated_at"],
        }

    def forget(self, id: str) -> bool:
        """
        Delete a memory.

        Args:
            id: Memory ID to delete

        Returns:
            True if deleted, False if not found
        """
        deleted = self.storage.delete(id)
        if deleted:
            logger.info(f"Forgot memory {id}")
        return deleted

    def summarize(self, project: Optional[str] = None) -> dict:
        """
        Get a summary of stored memories.

        Args:
            project: Project to summarize (optional)

        Returns:
            Summary with stats
        """
        stats = self.storage.stats()

        if project:
            stats["project"] = project
            memories = self.storage.list(project=project, limit=100)
        else:
            memories = self.storage.list(limit=100)

        # Group by type
        by_type = {}
        for m in memories:
            t = m["metadata"].get("type", "untyped")
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(m["text"][:100])

        # Get recent
        recent = [
            {"text": m["text"][:100], "type": m["metadata"].get("type")}
            for m in memories[:10]
        ]

        return {
            "stats": stats,
            "by_type": {k: len(v) for k, v in by_type.items()},
            "recent": recent,
        }

    def stats(self) -> dict:
        """Get storage statistics."""
        return self.storage.stats()

    # ------------------------------------------------------------------
    # LLM-powered methods (new in v0.2)
    # ------------------------------------------------------------------

    def think(
        self,
        question: str,
        project: Optional[str] = None,
        top_k: int = 10,
    ) -> dict:
        """
        Use the local LLM to reason about memories and answer a question.

        Retrieves the most relevant memories and feeds them to the LLM
        along with the question.

        Args:
            question: The question to answer about stored context.
            project: Optional project filter.
            top_k: Number of memories to feed to the LLM.

        Returns:
            Dict with 'answer', 'memories_used', and 'inference_ms'.
        """
        # Recall relevant memories
        memories = self.recall(
            query=question,
            top_k=top_k,
            project=project,
            min_similarity=0.2,  # Lower threshold for broader context
        )

        if not memories:
            return {
                "answer": "No relevant memories found. Try storing some context first.",
                "memories_used": 0,
                "inference_ms": 0,
            }

        # LLM reasoning
        result = self.llm.analyze_memories(
            memories=memories,
            query=question,
            project=project,
        )

        return {
            "answer": result.text,
            "memories_used": len(memories),
            "inference_ms": result.inference_ms,
            "tokens_generated": result.tokens_generated,
            "model": result.model,
        }

    def analyze(
        self,
        project: Optional[str] = None,
        focus: Optional[str] = None,
        limit: int = 50,
    ) -> dict:
        """
        Use the local LLM to analyze memories for patterns and conflicts.

        Args:
            project: Project to analyze.
            focus: Optional focus area (e.g., "architecture", "testing").
            limit: Max memories to analyze.

        Returns:
            Dict with structured analysis.
        """
        # Get all memories for the project
        memories = self.list(project=project, limit=limit)

        if not memories:
            return {
                "analysis": "No memories found for analysis.",
                "memories_analyzed": 0,
                "inference_ms": 0,
            }

        result = self.llm.detect_patterns(memories=memories, focus=focus)

        return {
            "analysis": result.text,
            "memories_analyzed": len(memories),
            "inference_ms": result.inference_ms,
            "tokens_generated": result.tokens_generated,
            "model": result.model,
        }

    def compress(self) -> dict:
        """
        Compress all uncompressed embeddings using TurboQuant.

        This is an idempotent operation — already-compressed memories
        are skipped.

        Returns:
            Dict with compression statistics.
        """
        start = time.time()

        uncompressed_ids = self.storage.get_uncompressed_ids()
        if not uncompressed_ids:
            return {
                "compressed": 0,
                "already_compressed": self.storage.count(),
                "message": "All embeddings are already compressed.",
            }

        compressed_count = 0
        errors = 0

        for mem_id in uncompressed_ids:
            try:
                mem = self.storage.get(mem_id)
                if mem and mem.get("embedding"):
                    vec = np.array(mem["embedding"], dtype=np.float32)
                    compressed = self.compressor.compress(vec)
                    self.storage.update_compressed_embedding(mem_id, compressed)
                    compressed_count += 1
            except Exception as e:
                logger.warning(f"Failed to compress memory {mem_id}: {e}")
                errors += 1

        elapsed = (time.time() - start) * 1000
        stats = self.storage.stats()

        return {
            "compressed": compressed_count,
            "errors": errors,
            "total_memories": stats["total_memories"],
            "compression_ratio": stats.get("compression_ratio", 0),
            "total_embedding_bytes": stats.get("total_embedding_bytes", 0),
            "total_compressed_bytes": stats.get("total_compressed_bytes", 0),
            "elapsed_ms": round(elapsed, 2),
            "message": (
                f"Compressed {compressed_count} embeddings "
                f"({errors} errors) in {elapsed:.0f}ms"
            ),
        }

    def llm_status(self) -> dict:
        """Get LLM engine status and info."""
        try:
            info = self.llm.info()
            return {
                "available": True,
                "model_name": info.model_name,
                "model_path": info.model_path,
                "context_size": info.context_size,
                "loaded": info.loaded,
                "ram_estimate_mb": info.ram_estimate_mb,
                "gpu_layers": info.gpu_layers,
                "platform": info.platform,
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "message": (
                    "LLM features require llama-cpp-python. "
                    "Install with: poetry install --extras llm"
                ),
            }

    def close(self):
        """Close storage connection and unload LLM."""
        self.storage.close()
        if self._llm_engine is not None:
            self._llm_engine.unload()
