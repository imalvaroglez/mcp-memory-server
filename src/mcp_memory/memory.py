"""
High-level memory interface.

Combines embedding generation with storage for a simple API.
"""

import uuid
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

from .embeddings import embed, cosine_similarity
from .storage import MemoryStorage

logger = logging.getLogger(__name__)


class Memory:
    """High-level memory management."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize memory system.
        
        Args:
            db_path: Path to storage database
        """
        self.storage = MemoryStorage(db_path)
    
    def store(
        self,
        text: str,
        project: Optional[str] = None,
        type: Optional[str] = None,
        importance: Optional[int] = None,
        file: Optional[str] = None,
        metadata: Optional[dict] = None
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
        
        # Store
        self.storage.store(
            id=id,
            text=text,
            embedding=embedding_result.embedding,
            metadata=full_metadata
        )
        
        logger.info(f"Stored memory {id} ({embedding_result.inference_ms:.1f}ms embedding)")
        
        return {
            "id": id,
            "text": text,
            "metadata": full_metadata,
            "embedding_time_ms": embedding_result.inference_ms
        }
    
    def recall(
        self,
        query: str,
        top_k: int = 5,
        project: Optional[str] = None,
        type: Optional[str] = None,
        min_similarity: float = 0.3
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
            filter=filter_dict if filter_dict else None
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
                "created_at": r["created_at"]
            }
            for r in filtered[:top_k]
        ]
    
    def list(
        self,
        project: Optional[str] = None,
        type: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100
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
            limit=limit
        )
        
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "metadata": r["metadata"],
                "created_at": r["created_at"]
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
            "updated_at": result["updated_at"]
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
            "recent": recent
        }
    
    def stats(self) -> dict:
        """Get storage statistics."""
        return self.storage.stats()
    
    def close(self):
        """Close storage connection."""
        self.storage.close()
