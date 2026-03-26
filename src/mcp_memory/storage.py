"""
SQLite-based storage for memory vectors.

Uses sqlite-vss for vector similarity search when available,
falls back to naive numpy-based search.

Supports TurboQuant-compressed embeddings for space-efficient storage
and fast approximate similarity search.
"""

import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".mcp-memory" / "memory.db"


class MemoryStorage:
    """SQLite-based storage for memories with vector search."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize storage.
        
        Args:
            db_path: Path to SQLite database (default: ~/.mcp-memory/memory.db)
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._init_db()
    
    @property
    def conn(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def _init_db(self):
        """Initialize database schema."""
        conn = self.conn
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                compressed_embedding BLOB,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
            CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(json_extract(metadata, '$.project'));
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(json_extract(metadata, '$.type'));
        """)
        conn.commit()
        self._migrate_add_compressed_column()
        logger.info(f"Database initialized at {self.db_path}")

    def _migrate_add_compressed_column(self):
        """Add compressed_embedding column if missing (migration for existing DBs)."""
        try:
            cursor = self.conn.execute("PRAGMA table_info(memories)")
            columns = [row[1] for row in cursor.fetchall()]
            if "compressed_embedding" not in columns:
                self.conn.execute(
                    "ALTER TABLE memories ADD COLUMN compressed_embedding BLOB"
                )
                self.conn.commit()
                logger.info("Migrated database: added compressed_embedding column")
        except Exception as e:
            logger.warning(f"Migration check failed: {e}")
    
    def store(
        self,
        id: str,
        text: str,
        embedding: list[float],
        metadata: dict,
        compressed_embedding: Optional[bytes] = None,
    ) -> str:
        """
        Store a memory.
        
        Args:
            id: Unique memory ID
            text: Memory text content
            embedding: Embedding vector
            metadata: Additional metadata (project, type, etc.)
            compressed_embedding: Optional TurboQuant-compressed embedding
            
        Returns:
            Memory ID
        """
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        metadata_json = json.dumps(metadata)
        
        self.conn.execute(
            """
            INSERT OR REPLACE INTO memories
                (id, text, embedding, compressed_embedding, metadata, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (id, text, embedding_bytes, compressed_embedding, metadata_json)
        )
        self.conn.commit()
        logger.debug(f"Stored memory {id}")
        return id
    
    def get(self, id: str) -> Optional[dict]:
        """Get a memory by ID."""
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (id,)
        ).fetchone()
        
        if row is None:
            return None
        
        return self._row_to_dict(row)
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: Optional[dict] = None
    ) -> list[dict]:
        """
        Search for similar memories.
        
        Args:
            query_embedding: Query vector
            top_k: Maximum results
            filter: Optional metadata filter
            
        Returns:
            List of memories with similarity scores
        """
        query_vec = np.array(query_embedding, dtype=np.float32)
        
        # Build query
        sql = "SELECT * FROM memories"
        params = []
        
        if filter:
            conditions = []
            if "project" in filter:
                conditions.append("json_extract(metadata, '$.project') = ?")
                params.append(filter["project"])
            if "type" in filter:
                conditions.append("json_extract(metadata, '$.type') = ?")
                params.append(filter["type"])
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
        
        rows = self.conn.execute(sql, params).fetchall()
        
        # Calculate similarities
        results = []
        for row in rows:
            stored_vec = np.frombuffer(row["embedding"], dtype=np.float32)
            similarity = float(np.dot(query_vec, stored_vec) / 
                              (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec)))
            
            result = self._row_to_dict(row)
            result["similarity"] = similarity
            results.append(result)
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
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
        conditions = []
        params = []
        
        if project:
            conditions.append("json_extract(metadata, '$.project') = ?")
            params.append(project)
        if type:
            conditions.append("json_extract(metadata, '$.type') = ?")
            params.append(type)
        if since:
            conditions.append("created_at >= ?")
            params.append(since)
        
        sql = "SELECT * FROM memories"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def search_compressed(
        self,
        query_compressed: bytes,
        compressor: "TurboQuantCompressor",
        top_k: int = 5,
        filter: Optional[dict] = None,
    ) -> list[dict]:
        """Search using TurboQuant-compressed embeddings.

        Falls back to full-precision search for memories that don't have
        compressed embeddings.

        Args:
            query_compressed: TurboQuant-compressed query vector.
            compressor: TurboQuantCompressor instance for similarity.
            top_k: Maximum results.
            filter: Optional metadata filter.

        Returns:
            List of memories with similarity scores.
        """
        sql = "SELECT * FROM memories"
        params: list = []

        if filter:
            conditions = []
            if "project" in filter:
                conditions.append("json_extract(metadata, '$.project') = ?")
                params.append(filter["project"])
            if "type" in filter:
                conditions.append("json_extract(metadata, '$.type') = ?")
                params.append(filter["type"])
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

        rows = self.conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            result = self._row_to_dict(row)
            compressed = row["compressed_embedding"]
            if compressed:
                similarity = compressor.similarity(query_compressed, compressed)
            else:
                # Fallback: decompress query and use full-precision
                from .quantization import turbo_decompress

                query_vec = turbo_decompress(query_compressed)
                stored_vec = np.frombuffer(row["embedding"], dtype=np.float32)
                norm_q = np.linalg.norm(query_vec)
                norm_s = np.linalg.norm(stored_vec)
                similarity = (
                    float(np.dot(query_vec, stored_vec) / (norm_q * norm_s))
                    if norm_q > 0 and norm_s > 0
                    else 0.0
                )
            result["similarity"] = similarity
            results.append(result)

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def update_compressed_embedding(
        self, id: str, compressed_embedding: bytes
    ) -> bool:
        """Update the compressed embedding for an existing memory.

        Args:
            id: Memory ID.
            compressed_embedding: TurboQuant-compressed bytes.

        Returns:
            True if updated, False if ID not found.
        """
        cursor = self.conn.execute(
            "UPDATE memories SET compressed_embedding = ? WHERE id = ?",
            (compressed_embedding, id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def get_uncompressed_ids(self) -> list[str]:
        """Get IDs of memories that lack compressed embeddings."""
        rows = self.conn.execute(
            "SELECT id FROM memories WHERE compressed_embedding IS NULL"
        ).fetchall()
        return [row["id"] for row in rows]

    def delete(self, id: str) -> bool:
        """Delete a memory by ID."""
        cursor = self.conn.execute("DELETE FROM memories WHERE id = ?", (id,))
        self.conn.commit()
        return cursor.rowcount > 0
    
    def count(self, project: Optional[str] = None) -> int:
        """Count memories."""
        if project:
            row = self.conn.execute(
                "SELECT COUNT(*) as count FROM memories WHERE json_extract(metadata, '$.project') = ?",
                (project,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) as count FROM memories").fetchone()
        return row["count"]
    
    def stats(self) -> dict:
        """Get storage statistics."""
        total = self.count()
        by_type = {}
        by_project = {}
        
        for row in self.conn.execute(
            "SELECT json_extract(metadata, '$.type') as type, COUNT(*) as count FROM memories GROUP BY type"
        ).fetchall():
            if row["type"]:
                by_type[row["type"]] = row["count"]
        
        for row in self.conn.execute(
            "SELECT json_extract(metadata, '$.project') as project, COUNT(*) as count FROM memories GROUP BY project"
        ).fetchall():
            if row["project"]:
                by_project[row["project"]] = row["count"]
        
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        # Compression stats
        compressed_row = self.conn.execute(
            "SELECT COUNT(*) as count FROM memories WHERE compressed_embedding IS NOT NULL"
        ).fetchone()
        compressed_count = compressed_row["count"] if compressed_row else 0

        uncompressed_row = self.conn.execute(
            "SELECT SUM(LENGTH(embedding)) as total FROM memories"
        ).fetchone()
        total_embedding_bytes = uncompressed_row["total"] or 0

        compressed_bytes_row = self.conn.execute(
            "SELECT SUM(LENGTH(compressed_embedding)) as total "
            "FROM memories WHERE compressed_embedding IS NOT NULL"
        ).fetchone()
        total_compressed_bytes = compressed_bytes_row["total"] or 0
        
        return {
            "total_memories": total,
            "by_type": by_type,
            "by_project": by_project,
            "db_size_bytes": db_size,
            "db_path": str(self.db_path),
            "compressed_count": compressed_count,
            "total_embedding_bytes": total_embedding_bytes,
            "total_compressed_bytes": total_compressed_bytes,
            "compression_ratio": (
                round(total_embedding_bytes / total_compressed_bytes, 2)
                if total_compressed_bytes > 0
                else 0.0
            ),
        }
    
    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert database row to dictionary."""
        embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
        metadata = json.loads(row["metadata"])
        
        return {
            "id": row["id"],
            "text": row["text"],
            "embedding": embedding,
            "metadata": metadata,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        }
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
