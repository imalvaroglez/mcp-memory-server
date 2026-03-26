"""
Tests for MCP Memory Server.
"""

import pytest
import tempfile
from pathlib import Path

from mcp_memory.memory import Memory
from mcp_memory.embeddings import embed, cosine_similarity


class TestEmbeddings:
    """Test embedding functionality."""
    
    def test_embed_single_text(self):
        """Test embedding a single text."""
        result = embed("Hello world")
        
        assert result.dimension == 384
        assert len(result.embedding) == 384
        assert result.model == "all-MiniLM-L6-v2"
        assert result.inference_ms > 0
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]
        
        sim_same = cosine_similarity(vec1, vec2)
        sim_ortho = cosine_similarity(vec1, vec3)
        
        assert sim_same == 1.0
        assert sim_ortho == 0.0


class TestMemory:
    """Test memory functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_memory.db"
    
    def test_store_and_recall(self, temp_db):
        """Test storing and recalling memories."""
        memory = Memory(temp_db)
        
        # Store some memories
        memory.store(
            text="Using PostgreSQL for the database",
            project="test-project",
            type="decision"
        )
        memory.store(
            text="API routes follow /api/v1/resource pattern",
            project="test-project",
            type="pattern"
        )
        memory.store(
            text="Don't use library X with Python 3.12",
            project="test-project",
            type="gotcha"
        )
        
        # Recall
        results = memory.recall(
            query="What database are we using?",
            project="test-project"
        )
        
        assert len(results) > 0
        assert "PostgreSQL" in results[0]["text"]
        
        memory.close()
    
    def test_list_memories(self, temp_db):
        """Test listing memories."""
        memory = Memory(temp_db)
        
        # Store memories
        for i in range(5):
            memory.store(
                text=f"Memory {i}",
                project="test-project",
                type="note"
            )
        
        # List all
        results = memory.list(project="test-project")
        assert len(results) == 5
        
        # List by type
        results = memory.list(project="test-project", type="note")
        assert len(results) == 5
        
        memory.close()
    
    def test_forget_memory(self, temp_db):
        """Test deleting memories."""
        memory = Memory(temp_db)
        
        # Store and get ID
        result = memory.store(
            text="Temporary memory",
            project="test-project"
        )
        memory_id = result["id"]
        
        # Verify stored
        assert memory.get(memory_id) is not None
        
        # Delete
        deleted = memory.forget(memory_id)
        assert deleted is True
        
        # Verify deleted
        assert memory.get(memory_id) is None
        
        memory.close()
    
    def test_stats(self, temp_db):
        """Test storage statistics."""
        memory = Memory(temp_db)
        
        # Store some memories
        memory.store(text="Memory 1", project="project-a", type="decision")
        memory.store(text="Memory 2", project="project-a", type="pattern")
        memory.store(text="Memory 3", project="project-b", type="note")
        
        # Get stats
        stats = memory.stats()
        
        assert stats["total_memories"] == 3
        assert "project-a" in stats["by_project"]
        assert "project-b" in stats["by_project"]
        assert stats["db_size_bytes"] > 0
        
        memory.close()
