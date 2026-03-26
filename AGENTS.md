# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

**MCP Memory Server** is a local Model Context Protocol (MCP) server that provides persistent memory capabilities for IDE code assistants (Cursor, Windsurf, Claude Code, etc.). It solves the problem of context loss in long coding sessions by storing and retrieving important information using semantic search.

### Core Architecture

```
IDE Assistant (via MCP Protocol)
    ↓
MCP Server (server.py)
    ↓
Memory Manager (memory.py)
    ↓
├─ Embeddings (embeddings.py) → all-MiniLM-L6-v2 model
└─ Storage (storage.py) → SQLite + vector search
```

### Technology Stack

- **Language**: Python 3.10+ (supports 3.10, 3.11, 3.12)
- **Package Manager**: Poetry
- **MCP Framework**: `mcp` library (>=1.0.0)
- **Embeddings**: `sentence-transformers` with all-MiniLM-L6-v2 model
  - 384-dimensional vectors
  - ~150MB RAM usage
  - ~10ms inference time (CPU)
- **Storage**: SQLite with vector similarity search
  - Cosine similarity for semantic search
  - JSON metadata storage
  - ~5KB per memory vector
- **Testing**: pytest with async support
- **Code Quality**: black (formatting), ruff (linting), mypy (type checking)

### Resource Requirements

- **RAM**: ~200MB (50MB base + 150MB model)
- **Disk**: ~150MB for 10K memories (80MB model + 70MB data)
- **CPU**: Any modern CPU (no GPU required)

## Building and Running

### Development Setup

```bash
# Install dependencies with Poetry
poetry install

# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=mcp_memory --cov-report=html

# Format code
poetry run black src/ tests/

# Lint code
poetry run ruff check src/ tests/

# Type check
poetry run mypy src/
```

### Running the Server

```bash
# Run MCP server (stdio mode)
poetry run mcp-memory-server serve

# Run with custom database path
poetry run mcp-memory-server serve --db /path/to/memory.db

# Run with verbose logging
poetry run mcp-memory-server serve -v
```

### CLI Commands

```bash
# Show storage statistics
poetry run mcp-memory-server stats

# List memories
poetry run mcp-memory-server list --project myproject --limit 20

# Search memories
poetry run mcp-memory-server search "authentication pattern" --top-k 5

# Store a memory
poetry run mcp-memory-server store "Using JWT for auth" --project myapp --type decision

# Show model information
poetry run mcp-memory-server info
```

### Container Deployment

```bash
# Build with Podman/Docker
podman build -t mcp-memory-server -f container/Containerfile .

# Run container
podman run -d \
  --name mcp-memory \
  -v ~/.mcp-memory:/data \
  -p 3000:3000 \
  mcp-memory-server
```

## MCP Tools Reference

The server exposes 6 MCP tools for memory management:

### 1. `memory_store`
Store important context for later retrieval.

**Use cases:**
- Important decisions ("Using PostgreSQL for this project")
- Established patterns ("API routes follow /api/v1/resource pattern")
- Useful code snippets ("Standard error handling wrapper")
- Gotchas and lessons learned ("Don't use library X with Python 3.12")

**Parameters:**
- `text` (required): Content to remember
- `project` (optional): Project name for organization
- `type` (optional): "decision" | "pattern" | "snippet" | "note" | "gotcha"
- `importance` (optional): 1-10 score (affects retrieval ranking)
- `file` (optional): Related file path

### 2. `memory_recall`
Search for relevant memories using semantic similarity.

**Use cases:**
- Starting a new session ("What patterns did we establish?")
- Debugging ("How did we handle this error before?")
- Consistency checks ("What's our auth approach?")

**Parameters:**
- `query` (required): Search query
- `top_k` (optional): Max results (default: 5)
- `project` (optional): Filter by project
- `type` (optional): Filter by type
- `min_similarity` (optional): Threshold 0-1 (default: 0.3)

### 3. `memory_list`
List memories by criteria (chronological order).

**Parameters:**
- `project` (optional): Filter by project
- `type` (optional): Filter by type
- `since` (optional): ISO date filter
- `limit` (optional): Max results (default: 100)

### 4. `memory_forget`
Delete a memory by ID.

**Parameters:**
- `id` (required): Memory ID to delete

### 5. `memory_summarize`
Get a summary of stored memories for a project.

**Parameters:**
- `project` (optional): Project to summarize

**Returns:** Stats, breakdown by type, and recent memories

### 6. `memory_stats`
Get storage statistics (total memories, size, breakdown by type/project).

## Development Conventions

### Code Style

- **Line length**: 100 characters (enforced by black and ruff)
- **Type hints**: Required (enforced by mypy with strict mode)
- **Imports**: Sorted and organized (enforced by ruff)
- **Docstrings**: Google-style docstrings for all public functions/classes

### Testing

- **Framework**: pytest with async support (`pytest-asyncio`)
- **Location**: `tests/` directory
- **Naming**: `test_*.py` files with `test_*` functions
- **Coverage**: Aim for >80% coverage
- **Run tests**: `poetry run pytest`

### Module Structure

```
src/mcp_memory/
├── __init__.py          # Package initialization
├── server.py            # MCP server implementation (tool handlers)
├── memory.py            # High-level memory management API
├── embeddings.py        # Embedding model wrapper (lazy loading)
├── storage.py           # SQLite vector storage backend
└── cli.py              # Command-line interface
```

### Key Design Patterns

1. **Lazy Loading**: Embedding model loads on first use (saves ~150MB RAM if unused)
2. **Separation of Concerns**: 
   - `server.py` handles MCP protocol
   - `memory.py` provides business logic
   - `storage.py` manages persistence
   - `embeddings.py` handles ML model
3. **Metadata Flexibility**: JSON-based metadata allows arbitrary fields
4. **Cosine Similarity**: Used for semantic search (normalized dot product)
5. **Connection Management**: SQLite connection with row factory for dict-like access

### Adding New Features

When adding new MCP tools:
1. Define tool schema in `server.py` `list_tools()`
2. Add handler in `server.py` `call_tool()`
3. Implement business logic in `memory.py`
4. Add storage methods in `storage.py` if needed
5. Write tests in `tests/`
6. Update README.md with tool documentation

### Performance Considerations

- Embedding generation: ~10ms per text (CPU)
- Vector search: O(n) naive search (acceptable for <100K memories)
- Database: SQLite with indexes on created_at, project, type
- Memory footprint: ~200MB total (model + data structures)

### Security Notes

- All data stored locally (no external API calls)
- Database path: `~/.mcp-memory/memory.db` by default
- No authentication required (local-only server)
- Embeddings generated locally (no data sent to external services)

## Project Status

**Current Version**: 0.1.0 (POC/Alpha)

**Completed**:
- ✅ Basic MCP server implementation
- ✅ SQLite vector storage
- ✅ Embedding integration (all-MiniLM-L6-v2)
- ✅ All 6 MCP tools (store, recall, list, forget, summarize, stats)
- ✅ CLI interface
- ✅ Container build support

**Roadmap**:
- v0.2.0: Memory summarization improvements, performance optimization
- v0.3.0: TurboQuant compression, multi-project support, memory expiration
- v1.0.0: Production-ready with full documentation

## Common Tasks

### Adding a New Memory Type

1. No code changes needed - types are stored as metadata strings
2. Update documentation in `server.py` tool descriptions
3. Consider adding validation if needed

### Changing Embedding Model

1. Update model name in `embeddings.py` `get_model()`
2. Update dimension in `get_model_info()`
3. Test with `poetry run pytest tests/test_embeddings.py`
4. Update README.md with new model specs

### Debugging Memory Issues

```bash
# Check storage stats
poetry run mcp-memory-server stats

# List recent memories
poetry run mcp-memory-server list --limit 10

# Search with verbose logging
poetry run mcp-memory-server serve -v
```

### Database Schema

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,           -- numpy float32 array
    metadata TEXT DEFAULT '{}',        -- JSON string
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast filtering
CREATE INDEX idx_memories_created ON memories(created_at);
CREATE INDEX idx_memories_project ON memories(json_extract(metadata, '$.project'));
CREATE INDEX idx_memories_type ON memories(json_extract(metadata, '$.type'));
```

## IDE Integration

Add to your IDE's `mcp_settings.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "poetry",
      "args": ["run", "mcp-memory-server", "serve"],
      "cwd": "/path/to/mcp-memory-server",
      "env": {}
    }
  }
}
```

Or with container:

```json
{
  "mcpServers": {
    "memory": {
      "command": "podman",
      "args": ["run", "-i", "--rm", "-v", "~/.mcp-memory:/data", "mcp-memory-server"],
      "env": {}
    }
  }
}
```
