# MCP Memory Server

A local Model Context Protocol (MCP) server that provides persistent memory for IDE code assistants.

## Problem

IDE code assistants (Cursor, Windsurf, Claude Code, etc.) struggle with long coding sessions:
- Context window limits (typically 200K tokens)
- Aggressive compression causes repetition
- Previous decisions, patterns, and context get lost
- Assistant re-explains things it already knew

## Solution

An MCP server that:
- **Stores** important context from your coding sessions
- **Retrieves** relevant memories when you need them
- **Runs locally** — no external APIs, full privacy
- **Fits on laptops** — designed for 8GB+ RAM machines

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      YOUR IDE                                │
│              (Cursor, Windsurf, Claude Code)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │ MCP Protocol
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  MCP Memory Server                           │
├─────────────────────────────────────────────────────────────┤
│  Tools:                                                     │
│  • memory_store(text, metadata) → id                        │
│  • memory_recall(query, top_k) → [memories]                 │
│  • memory_forget(id) → success                              │
│  • memory_list(filter) → [memories]                         │
│  • memory_summarize(project) → summary                      │
├─────────────────────────────────────────────────────────────┤
│  Storage:                                                   │
│  • Embedding: all-MiniLM-L6-v2 (80MB, local)                │
│  • Vector DB: SQLite with vector extension                  │
│  • Compression: TurboQuant (when available)                 │
├─────────────────────────────────────────────────────────────┤
│  Resources:                                                 │
│  • Model RAM: ~150MB                                        │
│  • Storage: ~50MB per 10K memories                          │
│  • Total: <300MB for most use cases                         │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Option 1: Podman (Recommended)

```bash
# Build container
podman build -t mcp-memory-server .

# Run server
podman run -d \
  --name mcp-memory \
  -v ~/.mcp-memory:/data \
  -p 3000:3000 \
  mcp-memory-server

# Add to your IDE's mcp_settings.json
```

### Option 2: Native Python

```bash
# Install
pip install -e .

# Run
mcp-memory-server --port 3000
```

### IDE Configuration

Add to your `mcp_settings.json`:

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

## MCP Tools

### `memory_store`

Store important context for later retrieval.

```
Arguments:
  - text (string): The content to remember
  - metadata (object, optional): Additional context
    - project: Project name
    - file: Related file path
    - type: "decision" | "pattern" | "snippet" | "note"
    - importance: 1-10 (affects retrieval ranking)

Returns:
  - id: Memory ID for later reference
```

**When to use:**
- Important decisions made ("Using PostgreSQL for this project")
- Established patterns ("API routes follow /api/v1/resource pattern")
- Useful snippets ("Standard error handling wrapper")
- Gotchas learned ("Don't use library X with Python 3.12")

### `memory_recall`

Search for relevant memories.

```
Arguments:
  - query (string): What you're looking for
  - top_k (number, optional): Max results (default: 5)
  - filter (object, optional): Filter by metadata

Returns:
  - memories: Array of matching memories with similarity scores
```

**When to use:**
- Starting a new session ("What patterns did we establish?")
- Debugging ("How did we handle this error before?")
- Consistency checks ("What's our auth approach?")

### `memory_list`

List memories by criteria.

```
Arguments:
  - project (string, optional): Filter by project
  - type (string, optional): Filter by type
  - since (string, optional): ISO date
  - limit (number, optional): Max results

Returns:
  - memories: Array of matching memories
```

### `memory_forget`

Remove a memory.

```
Arguments:
  - id (string): Memory ID to delete

Returns:
  - success: boolean
```

### `memory_summarize`

Get a summary of stored context for a project.

```
Arguments:
  - project (string): Project name

Returns:
  - summary: Generated summary of key memories
  - stats: Memory count, types, recency
```

## Resource Requirements

| Component | RAM | Disk | Notes |
|-----------|-----|------|-------|
| Base server | 50MB | 10MB | Python + SQLite |
| Embedding model | 150MB | 80MB | all-MiniLM-L6-v2 |
| Vector storage | — | ~5KB/vector | ~50MB per 10K |
| **Total (10K memories)** | **~200MB** | **~150MB** | Fits on laptop |

## Development

### Setup

```bash
# Clone
git clone https://github.com/tailor-made/mcp-memory-server.git
cd mcp-memory-server

# Create venv
python -m venv .venv
source .venv/bin/activate

# Install deps
pip install -e ".[dev]"

# Run tests
pytest

# Run server
mcp-memory-server
```

### Project Structure

```
mcp-memory-server/
├── src/
│   └── mcp_memory/
│       ├── __init__.py
│       ├── server.py          # MCP server implementation
│       ├── memory.py          # Memory storage + retrieval
│       ├── embeddings.py      # Embedding model wrapper
│       ├── storage.py         # SQLite vector storage
│       └── cli.py             # Command-line interface
├── tests/
│   ├── test_memory.py
│   ├── test_embeddings.py
│   └── test_server.py
├── container/
│   ├── Containerfile          # Podman/Docker build
│   └── entrypoint.sh
├── pyproject.toml
├── README.md
└── LICENSE
```

## Roadmap

### v0.1.0 — POC (This Week)
- [x] Basic MCP server
- [ ] SQLite vector storage
- [ ] Embedding integration
- [ ] Store + recall tools
- [ ] Container build

### v0.2.0 — Usable
- [ ] All MCP tools implemented
- [ ] Memory summarization
- [ ] Project filtering
- [ ] Performance optimization

### v0.3.0 — Production Ready
- [ ] TurboQuant compression
- [ ] Multi-project support
- [ ] Memory expiration
- [ ] Import/export

### v1.0.0 — Release
- [ ] Documentation
- [ ] IDE integration guides
- [ ] Community feedback incorporated

## License

MIT — Use freely, modify, distribute.

## Contributing

Issues and PRs welcome. This is designed to be a community resource for better IDE AI experiences.
