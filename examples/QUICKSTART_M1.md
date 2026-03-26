# Quick Start: MacBook Pro M1 Max (32GB)

This guide is optimized for your M1 Max with 32GB RAM. The server will use <300MB of that.

## Installation

```bash
# Clone
cd ~/projects
git clone https://github.com/imalvaroglez/mcp-memory-server.git
cd mcp-memory-server

# Install with Poetry
poetry install

# Verify installation
poetry run mcp-memory-server info
```

## Testing the Server

```bash
# In one terminal, run the server (MCP protocol over stdio)
poetry run mcp-memory-server serve

# In another terminal, test CLI commands:

# Store a memory
poetry run mcp-memory-server store "Using PostgreSQL for this project" --project my-app --type decision

# Search memories
poetry run mcp-memory-server search "database" --project my-app

# List all memories
poetry run mcp-memory-server list --project my-app

# Show stats
poetry run mcp-memory-server stats
```

## IDE Integration

### Cursor IDE

Add to `~/.cursor/mcp_settings.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "/Users/YOUR_USERNAME/projects/mcp-memory-server/.venv/bin/mcp-memory-server",
      "args": ["serve"],
      "env": {
        "MCP_MEMORY_DB": "~/.mcp-memory/memory.db"
      }
    }
  }
}
```

### Claude Code
Add to your MCP settings:

```json
{
  "mcpServers": {
    "memory": {
      "command": "/Users/YOUR_USERNAME/projects/mcp-memory-server/.venv/bin/mcp-memory-server",
      "args": ["serve"]
    }
  }
}
```

## Podman Container (Alternative)

```bash
# Build the container
podman build -f container/Containerfile -t mcp-memory-server .

# Run with persistent storage
podman run -i --rm \
  -v ~/.mcp-memory:/data \
  mcp-memory-server

# Use in IDE config
{
  "mcpServers": {
    "memory": {
      "command": "podman",
      "args": ["run", "-i", "--rm", "-v", "~/.mcp-memory:/data", "mcp-memory-server"]
    }
  }
}
```

## Performance on M1 Max

| Metric | Expected |
|--------|----------|
| Model load | ~2 seconds (first time) |
| Embedding generation | ~5-10ms per text |
| Vector search (1000 vectors) | <50ms |
| RAM usage | ~200MB |
| Storage per 10K memories | ~50MB |

Your M1 Max has 32GB RAM. This server will use <1% of that.

## Memory Management
The server stores memories in `~/.mcp-memory/memory.db` by default.

```bash
# Check database size
ls -lh ~/.mcp-memory/

# Backup
cp ~/.mcp-memory/memory.db ~/.mcp-memory/memory.db.backup

# Clear all memories (be careful!)
rm ~/.mcp-memory/memory.db
```

## Development

```bash
# Run tests
poetry run pytest

# Format code
poetry run black src/

# Type check
poetry run mypy src/

# Build distribution
poetry build
```
