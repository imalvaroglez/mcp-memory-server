# Quick Start: MacBook Pro M1 Max (32GB)

This guide is optimized for your M1 Max with 32GB RAM. The server will use <300MB of that.

## Native Installation (Recommended for Development)

```bash
# Clone or copy the project
cd ~/projects
git clone <repo-url> mcp-memory-server
cd mcp-memory-server

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Verify installation
mcp-memory-server info

# Run the server
mcp-memory-server serve
```

## Testing the Server

```bash
# In one terminal, run the server
mcp-memory-server serve

# In another terminal, test commands:

# Store a memory
mcp-memory-client store "Using PostgreSQL for this project" --project my-app --type decision

# Search memories
mcp-memory-client search "database" --project my-app

# List all memories
mcp-memory-client list --project my-app

# Show stats
mcp-memory-client stats
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
pytest

# Format code
black src/

# Type check
mypy src/

# Build distribution
pip install build
python -m build
```
