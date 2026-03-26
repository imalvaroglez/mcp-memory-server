"""
MCP Server implementation for memory tools.

Exposes memory_store, memory_recall, memory_list, memory_forget,
memory_summarize, memory_stats as core MCP tools.

New in v0.2:
- memory_think: LLM-powered reasoning about stored memories
- memory_analyze: Pattern detection and conflict analysis via local LLM
- memory_compress: TurboQuant compression for embedding storage optimization
"""

import logging
from typing import Optional
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .memory import Memory

logger = logging.getLogger(__name__)

# Global memory instance
_memory: Optional[Memory] = None


def get_memory(db_path: Optional[Path] = None) -> Memory:
    """Get or create memory instance."""
    global _memory
    if _memory is None:
        _memory = Memory(db_path)
    return _memory


# Create MCP server
server = Server("mcp-memory-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="memory_store",
            description="""Store important context for later retrieval.

WHEN TO USE:
- Important decisions made (e.g., "Using PostgreSQL for this project")
- Established patterns (e.g., "API routes follow /api/v1/resource pattern")
- Useful snippets (e.g., "Standard error handling wrapper")
- Gotchas learned (e.g., "Don't use library X with Python 3.12")
- Architecture decisions
- Config choices
- Dependencies and versions

The memory will be semantically indexed and retrievable via similarity search.
Embeddings are automatically compressed using TurboQuant for storage efficiency.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The content to remember",
                    },
                    "project": {
                        "type": "string",
                        "description": "Project name for organization",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["decision", "pattern", "snippet", "note", "gotcha"],
                        "description": "Type of memory",
                    },
                    "importance": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Importance score (affects retrieval ranking)",
                    },
                    "file": {
                        "type": "string",
                        "description": "Related file path",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="memory_recall",
            description="""Search for relevant memories.

WHEN TO USE:
- Starting a new session ("What patterns did we establish?")
- Debugging ("How did we handle this error before?")
- Consistency checks ("What's our auth approach?")
- Looking for related decisions

Returns memories ranked by semantic similarity to your query.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What you're looking for",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum results to return",
                    },
                    "project": {
                        "type": "string",
                        "description": "Filter by project",
                    },
                    "type": {
                        "type": "string",
                        "description": "Filter by type",
                    },
                    "min_similarity": {
                        "type": "number",
                        "default": 0.3,
                        "description": "Minimum similarity threshold (0-1)",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="memory_list",
            description="""List memories by criteria.

WHEN TO USE:
- Reviewing all memories for a project
- Finding all decisions
- Recent activity check

Returns memories in reverse chronological order.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Filter by project",
                    },
                    "type": {
                        "type": "string",
                        "description": "Filter by type",
                    },
                    "since": {
                        "type": "string",
                        "description": "Filter by date (ISO format)",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 100,
                        "description": "Maximum results",
                    },
                },
            },
        ),
        Tool(
            name="memory_forget",
            description="""Delete a memory by ID.

Use when a memory is no longer relevant or was stored incorrectly.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Memory ID to delete",
                    }
                },
                "required": ["id"],
            },
        ),
        Tool(
            name="memory_summarize",
            description="""Get a summary of stored memories.

WHEN TO USE:
- Quick overview of project context
- Before starting a new session
- Understanding what's been discussed

Returns stats, breakdown by type, and recent memories.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project to summarize (optional)",
                    }
                },
            },
        ),
        Tool(
            name="memory_stats",
            description="""Get storage statistics.

Returns total memories, breakdown by type/project, storage size,
and TurboQuant compression statistics.""",
            inputSchema={"type": "object", "properties": {}},
        ),
        # --- New LLM-powered tools (v0.2) ---
        Tool(
            name="memory_think",
            description="""Use the local LLM to reason about stored memories and answer questions.

WHEN TO USE:
- Deep questions about past decisions ("Why did we choose Redis over Memcached?")
- Synthesizing multiple memories ("What's our overall auth strategy?")
- Getting reasoned advice based on stored context
- Understanding the evolution of project decisions

This tool retrieves relevant memories and uses a local LLM (Gemma 2B)
to reason about them. The LLM runs entirely on-device — no data leaves
your machine.

REQUIRES: llama-cpp-python (install with: poetry install --extras llm)""",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to reason about",
                    },
                    "project": {
                        "type": "string",
                        "description": "Filter memories by project",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 10,
                        "description": "Number of memories to feed to the LLM",
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="memory_analyze",
            description="""Use the local LLM to analyze memories for patterns, conflicts, and gaps.

WHEN TO USE:
- Start of a new coding session (get a full context briefing)
- Architecture review (find conflicting decisions)
- Knowledge audit (identify what's missing)
- Before major refactoring (understand all related decisions)

Analyzes all memories for a project using a local LLM and returns:
- Patterns found (recurring conventions, approaches)
- Potential conflicts (contradicting decisions)
- Knowledge gaps (missing context)
- Recommendations (what to clarify or remember next)

REQUIRES: llama-cpp-python (install with: poetry install --extras llm)""",
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project to analyze",
                    },
                    "focus": {
                        "type": "string",
                        "description": "Focus area (e.g., 'architecture', 'testing', 'auth')",
                    },
                },
            },
        ),
        Tool(
            name="memory_compress",
            description="""Compress all embeddings using TurboQuant for storage optimization.

WHEN TO USE:
- After storing many memories (optimize storage)
- When checking compression efficiency
- Performance tuning

Uses the TurboQuant algorithm (PolarQuant + QJL) to compress embedding
vectors from float32 (1,536 bytes each) to ~168-216 bytes each,
achieving 7-9x compression with >99% similarity accuracy.

This is idempotent — already-compressed memories are skipped.""",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    memory = get_memory()

    try:
        if name == "memory_store":
            result = memory.store(
                text=arguments["text"],
                project=arguments.get("project"),
                type=arguments.get("type"),
                importance=arguments.get("importance"),
                file=arguments.get("file"),
            )
            compressed_info = ""
            if result.get("compressed"):
                compressed_info = (
                    f"\nTurboQuant: compressed in {result['compression_time_ms']:.1f}ms"
                )
            return [
                TextContent(
                    type="text",
                    text=(
                        f"✅ Memory stored\n\n"
                        f"ID: {result['id']}\n"
                        f"Text: {result['text'][:100]}...\n"
                        f"Embedding time: {result['embedding_time_ms']:.1f}ms"
                        f"{compressed_info}"
                    ),
                )
            ]

        elif name == "memory_recall":
            results = memory.recall(
                query=arguments["query"],
                top_k=arguments.get("top_k", 5),
                project=arguments.get("project"),
                type=arguments.get("type"),
                min_similarity=arguments.get("min_similarity", 0.3),
            )

            if not results:
                return [
                    TextContent(
                        type="text",
                        text=(
                            "No relevant memories found. "
                            "Try a different query or lower the similarity threshold."
                        ),
                    )
                ]

            output = [f"Found {len(results)} relevant memories:\n"]
            for i, r in enumerate(results, 1):
                output.append(f"\n{i}. [{r['similarity']:.3f}] {r['text'][:150]}...")
                if r.get("metadata", {}).get("project"):
                    output.append(f"   Project: {r['metadata']['project']}")
                if r.get("metadata", {}).get("type"):
                    output.append(f"   Type: {r['metadata']['type']}")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "memory_list":
            results = memory.list(
                project=arguments.get("project"),
                type=arguments.get("type"),
                since=arguments.get("since"),
                limit=arguments.get("limit", 100),
            )

            if not results:
                return [
                    TextContent(
                        type="text",
                        text="No memories found matching criteria.",
                    )
                ]

            output = [f"Found {len(results)} memories:\n"]
            for i, r in enumerate(results, 1):
                output.append(f"\n{i}. {r['text'][:100]}...")
                output.append(f"   ID: {r['id']}")
                if r.get("metadata", {}).get("type"):
                    output.append(f"   Type: {r['metadata']['type']}")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "memory_forget":
            deleted = memory.forget(arguments["id"])
            if deleted:
                return [
                    TextContent(
                        type="text",
                        text=f"✅ Memory {arguments['id']} deleted",
                    )
                ]
            else:
                return [
                    TextContent(
                        type="text",
                        text=f"❌ Memory {arguments['id']} not found",
                    )
                ]

        elif name == "memory_summarize":
            summary = memory.summarize(arguments.get("project"))

            output = ["📊 Memory Summary\n"]
            output.append(f"Total memories: {summary['stats']['total_memories']}")
            output.append(
                f"Database size: {summary['stats']['db_size_bytes'] / 1024 / 1024:.2f} MB"
            )

            if summary["by_type"]:
                output.append("\nBy type:")
                for t, count in summary["by_type"].items():
                    output.append(f"  • {t}: {count}")

            if summary["recent"]:
                output.append("\nRecent memories:")
                for r in summary["recent"][:5]:
                    output.append(
                        f"  • [{r.get('type', 'note')}] {r['text'][:80]}..."
                    )

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "memory_stats":
            stats = memory.stats()

            output = ["📊 Storage Statistics\n"]
            output.append(f"Total memories: {stats['total_memories']}")
            output.append(
                f"Database size: {stats['db_size_bytes'] / 1024 / 1024:.2f} MB"
            )
            output.append(f"Database path: {stats['db_path']}")

            if stats["by_type"]:
                output.append("\nBy type:")
                for t, count in stats["by_type"].items():
                    output.append(f"  • {t}: {count}")

            if stats["by_project"]:
                output.append("\nBy project:")
                for p, count in stats["by_project"].items():
                    output.append(f"  • {p}: {count}")

            # TurboQuant compression stats
            if stats.get("compressed_count", 0) > 0:
                output.append("\n🗜️ TurboQuant Compression:")
                output.append(
                    f"  Compressed: {stats['compressed_count']}/{stats['total_memories']}"
                )
                output.append(f"  Compression ratio: {stats['compression_ratio']:.1f}x")
                output.append(
                    f"  Embedding storage: "
                    f"{stats['total_embedding_bytes'] / 1024:.1f} KB "
                    f"→ {stats['total_compressed_bytes'] / 1024:.1f} KB"
                )

            return [TextContent(type="text", text="\n".join(output))]

        # --- New LLM-powered tools (v0.2) ---

        elif name == "memory_think":
            result = memory.think(
                question=arguments["question"],
                project=arguments.get("project"),
                top_k=arguments.get("top_k", 10),
            )

            output = ["🧠 Memory Reasoning\n"]
            output.append(result["answer"])
            output.append(
                f"\n---\nMemories used: {result['memories_used']} | "
                f"Inference: {result['inference_ms']:.0f}ms"
            )
            if result.get("model"):
                output.append(f"Model: {result['model']}")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "memory_analyze":
            result = memory.analyze(
                project=arguments.get("project"),
                focus=arguments.get("focus"),
            )

            output = ["🔍 Memory Analysis\n"]
            output.append(result["analysis"])
            output.append(
                f"\n---\nMemories analyzed: {result['memories_analyzed']} | "
                f"Inference: {result['inference_ms']:.0f}ms"
            )
            if result.get("model"):
                output.append(f"Model: {result['model']}")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "memory_compress":
            result = memory.compress()

            output = ["🗜️ TurboQuant Compression\n"]
            output.append(result["message"])

            if result.get("compression_ratio", 0) > 0:
                output.append(f"\nCompression ratio: {result['compression_ratio']:.1f}x")
                saved_kb = (
                    result.get("total_embedding_bytes", 0)
                    - result.get("total_compressed_bytes", 0)
                ) / 1024
                output.append(f"Storage saved: {saved_kb:.1f} KB")

            return [TextContent(type="text", text="\n".join(output))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Tool error: {e}", exc_info=True)
        return [TextContent(type="text", text=f"❌ Error: {str(e)}")]


async def run_server(db_path: Optional[Path] = None):
    """Run the MCP server."""
    # Initialize memory
    get_memory(db_path)

    # Run via stdio
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
