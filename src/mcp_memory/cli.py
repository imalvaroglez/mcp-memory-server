"""
Command-line interface for MCP Memory Server.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .server import run_server
from .memory import Memory
from .embeddings import get_model_info

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MCP Memory Server - Persistent memory for IDE assistants"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Server command
    server_parser = subparsers.add_parser("serve", help="Run MCP server")
    server_parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Database path (default: ~/.mcp-memory/memory.db)",
    )
    server_parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to a GGUF model file for the local LLM",
    )
    server_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show storage statistics")
    stats_parser.add_argument("--db", type=Path, default=None)

    # List command
    list_parser = subparsers.add_parser("list", help="List memories")
    list_parser.add_argument("--db", type=Path, default=None)
    list_parser.add_argument("--project", type=str, default=None)
    list_parser.add_argument("--type", type=str, default=None)
    list_parser.add_argument("--limit", type=int, default=20)

    # Search command
    search_parser = subparsers.add_parser("search", help="Search memories")
    search_parser.add_argument("--db", type=Path, default=None)
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--project", type=str, default=None)
    search_parser.add_argument("--top-k", type=int, default=5)

    # Store command
    store_parser = subparsers.add_parser("store", help="Store a memory")
    store_parser.add_argument("--db", type=Path, default=None)
    store_parser.add_argument("text", type=str, help="Text to remember")
    store_parser.add_argument("--project", type=str, default=None)
    store_parser.add_argument("--type", type=str, default=None)
    store_parser.add_argument("--importance", type=int, default=None)

    # Info command
    subparsers.add_parser("info", help="Show embedding model info")

    # Think command (LLM-powered)
    think_parser = subparsers.add_parser(
        "think", help="Use local LLM to reason about memories"
    )
    think_parser.add_argument("--db", type=Path, default=None)
    think_parser.add_argument("question", type=str, help="Question to reason about")
    think_parser.add_argument("--project", type=str, default=None)
    think_parser.add_argument("--top-k", type=int, default=10)

    # Analyze command (LLM-powered)
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze memories for patterns and conflicts"
    )
    analyze_parser.add_argument("--db", type=Path, default=None)
    analyze_parser.add_argument("--project", type=str, default=None)
    analyze_parser.add_argument(
        "--focus", type=str, default=None, help="Focus area for analysis"
    )

    # Compress command (TurboQuant)
    compress_parser = subparsers.add_parser(
        "compress", help="Compress embeddings with TurboQuant"
    )
    compress_parser.add_argument("--db", type=Path, default=None)

    # LLM info command
    llm_info_parser = subparsers.add_parser(
        "llm-info", help="Show local LLM model info"
    )
    llm_info_parser.add_argument("--db", type=Path, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    setup_logging(getattr(args, "verbose", False))

    if args.command == "serve":
        asyncio.run(run_server(args.db))

    elif args.command == "stats":
        memory = Memory(args.db)
        stats = memory.stats()
        print("\n📊 Memory Statistics")
        print(f"  Total memories: {stats['total_memories']}")
        print(f"  Database size: {stats['db_size_bytes'] / 1024 / 1024:.2f} MB")
        print(f"  Database path: {stats['db_path']}")
        if stats["by_type"]:
            print("\n  By type:")
            for t, count in stats["by_type"].items():
                print(f"    • {t}: {count}")
        if stats["by_project"]:
            print("\n  By project:")
            for p, count in stats["by_project"].items():
                print(f"    • {p}: {count}")
        if stats.get("compressed_count", 0) > 0:
            print("\n  🗜️ TurboQuant Compression:")
            print(
                f"    Compressed: {stats['compressed_count']}/{stats['total_memories']}"
            )
            print(f"    Ratio: {stats['compression_ratio']:.1f}x")

    elif args.command == "list":
        memory = Memory(args.db)
        results = memory.list(
            project=args.project,
            type=args.type,
            limit=args.limit,
        )
        print(f"\n📚 {len(results)} memories:\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['text'][:100]}...")
            print(f"   ID: {r['id']}")
            if r.get("metadata", {}).get("type"):
                print(f"   Type: {r['metadata']['type']}")
            print()

    elif args.command == "search":
        memory = Memory(args.db)
        results = memory.recall(
            query=args.query,
            top_k=args.top_k,
            project=args.project,
        )
        print(f"\n🔍 Found {len(results)} results for '{args.query}':\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['similarity']:.3f}] {r['text'][:150]}...")
            if r.get("metadata", {}).get("type"):
                print(f"   Type: {r['metadata']['type']}")
            print()

    elif args.command == "store":
        memory = Memory(args.db)
        result = memory.store(
            text=args.text,
            project=args.project,
            type=args.type,
            importance=args.importance,
        )
        print("\n✅ Memory stored")
        print(f"  ID: {result['id']}")
        print(f"  Text: {result['text'][:100]}...")
        print(f"  Embedding time: {result['embedding_time_ms']:.1f}ms")
        if result.get("compressed"):
            print(f"  TurboQuant: {result['compression_time_ms']:.1f}ms")

    elif args.command == "info":
        info = get_model_info()
        print("\n🧠 Embedding Model Information")
        print(f"  Model: {info['model_name']}")
        print(f"  Dimension: {info['dimension']}")
        print(f"  Max sequence length: {info['max_sequence_length']}")
        print(f"  Loaded: {info['loaded']}")

    elif args.command == "think":
        memory = Memory(args.db)
        print(f"\n🧠 Thinking about: '{args.question}'...\n")
        result = memory.think(
            question=args.question,
            project=args.project,
            top_k=args.top_k,
        )
        print(result["answer"])
        print(
            f"\n---\nMemories used: {result['memories_used']} | "
            f"Inference: {result['inference_ms']:.0f}ms"
        )
        if result.get("model"):
            print(f"Model: {result['model']}")

    elif args.command == "analyze":
        memory = Memory(args.db)
        project_str = f" for '{args.project}'" if args.project else ""
        focus_str = f" (focus: {args.focus})" if args.focus else ""
        print(f"\n🔍 Analyzing memories{project_str}{focus_str}...\n")
        result = memory.analyze(
            project=args.project,
            focus=args.focus,
        )
        print(result["analysis"])
        print(
            f"\n---\nMemories analyzed: {result['memories_analyzed']} | "
            f"Inference: {result['inference_ms']:.0f}ms"
        )

    elif args.command == "compress":
        memory = Memory(args.db)
        print("\n🗜️ Running TurboQuant compression...\n")
        result = memory.compress()
        print(result["message"])
        if result.get("compression_ratio", 0) > 0:
            print(f"  Compression ratio: {result['compression_ratio']:.1f}x")
            saved = (
                result.get("total_embedding_bytes", 0)
                - result.get("total_compressed_bytes", 0)
            )
            print(f"  Storage saved: {saved / 1024:.1f} KB")

    elif args.command == "llm-info":
        memory = Memory(args.db)
        status = memory.llm_status()
        print("\n🤖 Local LLM Information")
        if status.get("available"):
            print(f"  Model: {status['model_name']}")
            print(f"  Path: {status['model_path']}")
            print(f"  Context: {status['context_size']} tokens")
            print(f"  Loaded: {status['loaded']}")
            print(f"  RAM estimate: {status['ram_estimate_mb']:.0f} MB")
            print(f"  GPU layers: {status['gpu_layers']}")
            print(f"  Platform: {status['platform']}")
        else:
            print(f"  ❌ {status.get('message', status.get('error', 'Unknown error'))}")


if __name__ == "__main__":
    main()
