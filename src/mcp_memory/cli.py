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
        stream=sys.stderr
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
        "--db", type=Path, default=None,
        help="Database path (default: ~/.mcp-memory/memory.db)"
    )
    server_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging"
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
    info_parser = subparsers.add_parser("info", help="Show model info")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    setup_logging(getattr(args, 'verbose', False))
    
    if args.command == "serve":
        asyncio.run(run_server(args.db))
    
    elif args.command == "stats":
        memory = Memory(args.db)
        stats = memory.stats()
        print(f"\n📊 Memory Statistics")
        print(f"  Total memories: {stats['total_memories']}")
        print(f"  Database size: {stats['db_size_bytes'] / 1024 / 1024:.2f} MB")
        print(f"  Database path: {stats['db_path']}")
        if stats['by_type']:
            print("\n  By type:")
            for t, count in stats['by_type'].items():
                print(f"    • {t}: {count}")
        if stats['by_project']:
            print("\n  By project:")
            for p, count in stats['by_project'].items():
                print(f"    • {p}: {count}")
    
    elif args.command == "list":
        memory = Memory(args.db)
        results = memory.list(
            project=args.project,
            type=args.type,
            limit=args.limit
        )
        print(f"\n📚 {len(results)} memories:\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['text'][:100]}...")
            print(f"   ID: {r['id']}")
            if r.get('metadata', {}).get('type'):
                print(f"   Type: {r['metadata']['type']}")
            print()
    
    elif args.command == "search":
        memory = Memory(args.db)
        results = memory.recall(
            query=args.query,
            top_k=args.top_k,
            project=args.project
        )
        print(f"\n🔍 Found {len(results)} results for '{args.query}':\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['similarity']:.3f}] {r['text'][:150]}...")
            if r.get('metadata', {}).get('type'):
                print(f"   Type: {r['metadata']['type']}")
            print()
    
    elif args.command == "store":
        memory = Memory(args.db)
        result = memory.store(
            text=args.text,
            project=args.project,
            type=args.type,
            importance=args.importance
        )
        print(f"\n✅ Memory stored")
        print(f"  ID: {result['id']}")
        print(f"  Text: {result['text'][:100]}...")
        print(f"  Embedding time: {result['embedding_time_ms']:.1f}ms")
    
    elif args.command == "info":
        info = get_model_info()
        print("\n🧠 Model Information")
        print(f"  Model: {info['model_name']}")
        print(f"  Dimension: {info['dimension']}")
        print(f"  Max sequence length: {info['max_sequence_length']}")
        print(f"  Loaded: {info['loaded']}")


if __name__ == "__main__":
    main()
