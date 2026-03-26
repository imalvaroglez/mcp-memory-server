"""
MCP Memory Server - Local persistent memory for IDE assistants.

A Model Context Protocol server that provides memory tools for
code assistants like Cursor, Windsurf, and Claude Code.
"""

__version__ = "0.1.0"
__author__ = "Tailor Made"

from mcp_memory.server import MCPServer
from mcp_memory.memory import MemoryStore

__all__ = ["MCPServer", "MemoryStore"]
