"""Memory MCP -- MCP server for Memory SDK."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("recollect-mcp")
except PackageNotFoundError:  # source-tree import before install
    __version__ = "0.0.0"
