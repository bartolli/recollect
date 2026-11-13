"""MCP serverInfo identity: name + version from this package, not the mcp lib.

An `MCPServer` built without `version` advertises an empty string.
"""

from __future__ import annotations

from recollect_mcp import __version__
from recollect_mcp.server import mcp


def test_server_name_is_distribution_name() -> None:
    assert mcp.name == "recollect-mcp"


def test_server_version_is_package_version() -> None:
    assert mcp.version == __version__
    assert __version__ != "0.0.0"  # installed distribution resolved
