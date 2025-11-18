from __future__ import annotations

"""Utilities and tool definitions used by :mod:`agent_core`."""

from .native import (
    GoogleCustomSearchInput,
    GoogleCustomSearchTool,
    WebFetchInput,
    WebFetchTool,
    load_native_tools,
)
from .mcp import (
    load_connections_from_config,
    load_mcp_tools_robust,
    load_mcp_tools,
)

__all__ = [
    "GoogleCustomSearchInput",
    "GoogleCustomSearchTool",
    "WebFetchInput",
    "WebFetchTool",
    "load_native_tools",
    "load_connections_from_config",
    "load_mcp_tools_robust",
    "load_mcp_tools",
]
