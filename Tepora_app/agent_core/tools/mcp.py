from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient, StdioConnection

logger = logging.getLogger(__name__)

__all__ = [
    "load_connections_from_config",
    "load_mcp_tools_robust",
    "load_mcp_tools",
]


def load_connections_from_config(config_path: Path) -> Dict[str, StdioConnection]:
    try:
        config_data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load MCP config from %s: %s", config_path, exc)
        return {}

    connections: Dict[str, StdioConnection] = {}
    for server_name, server_config in config_data.get("mcpServers", {}).items():
        command = server_config.get("command")
        if not command:
            logger.warning("Skipping server '%s': 'command' key is missing.", server_name)
            continue
        connections[server_name] = StdioConnection(
            transport="stdio",
            command=command,
            args=server_config.get("args", []),
            env=server_config.get("env"),
        )
        logger.info("Loaded MCP server config: %s", server_name)
    return connections


def load_mcp_tools(config_path: Path) -> List[BaseTool]:
    connections = load_connections_from_config(config_path)
    if not connections:
        logger.warning("No MCP server connections found.")
        return []

    try:
        client = MultiServerMCPClient(connections=connections)
        tools = asyncio.run(client.get_tools())
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to get tools from MCP servers: %s", exc, exc_info=True)
        return []

    unique_tools: List[BaseTool] = []
    for tool in tools:
        server_name = tool.metadata.get("server_name") if tool.metadata else None
        if server_name:
            tool.name = f"{server_name}_{tool.name}"
        unique_tools.append(tool)
    for tool in unique_tools:
        logger.info("MCP tool available: %s", tool.name)
    return unique_tools


def load_mcp_tools_robust(config_path: Path) -> List[BaseTool]:
    connections = load_connections_from_config(config_path)
    if not connections:
        logger.warning("No MCP server connections found.")
        return []

    async def _load() -> List[BaseTool]:
        tools: List[BaseTool] = []
        for server_name, connection in connections.items():
            try:
                tools.extend(await _load_single_server(server_name, connection))
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load tools from MCP server '%s': %s", server_name, exc)
        return tools

    return asyncio.run(_load())


async def _load_single_server(server_name: str, connection: StdioConnection, max_retries: int = 3) -> List[BaseTool]:
    for attempt in range(max_retries):
        try:
            if attempt:
                delay = 2 ** attempt
                logger.info("Waiting %s seconds before retrying server %s", delay, server_name)
                await asyncio.sleep(delay)

            client = MultiServerMCPClient(connections={server_name: connection})
            discovered_tools = await client.get_tools()
            for tool in discovered_tools:
                tool.name = f"{server_name}_{tool.name}"
            return list(discovered_tools)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Attempt %s failed for server '%s': %s", attempt + 1, server_name, exc)
            if attempt == max_retries - 1:
                raise
    return []
