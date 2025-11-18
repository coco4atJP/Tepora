"""
Backward compatibility layer for legacy graph.py imports.

This module re-exports all components from the new modular graph package.
The monolithic graph.py has been refactored into:
- graph/constants.py - All constants
- graph/utils.py - Helper functions
- graph/routing.py - Routing logic
- graph/nodes/ - Node implementations (memory, conversation, react)
- graph/core.py - AgentCore class

All existing imports from agent_core.graph should continue to work unchanged.
"""

from __future__ import annotations

# Re-export everything from the new graph package
from .graph import (
    ATTENTION_SINK_PREFIX,
    PROFESSIONAL_ATTENTION_SINK,
    AgentCore,
    CommandPrefixes,
    GraphNodes,
    GraphRoutes,
    MemoryLimits,
    RAGConfig,
    append_context_timestamp,
    clone_message_with_timestamp,
    format_episode_list,
    format_scratchpad,
    route_by_command,
    should_continue_react_loop,
    truncate_json_bytes,
)

# Legacy private names for internal use (if any)
_GraphNodes = GraphNodes
_format_scratchpad = format_scratchpad
_append_context_timestamp = append_context_timestamp
_clone_message_with_timestamp = clone_message_with_timestamp

# Memory limits (legacy names)
_MAX_MEMORY_MESSAGES = MemoryLimits.MAX_MEMORY_MESSAGES
_MAX_MEMORY_JSON_BYTES = MemoryLimits.MAX_MEMORY_JSON_BYTES

# Command prefixes (legacy)
_EM_STATS_COMMAND_PREFIXES = CommandPrefixes.EM_STATS

# RAG config (legacy)
_RAG_CHUNK_SIZE = RAGConfig.CHUNK_SIZE

__all__ = [
    "AgentCore",
    "GraphNodes",
    "GraphRoutes",
    "CommandPrefixes",
    "MemoryLimits",
    "RAGConfig",
    "ATTENTION_SINK_PREFIX",
    "PROFESSIONAL_ATTENTION_SINK",
    "route_by_command",
    "should_continue_react_loop",
    "format_scratchpad",
    "append_context_timestamp",
    "clone_message_with_timestamp",
    "format_episode_list",
    "truncate_json_bytes",
]
