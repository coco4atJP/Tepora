"""
Agent graph package.

This package provides a modular implementation of the agent's
execution graph using LangGraph.

Main components:
- AgentCore: Main graph orchestrator
- EMEnabledAgentCore: EM-LLM integrated graph
- Node implementations: Memory, Conversation, ReAct, EM-LLM
- Routing logic
- Constants and utilities

The legacy monolithic graph.py is replaced by this modular structure.
"""

from __future__ import annotations

from .constants import (
    ATTENTION_SINK_PREFIX,
    PROFESSIONAL_ATTENTION_SINK,
    CommandPrefixes,
    GraphNodes,
    GraphRoutes,
    MemoryLimits,
    RAGConfig,
)
from .core import AgentCore
from .em_llm_core import EMEnabledAgentCore
from .routing import route_by_command, should_continue_react_loop
from .utils import (
    append_context_timestamp,
    clone_message_with_timestamp,
    format_episode_list,
    format_scratchpad,
    truncate_json_bytes,
)

# For backward compatibility with existing imports
__all__ = [
    "AgentCore",
    "EMEnabledAgentCore",
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
