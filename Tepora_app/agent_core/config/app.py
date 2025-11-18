"""
Application-level configuration and constants.

This module provides general application settings including:
- Input validation limits
- Command prefixes
- Display settings
"""

from __future__ import annotations

# Input validation
MAX_INPUT_LENGTH = 10000  # Maximum user input length in characters

# User command prefixes
CMD_AGENT_MODE = "/agentmode"
CMD_SEARCH = "/search"
CMD_EM_STATS = "/emstats"
CMD_EM_STATS_PROF = "/emstats_prof"
CMD_EXIT = ("exit", "quit")

# Prompt injection patterns to detect
DANGEROUS_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"system\s*:",
    r"<\|im_start\|>",
    # More patterns can be added as needed
]

# Graph execution settings
GRAPH_RECURSION_LIMIT = 50

# Streaming event types
STREAM_EVENT_CHAT_MODEL = "on_chat_model_stream"
STREAM_EVENT_GRAPH_END = "on_graph_end"
