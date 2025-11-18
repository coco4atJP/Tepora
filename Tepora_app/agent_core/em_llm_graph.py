"""
Backward compatibility layer for legacy em_llm_graph.py imports.

This module re-exports EMEnabledAgentCore from the new modular graph package.
The EM-LLM graph implementation has been moved to:
- graph/nodes/em_llm.py - EM-LLM specific node implementations
- graph/em_llm_core.py - EMEnabledAgentCore class

All existing imports from agent_core.em_llm_graph should continue to work unchanged.
"""

from __future__ import annotations

# Re-export from the new graph package
from .graph import EMEnabledAgentCore

__all__ = [
    "EMEnabledAgentCore",
]
