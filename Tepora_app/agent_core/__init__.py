"""Core runtime for the Tepora agent.

Provided submodules:

* :mod:`agent_core.config` - structured configuration package
* :mod:`agent_core.graph` - LangGraph assembly helpers
* :mod:`agent_core.llm_manager` - llama.cpp-backed LLM lifecycle management
* :mod:`agent_core.tools` - native and MCP tool loaders
* :mod:`agent_core.tool_manager` - orchestration layer combining tool loaders
"""

from .graph import AgentCore
from .llm_manager import LLMManager
from .tool_manager import ToolManager

__all__ = [
    "AgentCore",
    "LLMManager",
    "ToolManager",
]
