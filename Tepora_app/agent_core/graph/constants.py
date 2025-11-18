"""
Constants and configuration values for the agent graph system.

This module centralizes all graph-related constants including:
- Node names
- Route identifiers  
- Memory limits
- Command prefixes
"""

from __future__ import annotations


class GraphNodes:
    """LangGraph node name constants."""
    
    # Memory operations
    MEMORY_RETRIEVAL = "memory_retrieval"
    SAVE_MEMORY = "save_memory_node"
    
    # Conversation modes
    DIRECT_ANSWER = "direct_answer"
    GENERATE_SEARCH_QUERY = "generate_search_query"
    EXECUTE_SEARCH = "execute_search"
    SUMMARIZE_SEARCH_RESULT = "summarize_search_result"
    
    # ReAct loop
    GENERATE_ORDER = "generate_order_node"
    AGENT_REASONING = "agent_reasoning_node"
    SYNTHESIZE_FINAL_RESPONSE = "synthesize_final_response_node"
    TOOL_NODE = "tool_node"
    UPDATE_SCRATCHPAD = "update_scratchpad_node"


class GraphRoutes:
    """LangGraph routing condition identifiers."""
    
    AGENT_MODE = "agent_mode"
    SEARCH = "search"
    DIRECT_ANSWER = "direct_answer"
    STATS = "stats"


class MemoryLimits:
    """Memory persistence and context limits."""
    
    # Maximum number of messages to save in persistent memory
    MAX_MEMORY_MESSAGES = 6
    
    # Maximum size in bytes for memory JSON payload
    MAX_MEMORY_JSON_BYTES = 4096
    
    # Maximum tokens for local context in direct answer
    MAX_LOCAL_CONTEXT_TOKENS = 4096


class CommandPrefixes:
    """User input command prefixes."""
    
    AGENT_MODE = "/agentmode"
    SEARCH = "/search"
    EM_STATS = ("/emstats", "/emstats_prof", "/emstats_char")


class RAGConfig:
    """Retrieval-Augmented Generation configuration."""
    
    # Chunk size for text splitting
    CHUNK_SIZE = 800
    
    # Number of top chunks to retrieve
    TOP_K_CHUNKS = 3
    
    # Chunk overlap for text splitting
    CHUNK_OVERLAP = 100


# Attention sink prefix for hierarchical prompting
ATTENTION_SINK_PREFIX = "This is a conversation between a user and an AI assistant."

# Professional agent attention sink
PROFESSIONAL_ATTENTION_SINK = (
    "This is a professional agent session. The agent will now reason and act to complete the given order."
)
