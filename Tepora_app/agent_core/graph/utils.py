"""
Utility functions for agent graph operations.

This module provides helper functions for:
- Message formatting
- Scratchpad management
- Timestamp handling
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

logger = logging.getLogger(__name__)


def format_scratchpad(scratchpad: List[BaseMessage]) -> str:
    """
    Format agent_scratchpad contents into a string for LLM consumption.
    
    Args:
        scratchpad: List of messages representing agent's work history
        
    Returns:
        Formatted string representation of the scratchpad
    """
    logger.debug(f"Formatting scratchpad with {len(scratchpad)} messages")
    
    if not scratchpad:
        logger.debug("Scratchpad is empty")
        return ""
    
    string_messages = []
    for i, message in enumerate(scratchpad):
        logger.debug(f"  Message {i+1}: {type(message).__name__}")
        
        if isinstance(message, AIMessage):
            if message.tool_calls:
                # Separate thought and tool call
                thought = message.content
                tool_call = message.tool_calls[0]
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                # Create dictionary and serialize as JSON
                action_obj = {
                    "thought": thought,
                    "action": {
                        "tool_name": tool_name,
                        "args": tool_args
                    }
                }
                formatted_msg = json.dumps(action_obj, ensure_ascii=False)
                string_messages.append(formatted_msg)
                logger.debug(f"    AI Message with tool call: {tool_name}")
            else:
                # AI message without tool call (e.g., error)
                string_messages.append(message.content)
                logger.debug(f"    AI Message without tool call: {message.content[:50]}...")
                
        elif isinstance(message, ToolMessage):
            # Tool execution result
            observation_obj = {"observation": message.content}
            formatted_msg = json.dumps(observation_obj, ensure_ascii=False)
            string_messages.append(formatted_msg)
            logger.debug(f"    Tool Message: {message.content[:50]}...")
    
    # Join each step with newline
    result = "\n".join(string_messages)
    logger.debug(f"Formatted scratchpad length: {len(result)} characters")
    return result


def append_context_timestamp(content: str) -> str:
    """
    Append current datetime to the end of context string.
    
    Args:
        content: Original content string
        
    Returns:
        Content with timestamp appended
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    normalized_content = content.rstrip()
    return f"{normalized_content}\n\n[Context generated at {timestamp}]"


def clone_message_with_timestamp(message: BaseMessage) -> BaseMessage:
    """
    Create a new message instance with timestamp added to content.
    
    Args:
        message: Original message
        
    Returns:
        New message with timestamped content
    """
    if not getattr(message, "content", ""):
        return message
    updated_content = append_context_timestamp(message.content)
    return message.copy(update={"content": updated_content})


def format_episode_list(episodes: List[dict]) -> str:
    """
    Format retrieved episodes into a readable string.
    
    Args:
        episodes: List of episode dictionaries from memory system
        
    Returns:
        Formatted string representation of episodes
    """
    if not episodes:
        return "No relevant memories found."
    
    formatted_parts = []
    for i, ep in enumerate(episodes):
        summary = ep.get('summary', 'N/A')
        formatted_parts.append(f"Recalled Episode {i+1}:\n- Summary: {summary}")
    
    return "\n\n".join(formatted_parts)


def truncate_json_bytes(json_str: str, max_bytes: int) -> str:
    """
    Truncate JSON string to maximum byte length.
    
    Args:
        json_str: JSON string to truncate
        max_bytes: Maximum allowed bytes
        
    Returns:
        Truncated JSON string that fits within byte limit
    """
    if len(json_str.encode("utf-8")) <= max_bytes:
        return json_str
    
    truncated = json_str.encode("utf-8")[:max_bytes]
    return truncated.decode("utf-8", errors="ignore")
