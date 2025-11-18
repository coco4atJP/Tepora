"""
Memory-related graph nodes.

This module provides nodes for:
- Memory retrieval
- Memory saving
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

from langchain_core.messages import AIMessage

from ..constants import MemoryLimits
from ..utils import format_episode_list, truncate_json_bytes

if TYPE_CHECKING:
    from ...memory.memory_system import MemorySystem
    from ...state import AgentState

logger = logging.getLogger(__name__)


class MemoryNodes:
    """Memory-related graph node implementations."""
    
    def __init__(self, memory_system: Optional[MemorySystem] = None):
        """
        Initialize memory nodes.
        
        Args:
            memory_system: Optional memory system for episode storage
        """
        self.memory_system = memory_system
    
    def memory_retrieval_node(self, state: AgentState) -> dict:
        """
        Retrieve relevant episodic memories based on user input.
        
        Args:
            state: Current agent state
            
        Returns:
            Dictionary with recalled_episodes and synthesized_memory
        """
        logger.info("--- Node: Memory Retrieval ---")
        
        if not self.memory_system:
            logger.warning("Memory system not available. Skipping retrieval.")
            return {
                "recalled_episodes": [],
                "synthesized_memory": "No memory system available."
            }
        
        try:
            recalled_episodes = self.memory_system.retrieve_similar_episodes(state["input"])
            
            if recalled_episodes:
                logger.info(f"Retrieved {len(recalled_episodes)} relevant episodes.")
                formatted_memory = format_episode_list(recalled_episodes)
                return {
                    "recalled_episodes": recalled_episodes,
                    "synthesized_memory": formatted_memory
                }
            else:
                logger.info("No relevant memories found.")
                return {
                    "recalled_episodes": [],
                    "synthesized_memory": "No relevant memories found."
                }
                
        except Exception as e:
            logger.warning(f"Failed to retrieve memories: {e}", exc_info=True)
            return {
                "recalled_episodes": [],
                "synthesized_memory": "An error occurred during memory retrieval."
            }
    
    def save_memory_node(self, state: AgentState) -> dict:
        """
        Save the final conversation content to memory system.
        
        Args:
            state: Current agent state
            
        Returns:
            Empty dictionary (no state updates)
        """
        logger.info("--- Node: Save Memory ---")
        
        if not self.memory_system:
            logger.warning("Memory system not available. Skipping save.")
            return {}
        
        try:
            # Get the last AI response as summary
            chat_history = state.get("chat_history", [])
            last_ai_message = next(
                (msg for msg in reversed(chat_history) if isinstance(msg, AIMessage)),
                None
            )
            
            if not last_ai_message:
                logger.warning("No AI message found to save.")
                return {}
            
            # Compact history for storage
            compact_history = []
            for msg in chat_history[-MemoryLimits.MAX_MEMORY_MESSAGES:]:
                entry = {
                    "type": type(msg).__name__,
                    "content": msg.content if isinstance(msg.content, str) else str(msg.content)
                }
                compact_history.append(entry)
            
            history_json = json.dumps(compact_history, ensure_ascii=False)
            
            # Truncate if too large
            if len(history_json.encode("utf-8")) > MemoryLimits.MAX_MEMORY_JSON_BYTES:
                history_json = truncate_json_bytes(
                    history_json,
                    MemoryLimits.MAX_MEMORY_JSON_BYTES
                )
            
            # Save to memory system
            self.memory_system.save_episode(
                summary=last_ai_message.content,
                history_json=history_json
            )
            logger.info("Memory saved successfully.")
            
        except Exception as e:
            logger.warning(f"Failed to save memory: {e}", exc_info=True)
        
        return {}
