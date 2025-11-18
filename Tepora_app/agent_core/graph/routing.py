"""
Graph routing logic.

This module provides routing functions that determine
which path the graph should take based on state.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import AIMessage

from .constants import CommandPrefixes, GraphRoutes

if TYPE_CHECKING:
    from ..state import AgentState

logger = logging.getLogger(__name__)


def route_by_command(state: AgentState) -> Literal["agent_mode", "search", "direct_answer", "stats"]:
    """
    Determine route based on user input command.
    
    Args:
        state: Current agent state
        
    Returns:
        Route identifier string
    """
    user_input = state["input"].strip().lower()
    logger.info(f"Routing decision for input: '{user_input}'")
    
    if user_input.startswith(CommandPrefixes.AGENT_MODE):
        logger.info("Route: agent_mode (ReAct loop)")
        return GraphRoutes.AGENT_MODE
    elif user_input.startswith(CommandPrefixes.SEARCH):
        logger.info("Route: search")
        return GraphRoutes.SEARCH
    elif any(user_input.startswith(prefix) for prefix in CommandPrefixes.EM_STATS):
        logger.info("Route: stats (EM diagnostics)")
        return GraphRoutes.STATS
    else:
        logger.info("Route: direct_answer")
        return GraphRoutes.DIRECT_ANSWER


def should_continue_react_loop(state: AgentState) -> Literal["continue", "end"]:
    """
    Determine if ReAct loop should continue based on tool calls or finish action.
    
    Args:
        state: Current agent state
        
    Returns:
        "continue" if loop should continue, "end" if it should terminate
    """
    logger.debug("Decision: Should Continue ReAct Loop?")
    
    if "agent_outcome" in state and state["agent_outcome"]:
        logger.info("Decision: End ReAct loop (finish action detected).")
        logger.debug(f"Final outcome: {state['agent_outcome']}")
        return "end"
    
    # If last message in scratchpad has tool calls, continue
    if state["agent_scratchpad"]:
        last_message = state["agent_scratchpad"][-1]
        logger.debug(f"Last message in scratchpad: {type(last_message).__name__}")
        
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            logger.info("Decision: Continue ReAct loop (last message has tool calls).")
            logger.debug(f"Tool calls: {[tc['name'] for tc in last_message.tool_calls]}")
            return "continue"
        else:
            logger.info("Decision: End ReAct loop (last message has no tool calls).")
            if isinstance(last_message, AIMessage):
                logger.debug(f"Last message content: {last_message.content[:100]}...")
            return "end"
    else:
        logger.info("Decision: End ReAct loop (empty scratchpad).")
        return "end"
