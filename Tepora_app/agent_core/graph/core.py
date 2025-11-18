"""
Core agent graph implementation.

This module provides the AgentCore class that orchestrates
all graph nodes and edges for the conversational agent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from langgraph.graph import END, StateGraph

from ..state import AgentState
from .constants import GraphNodes, GraphRoutes
from .nodes import ConversationNodes, MemoryNodes, ReActNodes
from .routing import route_by_command, should_continue_react_loop

if TYPE_CHECKING:
    from ..llm_manager import LLMManager
    from ..memory.memory_system import MemorySystem
    from ..tool_manager import ToolManager

logger = logging.getLogger(__name__)


class AgentCore:
    """
    Facade for building and executing the application graph.
    
    This class orchestrates:
    - Memory operations (retrieval and persistence)
    - Multiple conversation modes (direct, search, agent)
    - ReAct loop for complex reasoning tasks
    - Tool execution through ToolManager
    """
    
    def __init__(
        self,
        llm_manager: LLMManager,
        tool_manager: ToolManager,
        memory_system: Optional[MemorySystem] = None
    ):
        """
        Initialize AgentCore with required managers.
        
        Args:
            llm_manager: Manager for LLM model loading and access
            tool_manager: Manager for tool discovery and execution
            memory_system: Optional memory system for episode storage
        """
        self.llm_manager = llm_manager
        self.tool_manager = tool_manager
        self.memory_system = memory_system
        
        # Initialize node implementations
        self._memory_nodes = MemoryNodes(memory_system)
        self._conversation_nodes = ConversationNodes(llm_manager, tool_manager)
        self._react_nodes = ReActNodes(llm_manager, tool_manager)
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info("AgentCore initialized successfully")
    
    # ========== Node Methods (delegate to node implementations) ==========
    
    def memory_retrieval_node(self, state: AgentState) -> dict:
        """Delegate to MemoryNodes."""
        return self._memory_nodes.memory_retrieval_node(state)
    
    def save_memory_node(self, state: AgentState) -> dict:
        """Delegate to MemoryNodes."""
        return self._memory_nodes.save_memory_node(state)
    
    async def direct_answer_node(self, state: AgentState) -> dict:
        """Delegate to ConversationNodes."""
        return await self._conversation_nodes.direct_answer_node(state)
    
    async def generate_search_query_node(self, state: AgentState) -> dict:
        """Delegate to ConversationNodes."""
        return await self._conversation_nodes.generate_search_query_node(state)
    
    def execute_search_node(self, state: AgentState) -> dict:
        """Delegate to ConversationNodes."""
        return self._conversation_nodes.execute_search_node(state)
    
    async def summarize_search_result_node(self, state: AgentState) -> dict:
        """Delegate to ConversationNodes."""
        return await self._conversation_nodes.summarize_search_result_node(state)
    
    def generate_order_node(self, state: AgentState) -> dict:
        """Delegate to ReActNodes."""
        return self._react_nodes.generate_order_node(state)
    
    def agent_reasoning_node(self, state: AgentState) -> dict:
        """Delegate to ReActNodes."""
        return self._react_nodes.agent_reasoning_node(state)
    
    def update_scratchpad_node(self, state: AgentState) -> dict:
        """Delegate to ReActNodes."""
        return self._react_nodes.update_scratchpad_node(state)
    
    def unified_tool_executor_node(self, state: AgentState) -> dict:
        """Delegate to ReActNodes."""
        return self._react_nodes.unified_tool_executor_node(state)
    
    async def synthesize_final_response_node(self, state: AgentState) -> dict:
        """Delegate to ReActNodes."""
        return await self._react_nodes.synthesize_final_response_node(state)
    
    # ========== Routing Methods ==========
    
    def route_by_command(self, state: AgentState):
        """Delegate to routing module."""
        return route_by_command(state)
    
    def should_continue_react_loop(self, state: AgentState):
        """Delegate to routing module."""
        return should_continue_react_loop(state)
    
    # ========== Graph Construction ==========
    
    def _build_graph(self):
        """
        Build and compile the LangGraph workflow.
        
        Returns:
            Compiled LangGraph
        """
        workflow = StateGraph(AgentState)
        
        # ========== Register Nodes ==========
        
        # Memory operations
        workflow.add_node(GraphNodes.MEMORY_RETRIEVAL, self.memory_retrieval_node)
        workflow.add_node(GraphNodes.SAVE_MEMORY, self.save_memory_node)
        
        # Conversation modes
        workflow.add_node(GraphNodes.DIRECT_ANSWER, self.direct_answer_node)
        workflow.add_node(GraphNodes.GENERATE_SEARCH_QUERY, self.generate_search_query_node)
        workflow.add_node(GraphNodes.EXECUTE_SEARCH, self.execute_search_node)
        workflow.add_node(GraphNodes.SUMMARIZE_SEARCH_RESULT, self.summarize_search_result_node)
        
        # ReAct loop
        workflow.add_node(GraphNodes.GENERATE_ORDER, self.generate_order_node)
        workflow.add_node(GraphNodes.AGENT_REASONING, self.agent_reasoning_node)
        workflow.add_node(GraphNodes.SYNTHESIZE_FINAL_RESPONSE, self.synthesize_final_response_node)
        workflow.add_node(GraphNodes.TOOL_NODE, self.unified_tool_executor_node)
        workflow.add_node(GraphNodes.UPDATE_SCRATCHPAD, self.update_scratchpad_node)
        
        # ========== Connect Graph ==========
        
        # 1. Entry point: memory retrieval
        workflow.set_entry_point(GraphNodes.MEMORY_RETRIEVAL)
        
        # 2. Route after memory retrieval based on command
        workflow.add_conditional_edges(
            GraphNodes.MEMORY_RETRIEVAL,
            self.route_by_command,
            {
                GraphRoutes.AGENT_MODE: GraphNodes.GENERATE_ORDER,
                GraphRoutes.SEARCH: GraphNodes.GENERATE_SEARCH_QUERY,
                GraphRoutes.DIRECT_ANSWER: GraphNodes.DIRECT_ANSWER,
                GraphRoutes.STATS: GraphNodes.DIRECT_ANSWER,
            }
        )
        
        # 3. Flow for each branch
        
        # Direct Answer Path
        workflow.add_edge(GraphNodes.DIRECT_ANSWER, GraphNodes.SAVE_MEMORY)
        
        # Search Path
        workflow.add_edge(GraphNodes.GENERATE_SEARCH_QUERY, GraphNodes.EXECUTE_SEARCH)
        workflow.add_edge(GraphNodes.EXECUTE_SEARCH, GraphNodes.SUMMARIZE_SEARCH_RESULT)
        workflow.add_edge(GraphNodes.SUMMARIZE_SEARCH_RESULT, GraphNodes.SAVE_MEMORY)
        
        # Agent (ReAct) Path
        workflow.add_edge(GraphNodes.GENERATE_ORDER, GraphNodes.AGENT_REASONING)
        workflow.add_conditional_edges(
            GraphNodes.AGENT_REASONING,
            self.should_continue_react_loop,
            {
                "continue": GraphNodes.TOOL_NODE,
                "end": GraphNodes.SYNTHESIZE_FINAL_RESPONSE
            },
        )
        workflow.add_edge(GraphNodes.TOOL_NODE, GraphNodes.UPDATE_SCRATCHPAD)
        workflow.add_edge(GraphNodes.UPDATE_SCRATCHPAD, GraphNodes.AGENT_REASONING)
        workflow.add_edge(GraphNodes.SYNTHESIZE_FINAL_RESPONSE, GraphNodes.SAVE_MEMORY)
        
        # 4. Final exit
        workflow.add_edge(GraphNodes.SAVE_MEMORY, END)
        
        logger.info("Graph construction complete")
        return workflow.compile()
