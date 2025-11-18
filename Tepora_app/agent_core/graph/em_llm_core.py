"""
EM-LLM enabled agent graph core.

This module provides the EMEnabledAgentCore class that integrates
EM-LLM functionality into the agent graph.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from ..state import AgentState
from .constants import GraphRoutes
from .nodes.em_llm import EMMemoryNodes

if TYPE_CHECKING:
    from ..em_llm import EMLLMIntegrator
    from ..llm_manager import LLMManager
    from ..tool_manager import ToolManager

logger = logging.getLogger(__name__)


class _GraphNodes:
    """LangGraph node names for EM-LLM graph."""
    EM_MEMORY_RETRIEVAL = "em_memory_retrieval"
    EM_MEMORY_FORMATION = "em_memory_formation"
    EM_STATS = "em_stats_node"


class _GraphRoutes:
    """LangGraph routing conditions for EM-LLM graph."""
    AGENT_MODE = "agent_mode"
    SEARCH = "search"
    DIRECT_ANSWER = "direct_answer"
    STATS = "stats"


class EMEnabledAgentCore:
    """
    EM-LLM feature-integrated AgentCore class.
    
    This class replaces traditional memory nodes with EM-LLM nodes:
    - memory_retrieval -> em_memory_retrieval (two-stage search)
    - save_memory -> em_memory_formation (surprise-based)
    - Adds em_stats for diagnostics
    """
    
    def __init__(
        self,
        llm_manager: LLMManager,
        tool_manager: ToolManager,
        char_em_llm_integrator: EMLLMIntegrator,
        prof_em_llm_integrator: EMLLMIntegrator = None
    ):
        """
        Initialize EM-LLM enabled agent core.
        
        Args:
            llm_manager: LLM manager for model access
            tool_manager: Tool manager for tool execution
            char_em_llm_integrator: EM-LLM integrator for character agent
            prof_em_llm_integrator: Optional EM-LLM integrator for professional agent
        """
        self.llm_manager = llm_manager
        self.tool_manager = tool_manager
        # Hold both character and professional integrators
        self.char_em_llm_integrator = char_em_llm_integrator
        # Fallback to character if professional not provided
        self.prof_em_llm_integrator = prof_em_llm_integrator if prof_em_llm_integrator else char_em_llm_integrator
        
        # Traditional features are preserved
        from .core import AgentCore
        # EM-LLM graph doesn't use AgentCore memory-related nodes,
        # so initialize without memory_system argument
        self.base_agent_core = AgentCore(llm_manager, tool_manager)
        
        # Initialize EM-LLM specific nodes
        self._em_memory_nodes = EMMemoryNodes(char_em_llm_integrator, prof_em_llm_integrator)
        
        # Rebuild graph (replace with EM-LLM nodes)
        self.graph = self._build_em_llm_graph()
        
        logger.info("EM-LLM enabled Agent Core initialized")
    
    # ========== EM-LLM Node Delegates ==========
    
    def em_memory_retrieval_node(self, state: AgentState) -> dict:
        """Delegate to EMMemoryNodes."""
        return self._em_memory_nodes.em_memory_retrieval_node(state)
    
    async def em_memory_formation_node(self, state: AgentState) -> dict:
        """Delegate to EMMemoryNodes."""
        return await self._em_memory_nodes.em_memory_formation_node(state)
    
    def em_stats_node(self, state: AgentState) -> dict:
        """Delegate to EMMemoryNodes."""
        return self._em_memory_nodes.em_stats_node(state)
    
    # ========== Graph Construction ==========
    
    def _build_em_llm_graph(self):
        """
        Build graph with EM-LLM functionality integrated.
        
        Returns:
            Compiled LangGraph
        """
        from .core import AgentCore
        
        base_core = self.base_agent_core
        workflow = StateGraph(AgentState)
        
        # ========== Register Nodes ==========
        
        # 1. EM-LLM specific nodes
        workflow.add_node(_GraphNodes.EM_MEMORY_RETRIEVAL, self.em_memory_retrieval_node)
        workflow.add_node(_GraphNodes.EM_MEMORY_FORMATION, self.em_memory_formation_node)
        workflow.add_node(_GraphNodes.EM_STATS, self.em_stats_node)
        
        # 2. Reuse nodes from traditional AgentCore
        workflow.add_node("direct_answer", base_core.direct_answer_node)
        workflow.add_node("execute_search", base_core.execute_search_node)
        workflow.add_node("summarize_search_result", base_core.summarize_search_result_node)
        workflow.add_node("generate_order_node", base_core.generate_order_node)
        workflow.add_node("agent_reasoning_node", base_core.agent_reasoning_node)
        workflow.add_node("synthesize_final_response_node", base_core.synthesize_final_response_node)
        workflow.add_node("tool_node", base_core.unified_tool_executor_node)
        workflow.add_node("update_scratchpad_node", base_core.update_scratchpad_node)
        workflow.add_node("generate_search_query", base_core.generate_search_query_node)
        
        # ========== Connect Graph ==========
        
        # 1. Entry point: EM-LLM memory retrieval
        workflow.set_entry_point(_GraphNodes.EM_MEMORY_RETRIEVAL)
        
        # 2. Route after memory retrieval based on command
        workflow.add_conditional_edges(
            _GraphNodes.EM_MEMORY_RETRIEVAL,
            base_core.route_by_command,
            {
                _GraphRoutes.AGENT_MODE: "generate_order_node",
                _GraphRoutes.SEARCH: "generate_search_query",
                _GraphRoutes.DIRECT_ANSWER: "direct_answer",
                _GraphRoutes.STATS: _GraphNodes.EM_STATS,  # Add edge for stats command
            }
        )
        
        # 3. Flow for each branch
        # Direct Answer and Search flows connect to memory formation node
        workflow.add_edge("direct_answer", _GraphNodes.EM_MEMORY_FORMATION)
        workflow.add_edge("generate_search_query", "execute_search")
        workflow.add_edge("execute_search", "summarize_search_result")
        workflow.add_edge("summarize_search_result", _GraphNodes.EM_MEMORY_FORMATION)
        
        # AgentMode (ReAct) Path
        workflow.add_edge("generate_order_node", "agent_reasoning_node")
        workflow.add_conditional_edges(
            "agent_reasoning_node",
            base_core.should_continue_react_loop,
            {
                "continue": "tool_node",
                "end": "synthesize_final_response_node"
            },
        )
        workflow.add_edge("tool_node", "update_scratchpad_node")
        workflow.add_edge("update_scratchpad_node", "agent_reasoning_node")
        # After final response generation in AgentMode, also connect to memory formation
        # This ensures agent task results are persisted as conversation memory
        workflow.add_edge("synthesize_final_response_node", _GraphNodes.EM_MEMORY_FORMATION)
        
        # 4. After memory formation, check stats and end
        workflow.add_edge(_GraphNodes.EM_MEMORY_FORMATION, _GraphNodes.EM_STATS)
        workflow.add_edge(_GraphNodes.EM_STATS, END)
        
        return workflow.compile()
