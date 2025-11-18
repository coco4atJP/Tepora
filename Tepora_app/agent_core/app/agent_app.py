"""
Main agent application class.

This module provides the AgentApplication class that manages
the complete lifecycle of the EM-LLM agent.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from .. import config
from ..em_llm import EMConfig, EMLLMIntegrator
from ..embedding_provider import EmbeddingProvider
from ..graph import AgentCore, EMEnabledAgentCore
from ..llm_manager import LLMManager
from ..memory.memory_system import MemorySystem
from ..tool_manager import ToolManager
from .utils import ainput, display_em_stats, sanitize_user_input

logger = logging.getLogger(__name__)


class AgentApplication:
    """
    Main application class for EM-LLM enhanced AI agent.
    
    This class manages:
    - Initialization of all components (LLM, tools, memory, graph)
    - Interactive conversation loop
    - Resource cleanup
    """
    
    def __init__(self):
        """Initialize the agent application (components created during run)."""
        self.llm_manager: Optional[LLMManager] = None
        self.tool_manager: Optional[ToolManager] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None
        self.char_em_llm_integrator: Optional[EMLLMIntegrator] = None
        self.prof_em_llm_integrator: Optional[EMLLMIntegrator] = None
        self.app = None
        self.chat_history: List = []
    
    async def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        try:
            print("Initializing EM-LLM Enhanced AI Agent...")
            print("=" * 60)
            
            # === Phase 1: Core systems ===
            print("Phase 1: Initializing core systems...")
            
            if not await self._initialize_core_systems():
                return False
            
            # === Phase 2: EM-LLM systems ===
            print("\nPhase 2: Initializing EM-LLM systems...")
            
            em_llm_success = await self._initialize_em_llm_systems()
            
            # === Phase 3: Application graph ===
            print("\nPhase 3: Building application graph...")
            
            if not await self._build_application_graph(em_llm_success):
                return False
            
            print("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"Critical error during initialization: {e}", exc_info=True)
            print(f"\n❌ Failed to start the AI agent: {e}")
            print("Please check the logs and configuration.")
            return False
    
    async def _initialize_core_systems(self) -> bool:
        """Initialize LLM manager and tool manager."""
        try:
            # LLM manager
            self.llm_manager = LLMManager()
            self.llm_manager.get_character_agent()  # Preload main LLM
            print("✓ LLM Manager initialized")
            
            # Tool manager
            self.tool_manager = ToolManager(config_file=config.MCP_CONFIG_FILE)
            self.tool_manager.initialize()
            print(f"✓ Tool Manager initialized with {len(self.tool_manager.tools)} tools")
            
            return True
        except Exception as e:
            logger.error(f"Core system initialization failed: {e}", exc_info=True)
            return False
    
    async def _initialize_em_llm_systems(self) -> bool:
        """Initialize EM-LLM memory systems."""
        try:
            # Load embedding model
            embedding_llm = self.llm_manager.get_embedding_model()
            self.embedding_provider = EmbeddingProvider(embedding_llm)
            print("✓ Embedding provider initialized")
            
            # Initialize EM config
            em_config = EMConfig(**config.EM_LLM_CONFIG)
            
            # Character agent memory system
            char_em_memory_system = MemorySystem(
                self.embedding_provider,
                db_path="./chroma_db_em_llm",
                collection_name="em_llm_events_char"
            )
            print("✓ Character EM-LLM memory system initialized")
            self.char_em_llm_integrator = EMLLMIntegrator(
                self.llm_manager,
                self.embedding_provider,
                em_config,
                char_em_memory_system
            )
            
            # Professional agent memory system
            prof_em_memory_system = MemorySystem(
                self.embedding_provider,
                db_path="./chroma_db_em_llm",
                collection_name="em_llm_events_prof"
            )
            print("✓ Professional EM-LLM memory system initialized")
            self.prof_em_llm_integrator = EMLLMIntegrator(
                self.llm_manager,
                self.embedding_provider,
                em_config,
                prof_em_memory_system
            )
            
            print("✓ EM-LLM configuration applied")
            print("✓ EM-LLM integrator initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"EM-LLM initialization failed: {e}", exc_info=True)
            print(f"⚠ EM-LLM initialization failed: {e}. Check logs for details.")
            print("Falling back to traditional memory system...")
            return False
    
    async def _build_application_graph(self, em_llm_enabled: bool) -> bool:
        """Build the application graph based on available features."""
        try:
            if em_llm_enabled and self.char_em_llm_integrator:
                # EM-LLM enhanced graph
                agent_core = EMEnabledAgentCore(
                    self.llm_manager,
                    self.tool_manager,
                    self.char_em_llm_integrator,
                    self.prof_em_llm_integrator
                )
                self.app = agent_core.graph
                print("✓ EM-LLM enhanced graph initialized")
                
                # Display initial statistics
                total_char_events = self.char_em_llm_integrator.memory_system.count()
                summary = (
                    f"{total_char_events} events loaded from persistent storage."
                    if total_char_events > 0
                    else "Ready (no prior events)."
                )
                print(f"✓ Character EM-LLM Memory: {summary}")
            else:
                # Fallback: traditional system
                if self.embedding_provider:
                    print("Re-using embedding provider for fallback memory system.")
                    memory_system = MemorySystem(
                        self.embedding_provider,
                        db_path="./chroma_db_fallback"
                    )
                    agent_core = AgentCore(self.llm_manager, self.tool_manager, memory_system)
                else:
                    print("⚠ Embedding provider is not available. Fallback memory system will be disabled.")
                    agent_core = AgentCore(self.llm_manager, self.tool_manager, None)
                
                self.app = agent_core.graph
                print("✓ Traditional graph initialized (fallback mode)")
            
            return True
            
        except Exception as e:
            logger.error(f"Graph construction failed: {e}", exc_info=True)
            return False
    
    def display_welcome_message(self):
        """Display welcome message and available commands."""
        if self.char_em_llm_integrator:
            print("🧠 EM-LLM Enhanced AI Agent is ready!")
            print("Features: Surprise-based memory formation, Two-stage retrieval, Episodic segmentation")
        else:
            print("🤖 AI Agent is ready (traditional mode)")
        
        print("\nCommands:")
        print("  • '/agentmode <request>' - Complex task with tools")
        print("  • '/search <query>' - Web search")
        print("  • '/emstats' - Character agent's memory statistics (if available)")
        print("  • '/emstats_prof' - Professional agent's memory statistics (if available)")
        print("  • Normal chat - Direct conversation")
        print("  • 'exit' - Quit")
        print("-" * 60)
    
    async def handle_stats_command(self, user_input: str) -> bool:
        """
        Handle statistics display commands.
        
        Args:
            user_input: User input string
            
        Returns:
            True if command was handled, False otherwise
        """
        if user_input.lower() == config.CMD_EM_STATS and self.char_em_llm_integrator:
            try:
                stats = self.char_em_llm_integrator.get_memory_statistics()
                display_em_stats(stats, "EM-LLM Memory System Statistics")
                return True
            except Exception as e:
                print(f"❌ Failed to retrieve EM-LLM statistics: {e}")
                return True
        
        elif user_input.lower() == config.CMD_EM_STATS_PROF and self.prof_em_llm_integrator:
            try:
                stats = self.prof_em_llm_integrator.get_memory_statistics()
                display_em_stats(stats, "Professional Agent EM-LLM Memory Statistics")
                return True
            except Exception as e:
                print(f"❌ Failed to retrieve Professional Agent EM-LLM statistics: {e}")
                return True
        
        return False
    
    async def process_user_input(self, user_input: str):
        """
        Process user input through the agent graph.
        
        Args:
            user_input: Sanitized user input
        """
        # Prepare initial state
        initial_state = {
            "input": user_input,
            "chat_history": self.chat_history,
            "agent_scratchpad": [],
            "messages": [],
        }
        
        print(f"\n--- Processing (EM-LLM: {'✓' if self.char_em_llm_integrator else '✗'}) ---")
        
        full_response = ""
        
        # Streaming execution
        async for event in self.app.astream_events(
            initial_state,
            version="v2",
            config={"recursion_limit": config.GRAPH_RECURSION_LIMIT}
        ):
            kind = event["event"]
            
            # LLM streaming output
            if kind == config.STREAM_EVENT_CHAT_MODEL:
                content = event["data"]["chunk"].content
                if content:
                    print(content, end="", flush=True)
                    full_response += content
            
            # Graph execution complete
            elif kind == config.STREAM_EVENT_GRAPH_END:
                final_output = event["data"]["output"]
        
        print()  # Newline
        
        # Update chat history
        if full_response:
            self.chat_history.append(HumanMessage(content=user_input))
            self.chat_history.append(AIMessage(content=full_response))
        else:
            # Fallback: response not generated
            print("\nAI: An unexpected error occurred.")
            self.chat_history.append(HumanMessage(content=user_input))
        
        # Truncate chat history if needed
        await self._truncate_chat_history()
    
    async def _truncate_chat_history(self):
        """Truncate chat history to stay within token limit."""
        try:
            if not self.llm_manager:
                return
            
            current_tokens = self.llm_manager.count_tokens_for_messages(self.chat_history)
            if current_tokens > config.MAX_CHAT_HISTORY_TOKENS:
                print(
                    f"INFO: Chat history exceeds token limit "
                    f"({current_tokens}/{config.MAX_CHAT_HISTORY_TOKENS}). Truncating..."
                )
                
                truncated_history = list(self.chat_history)
                # Remove old message pairs (Human & AI) until below limit
                while (
                    self.llm_manager.count_tokens_for_messages(truncated_history) > config.MAX_CHAT_HISTORY_TOKENS
                    and len(truncated_history) > 2
                ):
                    truncated_history = truncated_history[2:]
                
                self.chat_history = truncated_history
                final_tokens = self.llm_manager.count_tokens_for_messages(self.chat_history)
                print(f"INFO: Chat history truncated. Final tokens: {final_tokens}")
                
        except Exception as e:
            logger.warning(f"Could not truncate chat history by tokens: {e}. The history may grow unchecked.")
    
    async def run_conversation_loop(self):
        """Run the main interactive conversation loop."""
        try:
            while True:
                try:
                    raw_user_input = (await ainput("You: ")).strip()
                    
                    if raw_user_input.lower() in ["exit", "quit"]:
                        break
                    if not raw_user_input:
                        continue
                    
                    # Sanitize user input
                    try:
                        user_input = sanitize_user_input(raw_user_input)
                    except ValueError as e:
                        print(f"Error: {e}")
                        continue
                    
                    # Handle statistics commands
                    if await self.handle_stats_command(user_input):
                        continue
                    
                    # Process through graph
                    await self.process_user_input(user_input)
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Exiting EM-LLM Agent. Goodbye!")
                    break
                except Exception as e:
                    logger.error(f"Error during conversation: {e}", exc_info=True)
                    print(f"\n❌ An error occurred: {e}")
                    print("Please try again or type 'exit' to quit.")
                finally:
                    print("-" * 60)
                    
        except Exception as e:
            logger.error(f"Conversation loop failed: {e}", exc_info=True)
    
    async def cleanup(self):
        """Clean up all resources."""
        print("\n🧹 Cleaning up resources...")
        
        if self.llm_manager:
            try:
                self.llm_manager.cleanup()
                print("✓ LLM resources cleaned up")
            except Exception as e:
                print(f"⚠ LLM cleanup warning: {e}")
        
        if self.tool_manager:
            try:
                self.tool_manager.cleanup()
                print("✓ Tool manager cleaned up")
            except Exception as e:
                print(f"⚠ Tool manager cleanup warning: {e}")
        
        if self.char_em_llm_integrator:
            try:
                stats = self.char_em_llm_integrator.get_memory_statistics()
                print(
                    f"✓ Final EM-LLM state: {stats.get('total_events', 0)} events, "
                    f"{stats.get('total_tokens_in_memory', 0)} tokens"
                )
            except Exception as e:
                print(f"⚠ Could not retrieve final EM-LLM statistics: {e}")
        
        print("Cleanup completed.")
    
    async def run(self):
        """
        Main application entry point.
        
        Handles complete lifecycle: initialization, conversation loop, cleanup.
        """
        try:
            # Initialize
            if not await self.initialize():
                return
            
            # Display welcome message
            self.display_welcome_message()
            
            # Run conversation loop
            await self.run_conversation_loop()
            
        finally:
            # Cleanup
            await self.cleanup()
