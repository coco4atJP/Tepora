"""
EM-LLM Enhanced AI Agent - Main Entry Point

This is the main entry point for the EM-LLM (Episodic Memory for Large Language Models)
enhanced AI agent application. The application provides:

- Surprise-based episodic memory formation
- Two-stage memory retrieval (similarity + contiguity)
- Multiple conversation modes (direct, search, agent/ReAct)
- Tool integration via MCP protocol

The core application logic has been refactored into modular components:
- agent_core.app.AgentApplication: Main application class
- agent_core.graph: Graph execution engine
- agent_core.em_llm: EM-LLM memory system
- agent_core.llm_manager: LLM model management
- agent_core.tool_manager: Tool discovery and execution

For detailed architecture documentation, see design_document.txt
"""

import asyncio
import logging
import os

# Disable TorchDynamo for stability
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from agent_core.app import AgentApplication

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main application entry point."""
    app = AgentApplication()
    await app.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Failed to run the EM-LLM agent application: {e}", exc_info=True)
        print(f"❌ Critical failure: {e}")
        print("Check logs for details.")
