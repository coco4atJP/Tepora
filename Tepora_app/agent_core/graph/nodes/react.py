"""
ReAct loop graph nodes.

This module provides nodes for:
- Order generation
- Agent reasoning (ReAct loop core)
- Scratchpad updates
- Final response synthesis
- Tool execution
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from ... import config
from ..constants import PROFESSIONAL_ATTENTION_SINK
from ..utils import format_scratchpad

if TYPE_CHECKING:
    from ...llm_manager import LLMManager
    from ...state import AgentState
    from ...tool_manager import ToolManager

logger = logging.getLogger(__name__)


class ReActNodes:
    """ReAct loop graph node implementations."""
    
    def __init__(self, llm_manager: LLMManager, tool_manager: ToolManager):
        """
        Initialize ReAct nodes.
        
        Args:
            llm_manager: LLM manager for model access
            tool_manager: Tool manager for tool execution
        """
        self.llm_manager = llm_manager
        self.tool_manager = tool_manager
    
    def generate_order_node(self, state: AgentState) -> dict:
        """
        Character Agent (Gemma) converts user request into a professional "Order".
        
        Args:
            state: Current agent state
            
        Returns:
            Dictionary with order JSON
        """
        logger.info("--- Node: Generate Order (using Gemma 3N) ---")
        llm = self.llm_manager.get_character_agent()
        
        # Use order generation prompt
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                config.BASE_SYSTEM_PROMPTS["order_generation"] +
                "\n\n--- Relevant Context from Past Conversations ---\n{synthesized_memory}"
            ),
            (
                "human",
                "Based on the user's request and the provided context, generate a structured plan (Order).\n\n"
                "--- User Request ---\n{input}\n\n"
                "--- Available Tools ---\n{tools}\n\n"
                "Please generate the JSON order now."
            )
        ])
        chain = prompt | llm
        
        response_message = chain.invoke({
            "input": state["input"],
            "synthesized_memory": state.get("synthesized_memory", "No relevant context."),
            "tools": config.format_tools_for_react_prompt(self.tool_manager.tools)
        })
        
        # Parse LLM-generated JSON string and save to state
        try:
            order_json = json.loads(response_message.content)
            return {"order": order_json}
        except json.JSONDecodeError:
            # Fallback on parse failure
            return {
                "order": {
                    "task_summary": state["input"],
                    "steps": [
                        "Research the user's request using available tools.",
                        "Synthesize the findings."
                    ]
                }
            }
    
    def agent_reasoning_node(self, state: AgentState) -> dict:
        """
        Core ReAct loop node. Prompts LLM for thought and tool use, decides next action.
        
        Processing flow:
        1. Initialize scratchpad if starting ReAct loop
        2. Format past thought/tool execution history
        3. Instruct LLM with REACT_SYSTEM_PROMPT
        4. Parse LLM output as JSON
        5. If "action": create tool call message
        6. If "finish": return result and end loop
        7. On error: add self-correction message to scratchpad
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent_scratchpad and messages, or agent_outcome if finished
        """
        logger.info("--- Node: Agent Reasoning (using Jan-nano) ---")
        logger.debug("Starting ReAct loop...")
        llm = self.llm_manager.get_professional_agent()
        
        if not state["agent_scratchpad"]:
            logger.debug("Initializing agent_scratchpad for new ReAct loop")
            state["agent_scratchpad"] = []
        
        # Build hierarchical prompt (EM-LLM compliant)
        # 1. Attention sink
        attention_sink_prefix = PROFESSIONAL_ATTENTION_SINK
        
        # 2. System prompt
        system_prompt_template = config.BASE_SYSTEM_PROMPTS["react_professional"]
        system_prompt = system_prompt_template.format(
            tools=config.format_tools_for_react_prompt(self.tool_manager.tools)
        )
        
        # 3. Order
        order_plan_str = json.dumps(state.get("order", {}), indent=2, ensure_ascii=False)
        
        # 4. Long-term memory (retrieved from EM-LLM)
        long_term_memory_str = state.get("synthesized_memory", "No relevant long-term memories found.")
        
        # 5. Short-term memory (ReAct loop work history)
        short_term_memory_str = format_scratchpad(state["agent_scratchpad"])
        
        # Build prompt template
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"{attention_sink_prefix}\n\n"
                f"--- System Instructions & Tools ---\n{system_prompt}"
            ),
            (
                "human",
                "You must now execute the following order. Use the provided memories and your reasoning abilities to complete the task.\n\n"
                "--- Order ---\n"
                "User's Original Request: {user_input}\n\n"
                "Execution Plan:\n{order_plan}\n\n"
                "--- Long-Term Memory (Context from past conversations) ---\n{long_term_memory}\n\n"
                "--- Short-Term Memory (Your work history for this order) ---\n{short_term_memory}"
            )
        ])
        chain = prompt | llm
        
        response_message = chain.invoke({
            "user_input": state["input"],
            "order_plan": order_plan_str,
            "long_term_memory": long_term_memory_str,
            "short_term_memory": short_term_memory_str
        })
        
        logger.debug("LLM Raw Output:")
        logger.debug(f"Response content: {response_message.content}")
        
        try:
            # Parse CoT + JSON format output
            content_str = response_message.content
            
            # Search for JSON block with regex
            json_match = re.search(r"```json\n(.*?)\n```", content_str, re.DOTALL)
            
            if not json_match:
                raise ValueError("Invalid format: JSON block not found in the output.")
            
            # Separate thought text and JSON string
            thought_text = content_str[:json_match.start()].strip()
            json_str = json_match.group(1).strip()
            
            logger.debug("Parsed CoT Output:")
            logger.debug(f"Thought: {thought_text}")
            logger.debug(f"JSON String: {json_str}")
            
            parsed_json = json.loads(json_str)
            logger.debug("Parsed JSON successfully:")
            logger.debug(f"{json.dumps(parsed_json, indent=2, ensure_ascii=False)}")
            
            # If "action": create tool call message
            if "action" in parsed_json:
                action = parsed_json["action"]
                
                # Put thought text in AIMessage content
                tool_call_message = AIMessage(
                    content=thought_text,
                    tool_calls=[{
                        "name": action["tool_name"],
                        "args": action.get("args", {}),
                        "id": f"tool_call_{len(state['agent_scratchpad'])}"
                    }]
                )
                
                logger.debug("Tool Call Message Created:")
                logger.debug(f"Content (Thought): {tool_call_message.content}")
                logger.debug(f"Tool calls: {tool_call_message.tool_calls}")
                
                return {
                    "agent_scratchpad": state["agent_scratchpad"] + [tool_call_message],
                    "messages": [tool_call_message]
                }
            
            # If "finish": return result and end loop
            elif "finish" in parsed_json:
                answer = parsed_json["finish"]["answer"]
                logger.info("Finish Action Detected")
                logger.debug(f"Thought: {thought_text}")
                logger.debug(f"Final answer: {answer}")
                
                # Include thought in final report for better context in summary node
                final_report = f"Thought Process:\n{thought_text}\n\nTechnical Report:\n{answer}"
                
                return {"agent_outcome": final_report, "messages": []}
            
            else:
                raise ValueError("Invalid JSON: missing 'action' or 'finish' key.")
        
        except (json.JSONDecodeError, ValueError) as e:
            # On error: add self-correction message to scratchpad
            logger.error(f"Error Parsing LLM Output: {e}")
            logger.debug(f"Raw content that failed to parse: {response_message.content}")
            error_ai_message = AIMessage(
                content=(
                    f"My last attempt failed. The response was not in the correct 'Thought then JSON' format. "
                    f"Error: {e}. I must correct my output to be a plain text thought, followed by a valid JSON "
                    f"block in ```json code fences."
                )
            )
            return {"agent_scratchpad": state["agent_scratchpad"] + [error_ai_message]}
    
    def update_scratchpad_node(self, state: AgentState) -> dict:
        """
        Transfer all ToolMessages added to messages by ToolNode to agent_scratchpad.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent_scratchpad
        """
        logger.info("--- Node: Update Scratchpad ---")
        
        # Collect consecutive ToolMessages from end of messages
        tool_messages = []
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, ToolMessage):
                tool_messages.insert(0, msg)
            else:
                break  # Stop at non-ToolMessage
        
        if not tool_messages:
            logger.warning("No ToolMessage found to update scratchpad.")
            return {}
        
        logger.info(f"Found {len(tool_messages)} tool result(s) to add to scratchpad.")
        logger.debug(f"Current scratchpad length: {len(state['agent_scratchpad'])}")
        
        for i, msg in enumerate(tool_messages):
            logger.debug(f"  - Tool Result {i+1}: {msg.content[:100]}...")
            logger.debug(f"    Tool Call ID: {msg.tool_call_id}")
        
        new_scratchpad = state["agent_scratchpad"] + tool_messages
        logger.debug(f"New scratchpad length: {len(new_scratchpad)}")
        
        return {"agent_scratchpad": new_scratchpad}
    
    def unified_tool_executor_node(self, state: AgentState) -> dict:
        """
        Simple node that delegates tool execution to ToolManager.
        
        Args:
            state: Current agent state
            
        Returns:
            Dictionary with tool result messages
        """
        logger.info("--- Node: Unified Tool Executor ---")
        last_message = state.get("messages", [])[-1]
        
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            logger.debug("No tool calls found in last message")
            return {}
        
        tool_calls = last_message.tool_calls
        logger.info(f"Executing {len(tool_calls)} tool call(s)")
        tool_messages = []
        
        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]
            
            logger.info(f"Tool Execution {i+1}/{len(tool_calls)}: {tool_name}")
            logger.debug(f"Arguments: {json.dumps(tool_args, indent=2, ensure_ascii=False)}")
            logger.debug(f"Call ID: {tool_call_id}")
            
            # Delegate to ToolManager
            logger.debug(f"Executing tool...")
            result_content = self.tool_manager.execute_tool(tool_name, tool_args)
            logger.debug(f"Tool result: {str(result_content)[:200]}...")
            
            tool_messages.append(
                ToolMessage(content=str(result_content), tool_call_id=tool_call_id)
            )
        
        logger.info(f"Tool Execution Complete: Generated {len(tool_messages)} tool result message(s)")
        
        return {"messages": tool_messages}
    
    async def synthesize_final_response_node(self, state: AgentState) -> dict:
        """
        Convert ReAct loop result (internal report) into natural user-facing response.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated messages, chat_history, and generation_logprobs
        """
        logger.info("--- Node: Synthesize Final Response (Streaming) ---")
        
        # Load Gemma-3n
        llm = self.llm_manager.get_character_agent()
        
        # Get internal report generated by ReAct loop
        internal_report = state.get("agent_outcome", "No report generated.")
        
        # Fallback if agent_outcome missing (e.g., ReAct loop error)
        if not state.get("agent_outcome"):
            logger.warning("No agent_outcome found. Synthesizing from scratchpad as a fallback.")
            internal_report = (
                f"The agent could not produce a final report. The following is the internal work log:\n"
                f"{format_scratchpad(state['agent_scratchpad'])}"
            )
        
        logger.debug(f"Internal report for synthesis: {internal_report}")
        logger.debug(f"Original user input: {state['input']}")
        
        # Build synthesis prompt
        persona = config.PERSONA_PROMPTS[config.ACTIVE_PERSONA]
        system_template = config.BASE_SYSTEM_PROMPTS["synthesis"]
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"{persona}\n\n{system_template}\n\n"
                f"--- Relevant Context from Past Conversations ---\n{{synthesized_memory}}"
            ),
            ("placeholder", "{chat_history}"),
            ("human", "Please provide the final response for my request: {original_request}")
        ])
        
        logger.debug("Generating Final Response using synthesis prompt")
        
        chain = prompt | llm
        
        response_message = await chain.ainvoke(
            {
                "chat_history": state["chat_history"],
                "synthesized_memory": state.get('synthesized_memory', 'No relevant memories found.'),
                "original_request": state["input"],
                "technical_report": internal_report
            },
            config={
                "configurable": {
                    "model_kwargs": {
                        "logprobs": True
                    }
                }
            }
        )
        
        # Extract logprobs
        logprobs = response_message.response_metadata.get("logprobs")
        
        return {
            "messages": [AIMessage(content=response_message.content)],
            "chat_history": state["chat_history"] + [
                HumanMessage(content=state["input"]),
                AIMessage(content=response_message.content)
            ],
            "generation_logprobs": logprobs,
        }
