"""
Conversation-related graph nodes.

This module provides nodes for:
- Direct answer generation
- Search query generation
- Search execution
- Search result summarization
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity

from ... import config
from ..constants import ATTENTION_SINK_PREFIX, MemoryLimits, RAGConfig
from ..utils import clone_message_with_timestamp

if TYPE_CHECKING:
    from ...llm_manager import LLMManager
    from ...state import AgentState
    from ...tool_manager import ToolManager

logger = logging.getLogger(__name__)


class ConversationNodes:
    """Conversation-related graph node implementations."""
    
    def __init__(self, llm_manager: LLMManager, tool_manager: ToolManager):
        """
        Initialize conversation nodes.
        
        Args:
            llm_manager: LLM manager for model access
            tool_manager: Tool manager for tool execution
        """
        self.llm_manager = llm_manager
        self.tool_manager = tool_manager
    
    async def direct_answer_node(self, state: AgentState) -> dict:
        """
        Generate a simple one-turn response with system prompt.
        
        Implements hierarchical context structure:
        1. Attention Sink (fixed prefix)
        2. System/Persona Context
        3. Retrieved Memory (long-term)
        4. Local Context (short-term)
        
        Args:
            state: Current agent state
            
        Returns:
            Updated chat_history and generation_logprobs
        """
        logger.info("--- Node: Direct Answer (Streaming, EM-LLM Context) ---")
        
        # Load Gemma-3N
        llm = self.llm_manager.get_character_agent()
        
        # Get persona and system prompt
        persona = config.PERSONA_PROMPTS[config.ACTIVE_PERSONA]
        system_prompt = config.resolve_system_prompt("direct_answer")
        
        # Build hierarchical context (EM-LLM & Attention Sink compliant)
        full_history = state.get("chat_history", [])
        
        # 1. Attention Sink (fixed prefix)
        attention_sink_message = SystemMessage(content=ATTENTION_SINK_PREFIX)
        
        # 2. System/Persona Context
        system_persona_message = SystemMessage(
            content=(
                "<instructions>\n"
                "Your persona and instructions for this conversation are defined as follows:\n\n"
                f"<persona_definition>\n{persona}\n</persona_definition>\n\n"
                f"<system_prompt>\n{system_prompt}\n</system_prompt>\n"
                "</instructions>"
            )
        )
        
        # 3. Retrieved Memory Context (long-term)
        retrieved_memory_str = state.get('synthesized_memory', 'No relevant memories found.')
        retrieved_memory_message = SystemMessage(
            content=(
                "--- Relevant Context from Past Conversations ---\n"
                f"{retrieved_memory_str}\n"
            )
        )
        
        # 4. Local Context (short-term) construction
        max_local_tokens = MemoryLimits.MAX_LOCAL_CONTEXT_TOKENS
        local_context = []
        current_local_tokens = 0
        
        for i in range(len(full_history) - 1, -1, -1):
            msg = full_history[i]
            msg_tokens = self.llm_manager.count_tokens_for_messages([msg])
            if current_local_tokens + msg_tokens > max_local_tokens and local_context:
                break
            local_context.insert(0, msg)  # Maintain order
            current_local_tokens += msg_tokens
        
        # 5. Combine all contexts
        base_context = [
            attention_sink_message,
            system_persona_message,
            retrieved_memory_message
        ]
        
        if len(local_context) == len(full_history):
            # Short history: use full history as local context
            context_messages = base_context + local_context
            logger.info(
                f"Context: History is short. Using full history as local context "
                f"({len(local_context)} messages)."
            )
        else:
            # Long history: insert omission notice
            omission_notice = SystemMessage(
                content=(
                    "... (omitted earlier conversation for brevity; rely on the provided "
                    "long-term memories above) ...\n"
                    "--- Returning to recent conversation context ---"
                )
            )
            context_messages = base_context + [omission_notice] + local_context
            logger.info("Context: Using hierarchical structure (Attention Sink > System/Persona > Retrieved > Local).")
            logger.debug(f"  - Local Context: {len(local_context)} messages (~{current_local_tokens} tokens)")
            logger.debug(f"  - Omitted: {len(full_history) - len(local_context)} messages")
        
        context_history = [clone_message_with_timestamp(msg) for msg in context_messages]
        
        # Build prompt and invoke LLM
        prompt = ChatPromptTemplate.from_messages([
            ("placeholder", "{context_history}"),
            ("human", "<user_input>{input}</user_input>")
        ])
        
        chain = prompt | llm
        
        # Request logprobs for surprise calculation
        response_message = await chain.ainvoke(
            {
                "context_history": context_history,
                "input": state["input"],
            },
            config={
                "configurable": {
                    "model_kwargs": {
                        "logprobs": True
                    }
                }
            }
        )
        
        # Extract logprobs from response
        logprobs = response_message.response_metadata.get("logprobs")
        
        return {
            "chat_history": state["chat_history"] + [
                HumanMessage(content=state["input"]),
                AIMessage(content=response_message.content)
            ],
            "generation_logprobs": logprobs,
        }
    
    async def generate_search_query_node(self, state: AgentState) -> dict:
        """
        Generate multiple search queries from user input.
        
        Args:
            state: Current agent state
            
        Returns:
            Dictionary with search_queries list
        """
        logger.info("--- Node: Generate Search Query (using Gemma 3N) ---")
        llm = self.llm_manager.get_character_agent()
        
        prompt = ChatPromptTemplate.from_template(
            "Based on the user's request, propose two diverse and effective web search queries "
            "separated by a newline.\n"
            "User request: \"{input}\""
        )
        chain = prompt | llm
        response_message = await chain.ainvoke({"input": state["input"]})
        
        raw_queries = response_message.content.strip().splitlines()
        queries = [q.strip('- ').strip() for q in raw_queries if q.strip()]
        
        if len(queries) > 2:
            queries = queries[:2]
        elif len(queries) < 2:
            # Fallback: supplement with user input if needed
            fallback_query = state["input"].strip()
            if fallback_query and fallback_query not in queries:
                queries.append(fallback_query)
        
        logger.info(f"Generated search queries: {queries}")
        return {"search_queries": queries}
    
    def execute_search_node(self, state: AgentState) -> dict:
        """
        Execute Google Custom Search API tool and aggregate results.
        
        Args:
            state: Current agent state
            
        Returns:
            Dictionary with search_results
        """
        logger.info("--- Node: Execute Search ---")
        queries = state.get("search_queries") or []
        aggregated_results = []
        
        for query in queries:
            logger.info("Executing search for query: '%s'", query)
            raw_result = self.tool_manager.execute_tool("native_google_search", {"query": query})
            
            if not isinstance(raw_result, str):
                logger.warning("Unexpected search result type for query '%s': %s", query, type(raw_result))
                aggregated_results.append({
                    "query": query,
                    "results": [{"error": "Received unexpected result format from search tool."}]
                })
                continue
            
            if raw_result.strip().startswith("Error:"):
                logger.warning("Search tool returned error for query '%s': %s", query, raw_result)
                aggregated_results.append({"query": query, "results": [{"error": raw_result}]})
                continue
            
            try:
                parsed = json.loads(raw_result)
                aggregated_results.append({"query": query, "results": parsed.get("results", [])})
            except json.JSONDecodeError:
                logger.warning("Failed to parse search result for query '%s'. payload=%s", query, raw_result[:200])
                aggregated_results.append({
                    "query": query,
                    "results": [{"error": "Failed to parse search results."}]
                })
        
        return {"search_results": aggregated_results}
    
    async def summarize_search_result_node(self, state: AgentState) -> dict:
        """
        Convert search results into a user-friendly summary using RAG.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated messages, chat_history, and generation_logprobs
        """
        logger.info("--- Node: Summarize Search Result (Streaming with RAG) ---")
        
        # Load Gemma-3n and embedding model
        llm = self.llm_manager.get_character_agent()
        embedding_llm = self.llm_manager.get_embedding_model()
        
        # Prepare search result snippets
        search_results_list = state.get("search_results", [])
        search_snippets = json.dumps(search_results_list, ensure_ascii=False, indent=2)
        
        # Identify most promising URL
        top_result_url = None
        if search_results_list and isinstance(search_results_list, list):
            for result_group in search_results_list:
                if (result_group.get("results") and
                    isinstance(result_group["results"], list) and
                    len(result_group["results"]) > 0):
                    top_result_url = result_group["results"][0].get("link")
                    if top_result_url:
                        break
        
        # RAG pipeline
        rag_context = "No relevant content found on the fetched page."
        if top_result_url:
            logger.info("--- Fetching most promising URL: %s ---", top_result_url)
            content = await self.tool_manager.aexecute_tool("native_web_fetch", {"url": top_result_url})
            
            if isinstance(content, str) and content and not content.startswith("Error:"):
                logger.info("--- Fetched content (%d chars). Starting RAG pipeline. ---", len(content))
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=RAGConfig.CHUNK_SIZE,
                    chunk_overlap=RAGConfig.CHUNK_OVERLAP
                )
                chunks = text_splitter.split_text(content)
                logger.info("Split content into %d chunks.", len(chunks))
                
                if chunks:
                    if not all(hasattr(embedding_llm, attr) for attr in ("embed_query", "embed_documents")):
                        logger.error("Embedding model does not expose embed_query/embed_documents. Skipping RAG.")
                        rag_context = "Embedding model unavailable for RAG."
                    else:
                        query_embedding = np.array(embedding_llm.embed_query(state["input"]))
                        query_embedding = query_embedding.reshape(1, -1)
                        chunk_embeddings = np.array(embedding_llm.embed_documents(chunks))
                        
                        if chunk_embeddings.size == 0:
                            logger.warning("Embedding model returned empty embeddings. Skipping similarity search.")
                        else:
                            similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
                            top_k = min(RAGConfig.TOP_K_CHUNKS, len(chunks))
                            top_indices = similarities.argsort()[-top_k:][::-1]
                            relevant_chunks = [chunks[i] for i in top_indices]
                            rag_context = "\n\n---\n\n".join(relevant_chunks)
                            logger.info("Extracted %d most relevant chunks.", len(relevant_chunks))
            else:
                rag_context = f"Failed to fetch content from {top_result_url}. Reason: {content}"
                logger.warning("Web fetch failed for URL '%s': %s", top_result_url, content)
        
        # Build summarization prompt
        persona = config.PERSONA_PROMPTS[config.ACTIVE_PERSONA]
        system_template = """You are a search summarization expert. Your task is to synthesize the provided search result snippets and the most relevant text chunks from a web page to answer the user's original question.
Base your answer *only* on the information given.

User's original question: {original_question}
Search result snippets: {search_snippets}
Relevant content from the web page: {rag_context}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{persona}\n\n{system_template}\n\n--- Relevant Context from Past Conversations ---\n{{synthesized_memory}}"),
            ("placeholder", "{chat_history}"),
            ("human", "Please summarize the search results for my request: {original_question}")
        ])
        
        chain = prompt | llm
        
        response_message = await chain.ainvoke(
            {
                "chat_history": state["chat_history"],
                "synthesized_memory": state.get('synthesized_memory', 'No relevant memories found.'),
                "original_question": state["input"],
                "search_snippets": search_snippets,
                "rag_context": rag_context
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
