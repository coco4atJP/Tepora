# agent_core/memory_processor.py
import json
import logging
from typing import List
from langchain_core.messages import BaseMessage, AIMessage
from ..llm_manager import LLMManager
from .memory_system import MemorySystem

logger = logging.getLogger(__name__)

class MemoryProcessor:
    """
    対話履歴からエピソードを形成し、MemorySystemに保存する役割を担う。
    """
    def __init__(self, llm_manager: LLMManager, memory_system: MemorySystem):
        self._llm_manager = llm_manager
        self._memory_system = memory_system
        # 要約用SLMは一度ロードしたら、インスタンス内で保持し続ける
        self._summarizer_llm = self._llm_manager.get_slm_summarizer()
        logger.info("MemoryProcessor initialized with pre-loaded SLM.")

    def _should_form_memory(self, chat_history: List[BaseMessage]) -> bool:
        """記憶を形成すべきかどうかの単純なヒューリスティック。"""
        # 4ターン以上の対話があった場合に記憶を形成する
        return len(chat_history) >= 4

    def _summarize_episode(self, chat_history_str: str) -> str:
        """SLMを使って対話履歴の要約を生成する。"""
        prompt = f"""Summarize the following conversation into a concise single paragraph. This summary will be used as a memory for a future AI assistant. Focus on the key topics, user intentions, and final outcomes.

Conversation:
{chat_history_str}

Concise Summary:"""
        
        response = self._summarizer_llm.invoke(prompt)
        return response.content

    def process_and_save_memory(self, chat_history: List[BaseMessage]):
        """対話履歴を処理し、必要であれば要約して記憶システムに保存する。"""
        if not self._should_form_memory(chat_history):
            logger.info("Skipping memory formation, conversation is too short.")
            return

        logger.info("Processing conversation to form a new memory episode...")
        try:
            # chat_historyを文字列に変換
            history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
            
            # 要約を生成
            summary = self._summarize_episode(history_str)
            logger.info(f"Generated summary: {summary[:100]}...")

            # 履歴をJSON文字列に変換して保存
            history_json = json.dumps([msg.to_json() for msg in chat_history])
            
            # 記憶システムに保存を指示
            self._memory_system.save_episode(summary, history_json)

        except Exception as e:
            logger.error(f"Failed during memory processing: {e}", exc_info=True)