# agent_core/em_llm_graph.py
"""
既存のLangGraphシステムにEM-LLMを統合するための修正版グラフ実装

主な変更点:
1. memory_retrieval_node: 従来のRAG検索をEM-LLMの2段階検索に置換
2. memory_synthesis_node: SLMによる記憶統合はそのまま活用
3. save_memory_node: 従来のメモリ保存をEM-LLMメモリ形成に置換
4. EM-LLM統計情報の追加

従来システムとの互換性を保ちながら、段階的にEM-LLM機能を導入します。
"""

import asyncio
import json
import logging
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

class EMEnabledAgentCore:
    """EM-LLM機能を統合した新しいAgentCoreクラス"""
    
    def __init__(self, llm_manager, tool_manager, em_llm_integrator):
        self.llm_manager = llm_manager
        self.tool_manager = tool_manager
        self.em_llm_integrator = em_llm_integrator
        
        # 従来の機能は保持
        from .graph import AgentCore
        # AgentCoreにEMMemorySystemを直接渡すのではなく、互換レイヤーを渡す。
        # これにより、AgentCoreが期待する `add_episode` メソッドが提供され、
        # 万が一古いメモリ保存ノードが呼ばれてもエラーが発生しなくなる。
        compatibility_layer = EMCompatibilityLayer(em_llm_integrator)
        self.base_agent_core = AgentCore(llm_manager, tool_manager, compatibility_layer)
        
        # グラフを再構築（EM-LLMノードで置換）
        self.graph = self._build_em_llm_graph()
        
        logger.info("EM-LLM enabled Agent Core initialized")
    
    def _build_em_llm_graph(self):
        """EM-LLM機能を統合したグラフを構築"""
        from langgraph.graph import StateGraph, END
        from .state import AgentState
        
        workflow = StateGraph(AgentState)
        
        # EM-LLM専用ノード
        workflow.add_node("em_memory_retrieval", self.em_memory_retrieval_node)
        workflow.add_node("em_memory_synthesis", self.em_memory_synthesis_node)
        workflow.add_node("em_memory_formation", self.em_memory_formation_node)
        
        # 従来のノードはそのまま使用
        workflow.add_node("direct_answer", self.base_agent_core.direct_answer_node)
        workflow.add_node("generate_search_query", self.base_agent_core.generate_search_query_node)
        workflow.add_node("execute_search", self.base_agent_core.execute_search_node)
        workflow.add_node("summarize_search_result", self.base_agent_core.summarize_search_result_node)
        workflow.add_node("generate_order_node", self.base_agent_core.generate_order_node)
        workflow.add_node("agent_reasoning_node", self.base_agent_core.agent_reasoning_node)
        workflow.add_node("synthesize_final_response_node", self.base_agent_core.synthesize_final_response_node)
        workflow.add_node("tool_node", self.base_agent_core.unified_tool_executor_node)
        workflow.add_node("update_scratchpad_node", self.base_agent_core.update_scratchpad_node)
        
        # EM-LLM統計ノード（デバッグ用）
        workflow.add_node("em_stats_node", self.em_stats_node)
        
        # グラフの接続（EM-LLMバージョン）
        
        # 1. エントリーポイント: EM-LLMメモリ検索
        workflow.set_entry_point("em_memory_retrieval")
        
        # 2. EM-LLMメモリパイプライン
        workflow.add_edge("em_memory_retrieval", "em_memory_synthesis")
        
        # 3. 統合後のルーティング
        workflow.add_conditional_edges(
            "em_memory_synthesis",
            self.base_agent_core.route_by_command,
            {
                "agent_mode": "generate_order_node",
                "search": "generate_search_query", 
                "direct_answer": "direct_answer",
            }
        )
        
        # 4. 各ブランチのフロー（変更なし）
        workflow.add_edge("direct_answer", "em_memory_formation")
        workflow.add_edge("generate_search_query", "execute_search")
        workflow.add_edge("execute_search", "summarize_search_result")
        workflow.add_edge("summarize_search_result", "em_memory_formation")
        
        # ReActパス
        workflow.add_edge("generate_order_node", "agent_reasoning_node")
        workflow.add_conditional_edges(
            "agent_reasoning_node",
            self.base_agent_core.should_continue_react_loop,
            {
                "continue": "tool_node",
                "end": "synthesize_final_response_node"
            },
        )
        workflow.add_edge("tool_node", "update_scratchpad_node")
        workflow.add_edge("update_scratchpad_node", "agent_reasoning_node")
        # AgentModeの実行結果はキャラクターの長期記憶に保存しないことで、記憶を完全に分離する。
        # これにより、キャラクターの記憶はユーザーとの直接対話のみから形成される。
        workflow.add_edge("synthesize_final_response_node", "em_stats_node")
        
        # 5. EM-LLM統計確認とエンド
        workflow.add_edge("em_memory_formation", "em_stats_node")
        workflow.add_edge("em_stats_node", END)
        
        return workflow.compile()
    
    def em_memory_retrieval_node(self, state) -> dict:
        """
        【EM-LLMバージョン】関連エピソード記憶の2段階検索
        
        従来のRAG検索をEM-LLMの驚異度ベース検索に置換
        """
        print("--- Node: EM-LLM Memory Retrieval (Two-Stage) ---")
        
        try:
            # EM-LLMの2段階検索を実行
            recalled_episodes = self.em_llm_integrator.retrieve_relevant_memories_for_query(state["input"])
            
            if recalled_episodes:
                print(f"EM-LLM retrieved {len(recalled_episodes)} relevant episodic events.")
                # 統計情報をログ出力
                for i, episode in enumerate(recalled_episodes):
                    surprise_stats = episode.get('surprise_stats', {})
                    print(f"  Event {i+1}: {episode.get('content', '')[:50]}... "
                          f"(surprise: {surprise_stats.get('mean_surprise', 0):.3f})")
                return {"recalled_episodes": recalled_episodes}
            else:
                print("No relevant episodic memories found.")
                return {"recalled_episodes": []}
                
        except Exception as e:
            print(f"Warning: EM-LLM memory retrieval failed: {e}")
            logger.error(f"EM-LLM memory retrieval error: {e}")
            return {"recalled_episodes": []}
    
    def em_memory_synthesis_node(self, state) -> dict:
        """
        【EM-LLMバージョン】エピソード記憶の統合
        
        従来のSLM統合機能を活用しつつ、EM-LLM固有の情報を追加
        """
        print("--- Node: EM-LLM Memory Synthesis (Enhanced SLM) ---")
        recalled_episodes = state.get("recalled_episodes")
        
        if not recalled_episodes:
            print("No episodes to synthesize. Skipping.")
            return {"synthesized_memory": "No relevant episodic memories found."}
        
        # エピソードを文字列にフォーマット（EM-LLM固有情報を含む）
        episodes_str = "\n".join([
            f"Episodic Event {i+1}:\n"
            f"- Content: {ep.get('content', 'N/A')}\n"
            f"- Summary: {ep.get('summary', 'N/A')}\n"
            f"- Surprise Statistics: Mean={ep.get('surprise_stats', {}).get('mean_surprise', 0):.3f}, "
            f"Max={ep.get('surprise_stats', {}).get('max_surprise', 0):.3f}\n"
            f"- Event Size: {ep.get('surprise_stats', {}).get('event_size', 0)} tokens\n"
            f"- Representative Tokens: {ep.get('representative_tokens', [])}\n"
            for i, ep in enumerate(recalled_episodes)
        ])
        
        print("Synthesizing EM-LLM episodic memories with SLM...")
        
        # SLMによる統合（既存機能を活用）
        slm = self.llm_manager.get_slm_summarizer()
        
        from langchain_core.prompts import ChatPromptTemplate
        from . import config
        
        # EM-LLM専用のプロンプトを使用
        enhanced_prompt = f"""
        {config.BASE_SYSTEM_PROMPTS["memory_synthesis"]}
        
        Additional Context: These are episodic memories formed through EM-LLM's surprise-based segmentation. 
        Each memory represents a coherent event with associated surprise statistics indicating the novelty 
        and importance of the information. Higher surprise scores suggest more significant or unexpected content.
        """
        
        prompt = ChatPromptTemplate.from_messages([("system", enhanced_prompt)])
        chain = prompt | slm
        
        response = chain.invoke({"retrieved_memories": episodes_str})
        synthesized_memory = response.content
        
        print(f"EM-LLM Synthesized Memory: {synthesized_memory[:200]}...")
        return {"synthesized_memory": synthesized_memory}
    
    async def em_memory_formation_node(self, state) -> dict:
        """
        【EM-LLMバージョン】対話の記憶形成（非同期実行）

        【変更】AIの応答全体ではなく、ユーザー入力とAI応答の「要約」を対象に
        意味的セグメンテーションを行い、軽量な記憶エピソードを形成します。
        """
        print()
        print("--- Node: EM-LLM Memory Formation (Summary-based) ---")

        # 最新の対話ペアを取得
        user_input = state.get("input")
        ai_response_message = next(
            (msg for msg in reversed(state.get("chat_history", [])) if isinstance(msg, AIMessage)), None
        )
        
        if not user_input or not ai_response_message:
            print("Warning: Could not find valid user input and AI response pair. Skipping EM-LLM memory formation.")
            return {}

        ai_response = ai_response_message.content

        print("Starting EM-LLM memory formation...")
        try:
            # 1. SLMを使って対話ターンを要約する
            print("  - Summarizing conversation turn with SLM...")
            slm = self.llm_manager.get_slm_summarizer()
            from . import config
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages([
                ("system", config.BASE_SYSTEM_PROMPTS["memory_consolidation"])
            ])
            chain = prompt | slm

            response = await chain.ainvoke({
                "user_input": user_input,
                "ai_response": ai_response
            })
            consolidated_summary = response.content.strip()
            print(f"  - Consolidated Summary: {consolidated_summary[:150]}...")

            # 2. 要約を対象に意味的変化検出に基づくメモリ形成を実行
            print(f"  - Analyzing summary for semantic change to form episodic memories.")

            # process_conversation_turn_for_memoryは第2引数(ai_response)を解析対象とするため、
            # ここに要約を渡す。
            formed_events = await self.em_llm_integrator.process_conversation_turn_for_memory(
                user_input, consolidated_summary
            )

            if formed_events:
                # 形成された事象の統計を表示
                total_tokens = sum(len(event.tokens) for event in formed_events)
                avg_surprise = sum(
                    sum(event.surprise_scores) / len(event.surprise_scores)
                    for event in formed_events if event.surprise_scores
                ) / len(formed_events) if formed_events else 0

                print(f"EM-LLM formed {len(formed_events)} new episodic events from summary.")
                print(f"  - Total tokens: {total_tokens}")
                print(f"  - Average surprise: {avg_surprise:.3f}")
            else:
                print("No episodic events were formed from this conversation turn.")
                
        except Exception as e:
            print(f"Error during EM-LLM memory formation: {e}")
            logger.error(f"EM-LLM memory formation error: {e}", exc_info=True)
        
        print("Memory formation completed. Graph continues.")
        return {}
    
    def em_stats_node(self, state) -> dict:
        """EM-LLMシステムの統計情報を表示（デバッグ用）"""
        print("--- Node: EM-LLM Statistics ---")
        
        try:
            stats = self.em_llm_integrator.get_memory_statistics()
            print("EM-LLM Memory System Statistics:")
            print(f"  Total Events: {stats.get('total_events', 0)}")
            print(f"  Total Tokens in Memory: {stats.get('total_tokens_in_memory', 0)}")
            print(f"  Mean Event Size: {stats.get('mean_event_size', 0):.1f} tokens")
            
            surprise_stats = stats.get('surprise_statistics', {})
            if surprise_stats:
                print(f"  Surprise Stats - Mean: {surprise_stats.get('mean', 0):.3f}, "
                      f"Std: {surprise_stats.get('std', 0):.3f}, Max: {surprise_stats.get('max', 0):.3f}")
            
            config_info = stats.get('configuration', {})
            print(f"  Configuration - Gamma: {config_info.get('surprise_gamma', 0)}, "
                  f"Event Size: {config_info.get('min_event_size', 0)}-{config_info.get('max_event_size', 0)}")
            
        except Exception as e:
            print(f"Failed to retrieve EM-LLM statistics: {e}")
        
        return {}

class EMCompatibilityLayer:
    """
    既存システムとの互換性を保つためのアダプターレイヤー
    
    段階的移行期間中に、従来のMemorySystemインターフェースを
    EM-LLM機能に透明にマッピングします。
    """
    
    def __init__(self, em_llm_integrator):
        self.em_llm_integrator = em_llm_integrator
        self.logger = logging.getLogger(__name__ + ".EMCompatibilityLayer")
    
    def retrieve_similar_episodes(self, query: str, k: int = 5) -> List[Dict]:
        """
        従来のMemorySystem.retrieve_similar_episodesと互換性のあるインターフェース
        
        内部的にはEM-LLMの2段階検索を使用
        """
        self.logger.debug(f"Compatibility layer: retrieving episodes for query: {query[:50]}...")
        
        try:
            # EM-LLMの検索機能を使用
            em_results = self.em_llm_integrator.retrieve_relevant_memories_for_query(query)
            
            # 結果数をk個に制限
            limited_results = em_results[:k] if em_results else []
            
            self.logger.info(f"Compatibility layer returned {len(limited_results)} episodes")
            return limited_results
            
        except Exception as e:
            self.logger.error(f"EM-LLM compatibility layer failed, falling back to empty result: {e}")
            return []
    
    def add_episode(self, summary: str, user_input: str, ai_response: str):
        """
        従来のMemorySystem.add_episodeと互換性のあるインターフェース
        
        内部的にはEM-LLMのメモリ形成パイプラインを使用
        """
        self.logger.debug(f"Compatibility layer: adding episode with summary: {summary[:50]}...")
        
        try:
            # EM-LLMのメモリ形成パイプラインを実行
            formed_events = self.em_llm_integrator.process_conversation_turn_for_memory(
                user_input, ai_response
            )
            
            self.logger.info(f"Compatibility layer formed {len(formed_events)} events from episode")
            
        except Exception as e:
            self.logger.error(f"EM-LLM compatibility layer episode formation failed: {e}")
    
    def get_memory_summary(self) -> str:
        """メモリシステムの現在の状態を要約"""
        try:
            stats = self.em_llm_integrator.get_memory_statistics()
            return (f"EM-LLM Memory: {stats.get('total_events', 0)} episodic events, "
                   f"{stats.get('total_tokens_in_memory', 0)} tokens, "
                   f"avg size {stats.get('mean_event_size', 0):.1f}")
        except Exception as e:
            return f"EM-LLM Memory: Statistics unavailable ({e})"