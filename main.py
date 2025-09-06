# main.py (EM-LLM統合版)
"""
EM-LLM対応エージェントアプリのエントリーポイント

主な変更点:
1. EM-LLM統合レイヤーの初期化
2. 既存システムとの互換性レイヤー使用
3. EM-LLM統計の出力
4. 段階的移行のためのフォールバック機能

流れ:
1) LLMロード
2) EM-LLM統合システム初期化
3) 互換性レイヤー経由でLangGraphアプリ構築
4) EM-LLM機能付き対話ループ開始
"""

import logging
import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import asyncio
import sys
from langchain_core.messages import HumanMessage, AIMessage

# EM-LLM関連のインポート
from agent_core.em_llm_core import EMLLMIntegrator, EMConfig
from agent_core.em_llm_graph import EMEnabledAgentCore, EMCompatibilityLayer
from agent_core.embedding_provider import LlamaCppEmbeddingProvider 

# 従来のインポート
from agent_core.config import MCP_CONFIG_FILE, MAX_CHAT_HISTORY_LENGTH, EM_LLM_CONFIG
from agent_core.llm_manager import LLMManager
from agent_core.tool_manager import ToolManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def ainput(prompt: str = "") -> str:
    print(prompt, end="", flush=True)
    return await asyncio.to_thread(sys.stdin.readline)

async def main():
    """EM-LLM対応エージェントのメイン関数"""
    
    llm_manager = None
    tool_manager = None
    em_llm_integrator = None
    app = None
    
    try:
        print("Initializing EM-LLM Enhanced AI Agent...")
        print("=" * 60)
        
        # === Phase 1: 基本システム初期化 ===
        print("Phase 1: Initializing core systems...")
        
        # LLMマネージャー初期化
        llm_manager = LLMManager()
        llm_manager.get_character_agent()  # メインLLMをプリロード
        print("✓ LLM Manager initialized")
        
        # ツールマネージャー初期化
        tool_manager = ToolManager(config_file=MCP_CONFIG_FILE)
        tool_manager.initialize()
        print(f"✓ Tool Manager initialized with {len(tool_manager.tools)} tools")
        
        # === Phase 2: EM-LLM システム初期化 ===
        print("\nPhase 2: Initializing EM-LLM systems...")
        
        try:
            # 埋め込みモデルをロード
            embedding_llm = llm_manager.get_embedding_model()
            embedding_provider = LlamaCppEmbeddingProvider(embedding_llm)
            print("✓ Embedding provider initialized")
            
            # EM-LLM統合レイヤーを初期化
            em_llm_integrator = EMLLMIntegrator(llm_manager, embedding_provider)
            print("✓ EM-LLM integrator initialized")
            
            # 設定値を更新
            em_config = EMConfig(
                surprise_window=EM_LLM_CONFIG["surprise_window_size"],
                surprise_gamma=EM_LLM_CONFIG["surprise_gamma"],
                min_event_size=EM_LLM_CONFIG["min_event_size"],
                max_event_size=EM_LLM_CONFIG["max_event_size"],
                similarity_buffer_ratio=EM_LLM_CONFIG["similarity_buffer_ratio"],
                contiguity_buffer_ratio=EM_LLM_CONFIG["contiguity_buffer_ratio"],
                total_retrieved_events=EM_LLM_CONFIG["total_retrieved_events"],
                recency_weight=EM_LLM_CONFIG["recency_weight"],
                use_boundary_refinement=EM_LLM_CONFIG["use_boundary_refinement"],
                refinement_metric=EM_LLM_CONFIG["refinement_metric"],
                refinement_search_range=EM_LLM_CONFIG["refinement_search_range"]
            )
            em_llm_integrator.config = em_config
            print("✓ EM-LLM configuration applied")
            
        except Exception as e:
            logger.error(f"EM-LLM initialization failed: {e}")
            print(f"⚠ EM-LLM initialization failed: {e}")
            print("Falling back to traditional memory system...")
            em_llm_integrator = None
        
        # === Phase 3: アプリケーショングラフ構築 ===
        print("\nPhase 3: Building application graph...")
        
        if em_llm_integrator:
            # EM-LLM対応グラフを構築
            agent_core = EMEnabledAgentCore(llm_manager, tool_manager, em_llm_integrator)
            app = agent_core.graph
            print("✓ EM-LLM enhanced graph initialized")
            
            # 初期統計を表示
            try:
                stats = em_llm_integrator.get_memory_statistics()
                print(f"✓ EM-LLM Memory System: {stats['status'] if stats.get('total_events') == 0 else f'{stats['total_events']} events'}")
            except Exception as e:
                print(f"⚠ Could not retrieve initial EM-LLM statistics: {e}")
        else:
            # フォールバック: 従来システム
            from agent_core.memory.memory_system import MemorySystem
            memory_system = MemorySystem(embedding_provider)
            from agent_core.graph import AgentCore
            agent_core = AgentCore(llm_manager, tool_manager, memory_system)
            app = agent_core.graph
            print("✓ Traditional graph initialized (fallback mode)")
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Critical error during initialization: {e}", exc_info=True)
        print(f"\n❌ Failed to start the AI agent: {e}")
        print("Please check the logs and configuration.")
        return
    
    # === 対話ループ開始 ===
    if em_llm_integrator:
        print("🧠 EM-LLM Enhanced AI Agent is ready!")
        print("Features: Surprise-based memory formation, Two-stage retrieval, Episodic segmentation")
    else:
        print("🤖 AI Agent is ready (traditional mode)")
    
    print("\nCommands:")
    print("  • '/agentmode <request>' - Complex task with tools")  
    print("  • '/search <query>' - Web search")
    print("  • '/emstats' - EM-LLM memory statistics (if available)")
    print("  • Normal chat - Direct conversation")
    print("  • 'exit' - Quit")
    print("-" * 60)
    
    chat_history = []
    
    try:
        while True:
            try:
                user_input = (await ainput("You: ")).strip()
                
                if user_input.lower() in ["exit", "quit"]:
                    break
                if not user_input:
                    continue
                
                # EM-LLM統計コマンド処理
                if user_input.lower() == '/emstats' and em_llm_integrator:
                    try:
                        stats = em_llm_integrator.get_memory_statistics()
                        print("\n📊 EM-LLM Memory System Statistics:")
                        print(f"   Total Events: {stats.get('total_events', 0)}")
                        print(f"   Total Tokens: {stats.get('total_tokens_in_memory', 0)}")
                        print(f"   Mean Event Size: {stats.get('mean_event_size', 0):.1f} tokens")
                        print()
                        
                        surprise_stats = stats.get('surprise_statistics', {})
                        if surprise_stats and surprise_stats.get('mean', 0) > 0:
                            print(f"   Surprise - Mean: {surprise_stats.get('mean', 0):.3f}, "
                                  f"Std: {surprise_stats.get('std', 0):.3f}, Max: {surprise_stats.get('max', 0):.3f}")
                        
                        config_info = stats.get('configuration', {})
                        print(f"   Config - γ: {config_info.get('surprise_gamma', 0)}, "
                              f"Event Size: {config_info.get('min_event_size', 0)}-{config_info.get('max_event_size', 0)}")
                        print()
                        continue
                    except Exception as e:
                        print(f"❌ Failed to retrieve EM-LLM statistics: {e}")
                        continue
                
                # LangGraphの実行
                initial_state = {
                    "input": user_input,
                    "chat_history": chat_history,
                    "agent_scratchpad": [],
                    "messages": [],
                }
                
                print(f"\n--- Processing (EM-LLM: {'✓' if em_llm_integrator else '✗'}) ---")
                
                full_response = ""
                final_output = None
                
                # ストリーミング実行
                async for event in app.astream_events(initial_state, version="v2", config={"recursion_limit": 50}):
                    kind = event["event"]
                    
                    # LLMストリーミング出力
                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            print(content, end="", flush=True)
                            full_response += content
                    
                    # グラフ実行完了
                    elif kind == "on_graph_end":
                        final_output = event["data"]["output"]
                
                print()  # 改行
                
                # チャット履歴更新
                if full_response:
                    chat_history.append(HumanMessage(content=user_input))
                    chat_history.append(AIMessage(content=full_response))
                elif final_output and final_output.get("agent_outcome"):
                    print(f"\nAI: Task completed. Outcome: {final_output['agent_outcome']}")
                    chat_history.append(HumanMessage(content=user_input))
                else:
                    print("\nAI: An unexpected error occurred.")
                    chat_history.append(HumanMessage(content=user_input))
                
                # チャット履歴の長さ制限
                if len(chat_history) > MAX_CHAT_HISTORY_LENGTH:
                    chat_history = chat_history[-MAX_CHAT_HISTORY_LENGTH:]
                    print(f"INFO: Chat history truncated to {MAX_CHAT_HISTORY_LENGTH} messages.")
                
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Exiting EM-LLM Agent. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error during conversation: {e}", exc_info=True)
                print(f"\n❌ An error occurred: {e}")
                print("Please try again or type 'exit' to quit.")
    
    finally:
        # === クリーンアップ ===
        print("\n🧹 Cleaning up resources...")
        
        if llm_manager:
            try:
                llm_manager.cleanup()
                print("✓ LLM resources cleaned up")
            except Exception as e:
                print(f"⚠ LLM cleanup warning: {e}")
        
        if tool_manager:
            try:
                tool_manager.cleanup()
                print("✓ Tool manager cleaned up")
            except Exception as e:
                print(f"⚠ Tool manager cleanup warning: {e}")
        
        if em_llm_integrator:
            try:
                stats = em_llm_integrator.get_memory_statistics()
                print(f"✓ Final EM-LLM state: {stats.get('total_events', 0)} events, "
                      f"{stats.get('total_tokens_in_memory', 0)} tokens")
            except Exception as e:
                print(f"⚠ Could not retrieve final EM-LLM statistics: {e}")
        
        print("Cleanup completed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Failed to run the EM-LLM agent application: {e}", exc_info=True)
        print(f"❌ Critical failure: {e}")
        print("Check logs for details.")