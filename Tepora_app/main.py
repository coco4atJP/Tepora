"""
エージェントアプリのエントリーポイント。

流れ:
1) LLMロード
2) ツール初期化(MCP含む)
3) LangGraphのアプリ(実行グラフ)を構築
4) 対話ループを開始し、ユーザー入力を受け付ける

終了時はリソース(モデル/ツール)を確実に解放する。
"""

# main.py

import logging
import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

from langchain_core.messages import HumanMessage, AIMessage
import json
from agent_core.config import MCP_CONFIG_FILE , MAX_CHAT_HISTORY_LENGTH
from agent_core.llm_manager import LLMManager
from agent_core.tool_manager import ToolManager
from agent_core.graph import AgentCore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    エージェントの初期化と対話ループを実行するメイン関数。
    
    処理の流れ:
    1. 初期化フェーズ: LLMロード、ツール初期化、グラフ構築
    2. 対話ループフェーズ: ユーザー入力受付、LangGraph実行、結果表示
    3. クリーンアップフェーズ: リソース解放、MCPセッション終了
    """
    # finallyブロックで参照できるよう、Noneで初期化しておきます
    llm_manager = None
    tool_manager = None
    app = None
    
    # --- 1. 初期化フェーズ ---
    # 初期化中に発生した致命的なエラーはここでキャッチします
    try:
        print("Initializing AI Agent...")
        
        # 1a. LLMManagerをインスタンス化
        llm_manager = LLMManager()

        # 1b. 最初にGemmaをロードしておく
        llm_manager.get_gemma_3n()
        
        # 1c. ツールマネージャーを初期化: MCPサーバー起動とツール発見
        tool_manager = ToolManager(config_file=MCP_CONFIG_FILE)
        tool_manager.initialize()
        
        # 1d. エージェントコアを構築し、実行可能なグラフ(app)を取得
        agent_core = AgentCore(llm_manager=llm_manager, tool_manager=tool_manager)
        app = agent_core.graph
    
    except Exception as e:
        logging.error(f"A critical error occurred during agent initialization: {e}", exc_info=True)
        print("\nFailed to start the AI agent. Please check the logs.")
        return # 初期化に失敗した場合はここでプログラムを終了

    print("\n--- AI Agent is ready. Type '/agentmode <your request>' ・ AI Search is ready. Type '/search <your query>' for complex tasks, or a simple chat message. Type 'exit' to quit. ---")
    
    # --- 2. 対話ループフェーズ ---    
    chat_history = []
    try:
        while True:
            try:
                # 2a. ユーザー入力受付: 終了コマンドと空入力をチェック
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                if not user_input:
                    continue

                # 2b. ユーザーメッセージをチャット履歴に追加
                user_message = HumanMessage(content=user_input)
                
                # 2c. チャット履歴は各ノードで適切に構築されるため、ここでは空のままにする
                current_chat_history = chat_history.copy()
                
                # 2d. LangGraphに渡す初期状態を構築
                initial_state = {
                    "input": user_input,                    # 現在のユーザー入力
                    "chat_history": current_chat_history,   # これまでの会話履歴
                    "agent_scratchpad": [],                 # AgentStateの必須項目であるscratchpadを初期化
                    "messages": [],                         # ToolNode用のmessagesフィールドを初期化
                }
                
                print(f"\n--- Initial State ---")
                print(f"Input: {initial_state['input']}")
                print(f"Chat history length: {len(initial_state['chat_history'])}")
                print(f"Agent scratchpad length: {len(initial_state['agent_scratchpad'])}")
                print(f"Messages length: {len(initial_state['messages'])}")
                
                # 2e. LangGraphの実行: ルーティングとノード処理
                print("\n--- Agent is thinking... ---")
                final_state = app.invoke(initial_state)
                
                print(f"\n--- Final State ---")
                print(f"Final state keys: {list(final_state.keys())}")
                # オーダーが生成されていれば内容を表示
                if "order" in final_state and final_state["order"] is not None:
                    try:
                        print("\n--- Generated Order ---")
                        print(json.dumps(final_state["order"], ensure_ascii=False, indent=2))
                    except Exception:
                        print(f"\n--- Generated Order ---\n{final_state['order']}")
                if "agent_outcome" in final_state:
                    print(f"Agent outcome: {final_state['agent_outcome']}")
                if "chat_history" in final_state:
                    print(f"Final chat history length: {len(final_state['chat_history'])}")
                if "agent_scratchpad" in final_state:
                    print(f"Final agent scratchpad length: {len(final_state['agent_scratchpad'])}")
                
                # 2f. 実行後の完全な履歴でローカルの履歴を更新
                # 新しいチャット履歴には、元の履歴 + ユーザーメッセージ + AIの応答が含まれている
                chat_history = final_state.get("chat_history", current_chat_history)

                # チャット履歴が上限を超えたら古いものから削除
                if len(chat_history) > MAX_CHAT_HISTORY_LENGTH:
                    print(f"INFO: Chat history truncated from {len(chat_history)} to {MAX_CHAT_HISTORY_LENGTH} messages.")
                    chat_history = chat_history[-MAX_CHAT_HISTORY_LENGTH:]

                # 2g. ユーザーへの最終応答を表示
                # グラフ実行後の最終的なチャット履歴から、最後のAIの応答を取得して表示する
                if chat_history:
                    last_message = chat_history[-1]
                    # 最後のメッセージがツール呼び出しを含まないAIの応答であれば表示
                    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                        print(f"\nAI: {last_message.content}")
                
                print("\n-----------------------------------------\n")

            except KeyboardInterrupt:
                print("\nExiting agent. Goodbye!")
                break
            except Exception as e:
                logging.error(f"An error occurred during the conversation loop: {e}", exc_info=True)
                print("\nAn error occurred. Please try again or type 'exit' to quit.")

    # --- 3. クリーンアップフェーズ ---
    # プログラム終了時にリソースを解放します
    finally:
        print("Cleaning up resources...")
        # 3a. LLMのVRAM解放: モデルとトークナイザーを削除
        if llm_manager:
            try:
                llm_manager.cleanup()
            except Exception as e:
                print(f"ERROR: Failed to cleanup LLM manager: {e}")
        # 3b. ツールマネージャーのクリーンアップ: MCPセッション終了、イベントループ停止
        if tool_manager:
            try:
                tool_manager.cleanup()
            except Exception as e:
                print(f"ERROR: Failed to cleanup tool manager: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Failed to run the agent application: {e}", exc_info=True)