"""
LangGraph で用いるエージェント状態の型定義をまとめたモジュール。

フィールド概要:
- `input`: 現在のユーザー入力
- `chat_history`: これまでのユーザー/AIメッセージ履歴
- `agent_scratchpad`: ReActループ専用のワークスペース(思考/ツール呼び出し/結果)
- `messages`: ノード間受け渡し専用のメッセージ(特にToolNodeで使用)
- `agent_outcome`: Agentモードの最終成果物(内部レポート等)
- `search_query`/`search_result`: 検索系ルートで使用
"""

# agent_core/state.py

from typing import List, TypedDict, Optional
from langchain_core.messages import BaseMessage

class AgentAction(TypedDict):
    tool: str
    tool_input: dict

class AgentFinish(TypedDict):
    return_values: dict
    log: str

class AgentState(TypedDict):
    # 初期入力と全体のチャット履歴
    input: str
    chat_history: list[BaseMessage]
    
    # AgentModeのReActループ専用の履歴 (思考、ツール呼び出し、ツール結果)
    agent_scratchpad: list[BaseMessage]
    
    # LangGraphのノード間通信用のメッセージリスト
    # ToolNodeが処理対象とするため、Noneではなく空配列で初期化する想定
    messages: list[BaseMessage]

    # AgentModeの最終的な成果物
    agent_outcome: Optional[str]

    # --- EM-LLM Memory Pipeline ---
    # 1. 検索された生のエピソード
    recalled_episodes: Optional[List[dict]]
    # 2. SLMによって統合された記憶
    synthesized_memory: Optional[str]

    # ストリーミング生成時に収集されたlogprobs
    generation_logprobs: Optional[List[dict]]

    # 検索用 (変更なし)
    search_query: Optional[str]
    search_result: Optional[str]

    # キャラクターが生成した、プロフェッショナル向けの指示書
    order: Optional[dict]