# agent_core/tool_manager.py
"""
ツール管理モジュール。

このモジュールは以下を担います:
- MCP(Server)接続設定の読み込みとクライアント初期化
- ネイティブツール(DuckDuckGoなど)・MCPツールの発見と登録
- 同期/非同期ツールの実行を単一のインターフェースで提供
- 非同期処理用に専用のイベントループをバックグラウンドスレッドで常駐

設計メモ:
- 非同期ツールは `asyncio` のイベントループで動作させ、同期コードからは
  `asyncio.run_coroutine_threadsafe` により安全に橋渡しします。
- ツール名の衝突を避けるため、MCPツールは「サーバー名_ツール名」に正規化します。
"""

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any, Coroutine, List

from langchain_core.tools import BaseTool

from .tools import load_mcp_tools_robust, load_native_tools

logger = logging.getLogger(__name__)

class ToolManager:
    """
    MCPツールおよびネイティブツールを統合的に管理するクラス。

    責務:
    - 設定ファイルからMCP接続を構築し、ツールを発見
    - ネイティブツール(DuckDuckGo検索など)の準備
    - 同期/非同期に関わらず、ツール実行の統一APIを提供
    - バックグラウンドのイベントループを維持し、非同期ツールを安全に実行
    """
    def __init__(self, config_file: str):
        """コンストラクタ

        引数:
            config_file: MCPサーバー設定ファイル(`mcp_tools_config.json`)への相対パス
        """
        project_root = Path(__file__).parent.parent
        self.config_path = project_root / config_file
        self.tools: List[BaseTool] = []
        self.tool_map: dict[str, BaseTool] = {}

        # 非同期処理専用のイベントループを作成し、
        # デーモンスレッドで永続的に回し続ける
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        logger.info("Async task runner thread started.")

    # 同期コードから非同期関数を安全に呼び出すためのブリッジメソッド 
    def _run_coroutine(self, coro: Coroutine) -> Any:
        """同期的なコードから非同期のコルーチンを実行し、結果を待つ。"""
        try:
            # バックグラウンドのイベントループにコルーチンを送信し、現在のスレッドで待機
            # ここで result() を呼ぶのは「ループスレッド以外」からのみ安全。
            # ループスレッド内で待機するとデッドロックするため、wrapper は使わない。
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return future.result(timeout=120)
        except Exception as e:
            raise e

    # 同期/非同期を問わず、全てのツールを実行するための統一インターフェース 
    def execute_tool(self, tool_name: str, tool_args: dict) -> Any:
        """
        指定されたツールを同期/非同期を自動で判断して実行する。
        graph.pyからはこのメソッドだけを呼び出せば良い。
        
        処理の流れ:
        1. ツール名でツールインスタンスを取得
        2. `aexecute_tool`をコルーチンとして取得
        3. バックグラウンドのイベントループで実行し、結果を待つ
        """
        try:
            # aexecute_toolは同期/非同期を内部で吸収してくれる
            # これを同期的に呼び出すだけで良い
            logger.info("Executing tool '%s' via sync bridge.", tool_name)
            coro = self.aexecute_tool(tool_name, tool_args)
            return self._run_coroutine(coro)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            return f"Error executing tool {tool_name}: {e}"

    async def aexecute_tool(self, tool_name: str, tool_args: dict) -> Any:
        """非同期コンテキストからツールを実行するためのヘルパー。"""
        tool_instance = self.get_tool(tool_name)
        if not tool_instance:
            # エラーメッセージもログに出力するとデバッグしやすい
            logger.error("Tool '%s' not found.", tool_name)
            return f"Error: Tool '{tool_name}' not found."

        try:
            if hasattr(tool_instance, "ainvoke"):
                logger.info("Executing ASYNC tool: %s", tool_name)
                future = asyncio.run_coroutine_threadsafe(tool_instance.ainvoke(tool_args), self._loop)
                return await asyncio.wrap_future(future)

            if hasattr(tool_instance, "invoke"):
                logger.info("Executing SYNC tool in executor: %s", tool_name)
                return await asyncio.to_thread(tool_instance.invoke, tool_args)

            return f"Error: Tool '{tool_name}' has no callable invoke or ainvoke method."
        except Exception as exc:  # noqa: BLE001
            logger.error("Error executing tool %s asynchronously: %s", tool_name, exc, exc_info=True)
            return f"Error executing tool {tool_name}: {exc}"

    def initialize(self):
        """
        ネイティブツールとMCPツールを初期化する。

        処理の流れ（堅牢化後）:
        1. ツールリストとマップを明示的にクリア
        2. ネイティブツールをtry-exceptブロック内で安全にロード
        3. MCPツールをtry-exceptブロック内で安全にロード
        4. 最終的なツールリストからtool_mapを再構築
        5. 最終的なツールリストをログに出力
        """
        logger.info("Initializing ToolManager...")
        # 1. ツールリストとマップを明示的にクリアし、再初期化の安全性を確保
        self.tools = []
        self.tool_map = {}
        
        # 2. ネイティブツールを安全にロード
        try:
            native_tools = load_native_tools()
            self.tools.extend(native_tools)
            logger.info(f"Successfully loaded {len(native_tools)} native tools.")
        except Exception as e:
            logger.error(f"An error occurred during native tool loading: {e}", exc_info=True)

        # 3. MCPツールを安全にロード（改善されたエラーハンドリング）
        try:
            mcp_tools = load_mcp_tools_robust(self.config_path)
            self.tools.extend(mcp_tools)
        except Exception as e:
            logger.error(f"An error occurred during MCP tool loading: {e}", exc_info=True)
        
        # 4. 最終的なツールリストからtool_mapを再構築
        self.tool_map = {tool.name: tool for tool in self.tools}
        logger.info(f"ToolManager initialized with {len(self.tools)} tools: {[t.name for t in self.tools]}")

    def get_tool(self, tool_name: str) -> BaseTool | None:
        """指定された名前のツールを取得する。

        見つからない場合は `None` を返す。
        """
        return self.tool_map.get(tool_name)

    def cleanup(self):
        """リソースのクリーンアップ。

        - バックグラウンドイベントループを停止
        """
        if self._loop.is_running():
            logger.info("Stopping async task runner thread...")
            try:
                shutdown_future = asyncio.run_coroutine_threadsafe(self._loop.shutdown_asyncgens(), self._loop)
                shutdown_future.result(timeout=5)
            except Exception as e:
                logger.warning("Failed to shutdown async generators gracefully: %s", e, exc_info=True)

            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning("Async runner thread did not stop gracefully.")
            else:
                logger.info("Async task runner thread stopped.")

        if not self._loop.is_closed():
            logger.debug("Closing async event loop.")
            self._loop.close()