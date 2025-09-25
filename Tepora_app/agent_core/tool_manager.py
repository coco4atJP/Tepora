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

import json
import logging
import asyncio
import threading
import requests
import time
from pathlib import Path
from concurrent.futures import Future
from typing import List, Dict, Coroutine, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from . import config
from langchain_mcp_adapters.client import MultiServerMCPClient, StdioConnection

logger = logging.getLogger(__name__)

class GoogleCustomSearchInput(BaseModel):
    """Google Custom Search APIの入力スキーマ"""
    query: str = Field(description="検索クエリ")

class GoogleCustomSearchTool(BaseTool):
    """Google Custom Search JSON APIを使用した検索ツール"""
    
    name: str = "native_google_search"
    description: str = "Google Custom Search APIを使用してWeb検索を実行し、複数の結果を返します。"
    args_schema: type[BaseModel] = GoogleCustomSearchInput
    session: Any = Field(None, exclude=True)  # requests.Sessionを保持するが、モデルスキーマからは除外
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # セッションとリトライ設定の初期化
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """リトライ機能付きのHTTPセッションを作成"""
        session = requests.Session()
        
        # リトライ戦略の設定
        retry_strategy = Retry(
            total=3,  # 最大3回リトライ
            backoff_factor=1,  # 指数バックオフ
            status_forcelist=[429, 500, 502, 503, 504],  # リトライ対象のHTTPステータス
            allowed_methods=["GET"]  # GETリクエストのみリトライ
        )
        
        # アダプターにリトライ戦略を適用
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # セッションのタイムアウト設定
        session.timeout = (10, 30)  # (接続タイムアウト, 読み取りタイムアウト)
        
        return session
    
    def _run(self, query: str) -> str:
        """同期実行"""
        return self._perform_search(query)
    
    async def _arun(self, query: str) -> str:
        """非同期実行"""
        return self._perform_search(query)
    
    def _perform_search(self, query: str) -> str:
        """Google Custom Search APIを使用して検索を実行"""
        try:
            # API設定の確認
            if not config.GOOGLE_CUSTOM_SEARCH_API_KEY:
                return "Error: Google Custom Search API Key is not configured."
            if not config.GOOGLE_CUSTOM_SEARCH_ENGINE_ID:
                return "Error: Google Custom Search Engine ID is not configured."
            
            # APIリクエストの構築
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': config.GOOGLE_CUSTOM_SEARCH_API_KEY,
                'cx': config.GOOGLE_CUSTOM_SEARCH_ENGINE_ID,
                'q': query,
                'num': min(config.GOOGLE_CUSTOM_SEARCH_MAX_RESULTS, 10),  # Google APIの最大値は10
                'safe': 'active'  # セーフサーチを有効化
            }
            
            # ヘッダーの設定
            headers = {
                'User-Agent': 'AI-Agent/1.0 (Google Custom Search Tool)',
                'Accept': 'application/json'
            }
            
            logger.info(f"Executing Google Custom Search for query: {query}")
            start_time = time.time()
            
            # 検索実行（セッションを使用）
            response = self.session.get(
                url, 
                params=params, 
                headers=headers,
                timeout=(10, 30)  # (接続タイムアウト, 読み取りタイムアウト)
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Search completed in {elapsed_time:.2f} seconds")
            
            # レスポンスの確認
            response.raise_for_status()
            
            # レスポンスの解析
            data = response.json()
            
            # エラーレスポンスの確認
            if 'error' in data:
                error_info = data['error']
                error_message = f"Google API Error: {error_info.get('code', 'Unknown')} - {error_info.get('message', 'Unknown error')}"
                logger.error(error_message)
                return f"Error: {error_message}"
            
            if 'items' not in data:
                return f"No search results found for: {query}"
            
            # 結果の整形
            results = []
            for i, item in enumerate(data['items'], 1):
                result = {
                    'title': item.get('title', 'No title'),
                    'link': item.get('link', 'No link'),
                    'snippet': item.get('snippet', 'No description')
                }
                results.append(result)
            
            # 結果の返却
            if results:
                logger.info(f"Successfully retrieved {len(results)} search results")
                return json.dumps({
                    'query': query,
                    'total_results': len(results),
                    'results': results,
                    'search_time': f"{elapsed_time:.2f}s"
                }, ensure_ascii=False, indent=2)
            else:
                return f"No search results found for: {query}"
                
        except requests.exceptions.Timeout as e:
            logger.error(f"Google Custom Search API timeout: {e}")
            return f"Error: Search request timed out. Please try again later."
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Google Custom Search API connection error: {e}")
            return f"Error: Connection failed. Please check your internet connection and try again."
        except requests.exceptions.HTTPError as e:
            # [セキュリティ修正] エラーログにリクエストURLやヘッダーが含まれないように、必要な情報だけを抽出する
            status_code = e.response.status_code if e.response else "N/A"
            logger.error(f"Google Custom Search API HTTP error: Status {status_code}")
            if status_code == 429:
                return f"Error: Rate limit exceeded. Please wait a moment and try again."
            elif status_code == 403:
                return f"Error: API access denied. Please check your API key and permissions."
            else:
                # [セキュリティ修正] 例外オブジェクトeを直接文字列化せず、安全な情報のみを返す
                return f"Error: HTTP error occurred with status code {status_code}."
        except requests.exceptions.RequestException as e:
            logger.error(f"Google Custom Search API request failed: {e}")
            return f"Error: Failed to perform search: {e}"
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Google API response: {e}")
            return f"Error: Invalid response from Google API. Please try again."
        except Exception as e:
            logger.error(f"Unexpected error in Google Custom Search: {e}")
            return f"Error: Unexpected error occurred: {e}"
    
    def __del__(self):
        """デストラクタでセッションをクローズ"""
        if hasattr(self, 'session'):
            self.session.close()

def _load_connections_from_config(config_path: Path) -> Dict[str, StdioConnection]:
    """
    設定ファイル(mcp_tools_config.json)からMCPサーバー接続情報を読み込む。
    
    処理の流れ:
    1. JSON設定ファイルを読み込み
    2. mcpServersセクションから各サーバー設定を抽出
    3. 必須項目(command, args)の存在確認
    4. StdioConnectionオブジェクトを構築
    5. 接続辞書に登録して返却
    """
    connections: Dict[str, StdioConnection] = {}
    try:
        # 1. JSON設定ファイルを読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # 2. mcpServersセクションから各サーバー設定を抽出
        for server_name, server_config in config_data.get('mcpServers', {}).items():
            # 3. 必須項目(command)の存在確認 (argsは任意)
            if "command" not in server_config:
                logger.warning(f"Skipping server '{server_name}': 'command' key is missing.")
                continue

            # 4. StdioConnectionオブジェクトを構築
            connections[server_name] = StdioConnection(
                transport="stdio",
                command=server_config['command'],        # 実行するコマンド
                args=server_config.get('args', []),     # コマンドライン引数
                env=server_config.get('env')            # 環境変数(オプション)
            )
            logger.info(f"Loaded MCP server config: {server_name}")
        
        # 5. 接続辞書に登録して返却
        return connections
    except Exception as e:
        logger.error(f"Failed to load MCP config from {config_path}: {e}")
        return {}


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
        self.mcp_client: MultiServerMCPClient | None = None
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
        1. ツール名からツールインスタンスを取得
        2. ツールの存在確認
        3. 非同期ツール(.ainvoke)を優先的に実行
        4. 同期ツール(.invoke)を実行
        5. エラーハンドリングと結果返却
        """
        # 1. ツール名からツールインスタンスを取得
        tool_instance = self.get_tool(tool_name)
        # 2. ツールの存在確認
        if not tool_instance:
            return f"Error: Tool '{tool_name}' not found."
        
        try:
            # 3. 非同期ツール(.ainvoke)を優先的に実行
            if hasattr(tool_instance, "ainvoke"):
                logger.info(f"Executing ASYNC tool: {tool_name}")
                return self._run_coroutine(tool_instance.ainvoke(tool_args))
            # 4. 同期ツール(.invoke)を実行
            elif hasattr(tool_instance, "invoke"):
                logger.info(f"Executing SYNC tool: {tool_name}")
                return tool_instance.invoke(tool_args)
            else:
                return f"Error: Tool '{tool_name}' has no callable invoke or ainvoke method."
        except Exception as e:
            # 5. エラーハンドリングと結果返却
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            return f"Error executing tool {tool_name}: {e}"

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
            native_tools = self._load_native_tools()
            self.tools.extend(native_tools)
            logger.info(f"Successfully loaded {len(native_tools)} native tools.")
        except Exception as e:
            logger.error(f"An error occurred during native tool loading: {e}", exc_info=True)
        
        # 3. MCPツールを安全にロード（改善されたエラーハンドリング）
        try:
            mcp_tools = self._load_mcp_tools_robust()
            self.tools.extend(mcp_tools)
        except Exception as e:
            logger.error(f"An error occurred during MCP tool loading: {e}", exc_info=True)
        
        # 4. 最終的なツールリストからtool_mapを再構築
        self.tool_map = {tool.name: tool for tool in self.tools}
        logger.info(f"ToolManager initialized with {len(self.tools)} tools: {[t.name for t in self.tools]}")

    def _load_native_tools(self) -> List[BaseTool]:
        """
        ネイティブ(ローカル)実装のツールをロードして返す。

        処理の流れ:
        1. Google Custom Search APIツールの準備
        2. 設定ファイルの最大結果数を適用
        3. ツール名を統一("google_search")して既存コードとの互換性を維持
        """
        logger.info("Loading native tools...")
        try:
            # 1. Google Custom Search APIツールの準備
            google_search_tool = GoogleCustomSearchTool()
            # ツール名を統一("native_google_search")して既存コードとの互換性を維持
            google_search_tool.name = "native_google_search"
            google_search_tool.description = (
                "Search the web with Google Custom Search API and return multiple results (list of findings). This is a native tool."
            )
            tools = [google_search_tool]
        except Exception as e:
            logger.error(f"Failed to load Google Custom Search tool: {e}")
            # フォールバック: 空のツールリスト
            tools = []
        
        # 2. 利用可能なツールをログ出力
        for tool in tools:
            logger.info(f"Native tool available: {tool.name}")
        return tools

    def _load_mcp_tools_robust(self) -> List[BaseTool]:
        """
        堅牢化されたMCPツールローダー。個別のサーバーごとにエラーハンドリングを行い、
        一部のサーバーが失敗しても他のサーバーからツールを取得できるようにする。
        """
        logger.info("Loading MCP tools with robust error handling...")
        connections_config = _load_connections_from_config(self.config_path)
        
        if not connections_config:
            logger.warning("No MCP server connections found.")
            return []

        all_tools = []
        
        # 各サーバーを個別に処理
        for server_name, connection in connections_config.items():
            try:
                logger.info(f"Attempting to connect to MCP server: {server_name}")
                
                # 個別のサーバーに対してツールを取得
                server_tools = self._load_single_mcp_server(server_name, connection)
                
                if server_tools:
                    logger.info(f"Successfully loaded {len(server_tools)} tools from server: {server_name}")
                    all_tools.extend(server_tools)
                else:
                    logger.warning(f"No tools found for server: {server_name}")
                    
            except Exception as e:
                logger.error(f"Failed to load tools from MCP server '{server_name}': {e}")
                # 特定のエラーパターンに対する詳細な情報を提供
                if "Shutdown signal received" in str(e):
                    logger.error(f"Server '{server_name}' sent non-JSON output during startup. This is likely a race condition.")
                elif "ValidationError" in str(e):
                    logger.error(f"Server '{server_name}' output was not valid JSON-RPC. Server may need time to start up properly.")
                continue  # このサーバーをスキップして次へ

        logger.info(f"MCP tool loading completed. Total tools loaded: {len(all_tools)}")
        return all_tools

    def _load_single_mcp_server(self, server_name: str, connection: StdioConnection, max_retries: int = 3) -> List[BaseTool]:
        """
        単一のMCPサーバーからツールを取得する。リトライロジックを含む。
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} for server: {server_name}")
                
                # サーバーの起動を待つための短い遅延
                if attempt > 0:
                    delay = 2 ** attempt  # 指数バックオフ
                    logger.info(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                
                # 個別のMCPクライアントを作成
                single_client = MultiServerMCPClient(connections={server_name: connection})
                
                # ツールを取得
                discovered_tools = self._run_coroutine(single_client.get_tools())
                
                # サーバー名をプレフィックスとして追加
                unique_tools = []
                for tool in discovered_tools:
                    tool.name = f"{server_name}_{tool.name}"
                    unique_tools.append(tool)
                
                # クライアントをクリーンアップ
                try:
                    self._run_coroutine(single_client.close_all_sessions())
                except Exception as e:
                    logger.exception("Error while closing MCP client sessions for %s: %s", server_name, e)

                
                return unique_tools
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for server '{server_name}': {e}")
                if attempt == max_retries - 1:
                    # 最後の試行でも失敗した場合
                    raise e
                    
        return []

    def _load_mcp_tools(self) -> List[BaseTool]:
        """
        MCPサーバー群へ接続し、公開されているツールを発見して返す。
        
        注意: この旧バージョンのメソッドは後方互換性のために残していますが、
        _load_mcp_tools_robust()を使用することを推奨します。

        処理の流れ:
        1. 設定ファイルからStdioConnectionを構築
        2. MultiServerMCPClient経由でツールを列挙
        3. 衝突回避のため「サーバー名_ツール名」にリネーム
        4. 発見したツールをログ出力して返却
        """
        logger.info("Loading MCP tools...")
        # 1. 設定ファイルからStdioConnectionを構築
        connections_config = _load_connections_from_config(self.config_path)
        
        if not connections_config:
            logger.warning("No MCP server connections found.")
            return []

        try:
            # 2. MultiServerMCPClient経由でツールを列挙
            self.mcp_client = MultiServerMCPClient(connections=connections_config)
            # asyncio.run() の代わりに、クラス内のイベントループで実行する
            discovered_tools = self._run_coroutine(self.mcp_client.get_tools())
            
            # 3. 衝突回避のため「サーバー名_ツール名」にリネーム
            unique_tools = []
            for tool in discovered_tools:
                # ツールにサーバー名がメタデータとして含まれているか確認
                server_name = tool.metadata.get("server_name") if tool.metadata else None
                if server_name:
                    # サーバー名とツール名を結合して一意なIDを作成
                    tool.name = f"{server_name}_{tool.name}"
                unique_tools.append(tool)

            # 4. 発見したツールをログ出力して返却
            logger.info(f"Successfully discovered and processed {len(unique_tools)} MCP tools.")
            for tool in unique_tools:
                logger.info(f"MCP tool available: {tool.name}")
            return unique_tools
        except Exception as e:
            logger.error(f"Failed to get tools from MCP servers: {e}", exc_info=True)
            return []

    def get_tool(self, tool_name: str) -> BaseTool | None:
        """指定された名前のツールを取得する。

        見つからない場合は `None` を返す。
        """
        return self.tool_map.get(tool_name)

    def format_tools_for_prompt(self) -> str:
        """プロンプトに埋め込むためのツールリストを整形する。

        返却値は関数ツールのスキーマ(JSON文字列)。
        一部ツールが `pydantic` モデルを持つ場合に備え、柔軟にスキーマを抽出する。
        """
        if not self.tools:
            return "No tools available."

        try:
            tool_schemas = []
            for tool in self.tools:
                # ★★★ 修正点: args_schemaが辞書であることを前提に処理 ★★★
                if hasattr(tool, 'args_schema') and isinstance(tool.args_schema, dict):
                    parameters_schema = tool.args_schema
                # DuckDuckGoSearchRunのようにPydanticモデルを持つ場合
                elif hasattr(tool, 'args_schema') and hasattr(tool.args_schema, 'model_json_schema'):
                    parameters_schema = tool.args_schema.model_json_schema()
                else:
                    parameters_schema = {"type": "object", "properties": {}}

                schema = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": parameters_schema
                    }
                }
                tool_schemas.append(schema)

            return json.dumps(tool_schemas, indent=2)
        except Exception as e:
            logger.error(f"Failed to format tools for prompt: {e}", exc_info=True)
            return "Error formatting tools."

    def format_tools_for_react_prompt(self) -> str:
        """
        ReActプロンプトのために、ツール一覧を人が読みやすいシグネチャ形式の文字列に整形する。

        例:
          - tool_name(arg1: string, arg2: number): 説明
        """
        if not self.tools:
            return "No tools available."

        tool_strings = []
        for tool in self.tools:
            # Pydanticモデルのスキーマから引数を取得
            if hasattr(tool, 'args_schema') and hasattr(tool.args_schema, 'model_json_schema'):
                schema = tool.args_schema.model_json_schema()
                properties = schema.get('properties', {})
                args_repr = ", ".join(
                    f"{name}: {prop.get('type', 'any')}"
                    for name, prop in properties.items()
                )
            else:
                # フォールバック（args_schemaがない、またはPydanticモデルでない場合）
                args_repr = ""
            # プロンプトのインデントに合わせて整形
            tool_strings.append(f"  - {tool.name}({args_repr}): {tool.description}")
        return "\n".join(tool_strings)

    def cleanup(self):
        """リソースのクリーンアップ。

        - MCPクライアントの全セッションをクローズ
        - バックグラウンドイベントループを停止
        """
        if self.mcp_client and hasattr(self.mcp_client, 'close_all_sessions'):
            print("INFO: Closing all MCP client sessions...")
            try:
                self._run_coroutine(self.mcp_client.close_all_sessions())
            except Exception as e:
                print(f"ERROR: Error during MCP client cleanup: {e}")
                
        #バックグラウンドスレッドのイベントループを停止
        if self._loop.is_running():
            print("INFO: Stopping async task runner thread...")
            self._loop.call_soon_threadsafe(self._loop.stop)
            # スレッドが完全に終了するのを最大5秒間待つ 
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                print("WARNING: Async runner thread did not stop gracefully.")