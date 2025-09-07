![log](Tepora_log.png)

# Tepora - マルチAIエージェントシステム (Beta v1.5)

洗練されたマルチエージェント対話型AIシステムを提供します。このプロジェクトでは、ローカルLLM、動的リソース管理、拡張可能なMCPツールシステム、そして**EM-LLM (arXiv:2407.09450)** アーキテクチャに基づく記憶システムを活用し、文脈を理解し学習する自律的なエージェントを構築します。

## ✨ 主な機能

* **マルチエージェントアーキテクチャ**: 2つの主要エージェントによる設計:
* **キャラクターエージェント (`Gemma-3N`)**: キャラクターとして、ユーザーと対話をします。Agent_Modeでは、ユーザーリクエストを解釈し、構造化されたJSON形式の「オーダー」を作成し、最終的なユーザーへの報告を行います。
* **エグゼキューターエージェント (`Jan-nano`)**: ReAct (推論+行動) ループを用いてオーダーを実行する、プロフェッショナルで実用的なエージェント。
* **動的LLM管理**: `LLMManager` は、GGUFモデルをVRAMまたはRAMに動的にロード/アンロードすることで、コンシューマーグレードのGPUもしくはCPU上で複数の強力なモデルを使用できるようにします。
* **拡張可能なツールシステム**: `ToolManager` は以下を統合します。
* **ネイティブツール**: `GoogleCustomSearchTool` などの Python ベースのツール。
* **MCP (Model Context Protocol) ツール**: 別プロセスとして実行されているツールと通信するためのカスタムプロトコル。言語に依存しないツール開発を可能にします。
* **ステートフルなグラフベースの実行**: `LangGraph` を基盤とするエージェントのロジックは状態グラフとして定義され、さまざまなユーザーコマンドに対して複雑な条件付きフローを実現します。
* **エピソード記憶 (EM-LLM)**: エージェントは長期的なエピソード記憶を持つようになりました。過去の関連する対話を取得して現在の応答に反映させ、新しい対話を将来のために記憶として定着させることで、継続的な学習ループを実現します。
* **複数のインタラクションモード**:
* **ダイレクトチャット**: シンプルで直接的な会話が可能です。
* **検索モード (`/search`)**: Web 検索と要約専用のフローです。
* **エージェントモード (`/agentmode`)**: 複雑なタスクのために、完全なマルチエージェント ReAct ループを実行します。
* **構成駆動型**: プロンプト、モデルパラメータ、API キー、ツール設定を一元的に構成します。

## 🏗️ アーキテクチャの概要

このアプリケーションは、状態駆動型のグラフベースの実行モデルに従います。

1. **`main.py`**: アプリケーションのエントリポイント。`LLMManager`と`ToolManager`を初期化した後、**EM-LLM（エピソード記憶）システムの初期化を試みます**。成功した場合は、EM-LLMの機能を完全に統合した`EMEnabledAgentCore`グラフを構築します。失敗した場合は、基本的な機能を備えた従来の`AgentCore`グラフにフォールバックします。その後、コマンドラインループでユーザー入力を受け付けます。
2. **`agent_core/graph.py`**: **従来の`AgentCore`を定義します**。これは、EM-LLMシステムが利用できない場合のフォールバックとして機能する、エージェントの基本的な実行グラフです。`LangGraph`を基盤とし、コマンドルーティング、ReActループ、各種ツール実行のロジックを含みます。EM-LLMが有効な場合は、このグラフは直接使用されず、`EMEnabledAgentCore`が優先されます。
    * **記憶パイプライン**: 全ての対話は以下のパイプラインを通過します。
        1.  **記憶検索 (Retrieval)**: 対話の開始時に、ユーザー入力に基づいて関連する過去のエピソードを長期記憶から検索します。
        2.  **記憶統合 (Synthesis)**: SLM（小規模言語モデル）が、検索されたエピソードを簡潔な文脈サマリーに統合します。
        3.  **タスク実行 (Task Execution)**: メインのLLMが、統合された記憶をコンテキストとして利用し、応答生成やツール実行などのタスクを遂行します。
        4.  **記憶定着 (Consolidation)**: 対話の最後に、SLMがその回のやり取りを要約し、新しいエピソードとして長期記憶に保存します。
    * **ルーティング**: 記憶パイプラインの後、`route_by_command` 関数がユーザー入力を3つの主要ブランチ (`direct_answer`、`search`、`agent_mode`) のいずれかに誘導します。
    * **エージェントモードフロー**:
        1.  `generate_order_node`: キャラクターエージェント (Gemma) が JSON プランを作成します。
        2.  `agent_reasoning_node`: Executor Agent (Jan-nano) は、ツールを使用して計画を実行する ReAct ループを開始します。
        3.  `tool_node`: エージェントが `ToolManager` を介して選択したツールを実行します。
        4.  `synthesize_final_response_node`: ReAct ループが完了すると、最終的な技術レポートがユーザーフレンドリーなレスポンスに変換されます。
3. **`agent_core/llm_manager.py`**: LLM のライフサイクルを管理します。メインのLLMに加えて、記憶の統合と定着を担当するSLM（小規模言語モデル）も管理します。
4. **`agent_core/tool_manager.py`**: すべてのツールのための統一インターフェースです。ネイティブ Python ツールと MCP 経由で接続された外部ツールを検出および管理します。
5. **`agent_core/memory/memory_system.py`**: SQLiteデータベースを利用して、エピソード記憶の保存と類似度検索を管理します。

## 🚀 はじめに

### 前提条件

* `Python 3.10` 以上 <sub> 開発ではpython3.12が使用されました。</sub>
* モデル高速化のために、CUDA 対応の NVIDIA GPU または ROCm 対応の AMD GPU。CPU のみのモードも利用可能ですが、GPUと比較すると遅くなります。
* `Node.js` 多くのMCPサーバーを使用するために必要です。
* `Git`
* `google Custom Search JSON API` 検索機能を使用するためにはGoogleAPIを取得する必要があります。

### 最低の構成スペック <sub>(理論値)</sub>
* 8 GB以上のデスク空き容量
* 16GB以上のRAMもしくは 6GB以上のVRAMと8GB以上のRAM : 展開されるMCPサーバー分のRAMとロードされるSLMのためのRAM or VRAMが必須です。
  <sub> `llama.cpp`の`n_ctx`を削減することでロードされるLLMやSLMのRAMは減らせますが、動作に支障をきたす可能性があります。 </sub>
* `Llama-cpp-python` が対応している計算環境。

### インストール

1. **リポジトリのクローンを作成します:**
```bash
git clone 
cd Tepora
```

2. **依存関係をインストールします:**
仮想環境の使用を推奨します。
```bash
python -m venv venv
source venv/bin/activate # Windows では `venv\Scripts\activate` を使用します
pip install -r requirements.txt
```

3. **環境変数の設定:**
サンプルファイルをコピーして、プロジェクトルートに `.env` ファイルを作成します:
```bash
cp .env.example .env
```
次に、`.env` ファイルを編集して API キーを追加します:
```
# .env
GOOGLE_CUSTOM_SEARCH_API_KEY="your_google_api_key"
GOOGLE_CUSTOM_SEARCH_ENGINE_ID="your_google_cx_id"
```

4. **プロジェクトルートにモデルを配置:**
main.pyと同じ階層に使用するllama.cpp対応GGUF形式ファイルを配置します。
デフォルトのモデルは下記のものです。

[unsloth/gemma-3n-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-3n-E4B-it-GGUF/blob/main/gemma-3n-E4B-it-IQ4_XS.gguf)

[Menlo/Jan-nano-128k-gguf](https://huggingface.co/Menlo/Jan-nano-128k-gguf/blob/main/jan-nano-128k-iQ4_XS.gguf)

[unsloth/gemma-3-270m-it-GGUF](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/blob/main/gemma-3-270m-it-Q8_0.gguf)

[Casual-Autopsy/snowflake-arctic-embed-l-v2.0-gguf](https://huggingface.co/Casual-Autopsy/snowflake-arctic-embed-l-v2.0-gguf/blob/main/snowflake-arctic-embed-l-v2.0-q6_k.gguf)

### エージェントの実行

プロジェクトルートディレクトリからエージェントを起動します:
```bash
python main.py
```

## 🤖使用方法

エージェントが起動したら、ターミナルで操作できます。

* **直接チャット:**
> YOU: `こんにちは、お元気ですか？`

* **検索モード:**
> YOU: `/search LangGraph とは？`

* **エージェントモード (複雑なタスク向け):**
> YOU: `/agentmode ビットコインの現在の価格を調べ、最新ニュースを見つけます。`

* **アプリケーションを終了する:**
> YOu: `exit`

## 🧩 コアコンポーネント

* **`main.py`**: アプリケーションのエントリポイント、初期化、およびメインの会話ループ。
* **`agent_core/graph.py`**: `LangGraph` 実行グラフ、ノード、エッジを定義します。すべてのエージェントモードのコアロジックが含まれています。
* **`agent_core/state.py`**: グラフ内のノード間で渡される状態を表す `AgentState` TypedDict を定義します。
* **`agent_core/llm_manager.py`**: GGUF モデルの動的なロード/アンロードを処理して VRAM もしくは RAM を管理します。メインのLLMと記憶処理用のSLMの両方を扱います。
* **`agent_core/tool_manager.py`**: すべてのツール (ネイティブおよび MCP) の統合実行インターフェースを検出、管理、および提供します。
* **`agent_core/memory/memory_system.py`**: SQLiteデータベースを利用して、エピソード記憶の保存と類似度検索を管理します。
* **`agent_core/config.py`**: モデルパス、生成パラメータ、プロンプト、ペルソナ、API キーの一元的な構成。EM-LLMの記憶統合・定着用プロンプトも含まれます。

## 🛠️ ツールシステム

エージェントは2種類のツールを使用できます。

### ネイティブツール

これらは、`tool_manager.py` の `GoogleCustomSearchTool` のように、`langchain_core.tools.BaseTool` から継承された Python クラスです。これらは `ToolManager` によって直接ロードされます。

### MCP (Model Context Protocol) ツール

このシステムにより、エージェントは別プロセスで実行されるツールを使用できます。ツールは任意の言語で記述できます。

1. **設定**: `mcp_tools_config.json` でツールサーバーを定義します。サーバー定義はClaudeDesktop方式で可能です。
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:\\Users\\username\\Desktop",
        "C:\\Users\\username\\Downloads"
      ]
    }
  }
}
```
2. **検出**: `ToolManager` は設定で定義されたプロセスを起動し、`stdio` 経由で接続し、MCP プロトコルを使用してそのプロセスが提供するツールを検出します。
3. **命名**: MCP ツールは、競合を避けるため、自動的に `server_name_tool_name` という名前が付けられます。

## ⚙️ 設定

* **`.env`**: API キーなどのシークレットを保存します。バージョン管理にはコミットされません。
* **`agent_core/config.py`**: メインの設定ファイルです。
* `MODELS_GGUF`: モデルパス、モデルパラメータを定義しています。生成パラメータは temperature Top.P Top.K max_tokens がデフォルト定義です。
* `PERSONA_PROMPTS`: キャラクターエージェントの異なるキャラクターペルソナを定義します。デフォルトでは`souha_yoi`(奏羽 茗伊) `bunny_girl`(マリナ)の2種類が用意されています。どちらも日本語で記述されているので、必要に応じて書き換えてください。
* `ACTIVE_PERSONA`: 現在のペルソナを選択します。
* `BASE_SYSTEM_PROMPTS`: 要約、ReAct推論、記憶の統合・定着などのタスクにおけるコア機能プロンプトを定義します。
* **`mcp_tools_config.json`**: 外部ツールサーバーを設定します。


## 📜 ライセンス


このプロジェクトはにApache License 2.0に基づきライセンスされています。詳細は`LICENSE`ファイルをご覧ください。









