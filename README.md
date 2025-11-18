![log](https://github.com/coco4atJP/tepora-alpha/blob/main/Tepora_logo.png)

# Tepora - マルチAIエージェントシステム (BATA v2)

TeporaはローカルLLM、EM-LLM（Episodic Memory for LLMs）、LangGraph、MCPツールチェーンを統合したマルチエージェント対話システムです。2025年11月の大規模リファクタリングで、コアがすべてモジュール化され、`AgentApplication`が初期化フェーズ → EM-LLMフェーズ → グラフ構築の3段階でライフサイクルを管理します。

## ✨ 主な機能

1. **協調型マルチエージェント** – Gemma-3N系キャラクターエージェントがユーザー対応とオーダー生成を担当し、Jan-nano系プロフェッショナルエージェントがReActループでタスクを遂行します。思考・ツール実行・観察ログはLangGraphノードで管理されます。
2. **EM-LLM 記憶システム** – 驚き度に基づくイベントセグメンテーション、境界精密化、2段階検索、統計表示コマンド（`/emstats`, `/emstats_prof`）を備えた長期記憶を装備。初期化に失敗した場合は自動で従来メモリへフォールバックします。
3. **LangGraph 状態管理** – `/agentmode` / `/search` / 通常チャット / 統計コマンドをコマンドルータで判別し、Direct Answer・検索・ReActループの各フローをグラフとして構築。ストリーミング出力と再帰制限も設定済みです。
4. **ツール実行ランタイム** – ネイティブツール（Google Custom Search + WebFetch）とMCPツールを単一インターフェースでロード。同期/非同期ツールをバックグラウンドイベントループでブリッジします。
5. **llama.cpp による動的モデル制御** – `LLMManager` がGemma / Jan / Embeddingモデルを必要に応じてロードし、プロセスのヘルスチェックやトークンカウントを管理します。
6. **構成駆動型** – すべての設定が `agent_core/config/` に分割され、プロンプトやモデル、ツール、EM-LLM閾値を個別に調整できます。

## 🏗️ アーキテクチャ概要

```
Tepora_app/
├── main.py                      # 3行で AgentApplication を起動
├── agent_core/
│   ├── app/                     # ライフサイクル＆CLI（AgentApplication, utils）
│   ├── graph/                   # LangGraphエンジン（core, nodes/, routing, constants）
│   ├── em_llm/                  # EM-LLM セグメンター/統合
│   ├── llm/                     # llama.cpp 実行補助
│   ├── llm_manager.py           # モデルロード・ヘルスチェック
│   ├── tool_manager.py          # ネイティブ+MCPツール統合
│   ├── memory/memory_system.py  # ChromaDB ベース永続記憶
│   ├── embedding_provider.py
│   └── config/                  # app / models / prompts / em / tools / ...
├── llama.cpp/                   # llama-server バイナリ格納ディレクトリ
├── mcp_tools_config.json
└── tests/
    └── test_llm_manager.py
```


## 💿 事前準備

| 必要項目 | 内容 |
| --- | --- |
| Python | 3.10 以上（開発時は 3.12） |
| ハードウェア | GGUFモデルを扱えるCPU/GPU環境（llama.cpp対応） |
| LLMモデル | `gemma-3n-E4B-it-IQ4_XS.gguf`, `jan-nano-128k-iQ4_XS.gguf`, `embeddinggemma-300M-Q8_0.gguf` などを `Tepora_app/` 直下に配置（`MODEL_BASE_PATH`未設定時） |
| llama.cpp バイナリ | 公式リリースをダウンロードし `Tepora_app/llama.cpp/<バージョン>/llama-server.exe` として配置。`agent_core/llm/executable.py` が最適な実行ファイルを自動検出します。 |
| Google API | `.env` に `GOOGLE_CUSTOM_SEARCH_API_KEY`, `GOOGLE_CUSTOM_SEARCH_ENGINE_ID` を設定（検索ツール用） |
| MCP ツール (任意) | `mcp_tools_config.json` にサーバーコマンドと引数を追記。Node.js が必要なサーバーもあります。 |

## 🚀 セットアップ手順

```bash
git clone https://github.com/coco4atJP/Tepora.git
cd Tepora

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

# APIキー設定
cp .env.example .env
# .env を開いて Google API などを入力
```

1. `Tepora_app/` 直下に GGUF モデルを配置します（大容量のためシンボリックリンクでも可）。
2. `llama.cpp/` に展開したバイナリを置き、`llama-server` が存在することを確認します。
3. MCP を使う場合は `mcp_tools_config.json` を編集し、サーバー起動コマンドを記述します。
4. ChromaDB（`./chroma_db_em_llm` / `./chroma_db_fallback`）は初回実行時に自動で生成されます。

## ▶️ 実行

```bash
cd Tepora_app
python main.py
```

初回起動時は以下のフェーズが表示されます。

1. **Phase 1** – LLM / ツール初期化（Gemmaをプリロードし、MCP + ネイティブツールを起動）
2. **Phase 2** – EM-LLM 構築（埋め込みモデル・ChromaDB を準備）
3. **Phase 3** – LangGraph 構築（EM有効時は `EMEnabledAgentCore`, 失敗時は従来コアに自動フォールバック）

## ⌨️ CLI コマンド

| コマンド | 説明 |
| --- | --- |
| `こんにちは` | 通常のチャット（Direct Answerフロー） |
| `/search LangGraph とは？` | Gemmaが検索クエリ生成 → Google Custom Search → RAG要約 |
| `/agentmode ビットコインの価格調査` | キャラクターがオーダー生成 → プロフェッショナルがReActループでツール実行 |
| `/emstats` | キャラクターEM-LLM統計（イベント数・サプライズ統計） |
| `/emstats_prof` | プロフェッショナルEM-LLM統計 |
| `exit` / `quit` | アプリ終了 |

## 🧩 コアモジュール詳細

| モジュール | 役割 |
| --- | --- |
| `main.py` | `AgentApplication` を起動するエントリーポイント。
| `agent_core/app/agent_app.py` | 3フェーズ初期化、EM-LLM切り替え、CLIループ、クリーンアップ。
| `agent_core/graph/core.py` & `graph/nodes/` | Direct Answer / Search / ReAct ルートとLangGraphノード実装。
| `agent_core/graph/em_llm_core.py` | EM-LLM版グラフ。メモリノードをEM専用実装に差し替え、統計ノードを追加。
| `agent_core/em_llm/` | セグメンテーション、境界精密化、統合、統計ロジック。
| `agent_core/tool_manager.py` | ネイティブ+MCPツールを同期/非同期問わず単一APIで実行。
| `agent_core/llm_manager.py` | llama.cpp サーバープロセス起動、ヘルスチェック、動的モデル切替、トークンカウント。
| `agent_core/memory/memory_system.py` | ChromaDB 永続記憶。EM-LLM統合からも共有。
| `agent_core/config/` | 入力制限、プロンプト、モデル設定、EM閾値、ツール設定を分割管理。

## 🛠️ ツール & MCP

- **ネイティブ**: `agent_core/tools/native.py` の Google Custom Search / WebFetch がデフォルト搭載。
- **MCP**: `mcp_tools_config.json` にサーバーを追加すると `ToolManager` が自動検出して `server_toolname` 形式で登録します。すべてのツール呼び出しは LangGraph のツールノードから行われ、同期コードでも `execute_tool` だけで利用できます。

```jsonc
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:/Users/username/Desktop"]
    }
  }
}
```

## 🧠 メモリとEM-LLM

- キャラクター / プロフェッショナルそれぞれに `./chroma_db_em_llm` 内の別コレクションを持ちます。
- EM-LLMが無効な場合は `./chroma_db_fallback` を使う従来メモリに切り替えます。
- `/emstats*` コマンドでイベント数・サプライズ統計・設定値をリアルタイムに確認できます。

## ⚙️ 主な設定ファイル

| ファイル | 内容 |
| --- | --- |
| `.env` | Google API などのシークレット。必要に応じて `MODEL_BASE_PATH` も指定可能。
| `agent_core/config/app.py` | 入力長、コマンドプレフィックス、再帰制限、ストリーミングイベント名。
| `agent_core/config/models.py` | 各GGUFモデルのパス・ポート・推論パラメータ。
| `agent_core/config/em.py` | サプライズウィンドウ、バッファ比、イベントサイズなどEM-LLM設定。
| `agent_core/config/tools.py` | Google Custom Searchのキーやタイムアウト。
| `mcp_tools_config.json` | MCPサーバーと引数。

## 🧪 テスト

```bash
python -m unittest discover tests
```

LangGraphノードやEM-LLM統合の追加テストは `tests/` 配下に拡張してください。


## 📜 ライセンス

Apache License 2.0。詳細は [`LICENSE`](LICENSE) を参照してください。各 GGUF モデルは提供元ライセンスに従います。
