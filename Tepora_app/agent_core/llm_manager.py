# agent_core/llm_manager.py
import gc
import logging
from typing import Dict, List
from pathlib import Path
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from . import config

logger = logging.getLogger(__name__)

class LLMManager:
    """
    GGUFモデルをLlama.cppで動的にロード・アンロードするためのマネージャークラス。
    """
    def __init__(self):
        self._current_model_key = None
        self._chat_llm: BaseChatModel | None = None
        self._embedding_llm: Embeddings | None = None
        self._current_model_config: Dict | None = None # ロードされたモデルの設定を保持
        self._tokenizer_llm: LlamaCpp | None = None # トークンカウント専用
        logger.info("LLMManager for Llama.cpp initialized.")

    def _unload_model(self):
        """現在ロードされているモデルを解放する。"""
        if self._current_model_key:
            logger.info(f"Unloading model: {self._current_model_key}")
            del self._chat_llm
            self._chat_llm = None
            self._current_model_key = None
            gc.collect() # ガベージコレクションを強制
            self._current_model_config = None
            logger.info("Model unloaded.")

    def _load_model(self, key: str):
        """指定されたGGUFモデルをLlama.cppでロードする。"""
        if self._current_model_key == key:
            return

        self._unload_model()
        
        model_config = config.MODELS_GGUF[key]
        self._current_model_config = model_config
        
        # プロジェクトのルートディレクトリを基準にモデルファイルの絶対パスを構築
        # これにより、どこからスクリプトを実行してもパスが安定します
        project_root = Path(__file__).parent.parent
        model_path = project_root / model_config["path"]

        if not model_path.exists():
            # デバッグしやすいように、エラーメッセージに絶対パスを含めます
            raise FileNotFoundError(f"Model file not found at the absolute path: {model_path.resolve()}")

        logger.info(f"Loading {key} model from: {model_path.resolve()}...")
        
        # ChatLlamaCppの初期化パラメータを構築
        init_kwargs = {
            "model_path": str(model_path.resolve()),
            "streaming": True,
            "verbose": False,
        }

        # config.pyから動的にパラメータをコピー
        # これにより、将来的にパラメータを追加する際にllm_manager.pyの変更が不要になります
        params_to_copy = [
            "n_ctx", "n_gpu_layers", "temperature", "top_p", "top_k", 
            "max_tokens", "repeat_penalty", "logprobs"
        ]
        for param in params_to_copy:
            if param in model_config:
                init_kwargs[param] = model_config[param]

        self._chat_llm = ChatLlamaCpp(**init_kwargs)
        self._current_model_key = key
        logger.info(f"{key} model loaded successfully (context size: {model_config['n_ctx']}).")

    def _load_tokenizer(self):
        """トークンカウント専用のLlamaCppインスタンスをロードする。"""
        if self._tokenizer_llm is None:
            logger.info("Loading tokenizer-only LLM instance...")
            # メインの対話モデルと同じトークナイザを使用する
            model_config = config.MODELS_GGUF["gemma_3n"]
            project_root = Path(__file__).parent.parent
            model_path = project_root / model_config["path"]

            if not model_path.exists():
                raise FileNotFoundError(f"Tokenizer model file not found: {model_path.resolve()}")

            # トークナイズのためだけにGPUメモリは使わない
            self._tokenizer_llm = LlamaCpp(
                model_path=str(model_path.resolve()),
                n_ctx=model_config.get("n_ctx", 4096),
                n_gpu_layers=0, # GPUオフロードなし
                verbose=False,
            )
            logger.info("Tokenizer-only LLM instance loaded.")

    def count_tokens_for_messages(self, messages: List[BaseMessage]) -> int:
        """メッセージリストの合計トークン数を数える。"""
        if not messages:
            return 0
            
        self._load_tokenizer()
        if self._tokenizer_llm:
            # ここでは単純に各メッセージのcontentのトークン数を合計する。
            # 実際のプロンプトテンプレートによる追加トークンは含まれないが、
            # 履歴の長さを管理する目的では十分な近似となる。
            total_tokens = 0
            for msg in messages:
                if isinstance(msg.content, str):
                    total_tokens += self._tokenizer_llm.get_num_tokens(msg.content)
            return total_tokens
        
        # フォールバックとして文字数ベースで概算
        logger.warning("Tokenizer LLM not available, falling back to character-based token estimation.")
        # 1トークンあたり平均4文字と仮定
        return sum(len(msg.content) for msg in messages if isinstance(msg.content, str)) // 4

    # 埋め込みモデルを取得するための専用メソッド 
    def get_embedding_model(self) -> Embeddings:
        """埋め込みモデル (Jina v3) を取得またはロードする。"""
        if self._embedding_llm is None:
            # LlamaCppのインスタンスを直接生成してキャッシュする
            model_config = config.MODELS_GGUF["embedding_model"]
            model_path = Path(__file__).parent.parent / model_config["path"]
            if not model_path.exists():
                raise FileNotFoundError(f"Embedding model not found: {model_path}")
            
            logger.info(f"Loading embedding model from: {model_path}...")
            self._embedding_llm = LlamaCppEmbeddings(
                model_path=str(model_path.resolve()),
                n_ctx=model_config["n_ctx"],
                n_gpu_layers=model_config["n_gpu_layers"],
                verbose=False
            )
            logger.info("Embedding model loaded and cached.")
        return self._embedding_llm

    def get_current_model_config_for_diagnostics(self) -> Dict:
        """
        診断用に、現在ロードされているメインのChatLLMモデルの設定を返す。
        """
        if self._current_model_key and self._current_model_config:
            # インスタンス変数に保持した設定から診断情報を返す
            # これにより、ChatLlamaCppの内部実装への依存がなくなる
            config_copy = self._current_model_config.copy()
            config_copy["key"] = self._current_model_key
            # streamingは常にTrueなので明示的に追加
            config_copy["streaming"] = True
            return config_copy
        return {}

    def get_character_agent(self) -> BaseChatModel:
        """キャラクター・エージェント (Gemma 3N) を取得する。"""
        self._load_model("gemma_3n")
        return self._chat_llm

    def get_professional_agent(self) -> BaseChatModel:
        """プロフェッショナル・エージェント (Jan-nano) を取得する。"""
        self._load_model("jan_nano")
        return self._chat_llm

    def get_slm_summarizer(self) -> BaseChatModel:
        """EM-LLM用SLMを取得する。"""
        self._load_model("slm_summarizer")
        return self._chat_llm

    def cleanup(self):
        """アプリケーション終了時にモデルをアンロードする。"""
        logger.info("Cleaning up LLMManager...")
        self._unload_model()
        # トークナイザインスタンスも解放
        if self._tokenizer_llm:
            logger.info("Unloading tokenizer-only LLM instance.")
            # LlamaCppオブジェクトには明示的なcloseメソッドがないため、delとGCに任せる
            del self._tokenizer_llm
            self._tokenizer_llm = None
            gc.collect()