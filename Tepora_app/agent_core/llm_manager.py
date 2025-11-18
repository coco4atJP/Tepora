# agent_core/llm_manager.py
import gc
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import LlamaCpp  # トークンカウント用に維持
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from . import config
from .llm import (
    build_server_command,
    find_server_executable,
    launch_server,
    perform_health_check,
    terminate_process,
)

logger = logging.getLogger(__name__)

class LLMManager:
    """
    GGUFモデルをLlama.cppで動的にロード・アンロードするためのマネージャークラス。
    """
    def __init__(self):
        self._lock = threading.RLock()  # モデルのロード/アンロード操作をスレッドセーフにする
        self._current_model_key = None
        self._chat_llm: ChatOpenAI | None = None
        self._active_process: subprocess.Popen | None = None # 起動中のサーバープロセスを保持
        self._embedding_process: subprocess.Popen | None = None # 埋め込みモデル専用のプロセス
        self._embedding_llm: Embeddings | None = None
        self._current_model_config: Dict | None = None # ロードされたモデルの設定を保持
        self._embedding_config: Dict | None = None # 埋め込みモデルの設定を保持
        self._tokenizer_llm: LlamaCpp | None = None # トークンカウント専用

        logger.info("LLMManager for Llama.cpp initialized.")

    def _unload_model(self):
        """現在ロードされているモデルを解放する。"""
        if self._current_model_key:
            logger.info(f"Unloading model: {self._current_model_key}")

        timeout_sec = config.LLAMA_CPP_CONFIG.get("process_terminate_timeout", 10)
        if self._active_process:
            logger.info(f"Terminating server process (PID: {self._active_process.pid})...")
            try:
                self._active_process.terminate()
                self._active_process.wait(timeout=timeout_sec)
                logger.info("Server process terminated gracefully.")
            except subprocess.TimeoutExpired:
                logger.warning("Process didn't terminate gracefully, forcing kill...")
                try:
                    self._active_process.kill()
                    self._active_process.wait()
                    logger.info("Server process killed forcefully.")
                except Exception as kill_error:  # noqa: BLE001
                    logger.error("Failed to kill server process cleanly: %s", kill_error, exc_info=True)
            except Exception as terminate_error:  # noqa: BLE001
                logger.error("Error while terminating server process: %s", terminate_error, exc_info=True)
                try:
                    self._active_process.kill()
                    self._active_process.wait()
                except Exception as kill_error:  # noqa: BLE001
                    logger.error("Failed to kill server process after termination error: %s", kill_error, exc_info=True)
            finally:
                self._active_process = None

        if self._chat_llm is not None:
            del self._chat_llm
        self._chat_llm = None
        self._current_model_key = None
        self._current_model_config = None
        gc.collect() # メモリを明示的に解放

    def _unload_embedding_model(self):
        """埋め込みモデルを解放する。"""
        if self._embedding_process:
            logger.info("Unloading embedding model...")
            logger.info(f"Terminating embedding server process (PID: {self._embedding_process.pid})...")
            timeout_sec = config.LLAMA_CPP_CONFIG.get("process_terminate_timeout", 10)
            try:
                self._embedding_process.terminate()
                self._embedding_process.wait(timeout=timeout_sec)
                logger.info("Embedding server process terminated gracefully.")
            except subprocess.TimeoutExpired:
                logger.warning("Embedding process didn't terminate gracefully, forcing kill...")
                try:
                    self._embedding_process.kill()
                    self._embedding_process.wait()
                except Exception as kill_error:  # noqa: BLE001
                    logger.error("Failed to kill embedding server process cleanly: %s", kill_error, exc_info=True)
            except Exception as terminate_error:  # noqa: BLE001
                logger.error("Error while terminating embedding server process: %s", terminate_error, exc_info=True)
                try:
                    self._embedding_process.kill()
                    self._embedding_process.wait()
                except Exception as kill_error:  # noqa: BLE001
                    logger.error("Failed to kill embedding server process after termination error: %s", kill_error, exc_info=True)
            finally:
                self._embedding_process = None
                self._embedding_llm = None
                self._embedding_config = None

    def _find_server_executable(self, llama_cpp_dir: Path) -> Path | None:
        return find_server_executable(llama_cpp_dir, logger=logger)

    def _perform_health_check(self, port: int, key: str, stderr_log_path: Path | None = None):
        process_ref = lambda: self._active_process  # noqa: E731
        perform_health_check(
            port,
            key,
            process_ref=process_ref,
            stderr_log_path=stderr_log_path,
            logger=logger,
        )

    def _load_model(self, key: str):
        """指定された対話用GGUFモデルをLlama.cppでロードする。"""
        with self._lock:
            if self._current_model_key == key:
                return

            # 現在の対話用モデルのみアンロード
            self._unload_model()
            
            model_config = config.MODELS_GGUF[key]
            self._current_model_config = model_config
            
            # プロジェクトのルートディレクトリを基準にパスを構築
            project_root = Path(__file__).parent.parent
            model_path = project_root / model_config["path"]

            # --- サーバー実行ファイルのパスを動的に決定 ---
            llama_cpp_dir = project_root / "llama.cpp"
            server_executable = self._find_server_executable(llama_cpp_dir)

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at the absolute path: {model_path.resolve()}")
            if not server_executable:
                error_message = (
                    f"llama.cpp server executable not found in the '{llama_cpp_dir}' directory.\n"
                    "Please download the pre-built binary for your system from the official llama.cpp releases,\n"
                    "unzip it, and place the resulting folder (e.g., 'llama-b2915-bin-win-avx2-x64') inside the 'llama.cpp' directory."
                )
                logger.error(error_message)
                raise FileNotFoundError(error_message)

            # --- ログファイルの準備 ---
            log_dir = project_root / "logs"
            log_dir.mkdir(exist_ok=True)
            stderr_log_path = log_dir / f"llama_server_{key}_{int(time.time())}.log"

            # --- サーバープロセスの起動 ---
            port = model_config["port"]
            extra_args = []
            if model_config.get("logprobs"):
                extra_args.append("--logprobs")
            command = build_server_command(
                server_executable,
                model_path,
                port=port,
                n_ctx=model_config.get("n_ctx", 4096),
                n_gpu_layers=model_config.get("n_gpu_layers", -1),
                extra_args=extra_args,
            )

            self._active_process = launch_server(command, stderr_log_path=stderr_log_path, logger=logger)

            try:
                # --- ヘルスチェックの実行 ---
                self._perform_health_check(port, key, stderr_log_path)
            except Exception:
                logger.error("Health check failed while starting server for '%s'. Cleaning up process.", key, exc_info=True)
                if self._active_process:
                    terminate_process(self._active_process, logger=logger)
                    self._active_process = None
                raise
            logger.info(f"Server for '{key}' started successfully with PID: {self._active_process.pid}")
            # --- ChatOpenAIクライアントの初期化 ---
            base_url = f"http://localhost:{port}/v1"
            init_kwargs = {
                "model": key,
                "base_url": base_url,
                "api_key": "dummy-key",
                "streaming": True,
            }

            # 標準のOpenAIパラメータ
            standard_params = [
                "temperature", "top_p", "max_tokens", "repeat_penalty"
            ]
            for param in standard_params:
                if param in model_config:
                    init_kwargs[param] = model_config[param]

            # 非標準だがLlama.cppがサポートするパラメータはextra_bodyに入れる
            extra_body = {}
            if model_config.get("logprobs"):
                extra_body["logprobs"] = True
                logger.info(f"Client for '{key}' will request logprobs via extra_body.")

            if "top_k" in model_config:
                extra_body["top_k"] = model_config["top_k"]
                logger.info(f"Client for '{key}' will use top_k={model_config['top_k']} via extra_body.")

            if extra_body:
                init_kwargs["extra_body"] = extra_body

            self._chat_llm = ChatOpenAI(**init_kwargs)
            self._current_model_key = key
            logger.info(f"LLM client for '{key}' connected to {base_url}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def _load_tokenizer(self):
        """トークンカウント専用のLlamaCppインスタンスをロードする。"""
        if self._tokenizer_llm is None:
            logger.info("Loading tokenizer-only LLM instance for the first time...")
            # メインの対話モデルと同じトークナイザを使用する
            model_config = config.MODELS_GGUF["gemma_3n"] # メインモデルのトークナイザを流用
            project_root = Path(config.MODEL_BASE_PATH)
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

    @property
    def tokenizer(self) -> LlamaCpp:
        """トークナイザのプロパティ。初回アクセス時にロードされる。"""
        if self._tokenizer_llm is None:
            self._load_tokenizer()
        return self._tokenizer_llm

    def count_tokens_for_messages(self, messages: List[BaseMessage]) -> int:
        """メッセージリストの合計トークン数を数える。"""
        if not messages:
            return 0

        if self.tokenizer:
            # ここでは単純に各メッセージのcontentのトークン数を合計する。
            # 実際のプロンプトテンプレートによる追加トークンは含まれないが、
            # 履歴の長さを管理する目的では十分な近似となる。
            return sum(
                self.tokenizer.get_num_tokens(msg.content)
                for msg in messages if isinstance(msg.content, str)
            )

        # フォールバックとして文字数ベースで概算
        logger.warning("Tokenizer LLM not available, falling back to character-based token estimation.")
        # 1トークンあたり平均4文字と仮定
        return sum(len(msg.content) for msg in messages if isinstance(msg.content, str)) // 4

    # 埋め込みモデルを取得するための専用メソッド 
    def get_embedding_model(self) -> Embeddings:
        """埋め込みモデルを取得またはロードする。"""
        if self._embedding_llm is None:
            with self._lock:
                # ダブルチェックロッキング
                if self._embedding_llm is None:
                    key = "embedding_model"
                    model_config = config.MODELS_GGUF[key]
                    self._embedding_config = model_config
                    
                    project_root = Path(__file__).parent.parent
                    model_path = project_root / model_config["path"]
                    llama_cpp_dir = project_root / "llama.cpp"
                    server_executable = self._find_server_executable(llama_cpp_dir)

                    if not model_path.exists():
                        raise FileNotFoundError(f"Embedding model file not found: {model_path.resolve()}")
                    if not server_executable:
                        raise FileNotFoundError("llama.cpp server executable not found.")

                    log_dir = project_root / "logs"
                    log_dir.mkdir(exist_ok=True)
                    stderr_log_path = log_dir / f"llama_server_{key}_{int(time.time())}.log"

                    port = model_config["port"]
                    command = build_server_command(
                        server_executable,
                        model_path,
                        port=port,
                        n_ctx=model_config.get("n_ctx", 4096),
                        n_gpu_layers=model_config.get("n_gpu_layers", -1),
                        extra_args=["--embedding"],
                    )

                    self._embedding_process = launch_server(command, stderr_log_path=stderr_log_path, logger=logger)

                    try:
                        self._perform_health_check(port, key, stderr_log_path)
                    except Exception:
                        logger.error("Health check failed while starting embedding server '%s'. Cleaning up process.", key, exc_info=True)
                        if self._embedding_process:
                            terminate_process(self._embedding_process, logger=logger)
                            self._embedding_process = None
                            self._embedding_llm = None
                            self._embedding_config = None
                        raise

                    logger.info(f"Server for '{key}' started successfully with PID: {self._embedding_process.pid}")

                    base_url = f"http://localhost:{port}/v1"
                    self._embedding_llm = OpenAIEmbeddings(
                        model=key,
                        base_url=base_url,
                        api_key="dummy-key",
                    )
                    logger.info("Embedding model client initialized and cached.")
        return self._embedding_llm

    def unload_embedding_model_if_loaded(self):
        """
        このメソッドは埋め込みモデルを永続化する方針に変更されたため、何もしません。
        互換性のために残されています。
        """
        pass

    def get_current_model_config_for_diagnostics(self) -> Dict:
        """
        診断用に、現在ロードされているメインのChatLLMモデルの設定を返す。
        """
        if self._current_model_key and self._current_model_config:
            # インスタンス変数に保持した設定から診断情報を返す
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

    def cleanup(self):
        """アプリケーション終了時にモデルをアンロードする。"""
        logger.info("Cleaning up LLMManager...")
        # 対話用モデルと埋め込みモデルの両方をアンロード
        self._unload_model()
        self._unload_embedding_model()
        
        # トークナイザインスタンスも解放
        if self._tokenizer_llm:
            logger.info("Unloading tokenizer-only LLM instance.")
            del self._tokenizer_llm
            self._tokenizer_llm = None
        
        gc.collect()