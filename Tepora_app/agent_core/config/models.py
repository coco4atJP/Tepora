from __future__ import annotations

from typing import Final

__all__ = ["CHAT_MODELS", "EMBEDDING_MODEL_KEY", "EMBEDDING_MODEL", "MODELS_GGUF"]

CHAT_MODELS: Final = {
    "gemma_3n": {
        "port": 8000,
        "path": "gemma-3n-E4B-it-IQ4_XS.gguf",
        "n_ctx": 16384,
        "n_gpu_layers": -1,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 60,
        "max_tokens": 4096,
        "logprobs": True,
    },
    "jan_nano": {
        "port": 8001,
        "path": "jan-nano-128k-iQ4_XS.gguf",
        "n_ctx": 64000,
        "n_gpu_layers": -1,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": 4096,
        "logprobs": True,
    },
}

EMBEDDING_MODEL_KEY: Final = "embedding_model"

EMBEDDING_MODEL: Final = {
    "key": EMBEDDING_MODEL_KEY,
    "port": 8003,
    "path": "embeddinggemma-300M-Q8_0",
    "n_ctx": 32768,
    "n_gpu_layers": -1,
}

MODELS_GGUF: Final = {
    **CHAT_MODELS,
    EMBEDDING_MODEL_KEY: {
        key: value
        for key, value in EMBEDDING_MODEL.items()
        if key != "key"
    },
}
