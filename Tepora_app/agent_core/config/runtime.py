from __future__ import annotations

from typing import Final

__all__ = ["LLAMA_CPP_CONFIG"]

LLAMA_CPP_CONFIG: Final = {
    "health_check_timeout": 30,
    "health_check_interval": 1.0,
    "process_terminate_timeout": 10,
    "embedding_health_check_timeout": 20,
}
