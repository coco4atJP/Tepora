from __future__ import annotations

import os
from pathlib import Path

__all__ = ["MODEL_BASE_PATH"]

# Default to the project root (Tepora_app/) when MODEL_BASE_PATH is not set.
_MODEL_BASE_DEFAULT = Path(__file__).resolve().parents[2]
MODEL_BASE_PATH = os.getenv("MODEL_BASE_PATH", str(_MODEL_BASE_DEFAULT))
