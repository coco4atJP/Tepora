from __future__ import annotations

import logging
import platform
import re
import sys
from pathlib import Path
from typing import Iterable

import torch

__all__ = ["find_server_executable"]


def _build_preference_list() -> Iterable[str]:
    """Return ordered substrings describing preferred binaries for the host."""
    arch = platform.machine().lower()
    sys_platform = sys.platform

    if sys_platform == "win32":
        if torch.cuda.is_available():
            return ("cuda", "vulkan", "sycl", "opencl", "cpu", "win")
        return ("vulkan", "sycl", "opencl", "cpu", "win")

    if sys_platform == "darwin":
        if "arm" in arch:
            return ("arm64", "macos")
        return ("x64", "macos")

    if sys_platform == "linux":
        if torch.cuda.is_available():
            return ("cuda", "vulkan", "x64", "linux")
        return ("vulkan", "x64", "linux")

    return ()


def _score_candidate(path: Path, preferences: Iterable[str]) -> tuple[int, int, float]:
    path_str = str(path.resolve()).lower()

    version = 0
    match = re.search(r"b(\d+)", path_str)
    if match:
        version = int(match.group(1))

    env_score = 0
    for idx, pref in enumerate(preferences):
        if pref in path_str:
            env_score = len(tuple(preferences)) - idx
            break

    mtime = path.stat().st_mtime
    return version, env_score, mtime


def find_server_executable(llama_cpp_dir: Path, logger: logging.Logger | None = None) -> Path | None:
    """Locate the most suitable llama.cpp server executable for the current host."""
    server_exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"

    search_patterns = (f"**/{server_exe_name}", f"*/{server_exe_name}")
    found_files: list[Path] = []
    for pattern in search_patterns:
        found_files.extend(llama_cpp_dir.glob(pattern))

    if not found_files:
        return None

    preferences = tuple(_build_preference_list())
    latest_file = max(found_files, key=lambda p: _score_candidate(p, preferences))
    version, _, _ = _score_candidate(latest_file, preferences)

    if logger:
        if version > 0:
            logger.info(
                "Found server executable (latest version b%s): %s",
                version,
                latest_file.resolve(),
            )
        else:
            logger.info(
                "Found server executable (latest by mtime): %s",
                latest_file.resolve(),
            )
    return latest_file
