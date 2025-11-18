from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Sequence

from .. import config

__all__ = [
    "build_server_command",
    "launch_server",
    "terminate_process",
]


def build_server_command(
    server_executable: Path,
    model_path: Path,
    *,
    port: int,
    n_ctx: int,
    n_gpu_layers: int,
    extra_args: Sequence[str] | None = None,
) -> list[str]:
    command = [
        str(server_executable),
        "-m",
        str(model_path),
        "--port",
        str(port),
        "-c",
        str(n_ctx),
        "--n-gpu-layers",
        str(n_gpu_layers),
    ]
    if extra_args:
        command.extend(extra_args)
    return command


def launch_server(command: Sequence[str], *, stderr_log_path: Path, logger: logging.Logger) -> subprocess.Popen:
    logger.info("Starting llama.cpp server: %s", " ".join(command))
    logger.info("Server stderr will be logged to: %s", stderr_log_path)
    with open(stderr_log_path, "w", encoding="utf-8") as log_file:
        return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log_file)


def terminate_process(process: subprocess.Popen, *, logger: logging.Logger) -> None:
    timeout = config.LLAMA_CPP_CONFIG.get("process_terminate_timeout", 10)
    logger.info("Terminating process PID=%s", process.pid)
    try:
        process.terminate()
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Process PID=%s did not terminate gracefully. Forcing kill...", process.pid)
        process.kill()
        process.wait()
