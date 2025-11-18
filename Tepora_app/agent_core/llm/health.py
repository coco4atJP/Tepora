from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from .. import config

__all__ = ["perform_health_check"]


def perform_health_check(port: int, key: str, *, process_ref, stderr_log_path: Path | None, logger: logging.Logger) -> None:
    health_check_url = f"http://localhost:{port}/health"
    timeout_config_key = "embedding_health_check_timeout" if "embedding" in key else "health_check_timeout"
    max_retries = config.LLAMA_CPP_CONFIG.get(timeout_config_key, 20)
    retry_interval = config.LLAMA_CPP_CONFIG.get("health_check_interval", 1.0)

    logger.info("Performing health check for '%s' on %s", key, health_check_url)

    for attempt in range(max_retries):
        process = process_ref()
        if stderr_log_path and process and process.poll() is not None:
            raise RuntimeError(
                f"Server process for '{key}' terminated unexpectedly. Review server log for details: {stderr_log_path}"
            )
        try:
            response = requests.get(health_check_url, timeout=0.5)
            if response.status_code == 200 and response.json().get("status") == "ok":
                logger.info("Server for '%s' is healthy (attempt %s/%s)", key, attempt + 1, max_retries)
                return
            if response.status_code != 503:
                logger.warning("Unexpected health check response %s for '%s'", response.status_code, key)
        except requests.exceptions.RequestException:
            pass
        time.sleep(retry_interval)

    raise TimeoutError(
        f"Server for '{key}' did not become healthy within {max_retries * retry_interval} seconds."
        + (f" Review server log for details: {stderr_log_path}" if stderr_log_path else "")
    )
