"""
Application utility functions.

This module provides helper functions for the main application:
- Input sanitization
- Async input handling
- Statistics display
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from typing import Dict

from .. import config

logger = logging.getLogger(__name__)


async def ainput(prompt: str = "") -> str:
    """
    Async input function for interactive console.
    
    Args:
        prompt: Prompt string to display
        
    Returns:
        User input string
    """
    print(prompt, end="", flush=True)
    return await asyncio.to_thread(sys.stdin.readline)


from .. import config


def sanitize_user_input(user_input: str, max_length: int = None) -> str:
    """
    Sanitize user input to mitigate potential prompt injection attacks.
    
    Args:
        user_input: Raw user input
        max_length: Maximum allowed input length (defaults to config.MAX_INPUT_LENGTH)
        
    Returns:
        Sanitized input string
        
    Raises:
        ValueError: If input exceeds max_length
    """
    if max_length is None:
        max_length = config.MAX_INPUT_LENGTH
    
    if len(user_input) > max_length:
        raise ValueError(f"Input too long: {len(user_input)} > {max_length}")
    
    # Detect dangerous patterns that may attempt system prompt injection
    sanitized_input = user_input
    for pattern in config.DANGEROUS_PATTERNS:
        if re.search(pattern, sanitized_input, re.IGNORECASE):
            logger.warning(
                "Potential prompt injection attempt detected; input will be sanitized. "
                "pattern=%s snippet='%s...'",
                pattern,
                sanitized_input[:100]
            )
            sanitized_input = re.sub(pattern, "[filtered]", sanitized_input, flags=re.IGNORECASE)
    
    if sanitized_input != user_input:
        sanitized_input += (
            "\n\n[Notice: parts of your message were filtered due to unsafe instructions. "
            "Please rephrase if needed.]"
        )
    
    return sanitized_input


def display_em_stats(stats: Dict, title: str = "EM-LLM Memory System Statistics"):
    """
    Display EM-LLM statistics in formatted output.
    
    Args:
        stats: Statistics dictionary from EM-LLM integrator
        title: Title for the statistics display
    """
    print(f"\n📊 {title}:")
    print(f"   Total Events: {stats.get('total_events', 0)}")
    print(f"   Total Tokens: {stats.get('total_tokens_in_memory', 0)}")
    print(f"   Mean Event Size: {stats.get('mean_event_size', 0):.1f} tokens")
    print()
    
    surprise_stats = stats.get('surprise_statistics', {})
    if surprise_stats and surprise_stats.get('mean', 0) > 0:
        print(
            f"   Surprise - Mean: {surprise_stats.get('mean', 0):.3f}, "
            f"Std: {surprise_stats.get('std', 0):.3f}, Max: {surprise_stats.get('max', 0):.3f}"
        )
    
    config_info = stats.get('configuration', {})
    print(
        f"   Config - γ: {config_info.get('surprise_gamma', 0)}, "
        f"Event Size: {config_info.get('min_event_size', 0)}-{config_info.get('max_event_size', 0)}"
    )
    print()
