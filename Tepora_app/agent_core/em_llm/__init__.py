"""
EM-LLM (Episodic Memory for Large Language Models) package.

This package implements the EM-LLM system from the paper:
"Human-inspired Episodic Memory for Infinite Context LLMs" (ICLR 2025)

Main components:
- Types: Data classes (EpisodicEvent, EMConfig)
- Segmenter: Event segmentation based on surprise/semantic change
- Boundary: Boundary refinement using graph-theoretic metrics
- Retrieval: Two-stage retrieval (similarity + contiguity)
- Integrator: Main integration point with existing system

The legacy monolithic em_llm_core.py is replaced by this modular structure.
"""

from __future__ import annotations

from .boundary import EMBoundaryRefiner
from .integrator import EMLLMIntegrator
from .retrieval import EMTwoStageRetrieval
from .segmenter import EMEventSegmenter
from .types import EMConfig, EpisodicEvent

__all__ = [
    "EMConfig",
    "EpisodicEvent",
    "EMEventSegmenter",
    "EMBoundaryRefiner",
    "EMTwoStageRetrieval",
    "EMLLMIntegrator",
]
