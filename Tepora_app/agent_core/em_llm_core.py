"""
Backward compatibility layer for legacy em_llm_core.py imports.

This module re-exports all components from the new modular em_llm package.
The monolithic em_llm_core.py has been refactored into:
- em_llm/types.py - Data classes (EpisodicEvent, EMConfig)
- em_llm/segmenter.py - Event segmentation
- em_llm/boundary.py - Boundary refinement
- em_llm/retrieval.py - Two-stage retrieval
- em_llm/integrator.py - Main integration class

All existing imports from agent_core.em_llm_core should continue to work unchanged.
"""

from __future__ import annotations

# Re-export everything from the new em_llm package
from .em_llm import (
    EMBoundaryRefiner,
    EMConfig,
    EMEventSegmenter,
    EMLLMIntegrator,
    EMTwoStageRetrieval,
    EpisodicEvent,
)

__all__ = [
    "EpisodicEvent",
    "EMConfig",
    "EMEventSegmenter",
    "EMBoundaryRefiner",
    "EMTwoStageRetrieval",
    "EMLLMIntegrator",
]
