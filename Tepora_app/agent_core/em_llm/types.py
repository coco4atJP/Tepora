"""
EM-LLM data types and configuration.

This module defines the core data structures for the EM-LLM system:
- EpisodicEvent: Represents a single episodic event in memory
- EMConfig: Configuration parameters for EM-LLM components
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class EpisodicEvent:
    """
    Represents a single episodic event in the EM-LLM system.
    
    An episodic event is a segment of text that has been identified as
    semantically coherent based on surprise scores or semantic change.
    
    Attributes:
        tokens: List of tokens comprising this event
        start_position: Starting position in the overall token sequence
        end_position: Ending position in the overall token sequence
        surprise_scores: Surprise score for each token (from -log P(x|...))
        attention_keys: Optional attention key vectors (seq_len, hidden_dim)
        representative_tokens: Indices of tokens that best represent this event
        summary: Optional text summary of this event
        representative_embeddings: Embeddings of representative tokens (num_repr, hidden_dim)
    """
    tokens: List[str]
    start_position: int
    end_position: int
    surprise_scores: List[float]
    attention_keys: Optional[np.ndarray] = None
    representative_tokens: Optional[List[int]] = None
    summary: Optional[str] = None
    representative_embeddings: Optional[np.ndarray] = None


@dataclass
class EMConfig:
    """
    Configuration parameters for the EM-LLM system.
    
    Based on the paper "Human-inspired Episodic Memory for Infinite Context LLMs" (ICLR 2025).
    
    Surprise-related parameters:
        surprise_window: Window size for surprise calculation
        surprise_gamma: Threshold adjustment parameter (γ in paper)
        min_event_size: Minimum tokens per event
        max_event_size: Maximum tokens per event
    
    Retrieval-related parameters:
        similarity_buffer_ratio: Ratio of similarity buffer (Ks/K in paper)
        contiguity_buffer_ratio: Ratio of contiguity buffer (Kc/K in paper)
        total_retrieved_events: Total number of events to retrieve (K in paper)
        repr_topk: Number of representative tokens per event
        recency_weight: Temporal recency weight (0.0 - 1.0)
    
    Boundary refinement parameters:
        use_boundary_refinement: Whether to apply boundary refinement
        refinement_metric: Metric to use ("modularity" or "conductance")
        refinement_search_range: Maximum search range for refinement
    """
    # Surprise-related
    surprise_window: int = 128
    surprise_gamma: float = 1.0
    min_event_size: int = 8
    max_event_size: int = 128
    
    # Retrieval-related
    similarity_buffer_ratio: float = 0.7
    contiguity_buffer_ratio: float = 0.3
    total_retrieved_events: int = 4
    repr_topk: int = 4
    recency_weight: float = 0.1
    
    # Boundary refinement
    use_boundary_refinement: bool = True
    refinement_metric: str = "modularity"
    refinement_search_range: int = 16
