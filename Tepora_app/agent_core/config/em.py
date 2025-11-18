from __future__ import annotations

from typing import Final

__all__ = ["EM_LLM_CONFIG", "EM_LLM_DEBUG"]

EM_LLM_CONFIG: Final = {
    "surprise_window": 64,
    "surprise_gamma": 1.0,
    "min_event_size": 8,
    "max_event_size": 64,
    "similarity_buffer_ratio": 0.7,
    "contiguity_buffer_ratio": 0.3,
    "total_retrieved_events": 4,
    "recency_weight": 0.1,
    "repr_topk": 4,
    "use_boundary_refinement": True,
    "refinement_metric": "modularity",
    "refinement_search_range": 16,
}

EM_LLM_DEBUG: Final = {
    "log_surprise_calculations": True,
    "log_boundary_detection": True,
    "log_memory_formation": True,
    "log_retrieval_details": True,
    "save_event_visualizations": False,
    "performance_monitoring": True,
}
