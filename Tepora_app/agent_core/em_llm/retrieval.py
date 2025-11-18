"""
EM-LLM two-stage retrieval system.

This module implements the two-stage retrieval system from the paper:
1. Similarity buffer (Ks): Retrieve similar events based on query embedding
2. Contiguity buffer (Kc): Retrieve temporally adjacent events
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from .types import EMConfig, EpisodicEvent

if TYPE_CHECKING:
    from ..memory.memory_system import MemorySystem

logger = logging.getLogger(__name__)


class EMTwoStageRetrieval:
    """
    Two-stage retrieval system (similarity buffer + contiguity buffer).
    
    This class implements the retrieval algorithm from the EM-LLM paper,
    combining semantic similarity with temporal contiguity.
    """
    
    def __init__(self, config: EMConfig, memory_system: MemorySystem):
        """
        Initialize the two-stage retrieval system.
        
        Args:
            config: EM-LLM configuration parameters
            memory_system: ChromaDB-backed memory system
        """
        self.config = config
        self.memory_system = memory_system
        logger.info("EMTwoStageRetrieval initialized with ChromaDB backend.")
    
    def add_events(self, events: List[EpisodicEvent]):
        """
        Add new events to memory.
        
        Args:
            events: List of episodic events to store
        """
        if not events:
            return
        
        for event in events:
            # Convert to ChromaDB storage format
            doc_id = f"em_event_{event.start_position}_{event.end_position}"
            summary = " ".join(event.tokens)
            
            # Use mean of representative embeddings as event embedding
            if event.representative_embeddings is not None and event.representative_embeddings.shape[0] > 0:
                embedding = np.mean(event.representative_embeddings, axis=0).tolist()
                
                # Add surprise as metadata
                avg_surprise = 0.0
                if event.surprise_scores:
                    avg_surprise = float(np.mean(event.surprise_scores))
                
                metadata = {
                    "start_position": event.start_position,
                    "end_position": event.end_position,
                    "created_ts": time.time(),
                    "avg_surprise": avg_surprise,
                    "token_count": len(event.tokens),
                }
                self.memory_system.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[summary],
                    metadatas=[metadata]
                )
        logger.info(f"Added {len(events)} EM-LLM events to ChromaDB. Total events: {self.memory_system.count()}")
    
    def retrieve_relevant_events(
        self,
        query_embedding: np.ndarray,
        k: Optional[int] = None
    ) -> List[EpisodicEvent]:
        """
        Execute two-stage retrieval against ChromaDB.
        
        Args:
            query_embedding: Query embedding vector
            k: Optional number of events to retrieve (defaults to config)
            
        Returns:
            List of retrieved episodic events
        """
        if self.memory_system.count() == 0:
            return []
        
        total_k = k or self.config.total_retrieved_events
        ks = int(total_k * self.config.similarity_buffer_ratio)  # Similarity buffer size
        kc = total_k - ks  # Contiguity buffer size
        
        # Stage 1: Similarity-based retrieval
        similarity_events = self._similarity_based_retrieval(query_embedding, ks)
        
        # Stage 2: Temporal contiguity buffer
        contiguity_events = self._contiguity_based_retrieval(similarity_events, kc)
        
        # Combine and deduplicate
        all_retrieved = similarity_events + contiguity_events
        unique_events = self._deduplicate_events(all_retrieved)
        
        logger.debug(
            f"Retrieved {len(unique_events)} unique events "
            f"(similarity: {len(similarity_events)}, contiguity: {len(contiguity_events)})"
        )
        
        # Return events sorted by time (oldest first)
        sorted_events = sorted(unique_events, key=lambda e: e.start_position)
        return sorted_events[:total_k]
    
    def _results_to_events(self, results: List[Dict]) -> List[EpisodicEvent]:
        """
        Convert ChromaDB search results to EpisodicEvent objects.
        
        Args:
            results: List of result dictionaries from ChromaDB
            
        Returns:
            List of EpisodicEvent objects
        """
        events = []
        for result in results:
            try:
                # Get position from metadata
                metadata = result.get('metadata', {}) or {}
                start_pos = metadata.get('start_position')
                end_pos = metadata.get('end_position')
                
                # Fallback: parse from ID if not in metadata
                if start_pos is None or end_pos is None:
                    parts = result['id'].split('_')
                    start_pos = int(parts[2])
                    end_pos = int(parts[3])
                
                summary = result.get('summary') or ""
                tokens = summary.split()
                
                surprise_scores: List[float] = []
                avg_surprise = metadata.get('avg_surprise')
                if avg_surprise is not None:
                    try:
                        avg_surprise_value = float(avg_surprise)
                        token_count_meta = metadata.get('token_count')
                        if isinstance(token_count_meta, (int, float)) and token_count_meta > 0:
                            approx_token_count = int(token_count_meta)
                        else:
                            approx_token_count = len(tokens)
                        if approx_token_count <= 0:
                            approx_token_count = 1
                        surprise_scores = [avg_surprise_value] * approx_token_count
                    except (TypeError, ValueError):
                        logger.debug("Skipping avg_surprise metadata due to conversion error: %s", avg_surprise)
                
                event = EpisodicEvent(
                    tokens=tokens,
                    start_position=start_pos,
                    end_position=end_pos,
                    surprise_scores=surprise_scores,
                    summary=summary
                )
                events.append(event)
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Could not parse event position from id: {result['id']}")
                continue
        return events
    
    def _deduplicate_events(self, events: List[EpisodicEvent]) -> List[EpisodicEvent]:
        """
        Remove duplicate events from list.
        
        Args:
            events: List potentially containing duplicates
            
        Returns:
            List with duplicates removed
        """
        seen_positions = set()
        unique_events = []
        for event in events:
            position_key = (event.start_position, event.end_position)
            if position_key not in seen_positions:
                unique_events.append(event)
                seen_positions.add(position_key)
        return unique_events
    
    def _similarity_based_retrieval(self, query_embedding: np.ndarray, ks: int) -> List[EpisodicEvent]:
        """
        Execute similarity search against ChromaDB.
        
        Args:
            query_embedding: Query embedding vector
            ks: Number of similar events to retrieve
            
        Returns:
            List of similar events
        """
        if ks <= 0 or self.memory_system.count() == 0:
            return []
        
        # Use memory system's retrieve method with temporality boost
        results = self.memory_system.retrieve(
            query="",  # Don't use query string
            k=ks,
            temporality_boost=self.config.recency_weight,
            query_embedding_override=query_embedding.tolist()  # Pass embedding directly
        )
        return self._results_to_events(results)
    
    def _contiguity_based_retrieval(
        self,
        similarity_events: List[EpisodicEvent],
        kc: int
    ) -> List[EpisodicEvent]:
        """
        Retrieve temporally adjacent events using ChromaDB metadata search.
        
        Uses $or operator to efficiently combine multiple adjacency searches.
        
        Args:
            similarity_events: Events from similarity search
            kc: Number of contiguous events to retrieve
            
        Returns:
            List of contiguous events
        """
        if kc <= 0 or not similarity_events:
            return []
        
        # 1. Build filter list combining multiple search conditions
        or_filters = []
        for event in similarity_events:
            # Previous event: end_position matches current start_position
            or_filters.append({"end_position": event.start_position})
            # Next event: start_position matches current end_position
            or_filters.append({"start_position": event.end_position})
        
        if not or_filters:
            return []
        
        # 2. Query using $or for batch retrieval
        combined_filter = {"$or": or_filters}
        results = self.memory_system.collection.get(where=combined_filter, include=["metadatas", "documents"])
        
        if not results or not results['ids']:
            return []
        
        # 3. Extract results that don't overlap with similarity search
        similarity_event_ids = {f"em_event_{e.start_position}_{e.end_position}" for e in similarity_events}
        
        contiguity_results = []
        for i in range(len(results['ids'])):
            if results['ids'][i] not in similarity_event_ids:
                contiguity_results.append({
                    "id": results['ids'][i],
                    "summary": results['documents'][i],
                    "metadata": results['metadatas'][i]
                })
        
        # 4. Convert to EpisodicEvent and deduplicate
        contiguity_events = self._results_to_events(contiguity_results)
        unique_contiguity_events = self._deduplicate_events(contiguity_events)
        logger.debug(f"Found {len(unique_contiguity_events)} contiguous events via batch query.")
        
        return unique_contiguity_events[:kc]
