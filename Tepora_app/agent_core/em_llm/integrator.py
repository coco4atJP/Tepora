"""
EM-LLM integrator - Main integration point.

This module provides the EMLLMIntegrator class that orchestrates
all EM-LLM components for memory formation and retrieval.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from .boundary import EMBoundaryRefiner
from .retrieval import EMTwoStageRetrieval
from .segmenter import EMEventSegmenter
from .types import EMConfig, EpisodicEvent

if TYPE_CHECKING:
    from ..embedding_provider import EmbeddingProvider
    from ..llm_manager import LLMManager
    from ..memory.memory_system import MemorySystem

logger = logging.getLogger(__name__)


class EMLLMIntegrator:
    """
    Integration class for EM-LLM with existing system.
    
    This class orchestrates:
    1. Event segmentation (surprise-based or semantic)
    2. Boundary refinement
    3. Representative token selection
    4. Memory storage and retrieval
    """
    
    def __init__(
        self,
        llm_manager: LLMManager,
        embedding_provider: EmbeddingProvider,
        config: EMConfig,
        memory_system: MemorySystem
    ):
        """
        Initialize EM-LLM integrator.
        
        Args:
            llm_manager: LLM manager for model access
            embedding_provider: Provider for text embeddings
            config: EM-LLM configuration parameters
            memory_system: ChromaDB-backed memory system
        """
        self.llm_manager = llm_manager
        self.embedding_provider = embedding_provider
        self.config = config
        self.memory_system = memory_system
        
        # Initialize EM-LLM memory system
        self.segmenter = EMEventSegmenter(self.config)
        self.boundary_refiner = EMBoundaryRefiner(self.config)
        self.retrieval_system = EMTwoStageRetrieval(self.config, self.memory_system)
        
        logger.info("EM-LLM Integrator initialized")
    
    def get_current_llm_config_for_diagnostics(self) -> Dict:
        """
        Get current LLM configuration for diagnostics.
        
        Returns:
            Dictionary with LLM configuration information
        """
        if self.llm_manager and hasattr(self.llm_manager, 'get_current_model_config_for_diagnostics'):
            return self.llm_manager.get_current_model_config_for_diagnostics()
        logger.warning("LLMManager not available or does not have the required diagnostics method.")
        return {}
    
    async def _finalize_and_store_events(self, events: List[EpisodicEvent]) -> List[EpisodicEvent]:
        """
        Common post-processing: select representative tokens, compute embeddings, store.
        
        Args:
            events: List of segmented events
            
        Returns:
            List of finalized events
        """
        if not events:
            return []
        
        # Select representative tokens and compute embeddings for each event
        for event in events:
            event.representative_tokens = self._select_representative_tokens(event)
            if self.embedding_provider and event.representative_tokens:
                repr_texts = [event.tokens[i] for i in event.representative_tokens]
                if repr_texts:
                    # Embedding computation is not async, so use asyncio.to_thread
                    embeddings = await asyncio.to_thread(self.embedding_provider.encode, repr_texts)
                    event.representative_embeddings = np.array(embeddings)
        
        # Store in memory
        self.retrieval_system.add_events(events)
        
        return events
    
    async def process_logprobs_for_memory(
        self,
        logprobs_content: List[Dict],
        state: Optional[Dict] = None
    ) -> List[EpisodicEvent]:
        """
        Process LLM logprobs directly for surprise-based memory formation.
        
        This is the main memory formation path from the EM-LLM paper.
        
        Args:
            logprobs_content: List of logprob dictionaries from LLM
            state: Optional state dictionary (unused)
            
        Returns:
            List of created episodic events
        """
        logger.info("Processing logprobs for surprisal-based memory formation.")
        try:
            if not logprobs_content:
                logger.warning("logprobs_content is empty. Skipping memory formation.")
                return []
            
            # Normalize logprob entries
            normalized_entries = []
            for idx, entry in enumerate(logprobs_content):
                token = entry.get('token') or entry.get('token_str')
                logprob = entry.get('logprob')
                if token is None or logprob is None:
                    logger.debug(
                        "Skipping logprob entry %d due to missing token/logprob fields: %s",
                        idx,
                        entry,
                    )
                    continue
                normalized_entries.append({'token': token, 'logprob': logprob})
            
            if not normalized_entries:
                logger.warning("No valid logprob entries after normalization. Skipping memory formation.")
                return []
            
            # 1. Calculate surprise scores
            surprise_scores = self.segmenter.calculate_surprise_from_logprobs(normalized_entries)
            tokens = [item['token'] for item in normalized_entries]
            
            # 2. Identify boundaries based on surprise
            boundaries = self.segmenter._identify_event_boundaries(surprise_scores)
            
            # 3. Build events from boundaries
            events = []
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                event = EpisodicEvent(
                    tokens=tokens[start:end],
                    start_position=start,
                    end_position=end,
                    surprise_scores=surprise_scores[start:end]
                )
                events.append(event)
            
            # Boundary refinement (using attention keys if available)
            attention_keys = None  # Currently dummy
            if self.config.use_boundary_refinement and attention_keys is not None:
                logger.info("Applying boundary refinement using attention keys.")
                events = self.boundary_refiner.refine_boundaries(
                    events, attention_keys=attention_keys
                )
            
            return await self._finalize_and_store_events(events)
        except Exception as e:
            logger.error(f"EM-LLM logprobs processing failed: {e}", exc_info=True)
            return []
    
    async def process_conversation_turn_for_memory(
        self,
        user_input: str,
        ai_response: str
    ) -> List[EpisodicEvent]:
        """
        Process conversation turn through EM-LLM memory formation pipeline.
        
        Args:
            user_input: User's input text
            ai_response: AI's response text
            
        Returns:
            List of created episodic events
        """
        logger.info("Processing conversation turn for EM-LLM memory formation (semantic change based).")
        
        try:
            if not ai_response:
                logger.warning("AI response is empty. Aborting memory formation.")
                return []
            
            # Semantic change-based memory formation pipeline
            logger.info(f"Processing text of {len(ai_response)} chars for memory formation")
            
            # Step 1 & 2: Semantic change-based segmentation
            events, sentence_embeddings = self.segmenter.segment_text_into_events(
                ai_response, self.embedding_provider
            )
            if not events:
                return []
            
            # Step 3: Boundary refinement
            if self.config.use_boundary_refinement and sentence_embeddings is not None:
                events = self.boundary_refiner.refine_boundaries(
                    events,
                    context_vectors=sentence_embeddings,
                    attention_keys=None
                )
            
            # Step 4 & 5: Common post-processing and memory storage
            final_events = await self._finalize_and_store_events(events)
            
            logger.info(
                f"Created {len(final_events)} episodic events from conversation turn "
                f"via semantic segmentation."
            )
            return final_events
            
        except Exception as e:
            logger.error(f"EM-LLM memory formation failed: {e}", exc_info=True)
            return []
    
    def _select_representative_tokens(self, event: EpisodicEvent) -> List[int]:
        """
        Select representative tokens within event (highest surprise scores).
        
        Args:
            event: Episodic event
            
        Returns:
            List of token indices (sorted)
        """
        if not event.surprise_scores:
            return []
        
        indexed_scores = [(score, i) for i, score in enumerate(event.surprise_scores)]
        indexed_scores.sort(key=lambda x: x[0], reverse=True)
        representative_indices = [i for _, i in indexed_scores[:self.config.repr_topk]]
        return sorted(representative_indices)
    
    def retrieve_relevant_memories_for_query(self, query: str) -> List[Dict]:
        """
        Retrieve relevant memories using EM-LLM two-stage retrieval.
        
        Args:
            query: Query string
            
        Returns:
            List of memory entries (compatible with existing system)
        """
        logger.info("Retrieving memories using EM-LLM two-stage retrieval")
        
        try:
            if not self.embedding_provider:
                logger.warning("No embedding provider available for retrieval")
                return []
            
            # Execute EM-LLM two-stage retrieval
            query_embedding = np.array(self.embedding_provider.encode([query])[0])
            relevant_events = self.retrieval_system.retrieve_relevant_events(query_embedding)
            
            # Convert to dictionary format for compatibility with existing system
            memory_entries = []
            for i, event in enumerate(relevant_events):
                surprise_scores = event.surprise_scores or []
                surprise_mean = float(np.mean(surprise_scores)) if surprise_scores else 0.0
                surprise_max = float(np.max(surprise_scores)) if surprise_scores else 0.0
                event_size = len(event.tokens)
                
                memory_entry = {
                    'id': f"em_event_{event.start_position}_{event.end_position}",
                    'content': " ".join(event.tokens),
                    'summary': event.summary or f"Episodic event from position {event.start_position} to {event.end_position}",
                    'surprise_stats': {
                        'mean_surprise': surprise_mean,
                        'max_surprise': surprise_max,
                        'event_size': event_size
                    },
                    'representative_tokens': event.representative_tokens or [],
                    'retrieval_rank': i + 1
                }
                memory_entries.append(memory_entry)
            
            logger.info(f"Retrieved {len(memory_entries)} EM-LLM memories")
            return memory_entries
        
        except AttributeError as e:
            logger.error(
                f"EM-LLM memory retrieval failed due to a missing component "
                f"(e.g., embedding_provider): {e}",
                exc_info=True
            )
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during EM-LLM memory retrieval: {e}", exc_info=True)
            return []
    
    def get_memory_statistics(self) -> Dict:
        """
        Get current statistics of EM-LLM memory system.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {}
        try:
            stored_events = self.retrieval_system.memory_system.get_all()
            if not stored_events:
                return {"status": "No events in memory."}
            
            total_events = len(stored_events)
            total_tokens = 0
            surprise_values = []
            
            for event in stored_events:
                summary = event.get("summary", "") or ""
                metadata = event.get("metadata", {}) or {}
                
                token_count = metadata.get("token_count")
                if isinstance(token_count, (int, float)) and token_count >= 0:
                    total_tokens += int(token_count)
                else:
                    total_tokens += len(summary.split())
                
                avg_surprise = metadata.get("avg_surprise")
                if avg_surprise is not None:
                    try:
                        surprise_values.append(float(avg_surprise))
                    except (TypeError, ValueError):
                        logger.debug("Skipping non-numeric avg_surprise value: %s", avg_surprise)
            
            total_events_from_db = self.memory_system.count()
            
            surprise_stats = {}
            if surprise_values:
                surprise_stats = {
                    "mean": float(np.mean(surprise_values)),
                    "std": float(np.std(surprise_values)),
                    "max": float(np.max(surprise_values)),
                }
            
            mean_event_size = float(total_tokens) / total_events if total_events > 0 else 0.0
            
            stats = {
                "total_events": total_events_from_db,
                "total_tokens_in_memory": total_tokens,
                "mean_event_size": mean_event_size,
                "surprise_statistics": surprise_stats,
                "configuration": {
                    "surprise_gamma": self.config.surprise_gamma,
                    "min_event_size": self.config.min_event_size,
                    "max_event_size": self.config.max_event_size,
                    "total_retrieved_events": self.config.total_retrieved_events,
                },
                "llm_config": self.get_current_llm_config_for_diagnostics()
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get memory statistics: {e}", exc_info=True)
            return {"status": f"Error retrieving statistics: {e}"}
