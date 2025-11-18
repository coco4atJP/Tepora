"""
EM-LLM boundary refinement.

This module provides boundary refinement functionality using
graph-theoretic metrics (modularity and conductance).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .types import EMConfig, EpisodicEvent

logger = logging.getLogger(__name__)


class EMBoundaryRefiner:
    """
    Refines event boundaries using graph-theoretic metrics.
    
    This class optimizes segmentation quality by:
    1. Evaluating boundary positions with modularity or conductance
    2. Searching for optimal boundary positions in local neighborhoods
    3. Rebuilding events with refined boundaries
    """
    
    def __init__(self, config: EMConfig):
        """
        Initialize the boundary refiner.
        
        Args:
            config: EM-LLM configuration parameters
        """
        self.config = config
    
    def _calculate_similarity_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """
        Calculate similarity matrix from attention keys or context vectors.
        
        Args:
            vectors: Vector matrix (seq_len, hidden_dim)
            
        Returns:
            Similarity matrix (seq_len, seq_len)
        """
        # Use cosine similarity (paper uses dot product, but normalization is more stable)
        return cosine_similarity(vectors)
    
    def calculate_modularity(self, similarity_matrix: np.ndarray, boundaries: List[int]) -> float:
        """
        Calculate modularity (Equation 3 from paper).
        
        Args:
            similarity_matrix: Similarity matrix between items
            boundaries: Boundary indices defining communities
            
        Returns:
            Modularity score
        """
        try:
            G = nx.from_numpy_array(similarity_matrix)
            
            # Create communities based on boundaries
            communities = []
            for i in range(len(boundaries) - 1):
                community = list(range(boundaries[i], boundaries[i + 1]))
                if community:  # Only add non-empty communities
                    communities.append(community)
            
            if len(communities) <= 1:
                return 0.0
            
            # Calculate modularity (may raise exceptions if invalid partition)
            return nx.algorithms.community.modularity(G, communities, weight='weight')
        except (nx.NetworkXError, ValueError) as e:
            logger.warning(f"Modularity calculation failed: {e}", exc_info=True)
            return 0.0
    
    def calculate_conductance(self, similarity_matrix: np.ndarray, boundaries: List[int]) -> float:
        """
        Calculate conductance (Equation 4 from paper).
        
        Args:
            similarity_matrix: Similarity matrix between items
            boundaries: Boundary indices defining communities
            
        Returns:
            Average conductance across all communities
        """
        try:
            total_conductance = 0.0
            num_communities = len(boundaries) - 1
            
            for i in range(num_communities):
                start, end = boundaries[i], boundaries[i + 1]
                
                # Internal community weight
                internal_weight = np.sum(similarity_matrix[start:end, start:end])
                
                # External weight to other communities
                external_weight = (
                    np.sum(similarity_matrix[start:end, :start]) +
                    np.sum(similarity_matrix[start:end, end:])
                )
                
                # Calculate conductance
                total_weight = internal_weight + external_weight
                if total_weight > 0:
                    conductance = external_weight / total_weight
                    total_conductance += conductance
            
            return total_conductance / max(1, num_communities)
        except (IndexError, ValueError) as e:
            logger.warning(f"Conductance calculation failed: {e}", exc_info=True)
            return 1.0  # Bad score
    
    def refine_boundaries(
        self,
        events: List[EpisodicEvent],
        context_vectors: Optional[np.ndarray] = None,
        attention_keys: Optional[np.ndarray] = None
    ) -> List[EpisodicEvent]:
        """
        Refine boundaries using graph-theoretic metrics.
        
        Based on paper concept: prioritize attention keys if available,
        fallback to sentence embeddings (context_vectors).
        
        Args:
            events: List of initial events
            context_vectors: Optional sentence embedding vectors
            attention_keys: Optional attention key vectors
            
        Returns:
            List of events with refined boundaries
        """
        if not self.config.use_boundary_refinement or len(events) <= 1:
            return events
        
        # Paper concept: refinement based on attention key similarity
        if attention_keys is not None and attention_keys.shape[0] > 1:
            logger.info("Refining boundaries using attention key similarity (as per paper).")
            similarity_matrix = self._calculate_similarity_matrix(attention_keys)
        # Fallback: refinement based on sentence embedding similarity
        elif context_vectors is not None and context_vectors.shape[0] > 1:
            logger.info("Refining boundaries using sentence embedding similarity (fallback).")
            similarity_matrix = self._calculate_similarity_matrix(context_vectors)
        else:
            logger.warning("Neither attention keys nor context vectors available. Skipping boundary refinement.")
            return events
        
        logger.info("Refining event boundaries using graph-theoretic metrics")
        
        # Extract current boundaries
        current_boundaries = [event.start_position for event in events] + [events[-1].end_position]
        
        # Search for optimal position for each boundary pair
        refined_boundaries = [current_boundaries[0]]  # First boundary is fixed
        
        for i in range(len(current_boundaries) - 2):
            start_boundary = refined_boundaries[-1]
            end_boundary = current_boundaries[i + 2]
            current_pos = current_boundaries[i + 1]
            
            best_pos = current_pos
            best_score = self._evaluate_boundary_position(
                similarity_matrix, refined_boundaries + [current_pos, end_boundary]
            )
            
            # Search neighboring positions
            # Dynamic search range based on event pair length, capped by config
            event_pair_length = end_boundary - start_boundary
            dynamic_range = event_pair_length // 4
            search_range = min(self.config.refinement_search_range, dynamic_range)
            
            for offset in range(-search_range, search_range + 1):  # Fine-grained search (step=1)
                test_pos = current_pos + offset
                if start_boundary < test_pos < end_boundary:
                    test_boundaries = refined_boundaries + [test_pos, end_boundary]
                    score = self._evaluate_boundary_position(similarity_matrix, test_boundaries)
                    
                    if self._is_better_score(score, best_score):
                        best_score = score
                        best_pos = test_pos
            
            refined_boundaries.append(best_pos)
        
        refined_boundaries.append(current_boundaries[-1])  # Last boundary is also fixed
        
        # Rebuild events with refined boundaries
        return self._rebuild_events_from_boundaries(events, refined_boundaries)
    
    def _evaluate_boundary_position(self, similarity_matrix: np.ndarray, boundaries: List[int]) -> float:
        """
        Evaluate a boundary position using configured metric.
        
        Args:
            similarity_matrix: Similarity matrix
            boundaries: Boundary indices to evaluate
            
        Returns:
            Score (higher is better)
        """
        if self.config.refinement_metric == "modularity":
            return self.calculate_modularity(similarity_matrix, boundaries)
        else:
            return -self.calculate_conductance(similarity_matrix, boundaries)  # Negative (lower is better)
    
    def _is_better_score(self, new_score: float, current_best: float) -> bool:
        """
        Determine if new score is better than current best.
        
        Args:
            new_score: New score to evaluate
            current_best: Current best score
            
        Returns:
            True if new score is better
        """
        return new_score > current_best
    
    def _rebuild_events_from_boundaries(
        self,
        original_events: List[EpisodicEvent],
        boundaries: List[int]
    ) -> List[EpisodicEvent]:
        """
        Rebuild events from refined boundaries.
        
        Args:
            original_events: Original event list
            boundaries: Refined boundary indices
            
        Returns:
            List of events with refined boundaries
        """
        refined_events = []
        all_tokens = []
        all_surprises = []
        
        # Combine all tokens and surprises
        for event in original_events:
            all_tokens.extend(event.tokens)
            all_surprises.extend(event.surprise_scores)
        
        for i in range(len(boundaries) - 1):
            start_pos = boundaries[i]
            end_pos = boundaries[i + 1]
            
            refined_event = EpisodicEvent(
                tokens=all_tokens[start_pos:end_pos],
                start_position=start_pos,
                end_position=end_pos,
                surprise_scores=all_surprises[start_pos:end_pos]
            )
            refined_events.append(refined_event)
        
        logger.info(f"Boundary refinement completed: {len(original_events)} -> {len(refined_events)} events")
        return refined_events
