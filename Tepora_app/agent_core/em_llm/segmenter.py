"""
EM-LLM event segmentation.

This module provides semantic segmentation of text into episodic events
based on surprise scores and semantic change detection.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from .types import EMConfig, EpisodicEvent

if TYPE_CHECKING:
    from ..embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class EMEventSegmenter:
    """
    Segments text into episodic events based on semantic change.
    
    This class implements the segmentation algorithm from the EM-LLM paper,
    which identifies event boundaries based on:
    1. Surprise scores (-log P(x|...)) from LLM logprobs
    2. Semantic change detection using sentence embeddings
    """
    
    def __init__(self, config: EMConfig):
        """
        Initialize the event segmenter.
        
        Args:
            config: EM-LLM configuration parameters
        """
        self.config = config
        self.sent_tokenizer = self._get_sentence_tokenizer()
        logger.info("EM-LLM Semantic Event Segmenter initialized")
    
    def _get_sentence_tokenizer(self):
        """
        Safely load NLTK 'punkt' tokenizer.
        
        Returns fallback tokenizer if NLTK data is unavailable.
        """
        try:
            return nltk.data.load('tokenizers/punkt/english.pickle')
        except LookupError:
            logger.info("NLTK 'punkt' tokenizer data not found. Downloading...")
            try:
                nltk.download('punkt', quiet=True)
                logger.info("'punkt' data downloaded successfully.")
                return nltk.data.load('tokenizers/punkt/english.pickle')
            except Exception as download_error:
                logger.warning(
                    "Failed to download 'punkt' tokenizer data: %s",
                    download_error,
                    exc_info=True
                )
                return self._fallback_sentence_tokenizer()
        except Exception as load_error:
            logger.warning(
                "Unexpected error while loading NLTK 'punkt': %s",
                load_error,
                exc_info=True
            )
            return self._fallback_sentence_tokenizer()
    
    def _fallback_sentence_tokenizer(self):
        """Simple fallback tokenizer for offline environments."""
        
        class _SimpleSentenceTokenizer:
            def tokenize(self, text: str) -> List[str]:
                # Split on punctuation or newlines, excluding empty strings
                segments = re.split(r'(?:[\.!?]+\s+|\n+)', text)
                return [seg.strip() for seg in segments if seg.strip()]
        
        logger.warning(
            "Falling back to simple regex-based sentence tokenizer. "
            "EM segmentation quality may degrade."
        )
        return _SimpleSentenceTokenizer()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        # Replace newlines with spaces for better NLTK processing
        text = re.sub(r'\n+', ' ', text).strip()
        return self.sent_tokenizer.tokenize(text)
    
    def calculate_surprise_from_logprobs(self, logprobs: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate surprise scores from LLM logprobs.
        
        This implements the -log P(xt|...) formula from the paper.
        
        Args:
            logprobs: List of logprob dictionaries from LLM output
                     Each element should be {'token_str': str, 'logprob': float}
        
        Returns:
            List of surprise scores (one per token)
        """
        if not logprobs:
            return []
        # Per paper definition, negative log likelihood is surprise
        # logprobs are usually negative, so multiply by -1 to get positive values
        return [-item.get('logprob', 0.0) for item in logprobs]
    
    def segment_text_into_events(
        self,
        text: str,
        embedding_provider: EmbeddingProvider
    ) -> Tuple[List[EpisodicEvent], Optional[np.ndarray]]:
        """
        Segment text into episodic events based on semantic change.
        
        Args:
            text: Text to segment
            embedding_provider: Provider for text embeddings
            
        Returns:
            Tuple of (list of events, sentence embeddings matrix)
        """
        if not text or not embedding_provider:
            return [], None
        
        # 1. Split text into sentences
        sentences = self._split_into_sentences(text)
        if not sentences or len(sentences) < 2:
            logger.info("Text too short for semantic segmentation, treating as a single event.")
            tokens = text.split()
            event = EpisodicEvent(
                tokens=tokens if tokens else [],
                start_position=0,
                end_position=len(tokens),
                surprise_scores=[0.0] * len(tokens)  # No surprise
            )
            return [event], None
        
        # 2. Convert each sentence to embeddings
        sentence_embeddings = np.array(embedding_provider.encode(sentences))
        
        # 3. Calculate cosine distance between adjacent sentences
        distances = [
            cosine_distances(
                sentence_embeddings[i].reshape(1, -1),
                sentence_embeddings[i + 1].reshape(1, -1)
            )[0][0] for i in range(len(sentences) - 1)
        ]
        # First sentence has change score of 0
        semantic_change_scores = [0.0] + distances
        
        # 4. Identify boundaries based on semantic change scores
        boundary_indices = self._identify_event_boundaries(semantic_change_scores, sentences)
        
        # 5. Build events from boundaries
        events = []
        total_token_offset = 0
        for i in range(len(boundary_indices) - 1):
            start_sentence_idx = boundary_indices[i]
            end_sentence_idx = boundary_indices[i + 1]
            
            event_sentences = sentences[start_sentence_idx:end_sentence_idx]
            event_text = " ".join(event_sentences)
            event_tokens = event_text.split()  # Simple tokenizer
            
            # Representative "surprise" score for this event (semantic change at boundary)
            event_surprise_score = semantic_change_scores[start_sentence_idx]
            
            event = EpisodicEvent(
                tokens=event_tokens,
                start_position=total_token_offset,
                end_position=total_token_offset + len(event_tokens),
                # Assign same surprise score to all tokens in event
                surprise_scores=[event_surprise_score] * len(event_tokens)
            )
            events.append(event)
            total_token_offset += len(event_tokens)
        
        logger.info(f"Created {len(events)} episodic events based on semantic change.")
        return events, sentence_embeddings
    
    def _identify_event_boundaries(self, scores: List[float], sentences: Optional[List[str]] = None) -> List[int]:
        """
        Identify event boundaries from surprise/change scores.
        
        Implements the formula from the paper: T = μt−τ + γσt−τ
        
        Args:
            scores: Time series of surprise/change scores
            sentences: Optional sentence list (unused, for compatibility)
            
        Returns:
            List of boundary indices
        """
        if len(scores) < self.config.surprise_window:
            logger.info(
                "Sequence too short for boundary detection (len=%d, window=%d). "
                "Treating as single event.",
                len(scores),
                self.config.surprise_window,
            )
            return [0, len(scores)]
        
        boundaries = [0]  # First position is always a boundary
        
        for i in range(self.config.surprise_window, len(scores)):
            # Calculate mean and std over moving window
            window_scores = scores[i - self.config.surprise_window : i]
            
            if len(window_scores) > 1:  # Ensure window is not empty
                mean_score = np.mean(window_scores)
                std_score = np.std(window_scores)
                
                # Threshold calculation: T = μ + γσ
                threshold = mean_score + self.config.surprise_gamma * std_score
                
                # If current score exceeds threshold, mark as boundary
                if scores[i] > threshold:
                    boundaries.append(i)
                    logger.debug(
                        f"Boundary detected at item index {i}, "
                        f"score: {scores[i]:.3f}, threshold: {threshold:.3f}"
                    )
        
        boundaries.append(len(scores))  # Last position is also a boundary
        
        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))
        logger.info(f"Identified {len(boundaries)-1} initial events from surprise")
        
        return boundaries
    
    def calculate_attention_similarity_matrix(self, attention_keys: np.ndarray) -> np.ndarray:
        """
        Calculate similarity matrix from attention keys.
        
        Args:
            attention_keys: Attention key matrix (seq_len, hidden_dim)
            
        Returns:
            Similarity matrix (seq_len, seq_len)
        """
        # Use cosine similarity (paper uses dot product, but normalization is more stable)
        similarity_matrix = cosine_similarity(attention_keys)
        return similarity_matrix
