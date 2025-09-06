# agent_core/em_llm_core.py
"""
EM-LLM (Episodic Memory for Large Language Models) の核心実装

このモジュールは以下のEM-LLM固有の機能を提供します：
1. 驚異度（Surprise）計算とセグメンテーション
2. 境界精密化（Boundary Refinement）
3. 階層的注意メカニズム
4. 2段階検索システム

論文: "Human-inspired Episodic Memory for Infinite Context LLMs" (ICLR 2025)
"""

import numpy as np
import logging
import asyncio
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.stats import zscore
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from langchain_core.messages import BaseMessage
from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.llms import LlamaCpp
import torch
import nltk
import json
import re

logger = logging.getLogger(__name__)

@dataclass
class EpisodicEvent:
    """EM-LLMにおける単一のエピソード事象を表現するデータクラス"""
    tokens: List[str]
    start_position: int
    end_position: int
    surprise_scores: List[float]
    attention_keys: Optional[np.ndarray] = None
    representative_tokens: Optional[List[int]] = None
    summary: Optional[str] = None
    embedding: Optional[np.ndarray] = None

@dataclass
class EMConfig:
    """EM-LLMの設定パラメータ"""
    # 驚異度関連
    surprise_window: int = 128  # 驚異度計算のウィンドウサイズ
    surprise_gamma: float = 1.0  # 閾値調整パラメータ
    min_event_size: int = 8     # 最小事象サイズ
    max_event_size: int = 128   # 最大事象サイズ
    
    # 検索関連
    similarity_buffer_ratio: float = 0.7  # 類似度バッファの比率
    contiguity_buffer_ratio: float = 0.3  # 連続性バッファの比率
    total_retrieved_events: int = 4    # 総検索事象数
    repr_topk: int = 4                 # 代表トークン数
    recency_weight: float = 0.1        # 時間的近接性の重み (0.0 - 1.0)
    
    # 境界精密化関連
    use_boundary_refinement: bool = True
    refinement_metric: str = "modularity"  # "modularity" or "conductance"
    refinement_search_range: int = 16      # 境界精密化の最大探索範囲

class EMEventSegmenter:
    """意味的な変化に基づいてテキストをイベントに分割するセグメンター"""
    
    def __init__(self, config: EMConfig):
        self.config = config
        self.sent_tokenizer = self._get_sentence_tokenizer()
        logger.info("EM-LLM Semantic Event Segmenter initialized")

    def _get_sentence_tokenizer(self):
        """NLTKの'punkt'トークナイザを安全にロードする。
        
        `nltk.sent_tokenize` を直接使わず、`nltk.data.load` を使用することで、
        NLTKのインストール状態に問題がある場合でも安定して動作させる。
        """
        try:
            return nltk.data.load('tokenizers/punkt/english.pickle')
        except LookupError:
            logger.info("NLTK 'punkt' tokenizer data not found. Downloading...")
            nltk.download('punkt')
            logger.info("'punkt' data downloaded successfully.")
            return nltk.data.load('tokenizers/punkt/english.pickle')

    def _split_into_sentences(self, text: str) -> List[str]:
        """NLTKを使用してテキストを文に分割する"""
        if not text:
            return []
        # 改行をスペースに置換して、NLTKが処理しやすくする
        text = re.sub(r'\n+', ' ', text).strip()
        return self.sent_tokenizer.tokenize(text)

    def segment_text_into_events(self, text: str, embedding_provider) -> Tuple[List[EpisodicEvent], Optional[np.ndarray]]:
        """テキストを意味的な変化に基づいてエピソード事象に分割する"""
        if not text or not embedding_provider:
            return [], None

        # 1. テキストを文に分割
        sentences = self._split_into_sentences(text)
        if len(sentences) < 2:
            logger.info("Text too short for semantic segmentation, treating as a single event.")
            tokens = text.split()
            event = EpisodicEvent(
                tokens=tokens,
                start_position=0,
                end_position=len(tokens),
                surprise_scores=[0.0] * len(tokens) # 驚きなし
            )
            return [event], None

        # 2. 各文を埋め込みに変換
        sentence_embeddings = np.array(embedding_provider.encode(sentences))

        # 3. 隣接する文の埋め込み間のコサイン距離を計算
        distances = [
            cosine_distances(
                sentence_embeddings[i].reshape(1, -1),
                sentence_embeddings[i + 1].reshape(1, -1)
            )[0][0] for i in range(len(sentences) - 1)
        ]
        # 最初の文の変化スコアは0とする
        semantic_change_scores = [0.0] + distances

        # 4. 意味的変化スコアに基づいて境界を特定
        boundary_indices = self._identify_event_boundaries(semantic_change_scores, sentences)

        # 5. 境界からイベントを構築
        events = []
        total_token_offset = 0
        for i in range(len(boundary_indices) - 1):
            start_sentence_idx = boundary_indices[i]
            end_sentence_idx = boundary_indices[i+1]

            event_sentences = sentences[start_sentence_idx:end_sentence_idx]
            event_text = " ".join(event_sentences)
            event_tokens = event_text.split() # シンプルなトークナイザ

            # このイベントを代表する「驚き」スコア（境界開始点の意味的変化）
            event_surprise_score = semantic_change_scores[start_sentence_idx]

            event = EpisodicEvent(
                tokens=event_tokens,
                start_position=total_token_offset,
                end_position=total_token_offset + len(event_tokens),
                # イベント内の全トークンに同じ驚きスコアを割り当てる
                surprise_scores=[event_surprise_score] * len(event_tokens)
            )
            events.append(event)
            total_token_offset += len(event_tokens)

        logger.info(f"Created {len(events)} episodic events based on semantic change.")
        return events, sentence_embeddings

    def _identify_event_boundaries(self, scores: List[float], items: List[Any]) -> List[int]:
        """
        スコアの時系列データからイベント境界を特定する
        
        論文の式: T = μt−τ + γσt−τ
        """
        if len(scores) < self.config.surprise_window:
            logger.warning("Sequence too short for boundary detection")
            return [0, len(scores)]
        
        boundaries = [0]  # 最初は常に境界
        
        for i in range(self.config.surprise_window, len(scores)):
            # 移動ウィンドウでの平均と標準偏差を計算
            window_start = max(0, i - self.config.surprise_window)
            window_scores = scores[window_start:i]
            
            if len(window_scores) > 1:
                mean_score = np.mean(window_scores)
                std_score = np.std(window_scores)
                
                # 閾値計算: T = μ + γσ
                threshold = mean_score + self.config.surprise_gamma * std_score
                
                # 現在のトークンが閾値を超えた場合、境界とする
                if scores[i] > threshold:
                    boundaries.append(i)
                    logger.debug(f"Boundary detected at item index {i}, score: {scores[i]:.3f}, threshold: {threshold:.3f}")
        
        boundaries.append(len(scores))  # 最後も境界
        
        # 重複削除とソート
        boundaries = sorted(list(set(boundaries)))
        logger.info(f"Identified {len(boundaries)-1} initial events from surprise")
        
        return boundaries

class EMBoundaryRefiner:
    """境界精密化によるセグメンテーション最適化"""
    
    def __init__(self, config: EMConfig):
        self.config = config
        # 近似類似度計算用のウィンドウサイズを追加
        self.approx_similarity_window = 16
    
    def _calculate_surprise_similarity_matrix(self, surprise_scores: List[float]) -> np.ndarray:
        """
        【代替案】驚異度スコアの時系列パターンから近似的な類似度行列を計算する。
        アテンションキーが利用できないストリーミング時などに使用する。
        """
        logger.info("Calculating approximate similarity matrix from surprise scores.")
        seq_len = len(surprise_scores)
        if seq_len < self.approx_similarity_window:
            logger.warning("Sequence too short for surprise-based similarity calculation. Returning identity matrix.")
            return np.identity(seq_len)

        # 各トークンの「驚異度コンテキスト」ベクトルを作成
        # ウィンドウ内の驚異度スコアのパターンをベクトルとする
        window = self.approx_similarity_window
        padded_scores = np.pad(surprise_scores, (window // 2, window // 2), 'edge')
        
        context_vectors = np.array([
            padded_scores[i : i + window] for i in range(seq_len)
        ])

        # z-scoreで正規化して、絶対値ではなくパターンの類似度を重視
        # 行（各ベクトル）ごとに正規化
        std_devs = np.std(context_vectors, axis=1, keepdims=True)
        std_devs[std_devs == 0] = 1.0 # ゼロ除算を避ける
        means = np.mean(context_vectors, axis=1, keepdims=True)
        normalized_vectors = (context_vectors - means) / std_devs

        # コサイン類似度を計算して類似度行列とする
        similarity_matrix = cosine_similarity(normalized_vectors)
        
        np.fill_diagonal(similarity_matrix, 1.0)

        logger.debug(f"Approximate similarity matrix of shape {similarity_matrix.shape} created.")
        return similarity_matrix

    def calculate_attention_similarity_matrix(self, attention_keys: np.ndarray) -> np.ndarray:
        """
        アテンションキーから類似度行列を計算
        
        Args:
            attention_keys: (seq_len, hidden_dim) のアテンションキー行列
            
        Returns:
            (seq_len, seq_len) の類似度行列
        """
        # コサイン類似度を使用（論文ではドット積だが、正規化された方が安定）
        similarity_matrix = cosine_similarity(attention_keys)
        return similarity_matrix
    
    def calculate_modularity(self, similarity_matrix: np.ndarray, boundaries: List[int]) -> float:
        """
        モジュラリティ（論文の式3）を計算
        """
        try:
            G = nx.from_numpy_array(similarity_matrix)
            
            # 境界に基づくコミュニティ作成
            communities = []
            for i in range(len(boundaries) - 1):
                community = list(range(boundaries[i], boundaries[i + 1]))
                if community:  # 空でない場合のみ追加
                    communities.append(community)
            
            if len(communities) <= 1:
                return 0.0
                
            return nx.algorithms.community.modularity(G, communities)
        except Exception as e:
            logger.warning(f"Modularity calculation failed: {e}")
            return 0.0
    
    def calculate_conductance(self, similarity_matrix: np.ndarray, boundaries: List[int]) -> float:
        """
        伝導性（論文の式4）を計算
        """
        try:
            total_conductance = 0.0
            num_communities = len(boundaries) - 1
            
            for i in range(num_communities):
                start, end = boundaries[i], boundaries[i + 1]
                
                # コミュニティ内部の重み
                internal_weight = np.sum(similarity_matrix[start:end, start:end])
                
                # コミュニティ外部への重み
                external_weight = (
                    np.sum(similarity_matrix[start:end, :start]) +
                    np.sum(similarity_matrix[start:end, end:])
                )
                
                # 伝導性計算
                total_weight = internal_weight + external_weight
                if total_weight > 0:
                    conductance = external_weight / total_weight
                    total_conductance += conductance
            
            return total_conductance / max(1, num_communities)
        except Exception as e:
            logger.warning(f"Conductance calculation failed: {e}")
            return 1.0  # 悪いスコア
    
    def refine_boundaries(self, events: List[EpisodicEvent], context_vectors: Optional[np.ndarray] = None) -> List[EpisodicEvent]:
        """
        グラフ理論メトリクスを使用して境界を精密化。
        context_vectorsは文の埋め込みベクトル、または将来的にアテンションキー。
        """
        if not self.config.use_boundary_refinement or len(events) <= 1:
            return events
        
        logger.info("Refining event boundaries using graph-theoretic metrics")
        
        # 類似度行列を計算
        if context_vectors is not None:
            logger.info("Using context vectors (e.g., sentence embeddings) for similarity matrix.")
            # このメソッドはコサイン類似度を計算するだけなので、そのまま使える
            similarity_matrix = self.calculate_attention_similarity_matrix(context_vectors)
        else:
            logger.warning("Context vectors not available. Skipping boundary refinement.")
            return events
        
        # 現在の境界を抽出
        current_boundaries = [event.start_position for event in events] + [events[-1].end_position]
        
        # 各境界ペアについて最適位置を探索
        refined_boundaries = [current_boundaries[0]]  # 最初の境界は固定
        
        for i in range(len(current_boundaries) - 2):
            start_boundary = refined_boundaries[-1]
            end_boundary = current_boundaries[i + 2]
            current_pos = current_boundaries[i + 1]
            
            best_pos = current_pos
            best_score = self._evaluate_boundary_position(
                similarity_matrix, refined_boundaries + [current_pos, end_boundary]
            )
            
            # 近隣位置を探索
            # イベント長に応じて探索範囲を動的に決定。設定ファイルの `refinement_search_range` を最大値とする。
            event_pair_length = end_boundary - start_boundary
            dynamic_range = event_pair_length // 4
            search_range = min(self.config.refinement_search_range, dynamic_range)
            
            for offset in range(-search_range, search_range + 1): # ステップを1にしてより細かく探索
                test_pos = current_pos + offset
                if start_boundary < test_pos < end_boundary:
                    test_boundaries = refined_boundaries + [test_pos, end_boundary]
                    score = self._evaluate_boundary_position(similarity_matrix, test_boundaries)
                    
                    if self._is_better_score(score, best_score):
                        best_score = score
                        best_pos = test_pos
            
            refined_boundaries.append(best_pos)
        
        refined_boundaries.append(current_boundaries[-1])  # 最後の境界も固定
        
        # 精密化された境界で事象を再構築
        return self._rebuild_events_from_boundaries(events, refined_boundaries)
    
    def _evaluate_boundary_position(self, similarity_matrix: np.ndarray, boundaries: List[int]) -> float:
        """境界位置の評価"""
        if self.config.refinement_metric == "modularity":
            return self.calculate_modularity(similarity_matrix, boundaries)
        else:
            return -self.calculate_conductance(similarity_matrix, boundaries)  # 負の値（小さいほど良い）
    
    def _is_better_score(self, new_score: float, current_best: float) -> bool:
        """スコアの改善判定"""
        return new_score > current_best
    
    def _rebuild_events_from_boundaries(self, original_events: List[EpisodicEvent], boundaries: List[int]) -> List[EpisodicEvent]:
        """精密化された境界から事象を再構築"""
        refined_events = []
        all_tokens = []
        all_surprises = []
        
        # 全トークンと驚異度を結合
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

class EMTwoStageRetrieval:
    """EM-LLMの2段階検索システム（類似度バッファ + 連続性バッファ）"""
    
    def __init__(self, config: EMConfig):
        self.config = config
        self.stored_events: List[EpisodicEvent] = []
        self.contiguity_queue: List[EpisodicEvent] = []
    
    def add_events(self, events: List[EpisodicEvent]):
        """新しい事象をメモリに追加"""
        self.stored_events.extend(events)
        logger.debug(f"Added {len(events)} events to memory. Total: {len(self.stored_events)}")
    
    def retrieve_relevant_events(self, query_embedding: np.ndarray, k: Optional[int] = None) -> List[EpisodicEvent]:
        """
        2段階検索: 類似度ベース + 時間的連続性
        
        Args:
            query_embedding: クエリの埋め込みベクトル
            k: 取得する事象数（Noneの場合は設定値を使用）
            
        Returns:
            取得された関連事象のリスト
        """
        if not self.stored_events:
            return []
        
        total_k = k or self.config.total_retrieved_events
        ks = int(total_k * self.config.similarity_buffer_ratio)  # 類似度バッファサイズ
        kc = total_k - ks  # 連続性バッファサイズ
        
        # Stage 1: 類似度ベース検索
        similarity_events = self._similarity_based_retrieval(query_embedding, ks)
        
        # Stage 2: 時間的連続性バッファ
        contiguity_events = self._contiguity_based_retrieval(similarity_events, kc)
        
        # 結合して重複除去
        all_retrieved = similarity_events + contiguity_events
        unique_events = self._deduplicate_events(all_retrieved)
        
        logger.debug(f"Retrieved {len(unique_events)} events (similarity: {len(similarity_events)}, contiguity: {len(contiguity_events)})")
        # 最終的なイベントリストを時間順（古い順）にソートして返すと、文脈が自然になる可能性がある
        sorted_events = sorted(unique_events, key=lambda e: e.start_position)
        return sorted_events[:total_k]
    
    def _similarity_based_retrieval(self, query_embedding: np.ndarray, ks: int) -> List[EpisodicEvent]:
        """類似度ベースの検索（k-NN）に時間的近接性（Recency）の重み付けを追加。"""
        if ks <= 0 or not self.stored_events:
            return []
        
        # 各事象の埋め込みとスコアを計算
        scored_events = []
        total_events = len(self.stored_events)
        
        for i, event in enumerate(self.stored_events):
            if event.embedding is not None:
                # 1. 類似度スコア (コサイン類似度)
                similarity_score = np.dot(query_embedding, event.embedding)
                
                # 2. 時間的近接性（Recency）スコア
                # インデックスを正規化して [0, 1] の範囲にする (新しいほど1に近づく)
                recency_score = i / (total_events - 1) if total_events > 1 else 0.0
                
                # 3. 最終スコアの計算
                final_score = similarity_score + self.config.recency_weight * recency_score
                
                scored_events.append((final_score, event, similarity_score, recency_score))
        
        # 最終スコアでソート
        scored_events.sort(key=lambda x: x[0], reverse=True)
        
        logger.debug("--- Recency-aware Similarity Retrieval Top-K ---")
        for final_score, event, sim, rec in scored_events[:ks]:
            logger.debug(
                f"  - Event(pos:{event.start_position}): final_score={final_score:.3f} "
                f"(sim={sim:.3f} + recency={rec:.3f} * {self.config.recency_weight})"
            )
        
        # 上位ks個を選択
        return [event for final_score, event, sim, rec in scored_events[:ks]]
    
    def _contiguity_based_retrieval(self, similarity_events: List[EpisodicEvent], kc: int) -> List[EpisodicEvent]:
        """時間的連続性に基づく検索"""
        if kc <= 0 or not similarity_events:
            return []
        
        contiguity_events = []
        
        for event in similarity_events:
            # 隣接する事象を取得
            event_index = self._find_event_index(event)
            if event_index >= 0:
                # 前後の事象を連続性バッファに追加
                for offset in [-1, 1]:
                    neighbor_index = event_index + offset
                    if 0 <= neighbor_index < len(self.stored_events):
                        neighbor_event = self.stored_events[neighbor_index]
                        if neighbor_event not in similarity_events:
                            contiguity_events.append(neighbor_event)
        
        # kc個まで制限
        return contiguity_events[:kc]
    
    def _find_event_index(self, target_event: EpisodicEvent) -> int:
        """事象のインデックスを検索"""
        for i, event in enumerate(self.stored_events):
            if (event.start_position == target_event.start_position and 
                event.end_position == target_event.end_position):
                return i
        return -1
    
    def _deduplicate_events(self, events: List[EpisodicEvent]) -> List[EpisodicEvent]:
        """重複する事象を除去"""
        seen_positions = set()
        unique_events = []
        
        for event in events:
            position_key = (event.start_position, event.end_position)
            if position_key not in seen_positions:
                seen_positions.add(position_key)
                unique_events.append(event)
        
        return unique_events

class EMMemorySystem:
    """EM-LLMメモリシステムの統合クラス"""
    
    def __init__(self, config: EMConfig, embedding_provider=None, llm_manager=None):
        self.config = config
        self.embedding_provider = embedding_provider
        self.llm_manager = llm_manager
        # 各コンポーネントを初期化
        self.segmenter = EMEventSegmenter(config)
        self.boundary_refiner = EMBoundaryRefiner(config)
        self.retrieval_system = EMTwoStageRetrieval(config)
        
        logger.info("EM-LLM Memory System initialized")
    
    async def process_text_for_memory_formation(self, text: str) -> List[EpisodicEvent]:
        """
        テキスト全体をメモリ形成パイプラインで処理
        
        Processing Pipeline:
        1. 意味的変化の計算
        2. セグメンテーション
        3. 境界精密化（オプション）
        4. 代表トークン選出
        5. 個別事象の要約
        """
        logger.info(f"Processing text of {len(text)} chars for memory formation")

        # Step 1 & 2: セマンティック変化に基づくセグメンテーション
        events, sentence_embeddings = self.segmenter.segment_text_into_events(text, self.embedding_provider)
        if not events:
            return []

        # Step 3: 境界精密化
        # sentence_embeddingsがNoneでない場合、それをコンテキストベクトルとして使用
        if self.config.use_boundary_refinement and sentence_embeddings is not None:
            # refine_boundariesは文の数とsentence_embeddingsの数が一致することを期待する
            # 今回の実装では一致するので問題ない
            events = self.boundary_refiner.refine_boundaries(events, context_vectors=sentence_embeddings)
        
        # Step 4 & 5: 各事象の代表トークン選出と要約生成
        summarization_tasks = []
        for event in events:
            event.representative_tokens = self._select_representative_tokens(event)
            # 要約タスクを作成
            if self.llm_manager:
                summarization_tasks.append(self._summarize_event_async(event))

        if summarization_tasks:
            summaries = await asyncio.gather(*summarization_tasks)
            for event, summary in zip(events, summaries):
                event.summary = summary

        for event in events:
            # 事象の埋め込みを計算（利用可能な場合）
            if self.embedding_provider:
                event.embedding = self._compute_event_embedding(event)
        
        # Step 6: メモリに格納
        self.retrieval_system.add_events(events)
        
        logger.info(f"Memory formation completed: {len(events)} episodic events created")
        return events
    
    def retrieve_for_query(self, query: str, k: Optional[int] = None) -> List[EpisodicEvent]:
        """クエリに対して関連事象を検索"""
        if not self.embedding_provider:
            logger.warning("No embedding provider available for retrieval")
            return []
        
        # クエリの埋め込みを計算
        query_embedding = self.embedding_provider.encode([query])[0]
        
        # 2段階検索を実行
        return self.retrieval_system.retrieve_relevant_events(query_embedding, k)
    
    def _select_representative_tokens(self, event: EpisodicEvent) -> List[int]:
        """事象内の代表的なトークンを選出（驚異度が高い順）"""
        if not event.surprise_scores:
            return []
        
        # 驚異度でソートしてトップkを選択
        indexed_scores = [(score, i) for i, score in enumerate(event.surprise_scores)]
        indexed_scores.sort(key=lambda x: x[0], reverse=True)
        
        representative_indices = [i for _, i in indexed_scores[:self.config.repr_topk]]
        return sorted(representative_indices)  # 位置順にソート
    
    async def _summarize_event_async(self, event: EpisodicEvent) -> Optional[str]:
        """【★新規】SLMを使って単一の事象を非同期で要約する"""
        if not self.llm_manager:
            logger.warning("LLM manager not available for event summarization. Skipping.")
            return "Summary not generated."
        
        try:
            slm = self.llm_manager.get_slm_summarizer()
            from langchain_core.prompts import ChatPromptTemplate
            from . import config
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", config.BASE_SYSTEM_PROMPTS["event_summarization"])
            ])
            chain = prompt | slm
            
            event_text = " ".join(event.tokens)
            response = await chain.ainvoke({"event_text": event_text})
            
            summary = response.content.strip()
            return summary
        except Exception as e:
            logger.error(f"Failed to summarize event {event.start_position}-{event.end_position}: {e}", exc_info=True)
            return "Summary generation failed."

    def _compute_event_embedding(self, event: EpisodicEvent) -> np.ndarray:
        """事象の埋め込みベクトルを計算"""
        if not event.tokens:
            return np.zeros(384)  # デフォルト次元
        
        # 事象のテキストを結合
        event_text = " ".join(event.tokens)
        
        try:
            # 埋め込みプロバイダーを使用
            embedding = self.embedding_provider.encode([event_text])[0]
            return np.array(embedding)
        except Exception as e:
            logger.warning(f"Failed to compute event embedding: {e}")
            return np.zeros(384)

class EMLLMIntegrator:
    """既存システムとEM-LLMの統合クラス"""
    
    def __init__(self, llm_manager, embedding_provider):
        self.llm_manager = llm_manager
        self.embedding_provider = embedding_provider
        
        # EM-LLM設定
        self.config = EMConfig(
            surprise_window=64,
            surprise_gamma=1.0,
            min_event_size=8,
            max_event_size=64,
            similarity_buffer_ratio=0.7,
            contiguity_buffer_ratio=0.3,
            total_retrieved_events=4,
            refinement_search_range=16
        )
        
        # EM-LLMメモリシステム初期化
        self.memory_system = EMMemorySystem(self.config, embedding_provider, llm_manager)
        
        logger.info("EM-LLM Integrator initialized")
    
    async def process_conversation_turn_for_memory(self, user_input: str, ai_response: str) -> List[EpisodicEvent]:
        """
        対話ターンをEM-LLMメモリ形成パイプラインで非同期に処理する。
        """
        logger.info("Processing conversation turn for EM-LLM memory formation (semantic change based).")
        
        try:
            if not ai_response:
                logger.warning("AI response is empty. Aborting memory formation.")
                return []

            # EM-LLMメモリ形成パイプラインを、AIの応答テキストで実行
            events = await self.memory_system.process_text_for_memory_formation(
                text=ai_response
            )
            
            logger.info(f"Created {len(events)} episodic events from conversation turn via semantic segmentation.")
            return events
            
        except Exception as e:
            logger.error(f"EM-LLM memory formation failed: {e}", exc_info=True)
            return []
    
    def get_current_llm_config_for_diagnostics(self) -> Dict:
        """診断用に現在のLLM設定を取得する"""
        if self.llm_manager and hasattr(self.llm_manager, 'get_current_model_config_for_diagnostics'):
            return self.llm_manager.get_current_model_config_for_diagnostics()
        logger.warning("LLMManager not available or does not have the required diagnostics method.")
        return {}

    def retrieve_relevant_memories_for_query(self, query: str) -> List[Dict]:
        """
        クエリに対してEM-LLM方式で関連記憶を検索
        
        Returns:
            既存のメモリシステムと互換性のある辞書形式のリスト
        """
        logger.info("Retrieving memories using EM-LLM two-stage retrieval")
        
        try:
            # EM-LLMの2段階検索を実行
            relevant_events = self.memory_system.retrieve_for_query(query)
            
            # 既存システムとの互換性のため辞書形式に変換
            memory_entries = []
            for i, event in enumerate(relevant_events):
                memory_entry = {
                    'id': f"em_event_{event.start_position}_{event.end_position}",
                    'content': " ".join(event.tokens),
                    'summary': f"Episodic event from position {event.start_position} to {event.end_position}",
                    'surprise_stats': {
                        'mean_surprise': float(np.mean(event.surprise_scores)),
                        'max_surprise': float(np.max(event.surprise_scores)),
                        'event_size': len(event.tokens)
                    },
                    'representative_tokens': event.representative_tokens or [],
                    'retrieval_rank': i + 1
                }
                memory_entries.append(memory_entry)
            
            logger.info(f"Retrieved {len(memory_entries)} EM-LLM memories")
            return memory_entries
            
        except Exception as e:
            logger.error(f"EM-LLM memory retrieval failed: {e}")
            return []
    
    def get_memory_statistics(self) -> Dict:
        """EM-LLMメモリシステムの統計情報"""
        total_events = len(self.memory_system.retrieval_system.stored_events)
        
        if total_events == 0:
            return {'total_events': 0, 'status': 'empty'}
        
        # 事象の統計計算
        event_sizes = [len(event.tokens) for event in self.memory_system.retrieval_system.stored_events]
        surprise_stats = []
        
        for event in self.memory_system.retrieval_system.stored_events:
            if event.surprise_scores:
                surprise_stats.extend(event.surprise_scores)
        
        statistics = {
            'total_events': total_events,
            'mean_event_size': float(np.mean(event_sizes)) if event_sizes else 0,
            'total_tokens_in_memory': sum(event_sizes),
            'surprise_statistics': {
                'mean': float(np.mean(surprise_stats)) if surprise_stats else 0,
                'std': float(np.std(surprise_stats)) if surprise_stats else 0,
                'max': float(np.max(surprise_stats)) if surprise_stats else 0,
            },
            'configuration': {
                'surprise_gamma': self.config.surprise_gamma,
                'min_event_size': self.config.min_event_size,
                'max_event_size': self.config.max_event_size,
                'total_retrieved_events': self.config.total_retrieved_events
            }
        }
        
        return statistics