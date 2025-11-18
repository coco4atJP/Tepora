# agent_core/memory/memory_system.py
"""
ChromaDB-based MemorySystem implementation.

Features:
- Stores episodic events in a local ChromaDB collection.
- Stores summary (document), history (metadata), timestamp (metadata), and embedding.
- Uses embedding_provider.encode(...) to produce embeddings (expects a list-of-lists).
- Retrieval supports k-nearest by cosine similarity and a temporally-contiguous boost.
"""
import json
import time
import logging
from typing import List, Dict, Any, Optional
import chromadb

logger = logging.getLogger(__name__)

class MemorySystem:
    def __init__(self, embedding_provider, db_path: str = "./chroma_db", collection_name: str = "tepora_memory"):
        """
        embedding_provider: object with .encode(List[str]) -> List[List[float]]
        db_path: path to the directory where ChromaDB data will be stored.
        collection_name: name of the collection to use.
        """
        self.embedding_provider = embedding_provider
        self.db_path = db_path
        self.collection_name = collection_name
        self._init_chroma()

    def _init_chroma(self):
        """Initializes the ChromaDB client and collection."""
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"ChromaDB client initialized. Collection '{self.collection_name}' is ready at {self.db_path}")
        except Exception as e:
            logger.exception("Failed to initialize ChromaDB")
            raise

    def save_episode(self, summary: str, history_json: str, metadata: Optional[Dict[str,Any]] = None):
        """
        Save an episode and compute/store its embedding.
        Returns generated id.
        """
        if not summary:
            logger.warning("Attempted to save an episode with an empty summary. Skipping.")
            return None
        try:
            doc_id = metadata.get("id") if metadata and "id" in metadata else str(time.time()).replace('.','')
            embedding = self.embedding_provider.encode([summary])[0]
            
            # Prepare metadata for ChromaDB
            # All values must be str, int, float, or bool.
            episode_metadata = {
                "created_ts": time.time(),
                "history_json": history_json,
                "metadata_json": json.dumps(metadata or {})
            }

            self.collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[summary],
                metadatas=[episode_metadata]
            )
            logger.info(f"Saved episode {doc_id} to ChromaDB (summary len={len(summary)})")
            return doc_id
        except Exception as e:
            logger.exception("Failed to save episode to ChromaDB MemorySystem")
            raise

    def retrieve(self, query: str, k: int = 5, temporality_boost: float = 0.15, query_embedding_override: Optional[List[float]] = None, where_filter: Optional[Dict[str, Any]] = None):
        """
        Retrieve top-k episodes for query.
        temporality_boost: adds a small score boost for more recent episodes (0..1)
        query_embedding_override: If provided, use this embedding instead of computing one from the query string.
        where_filter: A ChromaDB 'where' filter dictionary to apply to the query.
        """
        if not query and query_embedding_override is None and where_filter is None:
            return []
        try:
            if query_embedding_override:
                query_embedding = [query_embedding_override]
            else:
                query_embedding = self.embedding_provider.encode([query])

            if query_embedding:
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=k,
                    where=where_filter
                )
            else: # where_filterのみの場合
                results = self.collection.get(
                    where=where_filter,
                    limit=k,
                    include=["metadatas", "documents", "distances"] # getでもdistancesは取れないが、互換性のため
                )

            if not results or not results['ids'][0]:
                return []

            scored = []
            # Chroma returns lists of lists, one for each query. We have one query.
            ids = results['ids'][0]
            raw_distances = results.get('distances')
            if raw_distances and raw_distances[0] is not None:
                distances = raw_distances[0]
            else:
                distances = [0.0] * len(ids)

            if len(distances) < len(ids):
                logger.debug("Distances length (%d) shorter than ids length (%d). Padding with zeros.", len(distances), len(ids))
                distances = list(distances) + [0.0] * (len(ids) - len(distances))

            metadatas = results['metadatas'][0]
            documents = results['documents'][0]

            for i in range(len(ids)):
                # Cosine distance to similarity: sim = 1 - dist
                sim = 1.0 - distances[i]
                meta = metadatas[i]
                scored.append({
                    "id": ids[i],
                    "ts": meta.get("created_ts", 0.0),
                    "summary": documents[i],
                    "history_json": meta.get("history_json", "{}"),
                    "metadata": json.loads(meta.get("metadata_json", "{}")),
                    "score": sim
                })

            if not scored:
                return []

            # Apply temporality boost
            if temporality_boost > 0 and len(scored) > 1:
                max_ts = max(x["ts"] for x in scored)
                if max_ts > 0:
                    for item in scored:
                        recency = item["ts"] / max_ts
                        item["score"] += temporality_boost * recency

            scored.sort(key=lambda x: x["score"], reverse=True)
            topk = scored[:k]
            logger.info(f"Retrieved {len(topk)} episodes from ChromaDB for query (k={k}). Top score={topk[0]['score'] if topk else None}")
            return topk
        except Exception as e:
            logger.exception("Failed during retrieval")
            return []

    def count(self):
        """Returns the number of episodes in the collection."""
        return self.collection.count()

    def get_all(self):
        """Returns all episodes, sorted by creation time."""
        # Note: ChromaDB's get() doesn't guarantee order. We fetch all and sort in Python.
        data = self.collection.get(include=["metadatas", "documents"])
        
        if not data or not data['ids']:
            return []
            
        items = []
        for i in range(len(data['ids'])):
            meta = data['metadatas'][i]

            metadata = {}
            metadata_json = meta.get("metadata_json")
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to decode metadata_json for id %s", data['ids'][i], exc_info=True
                    )
                    metadata = {}

            # Merge in fields stored directly on the metadata payload (e.g., EM-LLM values)
            for key, value in meta.items():
                if key not in {"created_ts", "history_json", "metadata_json"}:
                    metadata.setdefault(key, value)

            items.append({
                "id": data['ids'][i],
                "ts": meta.get("created_ts", 0.0),
                "summary": data['documents'][i],
                "history": meta.get("history_json", "{}"),
                "metadata": metadata
            })
        
        # Sort by timestamp
        items.sort(key=lambda x: x['ts'])
        return items

    def cleanup_old_events(self, max_age_days: int = 30, max_events: int = 10000000):
        """古いイベントや最大数を超えたイベントを削除してメモリを管理する。"""
        current_count = self.count()
        if current_count <= max_events:
            logger.info(f"Cleanup not needed. Current events ({current_count}) are within the limit ({max_events}).")
            return

        # 削除するイベント数を計算
        num_to_delete = current_count - max_events

        # 最も古いイベントを取得
        # ChromaDBは直接ソートして取得する機能が限定的なため、全件取得してソートする
        all_items = self.get_all() # get_allは 'ts' でソート済み

        # 削除対象のIDリストを作成
        ids_to_delete = [item['id'] for item in all_items[:num_to_delete]]

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            logger.info(f"Cleaned up {len(ids_to_delete)} oldest events to meet the max_events limit.")
        else:
            logger.info("No events to delete for cleanup.")

    def retrieve_similar_episodes(self, query: str, k: int = 5) -> List[Dict]:
        """retrieveメソッドをラッパーしてretrieve_similar_episodesと互換性のあるインターフェースを提供"""
        return self.retrieve(query, k)

    def close(self):
        """ChromaDB PersistentClient does not require explicit closing."""
        logger.info("ChromaDB client does not require explicit close(). Skipping.")
        pass
