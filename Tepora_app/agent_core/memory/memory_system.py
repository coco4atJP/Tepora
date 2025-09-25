# agent_core/memory/memory_system.py
"""
Simple EM-LLM MemorySystem fallback implementation.

Features:
- Stores episodic events in a local SQLite DB.
- Stores summary, raw history (JSON), timestamp, and embedding (as JSON array).
- Uses embedding_provider.encode(...) to produce embeddings (expects a list-of-lists).
- Retrieval supports k-nearest by cosine similarity and a temporally-contiguous boost.
- Simple schema and helper functions so it can be swapped with a ChromaDB backend later.
"""
import sqlite3
import json
import time
import os
import math
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class MemorySystem:
    def __init__(self, embedding_provider, db_path: str = "./tepora_memory.db"):
        """
        embedding_provider: object with .encode(List[str]) -> List[List[float]]
        db_path: path to sqlite file
        """
        self.embedding_provider = embedding_provider
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        created = not os.path.exists(self.db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = self._conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY,
            created_ts REAL,
            summary TEXT,
            history_json TEXT,
            embedding_json TEXT,
            metadata_json TEXT
        )''')
        self._conn.commit()
        if created:
            logger.info(f"Created new memory DB at {self.db_path}")

    def _vec_from_json(self, json_str: str):
        return json.loads(json_str)

    def _json_from_vec(self, vec: List[float]):
        return json.dumps(vec)

    def save_episode(self, summary: str, history_json: str, metadata: Optional[Dict[str,Any]] = None):
        """
        Save an episode and compute/store its embedding.
        Returns generated id.
        """
        try:
            doc_id = metadata.get("id") if metadata and "id" in metadata else str(time.time()).replace('.','')  # fallback id
            # produce embedding
            emb = self.embedding_provider.encode([summary])[0]
            cur = self._conn.cursor()
            cur.execute(
                """INSERT OR REPLACE INTO episodes (id, created_ts, summary, history_json, embedding_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)""", 
                (doc_id, time.time(), summary, history_json, self._json_from_vec(emb), json.dumps(metadata or {}))
            )
            self._conn.commit()
            logger.info(f"Saved episode {doc_id} (summary len={len(summary)})")
            return doc_id
        except Exception as e:
            logger.exception("Failed to save episode to MemorySystem")
            raise

    def _cosine(self, a, b):
        # handle zero vectors gracefully
        za = sum(x*x for x in a) ** 0.5
        zb = sum(x*x for x in b) ** 0.5
        if za == 0 or zb == 0:
            return 0.0
        dot = sum(x*y for x,y in zip(a,b))
        return dot / (za*zb)

    def retrieve(self, query: str, k: int = 5, temporality_boost: float = 0.15):
        """
        Retrieve top-k episodes for query.
        temporality_boost: adds a small score boost for more recent episodes (0..1)
        """
        try:
            q_emb = self.embedding_provider.encode([query])[0]
            cur = self._conn.cursor()
            cur.execute("SELECT id, created_ts, summary, history_json, embedding_json, metadata_json FROM episodes")
            rows = cur.fetchall()
            scored = []
            for r in rows:
                eid, ts, summary, history_json, embedding_json, metadata_json = r
                emb = self._vec_from_json(embedding_json)
                sim = self._cosine(q_emb, emb)
                # temporality boost: normalize timestamp into [0,1] relative to max ts
                scored.append({"id": eid, "ts": ts, "summary": summary, "history_json": history_json, "metadata": json.loads(metadata_json), "score": sim})
            if not scored:
                return []
            # normalize ts
            max_ts = max(x["ts"] for x in scored) or 1.0
            for item in scored:
                recency = (item["ts"] / max_ts)
                item["score"] = item["score"] + temporality_boost * recency
            scored.sort(key=lambda x: x["score"], reverse=True)
            topk = scored[:k]
            logger.info(f"Retrieved {len(topk)} episodes for query (k={k}). Top score={topk[0]['score'] if topk else None}")
            return topk
        except Exception as e:
            logger.exception("Failed during retrieval")
            return []

    def count(self):
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(1) FROM episodes")
        return cur.fetchone()[0]

    def get_all(self):
        cur = self._conn.cursor()
        cur.execute("SELECT id, created_ts, summary, history_json, metadata_json FROM episodes ORDER BY created_ts ASC")
        rows = cur.fetchall()
        return [{"id": r[0], "ts": r[1], "summary": r[2], "history": r[3], "metadata": json.loads(r[4] or "{}") } for r in rows]

    def retrieve_similar_episodes(self, query: str, k: int = 5) -> List[Dict]:
        """retrieveメソッドをラッパーしてretrieve_similar_episodesと互換性のあるインターフェースを提供"""
        return self.retrieve(query, k)

    def close(self):
        try:
            self._conn.close()
        except Exception as e:
            logger.exception("Failed to close memory DB connection: %s", e)

