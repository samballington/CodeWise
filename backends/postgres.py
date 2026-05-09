"""
backends/postgres.py — PostgreSQL + pgvector backend for CodeWise.

Contribution to samballington/CodeWise.

CodeWise currently uses SQLite + SQLite-VSS (FAISS-backed) for vector storage.
This module adds a PostgreSQL / pgvector alternative that:

  - Stores code chunks and embeddings in a Postgres table
  - Uses the pgvector HNSW index for fast ANN queries
  - Supports the same VectorStore.query() interface as the existing FAISS backend
  - Adds optional BM25 hybrid search (using rank_bm25 in-process, same as claude-echoes)
  - Works with the BGE-large-en-v1.5 (1024-dim) embeddings already used by CodeWise

Why Postgres instead of SQLite-VSS?
  - SQLite-VSS is experimental; the vss_version() function is not always available
  - Postgres + pgvector is production-grade, supports HNSW + IVFFlat indexes
  - Multi-user CodeWise deployments share a single Postgres instance
  - pgvector cosine ops match the BGE L2 distance used by CodeWise's FAISS index

Usage:
    from backends.postgres import PgVectorStore
    store = PgVectorStore(dsn="postgresql://user:pass@localhost/codewise")
    store.build(workspace_dir="/workspace")
    results = store.query("find the authentication middleware", k=5)

Environment variables:
    CODEWISE_PG_DSN  — Postgres connection string (required)
    CODEWISE_PG_TABLE — Table name for chunks (default: codewise_chunks)
    CODEWISE_EMBED_DIM — Embedding dimension (default: 1024 for BGE-large)
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_DSN   = os.environ.get("CODEWISE_PG_DSN", "")
_DEFAULT_TABLE = os.environ.get("CODEWISE_PG_TABLE", "codewise_chunks")
_EMBED_DIM     = int(os.environ.get("CODEWISE_EMBED_DIM", "1024"))
_CHUNK_SIZE    = 400   # characters — matches VectorStore._chunk_text()

# File extensions to index (mirrors VectorStore._list_files)
_INDEXED_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".md", ".txt", ".json", ".html", ".css",
}


class PgVectorStore:
    """
    PostgreSQL + pgvector code chunk store.

    Drop-in complement to the existing FAISS VectorStore.
    Implements the same query() interface so it can be used interchangeably
    in the SmartSearchEngine and HybridSearchEngine strategies.
    """

    def __init__(
        self,
        dsn: str = _DEFAULT_DSN,
        table: str = _DEFAULT_TABLE,
        embed_dim: int = _EMBED_DIM,
    ) -> None:
        if not dsn:
            raise ValueError(
                "CODEWISE_PG_DSN environment variable is required for PgVectorStore"
            )
        self.dsn = dsn
        self.table = table
        self.embed_dim = embed_dim
        self._embedder = None   # lazy-loaded
        self._conn = None       # lazy-connected

    # ── Connection ────────────────────────────────────────────────────────────

    def _ensure_conn(self) -> None:
        if self._conn is None or self._conn.closed:
            import psycopg2
            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = True
            self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create pgvector extension and chunks table if they don't exist."""
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id         TEXT PRIMARY KEY,
                    file_path  TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding  vector({self.embed_dim})
                )
            """)
            # HNSW index for cosine ANN (matches BGE training objective)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table}_hnsw
                ON {self.table}
                USING hnsw (embedding vector_cosine_ops)
            """)
        logger.info("PgVectorStore schema ready (table=%s, dim=%d)", self.table, self.embed_dim)

    # ── Embedder ──────────────────────────────────────────────────────────────

    def _get_embedder(self):
        """Lazy-load the BGE sentence transformer (same model as VectorStore)."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
        return self._embedder

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (n, embed_dim) float32 array."""
        embedder = self._get_embedder()
        vecs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype="float32")

    # ── File discovery ────────────────────────────────────────────────────────

    @staticmethod
    def _list_files(workspace_dir: str) -> List[Path]:
        paths = []
        for root, _, files in os.walk(workspace_dir):
            for f in files:
                if f.startswith("."):
                    continue
                p = Path(root) / f
                if p.suffix in _INDEXED_EXTS:
                    paths.append(p)
        return paths

    @staticmethod
    def _chunk_text(text: str) -> List[str]:
        return [text[i: i + _CHUNK_SIZE] for i in range(0, len(text), _CHUNK_SIZE)]

    @staticmethod
    def _chunk_id(file_path: str, offset: int) -> str:
        key = f"{file_path}::{offset}"
        return hashlib.sha1(key.encode()).hexdigest()[:16]

    # ── Build / index ─────────────────────────────────────────────────────────

    def build(self, workspace_dir: str = "/workspace", batch_size: int = 64) -> int:
        """
        Index all eligible files in workspace_dir into Postgres.

        Files are chunked at _CHUNK_SIZE character boundaries (matching
        VectorStore._chunk_text).  Chunks are upserted so re-indexing is
        idempotent.

        Returns the number of chunks inserted/updated.
        """
        self._ensure_conn()

        files = self._list_files(workspace_dir)
        logger.info("PgVectorStore.build: found %d files in %s", len(files), workspace_dir)

        total = 0
        batch_texts: List[str] = []
        batch_rows: List[tuple] = []   # (chunk_id, file_path, chunk_text)

        def _flush():
            nonlocal total
            if not batch_texts:
                return
            vecs = self._embed(batch_texts)
            import psycopg2.extras
            with self._conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    f"""
                    INSERT INTO {self.table} (id, file_path, chunk_text, embedding)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE
                        SET chunk_text = EXCLUDED.chunk_text,
                            embedding  = EXCLUDED.embedding
                    """,
                    [
                        (row[0], row[1], row[2], f"[{','.join(str(x) for x in vec)}]")
                        for row, vec in zip(batch_rows, vecs)
                    ],
                )
            total += len(batch_texts)
            batch_texts.clear()
            batch_rows.clear()

        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                logger.debug("skip %s: %s", file_path, exc)
                continue

            chunks = self._chunk_text(text)
            for offset, chunk in enumerate(chunks):
                chunk_id = self._chunk_id(str(file_path), offset)
                batch_texts.append(chunk)
                batch_rows.append((chunk_id, str(file_path), chunk))

                if len(batch_texts) >= batch_size:
                    _flush()

        _flush()
        logger.info("PgVectorStore.build: indexed %d chunks", total)
        return total

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query: str,
        k: int = 3,
        min_relevance: float = 0.3,
        allowed_projects: Optional[List[str]] = None,
        return_scores: bool = False,
    ) -> List[Tuple[str, str]] | List[Tuple[str, str, float]]:
        """
        Query the pgvector store for the k most relevant chunks.

        Matches the VectorStore.query() interface exactly so this backend
        can be passed to SmartSearchEngine / HybridSearchEngine.

        Parameters
        ----------
        query : str
            Natural language search query.
        k : int
            Maximum results to return.
        min_relevance : float
            Minimum cosine similarity threshold (0-1).  Chunks below this
            score are filtered out.
        allowed_projects : list[str], optional
            If provided, restrict results to chunks whose file_path starts
            with one of these project directory names.
        return_scores : bool
            If True return (path, snippet, score) 3-tuples instead of 2-tuples.

        Returns
        -------
        List of (file_path, snippet) or (file_path, snippet, score) tuples.
        """
        self._ensure_conn()

        vec = self._embed([query])[0]
        vec_str = f"[{','.join(str(x) for x in vec)}]"

        project_filter = ""
        params: list = [vec_str, vec_str]

        if allowed_projects:
            like_clauses = " OR ".join(
                f"file_path LIKE %s" for _ in allowed_projects
            )
            project_filter = f"AND ({like_clauses})"
            for proj in allowed_projects:
                params.append(f"{proj}/%")

        # Fetch more candidates than k to apply min_relevance filter
        fetch_k = k * 3
        params.append(fetch_k)

        import psycopg2.extras
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT file_path, chunk_text,
                       1 - (embedding <=> %s::vector) AS cosine_sim
                FROM {self.table}
                WHERE embedding IS NOT NULL {project_filter}
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s
                """,
                params,
            )
            rows = cur.fetchall()

        results = []
        for r in rows:
            score = float(r["cosine_sim"])
            if score < min_relevance:
                continue
            if return_scores:
                results.append((r["file_path"], r["chunk_text"], score))
            else:
                results.append((r["file_path"], r["chunk_text"]))
            if len(results) >= k:
                break

        logger.debug(
            "PgVectorStore.query: %d results for %r (min_relevance=%.2f)",
            len(results), query, min_relevance,
        )
        return results

    # ── Hybrid search (BM25 + pgvector RRF) ──────────────────────────────────

    def hybrid_query(
        self,
        query: str,
        k: int = 5,
        rrf_k: int = 60,
    ) -> List[Tuple[str, str, float]]:
        """
        Hybrid BM25 + pgvector search with Reciprocal Rank Fusion.

        Fetches a candidate pool from Postgres (up to k*5 rows), scores
        them with both BM25 (in-process) and pgvector cosine similarity,
        and fuses the ranked lists using RRF.

        Requires rank_bm25: pip install rank-bm25

        Returns (file_path, snippet, rrf_score) tuples.
        """
        # Vector leg
        vec_results = self.query(query, k=k * 5, min_relevance=0.0, return_scores=True)

        # BM25 leg over candidate pool
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed — returning vector-only results")
            return [(fp, sn, sc) for fp, sn, sc in vec_results[:k]]

        # Build BM25 index over the candidate pool
        corpus_tokens = [r[1].lower().split() for r in vec_results]
        bm25 = BM25Okapi(corpus_tokens)
        bm25_scores = bm25.get_scores(query.lower().split())

        # RRF fusion
        vec_ranks = {i: rank for rank, i in enumerate(range(len(vec_results)))}
        bm25_ranks = {
            idx: rank
            for rank, idx in enumerate(
                sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])
            )
        }

        rrf_scores: dict[int, float] = {}
        for i in range(len(vec_results)):
            rrf_scores[i] = (
                1.0 / (rrf_k + vec_ranks.get(i, len(vec_results)))
                + 1.0 / (rrf_k + bm25_ranks.get(i, len(vec_results)))
            )

        top_k = sorted(rrf_scores, key=lambda i: -rrf_scores[i])[:k]
        return [
            (vec_results[i][0], vec_results[i][1], rrf_scores[i])
            for i in top_k
        ]

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all indexed chunks."""
        self._ensure_conn()
        with self._conn.cursor() as cur:
            cur.execute(f"TRUNCATE {self.table}")
        logger.info("PgVectorStore: cleared all chunks from %s", self.table)

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
