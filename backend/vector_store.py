import os
import json
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import faiss
# type: ignore
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer

# Set up logging for vector store operations
logger = logging.getLogger(__name__)

# REQ-CACHE-6: BGE Embedding Cache Integration
try:
    from cache.embedding_cache import get_global_embedding_cache
except ImportError:
    # Graceful fallback if cache module is not available
    logger.warning("BGE Embedding Cache not available - running without cache")
    def get_global_embedding_cache():
        return None


# BGE encoder for high-quality embeddings (REQ-1.3.1)
_vs_embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")

# REQ-CACHE-6: Global embedding cache instance
_embedding_cache = None

WORKSPACE_DIR = "/workspace"
CACHE_DIR = Path(WORKSPACE_DIR) / ".vector_cache"
INDEX_FILE = CACHE_DIR / "index.faiss"
META_FILE = CACHE_DIR / "meta.json"
CHUNK_SIZE = 400  # characters per chunk
EMBED_MODEL = "text-embedding-3-small"


class VectorStore:
    """Simple FAISS-backed vector store for workspace code retrieval."""

    def __init__(self, workspace_dir: str = "/workspace"):
        self.workspace_dir = workspace_dir
        self.index_path = INDEX_FILE
        self.meta_path = META_FILE
        self.index = None  # type: ignore
        self.meta: List = []  # Can be tuples or dicts depending on format
        self._index_mtime = None  # Track last modified time of the loaded index

        if self.index_path.exists() and self.meta_path.exists():
            logger.info("Loading existing vector index from cache")
            self._load()
        else:
            # If index not present, fallback to empty index (indexer builds asynchronously)
            logger.warning("No vector index found, creating empty index")
            self.index = faiss.IndexFlatL2(1024)  # BGE uses 1024 dimensions

    # --------------------------- internal helpers ---------------------------
    def _chunk_text(self, text: str) -> List[str]:
        return [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    def _list_files(self) -> List[Path]:
        paths = []
        for root, _, files in os.walk(self.workspace_dir):
            for f in files:
                if f.startswith("."):
                    continue
                if any(f.endswith(ext) for ext in [
                    ".py",
                    ".js",
                    ".ts",
                    ".tsx",
                    ".jsx",
                    ".md",
                    ".txt",
                    ".json",
                    ".html",
                    ".css",
                ]):
                    paths.append(Path(root) / f)
        return paths

    def _embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Generate embeddings with cache integration (REQ-CACHE-6).
        
        Args:
            texts: List of text strings to embed
            is_query: Whether these are query embeddings (vs document embeddings)
        """
        # REQ-CACHE-6: Initialize embedding cache if needed
        global _embedding_cache
        if _embedding_cache is None:
            try:
                _embedding_cache = get_global_embedding_cache()
                logger.info("BGE Embedding Cache initialized for vector store")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding cache: {e}")
                _embedding_cache = None
        
        embeddings: List[List[float]] = []
        cache_hits = 0
        cache_misses = 0
        
        # REQ-CACHE-6: Check cache for each text if cache is available
        uncached_texts = []
        uncached_indices = []
        
        if _embedding_cache:
            for i, text in enumerate(texts):
                cached_embedding = _embedding_cache.get(text, is_query=is_query)
                if cached_embedding is not None:
                    embeddings.append(cached_embedding.tolist())
                    cache_hits += 1
                else:
                    embeddings.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    cache_misses += 1
        else:
            # No cache available, process all texts
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            embeddings = [None] * len(texts)
        
        if cache_hits > 0:
            hit_rate = cache_hits/(cache_hits+cache_misses)*100
            logger.info(f"BGE Cache: {cache_hits} hits, {cache_misses} misses ({hit_rate:.1f}% hit rate)")
            
            # Report batch cache performance to metrics
            if _embedding_cache and _embedding_cache.global_metrics:
                layer_name = 'bge_embeddings_query' if is_query else 'bge_embeddings_document'
                # Record overall batch performance
                avg_time_saved = 50 * (cache_hits / max(1, cache_hits + cache_misses))
                _embedding_cache.global_metrics.record_cache_hit(
                    f"{layer_name}_batch", 
                    time_saved_ms=avg_time_saved,
                    response_time_ms=1.0
                )
        
        # Process uncached texts in batches
        if uncached_texts:
            batch_size = 100  # Reduced from 256 to prevent memory exhaustion
            total_batches = (len(uncached_texts) + batch_size - 1) // batch_size
            
            logger.info(f"Processing {len(uncached_texts)} uncached texts in {total_batches} batches of {batch_size}")
            
            uncached_embeddings = []
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                
                try:
                    logger.debug(f"Processing embedding batch {batch_num}/{total_batches}")
                    vecs = _vs_embedder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
                    uncached_embeddings.extend(vecs.tolist())
                    
                    # REQ-CACHE-6: Cache the newly generated embeddings
                    if _embedding_cache:
                        try:
                            batch_arrays = [np.array(vec) for vec in vecs.tolist()]
                            _embedding_cache.set_batch(batch, batch_arrays, is_query=is_query)
                        except Exception as cache_error:
                            logger.warning(f"Failed to cache batch {batch_num}: {cache_error}")
                    
                    # Small delay to prevent memory pressure
                    import time
                    time.sleep(0.05)
                    
                except Exception as e:
                    logger.error(f"Batch {batch_num} embedding error: {e}")
                    uncached_embeddings.extend([[0.0] * 1024] * len(batch))  # BGE uses 1024 dimensions
            
            # Fill in the uncached embeddings at their original positions
            for idx, embedding in zip(uncached_indices, uncached_embeddings):
                embeddings[idx] = embedding
        
        logger.info(f"Successfully generated {len(embeddings)} embeddings (cache: {cache_hits} hits, {cache_misses} misses)")
        return embeddings

    def _build(self):
        logger.info("Starting vector index build process")
        texts = []
        self.meta = []
        
        files_processed = 0
        total_chunks = 0
        
        for path in self._list_files():
            try:
                content = Path(path).read_text(encoding="utf-8", errors="ignore")
                chunks = self._chunk_text(content)
                
                if chunks:
                    files_processed += 1
                    file_chunks = len(chunks)
                    total_chunks += file_chunks
                    
                    rel_path = str(Path(path).relative_to(self.workspace_dir))
                    logger.debug(f"Processing {rel_path}: {file_chunks} chunks, {len(content)} chars")
                    
                    for chunk in chunks:
                        texts.append(chunk)
                        self.meta.append((rel_path, chunk))
                        
            except Exception as e:
                logger.warning(f"Failed to process file {path}: {e}")
                continue

        logger.info(f"Index build complete: {files_processed} files processed, {total_chunks} chunks created")

        if not texts:
            logger.warning("No text content found for indexing - creating empty index")
            self.index = faiss.IndexFlatL2(1024)  # BGE uses 1024 dimensions
            return

        logger.info(f"Generating embeddings for {len(texts)} text chunks")
        embeddings = np.array(self._embed_batch(texts)).astype("float32")
        dim = embeddings.shape[1]
        
        logger.info(f"Creating FAISS index with dimension {dim}")
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        self.index = index
        
        logger.info(f"Saving index to {self.index_path}")
        faiss.write_index(index, str(self.index_path))
        
        logger.info(f"Saving metadata to {self.meta_path}")
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f)
        
        logger.info("Vector index build and save completed successfully")

    def _load(self):
        self.index = faiss.read_index(str(self.index_path))
        # Record the mtime we just loaded
        try:
            self._index_mtime = self.index_path.stat().st_mtime
        except Exception:
            self._index_mtime = None
        with open(self.meta_path, "r", encoding="utf-8") as f:
            raw_meta = json.load(f)
            
            # Handle both old format (tuples) and new format (dicts)
            if raw_meta and isinstance(raw_meta[0], dict):
                # New enhanced metadata format - keep as dicts for richer information
                self.meta = raw_meta
                logger.info(f"Loaded enhanced metadata with {len(self.meta)} chunks")
            else:
                # Old simple format - convert to dict format for consistency
                self.meta = []
                for item in raw_meta:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        self.meta.append({
                            "relative_path": item[0],
                            "chunk_text": item[1],
                            "file_type": "unknown",
                            "chunk_type": "legacy"
                        })
                    else:
                        logger.warning(f"Skipping invalid metadata item: {item}")
                logger.info(f"Converted legacy metadata to enhanced format with {len(self.meta)} chunks")
        
        logger.info(f"Loaded vector index with {len(self.meta)} chunks from {self.index.ntotal} embeddings")

    def _refresh_if_updated(self):
        """Reload the FAISS index & metadata if the on-disk file has changed."""
        try:
            current_mtime = self.index_path.stat().st_mtime
        except FileNotFoundError:
            logger.warning("Index file not found; using empty index")
            self.index = faiss.IndexFlatL2(1024)  # BGE uses 1024 dimensions
            self.meta = []
            self._index_mtime = None
            return

        if self._index_mtime is None or current_mtime != self._index_mtime:
            logger.info("Detected updated vector index on disk – reloading")
            self._load()
    
    def force_refresh(self):
        """Force reload the index from disk regardless of mtime"""
        if self.index_path.exists() and self.meta_path.exists():
            logger.info("Force refreshing vector index from disk")
            self._load()
        else:
            logger.warning("Cannot force refresh - index files do not exist")

    def _calculate_relevance_score(self, distance: float, file_path: str, query: str) -> float:
        """Calculate relevance score combining distance with contextual factors (optimized)"""
        # Start with similarity score (lower distance = higher similarity)
        base_score = max(0.1, 1.0 - (distance / 3.0))
        
        # Simplified boosting for performance
        boost = 0.0
        query_lower = query.lower()
        file_lower = file_path.lower()
        
        # Quick project name boost
        path_parts = file_path.split('/')
        if len(path_parts) > 0:
            project_name = path_parts[0].lower()
            if project_name in query_lower or query_lower in project_name:
                boost += 0.15
        
        # Quick file type boost
        if file_path.endswith(('.py', '.js', '.ts', '.tsx', '.jsx')):
            boost += 0.08
        elif file_path.endswith(('.md', '.txt', '.rst')):
            boost += 0.05
        
        # Quick context boost
        if 'readme' in file_lower and any(word in query_lower for word in ['what', 'how', 'explain']):
            boost += 0.1
        
        final_score = base_score + boost
        return min(1.0, final_score)

    # --------------------------- public API ---------------------------
    def query(
        self,
        query: str,
        k: int = 3,
        min_relevance: float = 0.3,
        allowed_projects: Optional[List[str]] = None,
        return_scores: bool = None,
    ) -> List[Tuple[str, str]] | List[Tuple[str, str, float]]:
        """Query the vector store with enhanced relevance scoring and adaptive filtering.

        If allowed_projects is provided, only return results whose file paths belong to those
        projects. This restriction is applied BEFORE scoring/logging to avoid cross-project noise.
        
        Args:
            query: Search query string
            k: Maximum number of results to return
            min_relevance: Minimum relevance score threshold
            allowed_projects: Optional list of project names to filter results
            return_scores: If True, return (path, snippet, score) tuples. If None, uses 
                          CODEWISE_VECTOR_RETURN_SCORES environment variable. Defaults to False.
        
        Returns:
            List of (path, snippet) tuples if return_scores=False, or
            List of (path, snippet, score) tuples if return_scores=True
        """
        # Determine whether to return scores based on parameter or environment variable
        if return_scores is None:
            return_scores = os.getenv("CODEWISE_VECTOR_RETURN_SCORES", "false").lower() == "true"
        
        # Ensure we are using the latest index built by the indexer
        self._refresh_if_updated()

        logger.debug(f"Vector search query: '{query}' (k={k}, min_relevance={min_relevance})")
        
        if self.index is None or not self.meta:
            logger.warning("No vector index available for search")
            return []
        
        try:
            # Generate embedding for query (REQ-CACHE-6: Mark as query embedding)
            emb_vec = np.array(self._embed_batch([query], is_query=True)[0]).astype("float32").reshape(1, -1)
            
            # Search for more candidates than requested to allow for in-project filtering
            if allowed_projects:
                # Be more generous to compensate for cross-project pruning
                search_k = min(max(k * 8, k), len(self.meta))
            else:
                # Search 2x more candidates (reduced from 5x for performance)
                search_k = min(k * 2, len(self.meta))
            distances, indices = self.index.search(emb_vec, search_k)
            
            # Calculate relevance scores for all candidates
            all_scored_results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.meta):
                    # Handle both tuple and dict metadata formats
                    meta_item = self.meta[idx]
                    if isinstance(meta_item, tuple):
                        # Old format: (file_path, snippet)
                        file_path, snippet = meta_item
                    else:
                        # New enhanced format: dict
                        file_path = meta_item.get("relative_path", meta_item.get("file_path", "unknown"))
                        snippet = meta_item.get("chunk_text", "")
                    
                    # Enforce project scoping early
                    if allowed_projects:
                        try:
                            project_dir = file_path.split('/')[0]
                        except Exception:
                            project_dir = ""
                        if project_dir not in set(allowed_projects):
                            continue
                    
                    relevance_score = self._calculate_relevance_score(distance, file_path, query)
                    all_scored_results.append((relevance_score, file_path, snippet))
            
            # Sort by relevance score (highest first)
            all_scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # Adaptive threshold logic
            current_threshold = min_relevance
            scored_results = [r for r in all_scored_results if r[0] >= current_threshold]
            
            # Simplified adaptive threshold logic for performance
            if not scored_results and all_scored_results:
                logger.debug(f"No results found with threshold {current_threshold:.3f}, using adaptive threshold")
                
                # Single adaptive threshold instead of multiple iterations
                adaptive_threshold = min_relevance * 0.5
                scored_results = [r for r in all_scored_results if r[0] >= adaptive_threshold]
                
                # If still no results, take the top results regardless of threshold
                if not scored_results:
                    scored_results = all_scored_results[:k]
                    current_threshold = all_scored_results[-1][0] if scored_results else 0.0
                else:
                    current_threshold = adaptive_threshold
            
            # Return top k results with or without scores based on feature flag
            if return_scores:
                final_results = [(path, snippet, score) for score, path, snippet in scored_results[:k]]
            else:
                final_results = [(path, snippet) for _, path, snippet in scored_results[:k]]
            
            logger.debug(f"Vector search returned {len(final_results)} results from {len(all_scored_results)} candidates "
                       f"(threshold: {current_threshold:.3f})")
            
            # Log details about top results with enhanced information (in-scope only)
            for i, (score, path, snippet) in enumerate(scored_results[:min(k, 3)]):  # Log top 3
                try:
                    dist_val = distances[0][i]
                except Exception:
                    dist_val = 0.0
                logger.info(
                    (
                        f"Result {i+1}: {path} (score: {score:.3f}, "
                        f"snippet: {len(snippet)} chars, distance: {dist_val:.3f})"
                    )
                )
            
            # Log threshold adaptation if it occurred
            if current_threshold != min_relevance:
                logger.info(f"Threshold adapted from {min_relevance:.3f} to {current_threshold:.3f}")
            
            return final_results
            
        except Exception as e:
            import traceback
            logger.error(f"Error during vector search: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []
    
    async def similarity_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Compatibility method for similarity search.
        
        This method provides compatibility with unified_query_pure which expects
        a similarity_search method. It wraps the existing query method.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of dictionaries with search results
        """
        try:
            # Use existing query method with scores enabled
            raw_results = self.query(query, k=k, return_scores=True)
            
            # Convert to expected format for unified_query_pure
            formatted_results = []
            for result in raw_results:
                if len(result) >= 3:  # (path, snippet, score)
                    path, content, score = result[0], result[1], result[2]
                else:  # (path, snippet) - fallback
                    path, content = result[0], result[1]
                    score = 0.5  # Default score
                
                formatted_results.append({
                    "content": content,
                    "file_path": path,
                    "score": float(score),
                    "metadata": {
                        "source": "vector_store",
                        "method": "similarity_search"
                    }
                })
            
            logger.debug(f"similarity_search returned {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"similarity_search failed: {e}")
            return []

    def remove_project_embeddings(self, project_path: str) -> bool:
        """Remove all embeddings for files in a specific project path."""
        if not self.meta or not self.index:
            return True  # Nothing to remove
        
        # Find indices of embeddings to remove
        indices_to_remove = []
        for i, meta_item in enumerate(self.meta):
            # Handle both tuple and dict formats
            if isinstance(meta_item, tuple):
                file_path = meta_item[0]
            else:
                file_path = meta_item.get("relative_path", meta_item.get("file_path", ""))
            
            # Check if file belongs to the project being deleted
            if file_path.startswith(project_path + "/") or file_path == project_path:
                indices_to_remove.append(i)
        
        if not indices_to_remove:
            return True  # No embeddings found for this project
        
        # Create new meta list and embeddings without the project's data
        new_meta = []
        embeddings_to_keep = []
        
        for i, meta_item in enumerate(self.meta):
            if i not in indices_to_remove:
                new_meta.append(meta_item)
                embeddings_to_keep.append(i)
        
        if not new_meta:
            # No embeddings left, create empty index
            self.index = faiss.IndexFlatL2(1024)  # BGE uses 1024 dimensions
            self.meta = []
        else:
            # Rebuild index with remaining embeddings
            try:
                # Read current index to get all embeddings
                current_embeddings = []
                for i in range(self.index.ntotal):
                    if i in embeddings_to_keep:
                        vec = self.index.reconstruct(embeddings_to_keep.index(i))
                        current_embeddings.append(vec)
                
                if current_embeddings:
                    embeddings_array = np.array(current_embeddings).astype("float32")
                    new_index = faiss.IndexFlatL2(embeddings_array.shape[1])
                    new_index.add(embeddings_array)
                    self.index = new_index
                else:
                    self.index = faiss.IndexFlatL2(1024)  # BGE uses 1024 dimensions
                
                self.meta = new_meta
            except Exception:
                # If reconstruction fails, rebuild from scratch
                self._rebuild_index()
                return True
        
        # Save updated index and metadata
        try:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.meta, f)
            return True
        except Exception:
            return False
    
    def _rebuild_index(self):
        """Rebuild the entire index from workspace files (fallback method)."""
        # Clear current state
        self.index = None
        self.meta = []
        
        # Rebuild from scratch
        self._build()

    def clear_all_embeddings(self):
        """Clear all embeddings and rebuild empty index."""
        self.index = faiss.IndexFlatL2(384)  # Fixed: MiniLM uses 384 dimensions, not 768
        self.meta = []
        
        # Save empty state
        try:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.meta, f)
        except Exception:
            pass

# Singleton for agent
_vector_store_instance: VectorStore | None = None

def get_vector_store() -> VectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance 