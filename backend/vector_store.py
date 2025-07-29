import os
import json
from typing import List, Tuple
import numpy as np
import faiss
# type: ignore
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer

# Set up logging for vector store operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Local MiniLM encoder (loads once)
_vs_embedder = SentenceTransformer("all-MiniLM-L6-v2")

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
            self.index = faiss.IndexFlatL2(768)

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

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        # encode locally in batches of 256
        embeddings: List[List[float]] = []
        batch_size = 256
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                vecs = _vs_embedder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
                embeddings.extend(vecs.tolist())
            except Exception as e:
                logger.error(f"Local embedding error: {e}")
                embeddings.extend([[0.0] * 384] * len(batch))
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
            self.index = faiss.IndexFlatL2(768)
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
            self.index = faiss.IndexFlatL2(768)
            self.meta = []
            self._index_mtime = None
            return

        if self._index_mtime is None or current_mtime != self._index_mtime:
            logger.info("Detected updated vector index on disk – reloading")
            self._load()

    def _calculate_relevance_score(self, distance: float, file_path: str, query: str) -> float:
        """Calculate relevance score combining distance with contextual factors"""
        # Start with similarity score (lower distance = higher similarity)
        # Use a more generous normalization to avoid filtering out good results
        base_score = max(0.1, 1.0 - (distance / 3.0))  # More generous normalization
        
        # Enhanced project name boosting
        project_boost = 0.0
        path_parts = file_path.split('/')
        query_lower = query.lower()
        
        if len(path_parts) > 0:
            project_name = path_parts[0].lower()
            
            # Multiple ways to match project names
            if (project_name in query_lower or 
                query_lower in project_name or
                any(word in project_name for word in query_lower.split() if len(word) > 2)):
                project_boost = 0.15  # Increased boost
                logger.debug(f"Project boost applied for {project_name} in query: {query}")
        
        # Enhanced file type boosting
        file_type_boost = 0.0
        if file_path.endswith(('.py', '.js', '.ts', '.tsx', '.jsx')):
            file_type_boost = 0.08  # Increased for code files
        elif file_path.endswith(('.md', '.txt', '.rst')):
            file_type_boost = 0.05  # Increased for documentation
        elif file_path.endswith(('.json', '.yaml', '.yml', '.toml')):
            file_type_boost = 0.03  # Config files
        
        # Additional context-based boosting
        context_boost = 0.0
        
        # Boost for README files when asking general questions
        if 'readme' in file_path.lower() and any(word in query_lower for word in ['what', 'how', 'explain', 'describe']):
            context_boost += 0.1
            
        # Boost for main/index files
        if any(name in file_path.lower() for name in ['main.', 'index.', 'app.', '__init__']):
            context_boost += 0.05
        
        # Combine scores
        final_score = base_score + project_boost + file_type_boost + context_boost
        
        logger.debug(f"Relevance score for {file_path}: {final_score:.3f} "
                    f"(base: {base_score:.3f}, project: {project_boost:.3f}, "
                    f"file: {file_type_boost:.3f}, context: {context_boost:.3f})")
        
        return min(1.0, final_score)  # Cap at 1.0

    # --------------------------- public API ---------------------------
    def query(self, query: str, k: int = 3, min_relevance: float = 0.3) -> List[Tuple[str, str]]:
        """Query the vector store with enhanced relevance scoring and adaptive filtering"""
        # Ensure we are using the latest index built by the indexer
        self._refresh_if_updated()

        logger.info(f"Vector search query: '{query}' (k={k}, min_relevance={min_relevance})")
        
        if self.index is None or not self.meta:
            logger.warning("No vector index available for search")
            return []
        
        try:
            # Generate embedding for query
            emb_vec = np.array(self._embed_batch([query])[0]).astype("float32").reshape(1, -1)
            
            # Search for more candidates than requested to allow for filtering
            search_k = min(k * 5, len(self.meta))  # Search 5x more candidates for better filtering
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
                    
                    relevance_score = self._calculate_relevance_score(distance, file_path, query)
                    all_scored_results.append((relevance_score, file_path, snippet))
            
            # Sort by relevance score (highest first)
            all_scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # Adaptive threshold logic
            current_threshold = min_relevance
            scored_results = [r for r in all_scored_results if r[0] >= current_threshold]
            
            # If no results meet the threshold, progressively lower it
            if not scored_results and all_scored_results:
                logger.info(f"No results found with threshold {current_threshold:.3f}, trying adaptive thresholds")
                
                # Try progressively lower thresholds
                adaptive_thresholds = [min_relevance * 0.8, min_relevance * 0.6, min_relevance * 0.4, 0.1]
                
                for threshold in adaptive_thresholds:
                    scored_results = [r for r in all_scored_results if r[0] >= threshold]
                    if scored_results:
                        logger.info(f"Found {len(scored_results)} results with adaptive threshold {threshold:.3f}")
                        current_threshold = threshold
                        break
                
                # If still no results, take the top results regardless of threshold
                if not scored_results and all_scored_results:
                    scored_results = all_scored_results[:k]
                    current_threshold = all_scored_results[-1][0] if scored_results else 0.0
                    logger.info(f"Using top {len(scored_results)} results with minimum threshold {current_threshold:.3f}")
            
            # Return top k results
            final_results = [(path, snippet) for _, path, snippet in scored_results[:k]]
            
            logger.info(f"Vector search returned {len(final_results)} results from {len(all_scored_results)} candidates "
                       f"(threshold: {current_threshold:.3f})")
            
            # Log details about top results with enhanced information
            for i, (score, path, snippet) in enumerate(scored_results[:min(k, 3)]):  # Log top 3
                logger.info(f"Result {i+1}: {path} (score: {score:.3f}, "
                           f"snippet: {len(snippet)} chars, distance: {distances[0][i]:.3f})")
            
            # Log threshold adaptation if it occurred
            if current_threshold != min_relevance:
                logger.info(f"Threshold adapted from {min_relevance:.3f} to {current_threshold:.3f}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
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
            self.index = faiss.IndexFlatL2(768)
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
                    self.index = faiss.IndexFlatL2(768)
                
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
        self.index = faiss.IndexFlatL2(768)
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