"""
BGE Embedding Cache

High-performance cache for BGE embedding vectors to improve vector search performance.
Eliminates redundant embedding computations for identical content while ensuring
consistency with model versions and normalization settings.

Architectural Design:
- Content hash + model version + settings based cache keys
- Persistent HDF5/NumPy storage for efficient vector operations
- Batch operations for optimal performance
- Cache warming for project initialization
- Memory-efficient vector storage with compression
"""

import os
import json
import hashlib
import logging
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import tempfile
import fcntl
import time

# Integration with global cache metrics
try:
    from .cache_metrics import get_global_cache_metrics
except ImportError:
    # Fallback for when module is imported directly
    from cache.cache_metrics import get_global_cache_metrics

logger = logging.getLogger(__name__)

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    logger.warning("h5py not available, falling back to NumPy storage")
    HDF5_AVAILABLE = False


@dataclass
class EmbeddingCacheEntry:
    """Cache entry for embedding vector with metadata"""
    embedding: np.ndarray
    content_hash: str
    model_version: str
    is_query: bool
    timestamp: float
    content_length: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize entry metadata (embedding stored separately)"""
        return {
            'content_hash': self.content_hash,
            'model_version': self.model_version,
            'is_query': self.is_query,
            'timestamp': self.timestamp,
            'content_length': self.content_length,
            'embedding_shape': list(self.embedding.shape),
            'embedding_dtype': str(self.embedding.dtype)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: np.ndarray) -> 'EmbeddingCacheEntry':
        """Deserialize entry from metadata dict and embedding array"""
        return cls(
            embedding=embedding,
            content_hash=data['content_hash'],
            model_version=data['model_version'],
            is_query=data['is_query'],
            timestamp=data['timestamp'],
            content_length=data['content_length']
        )


class BGEEmbeddingCache:
    """
    High-performance cache for BGE embedding vectors.
    
    This cache optimizes expensive BGE embedding computations by storing
    vectors with content-based keys that include model version and settings.
    Uses efficient storage formats (HDF5 preferred, NumPy fallback) for
    vector operations.
    
    Key Features:
    - Content hash + model version + settings cache keys
    - Batch operations for optimal performance  
    - Persistent storage with corruption protection
    - Memory-efficient vector storage
    - Cache warming for project initialization
    """
    
    def __init__(self, cache_dir: str = "/workspace/.vector_cache", 
                 model_version: str = "BAAI/bge-large-en-v1.5",
                 embedding_dim: int = 1024,
                 max_cache_size_gb: float = 2.0):
        """
        Initialize BGE embedding cache.
        
        Args:
            cache_dir: Directory for cache storage
            model_version: BGE model version identifier
            embedding_dim: Embedding vector dimension
            max_cache_size_gb: Maximum cache size in GB
        """
        self.cache_dir = Path(cache_dir)
        self.embedding_cache_dir = self.cache_dir / "embeddings"
        self.model_version = model_version
        self.embedding_dim = embedding_dim
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024 * 1024 * 1024)
        
        # Cache files
        self.metadata_file = self.embedding_cache_dir / "metadata.json"
        self.embeddings_file = self.embedding_cache_dir / "embeddings.h5" if HDF5_AVAILABLE else self.embedding_cache_dir / "embeddings.npz"
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.cache_saves = 0
        self.cache_loads = 0
        
        # Integration with global cache metrics system
        try:
            self.global_metrics = get_global_cache_metrics()
        except Exception as e:
            logger.warning(f"Could not initialize global cache metrics: {e}")
            self.global_metrics = None
        
        # In-memory cache for frequently accessed embeddings
        self._memory_cache: Dict[str, EmbeddingCacheEntry] = {}
        self._memory_cache_max_size = 1000  # Keep 1000 most recent embeddings in memory
        
        # Ensure cache directory exists
        self._ensure_cache_directory()
        
        # Load existing cache
        self._load_cache_metadata()
        
        logger.info(f"BGEEmbeddingCache initialized: {self.embedding_cache_dir}")
        logger.info(f"Model: {self.model_version}, Dimension: {self.embedding_dim}")
        logger.info(f"Storage: {'HDF5' if HDF5_AVAILABLE else 'NumPy'}")
    
    def _ensure_cache_directory(self) -> None:
        """Ensure cache directory exists with proper permissions"""
        try:
            self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Embedding cache directory ensured: {self.embedding_cache_dir}")
        except Exception as e:
            logger.error(f"Failed to create embedding cache directory {self.embedding_cache_dir}: {e}")
            raise
    
    def _load_cache_metadata(self) -> None:
        """Load cache metadata from persistent storage"""
        if not self.metadata_file.exists():
            return
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    metadata = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            logger.debug(f"Loaded embedding cache metadata: {len(metadata)} entries")
            self.cache_loads += 1
            
        except Exception as e:
            logger.error(f"Failed to load embedding cache metadata: {e}")
    
    def get_embedding_key(self, content: str, is_query: bool = False) -> str:
        """
        Generate cache key for content + model settings.
        
        Args:
            content: Text content to embed
            is_query: Whether this is a query embedding (vs document)
            
        Returns:
            Deterministic cache key
        """
        # Normalize content for consistent hashing
        normalized_content = content.strip()
        content_hash = hashlib.sha256(normalized_content.encode('utf-8')).hexdigest()
        
        # Include model version and embedding type in key
        settings = f"{self.model_version}:{'query' if is_query else 'doc'}:normalized"
        cache_key = f"{content_hash}:{settings}"
        
        return hashlib.sha256(cache_key.encode('utf-8')).hexdigest()
    
    def get(self, content: str, is_query: bool = False) -> Optional[np.ndarray]:
        """
        Retrieve cached embedding for content.
        
        Args:
            content: Text content
            is_query: Whether this is a query embedding
            
        Returns:
            Cached embedding vector or None if cache miss
        """
        with self._lock:
            try:
                cache_key = self.get_embedding_key(content, is_query)
                
                # Check in-memory cache first
                if cache_key in self._memory_cache:
                    entry = self._memory_cache[cache_key]
                    self.hits += 1
                    
                    # Report to global metrics
                    if self.global_metrics:
                        layer_name = 'bge_embeddings_query' if is_query else 'bge_embeddings_document'
                        self.global_metrics.record_cache_hit(layer_name, time_saved_ms=0, response_time_ms=0.1)
                    
                    logger.debug(f"Memory cache hit for content ({len(content)} chars)")
                    return entry.embedding.copy()
                
                # Check persistent storage
                embedding = self._load_embedding_from_storage(cache_key)
                if embedding is not None:
                    # Add to memory cache
                    self._add_to_memory_cache(cache_key, EmbeddingCacheEntry(
                        embedding=embedding,
                        content_hash=hashlib.sha256(content.encode()).hexdigest(),
                        model_version=self.model_version,
                        is_query=is_query,
                        timestamp=time.time(),
                        content_length=len(content)
                    ))
                    
                    self.hits += 1
                    
                    # Report to global metrics
                    if self.global_metrics:
                        layer_name = 'bge_embeddings_query' if is_query else 'bge_embeddings_document'
                        self.global_metrics.record_cache_hit(layer_name, time_saved_ms=50, response_time_ms=1.0)
                    
                    logger.debug(f"Persistent cache hit for content ({len(content)} chars)")
                    return embedding.copy()
                
                # Cache miss
                self.misses += 1
                
                # Report to global metrics
                if self.global_metrics:
                    layer_name = 'bge_embeddings_query' if is_query else 'bge_embeddings_document'
                    self.global_metrics.record_cache_miss(layer_name, response_time_ms=0)
                
                logger.debug(f"Cache miss for content ({len(content)} chars)")
                return None
                
            except Exception as e:
                logger.error(f"Error during embedding cache get: {e}")
                self.misses += 1
                return None
    
    def set(self, content: str, embedding: np.ndarray, is_query: bool = False) -> None:
        """
        Store embedding in cache.
        
        Args:
            content: Text content
            embedding: Embedding vector
            is_query: Whether this is a query embedding
        """
        with self._lock:
            try:
                cache_key = self.get_embedding_key(content, is_query)
                
                # Create cache entry
                entry = EmbeddingCacheEntry(
                    embedding=embedding.copy(),
                    content_hash=hashlib.sha256(content.encode()).hexdigest(),
                    model_version=self.model_version,
                    is_query=is_query,
                    timestamp=time.time(),
                    content_length=len(content)
                )
                
                # Add to memory cache
                self._add_to_memory_cache(cache_key, entry)
                
                # Save to persistent storage
                self._save_embedding_to_storage(cache_key, entry)
                
                logger.debug(f"Cached embedding for content ({len(content)} chars)")
                
            except Exception as e:
                logger.error(f"Error during embedding cache set: {e}")
    
    def get_batch(self, contents: List[str], is_query: bool = False) -> Tuple[List[Optional[np.ndarray]], List[str]]:
        """
        Get cached embeddings for batch of contents.
        
        Args:
            contents: List of text contents
            is_query: Whether these are query embeddings
            
        Returns:
            Tuple of (cached_embeddings, cache_miss_contents)
            cached_embeddings contains None for cache misses
        """
        with self._lock:
            cached_embeddings = []
            cache_miss_contents = []
            
            for content in contents:
                embedding = self.get(content, is_query)
                cached_embeddings.append(embedding)
                
                if embedding is None:
                    cache_miss_contents.append(content)
            
            logger.debug(f"Batch cache: {len(contents)} requested, {len(cache_miss_contents)} misses")
            return cached_embeddings, cache_miss_contents
    
    def set_batch(self, contents: List[str], embeddings: List[np.ndarray], is_query: bool = False) -> None:
        """
        Store batch of embeddings efficiently.
        
        Args:
            contents: List of text contents
            embeddings: List of embedding vectors
            is_query: Whether these are query embeddings
        """
        if len(contents) != len(embeddings):
            raise ValueError(f"Contents and embeddings length mismatch: {len(contents)} vs {len(embeddings)}")
        
        with self._lock:
            for content, embedding in zip(contents, embeddings):
                self.set(content, embedding, is_query)
            
            logger.debug(f"Batch cached {len(contents)} embeddings")
    
    def _add_to_memory_cache(self, cache_key: str, entry: EmbeddingCacheEntry) -> None:
        """Add entry to in-memory cache with LRU eviction"""
        self._memory_cache[cache_key] = entry
        
        # Evict oldest entries if cache is full
        if len(self._memory_cache) > self._memory_cache_max_size:
            # Remove oldest 10% of entries
            remove_count = max(1, len(self._memory_cache) // 10)
            oldest_keys = sorted(self._memory_cache.keys(), 
                               key=lambda k: self._memory_cache[k].timestamp)[:remove_count]
            
            for key in oldest_keys:
                del self._memory_cache[key]
            
            logger.debug(f"Memory cache evicted {remove_count} oldest entries")
    
    def _load_embedding_from_storage(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from persistent storage"""
        try:
            if HDF5_AVAILABLE and self.embeddings_file.exists():
                return self._load_from_hdf5(cache_key)
            elif self.embeddings_file.exists():
                return self._load_from_npz(cache_key)
            return None
        except Exception as e:
            logger.error(f"Failed to load embedding from storage: {e}")
            return None
    
    def _save_embedding_to_storage(self, cache_key: str, entry: EmbeddingCacheEntry) -> None:
        """Save embedding to persistent storage"""
        try:
            if HDF5_AVAILABLE:
                self._save_to_hdf5(cache_key, entry)
            else:
                self._save_to_npz(cache_key, entry)
            
            self._save_metadata(cache_key, entry)
            self.cache_saves += 1
            
        except Exception as e:
            logger.error(f"Failed to save embedding to storage: {e}")
    
    def _load_from_hdf5(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from HDF5 file"""
        try:
            with h5py.File(self.embeddings_file, 'r') as f:
                if cache_key in f:
                    return f[cache_key][:]
            return None
        except Exception as e:
            logger.debug(f"HDF5 load failed: {e}")
            return None
    
    def _save_to_hdf5(self, cache_key: str, entry: EmbeddingCacheEntry) -> None:
        """Save embedding to HDF5 file"""
        # Create file if it doesn't exist
        if not self.embeddings_file.exists():
            with h5py.File(self.embeddings_file, 'w') as f:
                pass
        
        with h5py.File(self.embeddings_file, 'a') as f:
            if cache_key in f:
                del f[cache_key]  # Replace existing
            
            # Store with compression
            f.create_dataset(cache_key, data=entry.embedding, 
                           compression='gzip', compression_opts=1)
    
    def _load_from_npz(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from NPZ file"""
        try:
            with np.load(self.embeddings_file, allow_pickle=False) as f:
                if cache_key in f:
                    return f[cache_key]
            return None
        except Exception as e:
            logger.debug(f"NPZ load failed: {e}")
            return None
    
    def _save_to_npz(self, cache_key: str, entry: EmbeddingCacheEntry) -> None:
        """Save embedding to NPZ file (append-only, rebuild periodically)"""
        # For NPZ, we need to load all existing data and save with new entry
        # This is less efficient than HDF5 but works without additional dependencies
        
        existing_data = {}
        if self.embeddings_file.exists():
            try:
                with np.load(self.embeddings_file, allow_pickle=False) as f:
                    existing_data = {key: f[key] for key in f.files}
            except Exception as e:
                logger.warning(f"Could not load existing NPZ data: {e}")
        
        # Add new entry
        existing_data[cache_key] = entry.embedding
        
        # Save atomically using temporary file
        with tempfile.NamedTemporaryFile(
            dir=self.embedding_cache_dir, 
            delete=False,
            suffix='.npz'
        ) as temp_f:
            np.savez_compressed(temp_f, **existing_data)
            temp_path = temp_f.name
        
        # Atomic move
        os.replace(temp_path, self.embeddings_file)
    
    def _save_metadata(self, cache_key: str, entry: EmbeddingCacheEntry) -> None:
        """Save cache metadata"""
        metadata = {}
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                pass
        
        metadata[cache_key] = entry.to_dict()
        
        # Atomic write
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=self.embedding_cache_dir,
            delete=False,
            encoding='utf-8'
        ) as temp_f:
            json.dump(metadata, temp_f, indent=2)
            temp_path = temp_f.name
        
        os.replace(temp_path, self.metadata_file)
    
    def clear_cache(self) -> None:
        """Clear all cached embeddings"""
        with self._lock:
            try:
                self._memory_cache.clear()
                
                if self.embeddings_file.exists():
                    self.embeddings_file.unlink()
                
                if self.metadata_file.exists():
                    self.metadata_file.unlink()
                
                self.hits = 0
                self.misses = 0
                self.cache_saves = 0
                self.cache_loads = 0
                
                logger.info("Cleared all embedding cache")
                
            except Exception as e:
                logger.error(f"Error clearing embedding cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0
        
        cache_size_bytes = 0
        if self.embeddings_file.exists():
            cache_size_bytes = self.embeddings_file.stat().st_size
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total_requests,
            'hit_rate_percent': hit_rate,
            'cache_saves': self.cache_saves,
            'cache_loads': self.cache_loads,
            'memory_cache_size': len(self._memory_cache),
            'cache_file_size_bytes': cache_size_bytes,
            'cache_file_size_mb': cache_size_bytes / (1024 * 1024),
            'storage_backend': 'HDF5' if HDF5_AVAILABLE else 'NumPy',
            'model_version': self.model_version,
            'embedding_dimension': self.embedding_dim
        }
    
    def cleanup_expired_entries(self, max_age_hours: int = 24 * 7) -> int:
        """Remove entries older than max_age_hours"""
        # This would require loading all metadata and rebuilding storage
        # Implementation depends on specific requirements
        logger.info(f"Cleanup not implemented yet - max_age_hours: {max_age_hours}")
        return 0


# Global instance for BGE embedding caching
_global_embedding_cache: Optional[BGEEmbeddingCache] = None


def get_global_embedding_cache() -> BGEEmbeddingCache:
    """Get the global BGE embedding cache instance"""
    global _global_embedding_cache
    if _global_embedding_cache is None:
        workspace_dir = os.getenv('WORKSPACE_DIR', '/workspace')
        _global_embedding_cache = BGEEmbeddingCache(cache_dir=f"{workspace_dir}/.vector_cache")
    return _global_embedding_cache


def configure_global_embedding_cache(cache_dir: str, model_version: str = "BAAI/bge-large-en-v1.5") -> None:
    """Configure the global embedding cache with custom settings"""
    global _global_embedding_cache
    _global_embedding_cache = BGEEmbeddingCache(cache_dir=cache_dir, model_version=model_version)