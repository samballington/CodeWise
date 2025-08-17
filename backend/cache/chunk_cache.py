"""
Hierarchical Chunk Cache

High-performance cache for hierarchical chunk relationships from Phase 1.2 to eliminate
redundant AST parsing and chunk creation. Stores complete bidirectional relationships
between parent and child chunks with content-based invalidation.

Architectural Design:
- File content hash + parser version + chunking settings cache keys
- Persistent relationship storage with complete parent-child maps
- Bidirectional link caching for context reconstruction
- Content-based invalidation on file changes
- Integration with existing HierarchicalChunker from Phase 1.2.3
"""

import os
import json
import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
import tempfile
import fcntl
import time

# Integration with global cache metrics
try:
    from .cache_metrics import get_global_cache_metrics
except ImportError:
    # Fallback for when module is imported directly
    try:
        from cache.cache_metrics import get_global_cache_metrics
    except ImportError:
        def get_global_cache_metrics():
            return None

logger = logging.getLogger(__name__)


@dataclass
class ChunkNode:
    """Represents a chunk with hierarchical relationships"""
    chunk_id: str
    content: str
    chunk_type: str  # 'file', 'class', 'function', 'method', 'comment', etc.
    start_line: int
    end_line: int
    start_char: int
    end_char: int
    file_path: str
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize chunk node for storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkNode':
        """Deserialize chunk node from storage"""
        return cls(**data)


@dataclass
class ChunkHierarchy:
    """Complete hierarchical chunk structure for a file"""
    file_path: str
    file_hash: str
    parser_version: str
    chunk_settings: Dict[str, Any]
    root_chunks: List[str]  # Top-level chunk IDs
    chunks: Dict[str, ChunkNode]  # chunk_id -> ChunkNode
    relationships: Dict[str, List[str]]  # parent_id -> [child_ids]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize hierarchy for storage"""
        return {
            'file_path': self.file_path,
            'file_hash': self.file_hash,
            'parser_version': self.parser_version,
            'chunk_settings': self.chunk_settings,
            'root_chunks': self.root_chunks,
            'chunks': {chunk_id: chunk.to_dict() for chunk_id, chunk in self.chunks.items()},
            'relationships': self.relationships,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkHierarchy':
        """Deserialize hierarchy from storage"""
        chunks = {chunk_id: ChunkNode.from_dict(chunk_data) 
                 for chunk_id, chunk_data in data['chunks'].items()}
        
        return cls(
            file_path=data['file_path'],
            file_hash=data['file_hash'],
            parser_version=data['parser_version'],
            chunk_settings=data['chunk_settings'],
            root_chunks=data['root_chunks'],
            chunks=chunks,
            relationships=data['relationships'],
            timestamp=data['timestamp']
        )
    
    def get_chunk_context(self, chunk_id: str, max_depth: int = 2) -> List[ChunkNode]:
        """
        Get hierarchical context for a chunk (parents + children).
        
        Args:
            chunk_id: Target chunk ID
            max_depth: Maximum depth to traverse (default: 2)
            
        Returns:
            List of related chunks in hierarchical order
        """
        context_chunks = []
        visited = set()
        
        def add_chunk_and_context(cid: str, depth: int):
            if cid in visited or depth > max_depth or cid not in self.chunks:
                return
            
            visited.add(cid)
            chunk = self.chunks[cid]
            context_chunks.append(chunk)
            
            # Add parent context
            if chunk.parent_id and depth < max_depth:
                add_chunk_and_context(chunk.parent_id, depth + 1)
            
            # Add children context
            if depth < max_depth:
                for child_id in chunk.children_ids:
                    add_chunk_and_context(child_id, depth + 1)
        
        add_chunk_and_context(chunk_id, 0)
        return context_chunks


class HierarchicalChunkCache:
    """
    Persistent cache for hierarchical chunk relationships.
    
    This cache eliminates redundant AST parsing by storing complete
    chunk hierarchies with bidirectional relationships. Cache keys
    are based on file content hash + parser version + settings.
    
    Key Features:
    - Content-based cache invalidation
    - Complete parent-child relationship preservation
    - Fast context reconstruction without re-parsing
    - Integration with Phase 1.2.3 HierarchicalChunker
    - Thread-safe operations with file locking
    """
    
    def __init__(self, cache_dir: str = "/workspace/.vector_cache", 
                 parser_version: str = "1.2.3", ttl_hours: int = 24 * 7):
        """
        Initialize hierarchical chunk cache.
        
        Args:
            cache_dir: Directory for cache storage
            parser_version: Version identifier for parser compatibility
            ttl_hours: Time-to-live for cache entries (hours)
        """
        self.cache_dir = Path(cache_dir)
        self.chunk_cache_dir = self.cache_dir / "chunks"
        self.parser_version = parser_version
        self.ttl_hours = ttl_hours
        
        # Cache files
        self.metadata_file = self.chunk_cache_dir / "metadata.json"
        self.chunks_dir = self.chunk_cache_dir / "hierarchies"
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.invalidations = 0
        self.cache_saves = 0
        
        # Integration with global cache metrics
        try:
            self.global_metrics = get_global_cache_metrics()
        except Exception as e:
            logger.warning(f"Could not initialize global cache metrics: {e}")
            self.global_metrics = None
        
        # Ensure cache directory exists
        self._ensure_cache_directory()
        
        logger.info(f"HierarchicalChunkCache initialized: {self.chunk_cache_dir}")
        logger.info(f"Parser version: {self.parser_version}, TTL: {self.ttl_hours}h")
    
    def _ensure_cache_directory(self) -> None:
        """Ensure cache directory exists with proper permissions"""
        try:
            self.chunk_cache_dir.mkdir(parents=True, exist_ok=True)
            self.chunks_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Chunk cache directory ensured: {self.chunk_cache_dir}")
        except Exception as e:
            logger.error(f"Failed to create chunk cache directory {self.chunk_cache_dir}: {e}")
            raise
    
    def get_file_hash(self, file_path: str) -> str:
        """
        Generate file content hash for cache key.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA256 hash of file content
        """
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to hash file {file_path}: {e}")
            return hashlib.sha256(str(time.time()).encode()).hexdigest()
    
    def get_cache_key(self, file_path: str, chunk_settings: Dict[str, Any]) -> str:
        """
        Generate cache key from file + parser version + settings.
        
        Args:
            file_path: Path to the file
            chunk_settings: Chunking configuration settings
            
        Returns:
            Deterministic cache key
        """
        file_hash = self.get_file_hash(file_path)
        settings_str = json.dumps(chunk_settings, sort_keys=True)
        
        key_string = f"{file_path}:{file_hash}:{self.parser_version}:{settings_str}"
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
    
    def get(self, file_path: str, chunk_settings: Dict[str, Any]) -> Optional[ChunkHierarchy]:
        """
        Retrieve cached chunk hierarchy if valid.
        
        Args:
            file_path: Path to the file
            chunk_settings: Chunking configuration settings
            
        Returns:
            Cached chunk hierarchy or None if cache miss
        """
        with self._lock:
            try:
                cache_key = self.get_cache_key(file_path, chunk_settings)
                hierarchy_file = self.chunks_dir / f"{cache_key}.json"
                
                if not hierarchy_file.exists():
                    self.misses += 1
                    
                    # Report to global metrics
                    if self.global_metrics:
                        self.global_metrics.record_cache_miss('hierarchical_chunks', response_time_ms=0)
                    
                    logger.debug(f"Chunk cache miss for {file_path}")
                    return None
                
                # Load and validate hierarchy
                with open(hierarchy_file, 'r', encoding='utf-8') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        hierarchy_data = json.load(f)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                hierarchy = ChunkHierarchy.from_dict(hierarchy_data)
                
                # Check if entry is expired
                if self._is_expired(hierarchy):
                    self.misses += 1
                    self.invalidations += 1
                    
                    # Remove expired entry
                    hierarchy_file.unlink()
                    logger.debug(f"Chunk cache expired for {file_path}")
                    
                    # Report to global metrics
                    if self.global_metrics:
                        self.global_metrics.record_cache_miss('hierarchical_chunks', response_time_ms=1.0)
                        self.global_metrics.record_cache_invalidation('hierarchical_chunks')
                    
                    return None
                
                # Validate file hasn't changed
                current_hash = self.get_file_hash(file_path)
                if hierarchy.file_hash != current_hash:
                    self.misses += 1
                    self.invalidations += 1
                    
                    # Remove invalidated entry
                    hierarchy_file.unlink()
                    logger.debug(f"Chunk cache invalidated due to file changes: {file_path}")
                    
                    # Report to global metrics
                    if self.global_metrics:
                        self.global_metrics.record_cache_miss('hierarchical_chunks', response_time_ms=2.0)
                        self.global_metrics.record_cache_invalidation('hierarchical_chunks')
                    
                    return None
                
                # Cache hit!
                self.hits += 1
                
                # Report to global metrics
                if self.global_metrics:
                    time_saved = 100  # Estimated AST parsing time saved (ms)
                    self.global_metrics.record_cache_hit('hierarchical_chunks', 
                                                       time_saved_ms=time_saved, 
                                                       response_time_ms=1.0)
                
                logger.info(f"🚀 CHUNK CACHE HIT for {file_path} ({len(hierarchy.chunks)} chunks)")
                return hierarchy
                
            except Exception as e:
                logger.error(f"Error during chunk cache get for {file_path}: {e}")
                self.misses += 1
                
                # Report to global metrics
                if self.global_metrics:
                    self.global_metrics.record_cache_error('hierarchical_chunks')
                
                return None
    
    def set(self, file_path: str, chunk_settings: Dict[str, Any], 
            hierarchy: ChunkHierarchy) -> None:
        """
        Store chunk hierarchy with relationships.
        
        Args:
            file_path: Path to the file
            chunk_settings: Chunking configuration settings
            hierarchy: Complete chunk hierarchy to cache
        """
        with self._lock:
            try:
                cache_key = self.get_cache_key(file_path, chunk_settings)
                hierarchy_file = self.chunks_dir / f"{cache_key}.json"
                
                # Atomic write using temporary file
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    dir=self.chunks_dir,
                    delete=False,
                    encoding='utf-8',
                    suffix='.json'
                ) as temp_f:
                    fcntl.flock(temp_f.fileno(), fcntl.LOCK_EX)
                    try:
                        json.dump(hierarchy.to_dict(), temp_f, indent=2)
                        temp_f.flush()
                        os.fsync(temp_f.fileno())
                    finally:
                        fcntl.flock(temp_f.fileno(), fcntl.LOCK_UN)
                    
                    temp_path = temp_f.name
                
                # Atomic move to final location
                os.replace(temp_path, hierarchy_file)
                self.cache_saves += 1
                
                logger.info(f"💾 CACHED chunk hierarchy for {file_path} ({len(hierarchy.chunks)} chunks)")
                
            except Exception as e:
                logger.error(f"Error during chunk cache set for {file_path}: {e}")
                
                # Clean up temp file if it exists
                try:
                    if 'temp_path' in locals():
                        os.unlink(temp_path)
                except:
                    pass
    
    def _is_expired(self, hierarchy: ChunkHierarchy) -> bool:
        """Check if hierarchy entry has exceeded TTL"""
        expiry_time = hierarchy.timestamp + (self.ttl_hours * 3600)
        return time.time() > expiry_time
    
    def clear_file_cache(self, file_path: str) -> int:
        """
        Clear all cache entries for a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Number of entries cleared
        """
        with self._lock:
            try:
                cleared_count = 0
                
                # Find all cache files for this file path
                for cache_file in self.chunks_dir.glob("*.json"):
                    try:
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            hierarchy_data = json.load(f)
                        
                        if hierarchy_data.get('file_path') == file_path:
                            cache_file.unlink()
                            cleared_count += 1
                            
                    except Exception as e:
                        logger.warning(f"Failed to check cache file {cache_file}: {e}")
                
                if cleared_count > 0:
                    logger.info(f"Cleared {cleared_count} chunk cache entries for {file_path}")
                
                return cleared_count
                
            except Exception as e:
                logger.error(f"Error clearing chunk cache for {file_path}: {e}")
                return 0
    
    def clear_all_cache(self) -> None:
        """Clear all cached chunk hierarchies"""
        with self._lock:
            try:
                cleared_count = 0
                
                for cache_file in self.chunks_dir.glob("*.json"):
                    cache_file.unlink()
                    cleared_count += 1
                
                self.hits = 0
                self.misses = 0
                self.invalidations = 0
                self.cache_saves = 0
                
                logger.info(f"Cleared all chunk cache ({cleared_count} entries)")
                
            except Exception as e:
                logger.error(f"Error clearing all chunk cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0
        
        cache_files = list(self.chunks_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'invalidations': self.invalidations,
            'total_requests': total_requests,
            'hit_rate_percent': hit_rate,
            'cache_saves': self.cache_saves,
            'cached_files': len(cache_files),
            'total_cache_size_bytes': total_size,
            'total_cache_size_mb': total_size / (1024 * 1024),
            'parser_version': self.parser_version,
            'ttl_hours': self.ttl_hours
        }
    
    def cleanup_expired_entries(self) -> int:
        """Remove all expired cache entries"""
        with self._lock:
            try:
                expired_count = 0
                
                for cache_file in self.chunks_dir.glob("*.json"):
                    try:
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            hierarchy_data = json.load(f)
                        
                        hierarchy = ChunkHierarchy.from_dict(hierarchy_data)
                        
                        if self._is_expired(hierarchy):
                            cache_file.unlink()
                            expired_count += 1
                            self.invalidations += 1
                            
                    except Exception as e:
                        logger.warning(f"Failed to check expiry for {cache_file}: {e}")
                
                logger.debug(f"Cleaned up {expired_count} expired chunk cache entries")
                return expired_count
                
            except Exception as e:
                logger.error(f"Error during chunk cache cleanup: {e}")
                return 0


# Global instance for hierarchical chunk caching
_global_chunk_cache: Optional[HierarchicalChunkCache] = None


def get_global_chunk_cache() -> HierarchicalChunkCache:
    """Get the global hierarchical chunk cache instance"""
    global _global_chunk_cache
    if _global_chunk_cache is None:
        workspace_dir = os.getenv('WORKSPACE_DIR', '/workspace')
        _global_chunk_cache = HierarchicalChunkCache(cache_dir=f"{workspace_dir}/.vector_cache")
    return _global_chunk_cache


def configure_global_chunk_cache(cache_dir: str, parser_version: str = "1.2.3") -> None:
    """Configure the global chunk cache with custom settings"""
    global _global_chunk_cache
    _global_chunk_cache = HierarchicalChunkCache(cache_dir=cache_dir, parser_version=parser_version)