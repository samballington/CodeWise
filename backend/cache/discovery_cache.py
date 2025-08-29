"""
Cross-Session Discovery Cache

High-performance persistent cache for discovery pipeline results across
different query sessions. Eliminates redundant discovery execution for
identical project content while ensuring data integrity through file-based
invalidation.

Architectural Design:
- Persistent JSON storage with atomic writes
- Deterministic cache keys based on project + file signatures  
- TTL with file modification-based invalidation
- Thread-safe operations with file locking
- Graceful degradation on cache failures
"""

import os
import json
import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import tempfile

# Windows-compatible import for fcntl
try:
    import fcntl
except ImportError:
    # Windows doesn't have fcntl, provide a mock for Docker compatibility
    class MockFcntl:
        LOCK_EX = 2
        LOCK_NB = 4
        LOCK_UN = 8
        
        def flock(self, fd, op):
            pass  # No-op on Windows
    
    fcntl = MockFcntl()

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Immutable cache entry with metadata"""
    data: Dict[str, Any]
    timestamp: float
    project: str
    file_signature: str
    ttl_hours: int = 24
    
    def is_expired(self) -> bool:
        """Check if entry has exceeded TTL"""
        expiry_time = self.timestamp + (self.ttl_hours * 3600)
        return time.time() > expiry_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize entry for JSON storage"""
        return {
            'data': self.data,
            'timestamp': self.timestamp,
            'project': self.project,
            'file_signature': self.file_signature,
            'ttl_hours': self.ttl_hours
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Deserialize entry from JSON storage"""
        return cls(
            data=data['data'],
            timestamp=data['timestamp'],
            project=data['project'],
            file_signature=data['file_signature'],
            ttl_hours=data.get('ttl_hours', 24)
        )


class CrossSessionDiscoveryCache:
    """
    Persistent cache for discovery pipeline results across query sessions.
    
    This cache solves the core architectural problem where QueryContext
    creates new instances per query, causing 0.0% cache hit rates even
    for identical content analysis.
    
    Key Features:
    - Project + file signature based cache keys
    - Automatic invalidation on file modifications
    - Atomic writes with corruption protection
    - Thread-safe operations
    - Configurable TTL with file-based override
    """
    
    def __init__(self, workspace_dir: str = "/workspace", ttl_hours: int = 24):
        """
        Initialize cross-session discovery cache.
        
        Args:
            workspace_dir: Base workspace directory
            ttl_hours: Time-to-live for cache entries (hours)
        """
        self.workspace_dir = Path(workspace_dir)
        self.cache_dir = self.workspace_dir / ".discovery_cache"
        self.cache_file = self.cache_dir / "cache.json"
        self.ttl_hours = ttl_hours
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.invalidations = 0
        
        # Ensure cache directory exists
        self._ensure_cache_directory()
        
        logger.info(f"CrossSessionDiscoveryCache initialized: {self.cache_dir}")
    
    def _ensure_cache_directory(self) -> None:
        """Ensure cache directory exists with proper permissions"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Cache directory ensured: {self.cache_dir}")
        except Exception as e:
            logger.error(f"Failed to create cache directory {self.cache_dir}: {e}")
            raise
    
    def get_file_signature(self, file_paths: List[str]) -> str:
        """
        Generate deterministic file signature based on paths + modification times.
        
        This signature changes when ANY file in the project is modified,
        ensuring cache invalidation on content changes.
        
        Args:
            file_paths: List of file paths to include in signature
            
        Returns:
            SHA256 hash of file signatures
        """
        file_sigs = []
        
        for fp in sorted(file_paths):  # Sort for determinism
            try:
                file_path = Path(fp)
                if file_path.exists():
                    stat = file_path.stat()
                    # Include path, mtime, and size for robust change detection
                    file_sigs.append(f"{fp}:{stat.st_mtime}:{stat.st_size}")
                else:
                    file_sigs.append(f"{fp}:missing:0")
            except OSError as e:
                logger.warning(f"Failed to stat file {fp}: {e}")
                file_sigs.append(f"{fp}:error:0")
        
        # Create deterministic signature
        signature_string = ":".join(file_sigs)
        return hashlib.sha256(signature_string.encode('utf-8')).hexdigest()
    
    def get_cache_key(self, project: str, file_paths: List[str]) -> str:
        """
        Generate deterministic cache key from project + file signatures.
        
        Cache key format: project_name:file_signature_hash
        
        Args:
            project: Project name/identifier
            file_paths: List of file paths in the project
            
        Returns:
            Deterministic cache key
        """
        file_signature = self.get_file_signature(file_paths)
        cache_key = f"{project}:{file_signature}"
        return hashlib.sha256(cache_key.encode('utf-8')).hexdigest()
    
    def _load_cache(self) -> Dict[str, CacheEntry]:
        """
        Load cache entries from persistent storage.
        
        Returns:
            Dictionary of cache entries keyed by cache key
        """
        if not self.cache_file.exists():
            return {}
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                # Use file locking for thread safety
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    raw_data = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Deserialize entries
            cache_entries = {}
            for key, entry_data in raw_data.items():
                try:
                    cache_entries[key] = CacheEntry.from_dict(entry_data)
                except Exception as e:
                    logger.warning(f"Failed to deserialize cache entry {key}: {e}")
                    continue
            
            logger.debug(f"Loaded {len(cache_entries)} cache entries")
            return cache_entries
            
        except Exception as e:
            logger.error(f"Failed to load cache from {self.cache_file}: {e}")
            return {}
    
    def _save_cache(self, cache_entries: Dict[str, CacheEntry]) -> None:
        """
        Save cache entries to persistent storage with atomic writes.
        
        Args:
            cache_entries: Dictionary of cache entries to save
        """
        try:
            # Serialize entries
            serialized_data = {
                key: entry.to_dict() 
                for key, entry in cache_entries.items()
            }
            
            # Atomic write using temporary file
            with tempfile.NamedTemporaryFile(
                mode='w', 
                dir=self.cache_dir, 
                delete=False,
                encoding='utf-8'
            ) as temp_f:
                # Use file locking for thread safety
                fcntl.flock(temp_f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(serialized_data, temp_f, indent=2)
                    temp_f.flush()
                    os.fsync(temp_f.fileno())  # Force write to disk
                finally:
                    fcntl.flock(temp_f.fileno(), fcntl.LOCK_UN)
                
                temp_path = temp_f.name
            
            # Atomic move to final location
            os.replace(temp_path, self.cache_file)
            logger.debug(f"Saved {len(cache_entries)} cache entries")
            
        except Exception as e:
            logger.error(f"Failed to save cache to {self.cache_file}: {e}")
            # Clean up temp file if it exists
            try:
                if 'temp_path' in locals():
                    os.unlink(temp_path)
            except:
                pass
    
    def get(self, project: str, file_paths: List[str]) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached discovery results if valid.
        
        Args:
            project: Project name/identifier
            file_paths: List of file paths in the project
            
        Returns:
            Cached discovery results or None if cache miss
        """
        with self._lock:
            try:
                cache_key = self.get_cache_key(project, file_paths)
                cache_entries = self._load_cache()
                
                if cache_key not in cache_entries:
                    self.misses += 1
                    logger.debug(f"Cache miss for project '{project}' (key: {cache_key[:8]})")
                    return None
                
                entry = cache_entries[cache_key]
                
                # Check if entry is expired
                if entry.is_expired():
                    self.misses += 1
                    self.invalidations += 1
                    logger.debug(f"Cache entry expired for project '{project}' (key: {cache_key[:8]})")
                    
                    # Remove expired entry
                    del cache_entries[cache_key]
                    self._save_cache(cache_entries)
                    return None
                
                # Validate file signature hasn't changed
                current_signature = self.get_file_signature(file_paths)
                if entry.file_signature != current_signature:
                    self.misses += 1
                    self.invalidations += 1
                    logger.debug(f"Cache invalidated due to file changes for project '{project}' (key: {cache_key[:8]})")
                    
                    # Remove invalidated entry
                    del cache_entries[cache_key]
                    self._save_cache(cache_entries)
                    return None
                
                # Cache hit!
                self.hits += 1
                logger.info(f"🚀 CROSS-SESSION CACHE HIT for project '{project}' (key: {cache_key[:8]})")
                return entry.data
                
            except Exception as e:
                logger.error(f"Error during cache get for project '{project}': {e}")
                self.misses += 1
                return None
    
    def set(self, project: str, file_paths: List[str], discovery_results: Dict[str, Any]) -> None:
        """
        Store discovery results with timestamp and file signature.
        
        Args:
            project: Project name/identifier
            file_paths: List of file paths in the project
            discovery_results: Discovery pipeline results to cache
        """
        with self._lock:
            try:
                cache_key = self.get_cache_key(project, file_paths)
                file_signature = self.get_file_signature(file_paths)
                
                # Create cache entry
                entry = CacheEntry(
                    data=discovery_results,
                    timestamp=time.time(),
                    project=project,
                    file_signature=file_signature,
                    ttl_hours=self.ttl_hours
                )
                
                # Load existing cache
                cache_entries = self._load_cache()
                
                # Add new entry
                cache_entries[cache_key] = entry
                
                # Save updated cache
                self._save_cache(cache_entries)
                
                logger.info(f"💾 CACHED discovery results for project '{project}' (key: {cache_key[:8]})")
                
            except Exception as e:
                logger.error(f"Error during cache set for project '{project}': {e}")
    
    def clear_project(self, project: str) -> int:
        """
        Clear all cache entries for a specific project.
        
        Args:
            project: Project name to clear
            
        Returns:
            Number of entries cleared
        """
        with self._lock:
            try:
                cache_entries = self._load_cache()
                
                # Find entries for this project
                entries_to_remove = [
                    key for key, entry in cache_entries.items()
                    if entry.project == project
                ]
                
                # Remove entries
                for key in entries_to_remove:
                    del cache_entries[key]
                
                # Save updated cache
                if entries_to_remove:
                    self._save_cache(cache_entries)
                
                logger.info(f"Cleared {len(entries_to_remove)} cache entries for project '{project}'")
                return len(entries_to_remove)
                
            except Exception as e:
                logger.error(f"Error clearing cache for project '{project}': {e}")
                return 0
    
    def clear_all(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            try:
                if self.cache_file.exists():
                    self.cache_file.unlink()
                
                self.hits = 0
                self.misses = 0
                self.invalidations = 0
                
                logger.info("Cleared all cross-session cache entries")
                
            except Exception as e:
                logger.error(f"Error clearing all cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache performance metrics
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'invalidations': self.invalidations,
            'total_requests': total_requests,
            'hit_rate_percent': hit_rate,
            'cache_file_exists': self.cache_file.exists(),
            'cache_file_size': self.cache_file.stat().st_size if self.cache_file.exists() else 0
        }
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired cache entries.
        
        Returns:
            Number of entries cleaned up
        """
        with self._lock:
            try:
                cache_entries = self._load_cache()
                
                # Find expired entries
                expired_keys = [
                    key for key, entry in cache_entries.items()
                    if entry.is_expired()
                ]
                
                # Remove expired entries
                for key in expired_keys:
                    del cache_entries[key]
                
                # Save updated cache
                if expired_keys:
                    self._save_cache(cache_entries)
                    self.invalidations += len(expired_keys)
                
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                return len(expired_keys)
                
            except Exception as e:
                logger.error(f"Error during cache cleanup: {e}")
                return 0


# Global instance for cross-session discovery caching
_global_discovery_cache: Optional[CrossSessionDiscoveryCache] = None


def get_global_discovery_cache() -> CrossSessionDiscoveryCache:
    """Get the global cross-session discovery cache instance"""
    global _global_discovery_cache
    if _global_discovery_cache is None:
        _global_discovery_cache = CrossSessionDiscoveryCache()
    return _global_discovery_cache


def configure_global_discovery_cache(workspace_dir: str, ttl_hours: int = 24) -> None:
    """Configure the global discovery cache with custom settings"""
    global _global_discovery_cache
    _global_discovery_cache = CrossSessionDiscoveryCache(workspace_dir, ttl_hours)