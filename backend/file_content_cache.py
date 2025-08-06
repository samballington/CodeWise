#!/usr/bin/env python3
"""
File Content Cache - Production-ready caching for file content operations

This module implements a thread-safe, TTL-based cache for file content
to eliminate redundant file I/O operations during discovery pipeline execution.
"""

import time
import threading
from typing import Dict, Optional, NamedTuple
from dataclasses import dataclass
import logging
import hashlib
import os

logger = logging.getLogger(__name__)


class FileContentEntry(NamedTuple):
    """Immutable cache entry with content and metadata"""
    content: str
    size: int
    mtime: float
    timestamp: float


@dataclass(frozen=True)
class FileContentCacheConfig:
    """Immutable configuration for file content cache"""
    max_size: int = 50
    ttl_seconds: float = 180.0  # 3 minutes
    max_file_size: int = 1024 * 1024  # 1MB max file size to cache
    enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration on creation"""
        if self.max_size < 0:
            raise ValueError("max_size must be non-negative")
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if self.max_file_size <= 0:
            raise ValueError("max_file_size must be positive")


class FileContentCache:
    """
    Thread-safe LRU cache with TTL for file content caching.
    
    Implements production-ready caching with:
    - Thread-safe operations using RLock
    - TTL-based expiration
    - File modification time checking
    - LRU eviction policy
    - Comprehensive metrics tracking
    - File size limits to prevent memory issues
    """
    
    def __init__(self, config: FileContentCacheConfig):
        self.config = config
        self._cache: Dict[str, FileContentEntry] = {}
        self._access_order: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        # Metrics for monitoring
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired_entries': 0,
            'stale_entries': 0,
            'oversized_files': 0,
            'total_requests': 0
        }
    
    def get(self, file_path: str) -> Optional[str]:
        """
        Get cached file content.
        
        Args:
            file_path: Absolute path to the file
            
        Returns:
            File content if cached and valid, None otherwise
        """
        if not self.config.enabled:
            return None
        
        with self._lock:
            self._stats['total_requests'] += 1
            
            if file_path not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[file_path]
            
            # Check if entry has expired
            if self._is_expired(entry):
                self._remove_key(file_path)
                self._stats['expired_entries'] += 1
                self._stats['misses'] += 1
                return None
            
            # Check if file has been modified since caching
            if self._is_stale(file_path, entry):
                self._remove_key(file_path)
                self._stats['stale_entries'] += 1
                self._stats['misses'] += 1
                return None
            
            # Update access order for LRU
            self._access_order[file_path] = time.time()
            self._stats['hits'] += 1
            
            return entry.content
    
    def put(self, file_path: str, content: str) -> bool:
        """
        Cache file content.
        
        Args:
            file_path: Absolute path to the file
            content: File content to cache
            
        Returns:
            True if cached successfully, False if skipped
        """
        if not self.config.enabled:
            return False
        
        # Check file size limit
        content_size = len(content.encode('utf-8'))
        if content_size > self.config.max_file_size:
            self._stats['oversized_files'] += 1
            logger.debug(f"File {file_path} too large to cache ({content_size} bytes > {self.config.max_file_size})")
            return False
        
        with self._lock:
            try:
                # Get file modification time
                mtime = os.path.getmtime(file_path) if os.path.exists(file_path) else 0.0
                current_time = time.time()
                
                # Create new entry
                entry = FileContentEntry(
                    content=content,
                    size=content_size,
                    mtime=mtime,
                    timestamp=current_time
                )
                
                # Add to cache
                self._cache[file_path] = entry
                self._access_order[file_path] = current_time
                
                # Evict if necessary
                self._evict_if_necessary()
                
                return True
                
            except Exception as e:
                logger.warning(f"Failed to cache file content for {file_path}: {e}")
                return False
    
    def clear(self) -> None:
        """Clear all cached entries"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._reset_stats()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics for monitoring"""
        with self._lock:
            total_requests = self._stats['total_requests']
            hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0.0
            
            total_size = sum(entry.size for entry in self._cache.values())
            
            return {
                **self._stats.copy(),
                'cache_size': len(self._cache),
                'hit_rate_percent': round(hit_rate, 2),
                'total_content_bytes': total_size
            }
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired and stale entries from cache.
        
        Returns:
            Number of entries removed
        """
        if not self.config.enabled:
            return 0
        
        with self._lock:
            expired_keys = []
            
            for file_path, entry in self._cache.items():
                if self._is_expired(entry) or self._is_stale(file_path, entry):
                    expired_keys.append(file_path)
            
            for key in expired_keys:
                self._remove_key(key)
            
            if expired_keys:
                self._stats['expired_entries'] += len(expired_keys)
                logger.debug(f"Cleaned up {len(expired_keys)} expired/stale cache entries")
            
            return len(expired_keys)
    
    def _is_expired(self, entry: FileContentEntry) -> bool:
        """Check if cache entry has exceeded TTL"""
        return (time.time() - entry.timestamp) > self.config.ttl_seconds
    
    def _is_stale(self, file_path: str, entry: FileContentEntry) -> bool:
        """Check if file has been modified since caching"""
        try:
            if not os.path.exists(file_path):
                return True
            current_mtime = os.path.getmtime(file_path)
            return current_mtime > entry.mtime
        except Exception:
            return True  # Assume stale if we can't check
    
    def _remove_key(self, key: str) -> None:
        """Remove key from both cache and access order tracking"""
        self._cache.pop(key, None)
        self._access_order.pop(key, None)
    
    def _evict_if_necessary(self) -> None:
        """Evict least recently used entries if cache is full"""
        if len(self._cache) <= self.config.max_size:
            return
        
        # Sort by access time (oldest first)
        sorted_keys = sorted(self._access_order.items(), key=lambda x: x[1])
        
        # Remove oldest entries until we're under the limit
        entries_to_remove = len(self._cache) - self.config.max_size
        for key, _ in sorted_keys[:entries_to_remove]:
            self._remove_key(key)
            self._stats['evictions'] += 1
        
        if entries_to_remove > 0:
            logger.debug(f"Evicted {entries_to_remove} file content cache entries due to size limit")
    
    def _reset_stats(self) -> None:
        """Reset all statistics to zero"""
        for key in self._stats:
            self._stats[key] = 0


# Global cache instance with default configuration
_default_content_cache_config = FileContentCacheConfig()
_global_content_cache = FileContentCache(_default_content_cache_config)


def get_global_content_cache() -> FileContentCache:
    """Get the global file content cache instance"""
    return _global_content_cache


def configure_global_content_cache(config: FileContentCacheConfig) -> None:
    """Configure the global content cache with new settings"""
    global _global_content_cache
    _global_content_cache = FileContentCache(config)