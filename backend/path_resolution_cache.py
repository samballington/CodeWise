#!/usr/bin/env python3
"""
Path Resolution Cache - Production-ready caching for path resolution operations

This module implements a thread-safe, TTL-based cache for path resolution
to eliminate redundant filesystem calls during discovery pipeline execution.
"""

import time
import threading
from typing import Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class CacheEntry(NamedTuple):
    """Immutable cache entry with TTL support"""
    resolved_path: str
    exists: bool
    timestamp: float


@dataclass(frozen=True)
class PathResolutionCacheConfig:
    """Immutable configuration for path resolution cache"""
    max_size: int = 1000
    ttl_seconds: float = 300.0  # 5 minutes
    enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration on creation"""
        if self.max_size < 0:
            raise ValueError("max_size must be non-negative")
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")


class PathResolutionCache:
    """
    Thread-safe LRU cache with TTL for path resolution results.
    
    Implements production-ready caching with:
    - Thread-safe operations using RLock
    - TTL-based expiration
    - LRU eviction policy
    - Comprehensive metrics tracking
    - Proper error handling
    """
    
    def __init__(self, config: PathResolutionCacheConfig):
        self.config = config
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        # Metrics for monitoring
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired_entries': 0,
            'total_requests': 0
        }
    
    def get(self, key: str) -> Optional[Tuple[str, bool]]:
        """
        Get cached path resolution result.
        
        Args:
            key: The original file path to resolve
            
        Returns:
            Tuple of (resolved_path, exists) if cached and valid, None otherwise
        """
        if not self.config.enabled:
            return None
        
        with self._lock:
            self._stats['total_requests'] += 1
            
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            # Check if entry has expired
            if self._is_expired(entry):
                self._remove_key(key)
                self._stats['expired_entries'] += 1
                self._stats['misses'] += 1
                return None
            
            # Update access order for LRU
            self._access_order[key] = time.time()
            self._stats['hits'] += 1
            
            return (entry.resolved_path, entry.exists)
    
    def put(self, key: str, resolved_path: str, exists: bool) -> None:
        """
        Cache a path resolution result.
        
        Args:
            key: The original file path
            resolved_path: The resolved absolute path
            exists: Whether the resolved path exists
        """
        if not self.config.enabled:
            return
        
        with self._lock:
            current_time = time.time()
            
            # Create new entry
            entry = CacheEntry(
                resolved_path=resolved_path,
                exists=exists,
                timestamp=current_time
            )
            
            # Add to cache
            self._cache[key] = entry
            self._access_order[key] = current_time
            
            # Evict if necessary
            self._evict_if_necessary()
    
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
            
            return {
                **self._stats.copy(),
                'cache_size': len(self._cache),
                'hit_rate_percent': round(hit_rate, 2)
            }
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        if not self.config.enabled:
            return 0
        
        with self._lock:
            expired_keys = []
            
            for key, entry in self._cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_key(key)
            
            if expired_keys:
                self._stats['expired_entries'] += len(expired_keys)
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry has exceeded TTL"""
        return (time.time() - entry.timestamp) > self.config.ttl_seconds
    
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
            logger.debug(f"Evicted {entries_to_remove} cache entries due to size limit")
    
    def _reset_stats(self) -> None:
        """Reset all statistics to zero"""
        for key in self._stats:
            self._stats[key] = 0


# Global cache instance with default configuration
_default_cache_config = PathResolutionCacheConfig()
_global_cache = PathResolutionCache(_default_cache_config)


def get_global_cache() -> PathResolutionCache:
    """Get the global path resolution cache instance"""
    return _global_cache


def configure_global_cache(config: PathResolutionCacheConfig) -> None:
    """Configure the global cache with new settings"""
    global _global_cache
    _global_cache = PathResolutionCache(config)