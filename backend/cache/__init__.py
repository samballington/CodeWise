"""
Cache module for CodeWise cross-session optimization.

This module implements enterprise-grade caching solutions to eliminate
redundant computations across query sessions while maintaining data integrity.
"""

from .discovery_cache import CrossSessionDiscoveryCache
from .cache_metrics import CacheMetricsAggregator
from .embedding_cache import BGEEmbeddingCache
from .chunk_cache import HierarchicalChunkCache
from .performance_monitor import CachePerformanceMonitor

__all__ = [
    'CrossSessionDiscoveryCache',
    'CacheMetricsAggregator',
    'BGEEmbeddingCache',
    'HierarchicalChunkCache',
    'CachePerformanceMonitor'
]