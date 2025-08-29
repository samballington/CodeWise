"""
Cache Metrics Aggregator

Comprehensive monitoring and optimization for all cache layers in CodeWise.
Provides unified metrics collection, analysis, and reporting across:
- Cross-session discovery cache
- Session-scoped query cache  
- BGE embedding cache
- Hierarchical chunk cache

Architectural Design:
- Unified metrics interface across all cache types
- Real-time performance monitoring
- Cache effectiveness analysis
- Automatic optimization recommendations
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


@dataclass
class CacheLayerMetrics:
    """Metrics for a single cache layer"""
    name: str
    hits: int = 0
    misses: int = 0
    invalidations: int = 0
    errors: int = 0
    total_time_saved_ms: float = 0.0
    average_hit_time_ms: float = 0.0
    average_miss_time_ms: float = 0.0
    last_hit_timestamp: Optional[float] = None
    last_miss_timestamp: Optional[float] = None
    
    @property
    def total_requests(self) -> int:
        """Total cache requests (hits + misses)"""
        return self.hits + self.misses
    
    @property
    def hit_rate_percent(self) -> float:
        """Cache hit rate as percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100
    
    @property
    def effectiveness_score(self) -> float:
        """
        Cache effectiveness score (0-100) based on:
        - Hit rate
        - Time savings
        - Error rate
        """
        if self.total_requests == 0:
            return 0.0
        
        hit_rate_component = self.hit_rate_percent
        error_penalty = (self.errors / self.total_requests) * 100
        time_savings_bonus = min(self.total_time_saved_ms / 1000, 10)  # Cap at 10s bonus
        
        effectiveness = hit_rate_component - error_penalty + time_savings_bonus
        return max(0.0, min(100.0, effectiveness))
    
    def record_hit(self, time_saved_ms: float = 0.0, response_time_ms: float = 0.0) -> None:
        """Record a cache hit"""
        self.hits += 1
        self.total_time_saved_ms += time_saved_ms
        self.last_hit_timestamp = time.time()
        
        if response_time_ms > 0:
            # Update average hit time (exponential moving average)
            if self.average_hit_time_ms == 0:
                self.average_hit_time_ms = response_time_ms
            else:
                self.average_hit_time_ms = (self.average_hit_time_ms * 0.9) + (response_time_ms * 0.1)
    
    def record_miss(self, response_time_ms: float = 0.0) -> None:
        """Record a cache miss"""
        self.misses += 1
        self.last_miss_timestamp = time.time()
        
        if response_time_ms > 0:
            # Update average miss time (exponential moving average)
            if self.average_miss_time_ms == 0:
                self.average_miss_time_ms = response_time_ms
            else:
                self.average_miss_time_ms = (self.average_miss_time_ms * 0.9) + (response_time_ms * 0.1)
    
    def record_invalidation(self) -> None:
        """Record a cache invalidation"""
        self.invalidations += 1
    
    def record_error(self) -> None:
        """Record a cache error"""
        self.errors += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting"""
        return {
            'name': self.name,
            'hits': self.hits,
            'misses': self.misses,
            'invalidations': self.invalidations,
            'errors': self.errors,
            'total_requests': self.total_requests,
            'hit_rate_percent': round(self.hit_rate_percent, 2),
            'effectiveness_score': round(self.effectiveness_score, 2),
            'total_time_saved_ms': round(self.total_time_saved_ms, 2),
            'average_hit_time_ms': round(self.average_hit_time_ms, 2),
            'average_miss_time_ms': round(self.average_miss_time_ms, 2),
            'last_hit_timestamp': self.last_hit_timestamp,
            'last_miss_timestamp': self.last_miss_timestamp
        }


class CacheMetricsAggregator:
    """
    Unified cache metrics collection and analysis across all cache layers.
    
    Provides comprehensive monitoring, performance analysis, and optimization
    recommendations for the entire cache system.
    """
    
    def __init__(self):
        """Initialize cache metrics aggregator"""
        self._metrics: Dict[str, CacheLayerMetrics] = {}
        self._lock = threading.RLock()
        self._session_start_time = time.time()
        
        # Initialize known cache layers
        self._initialize_cache_layers()
        
        logger.info("CacheMetricsAggregator initialized")
    
    def _initialize_cache_layers(self) -> None:
        """Initialize metrics for known cache layers"""
        cache_layers = [
            'discovery_cross_session',
            'discovery_session',
            'bge_embeddings_query',      # REQ-CACHE-6: Query embeddings
            'bge_embeddings_document',   # REQ-CACHE-6: Document embeddings  
            'hierarchical_chunks',
            'conversation_session'
        ]
        
        for layer_name in cache_layers:
            self._metrics[layer_name] = CacheLayerMetrics(name=layer_name)
    
    def get_layer_metrics(self, layer_name: str) -> CacheLayerMetrics:
        """
        Get metrics for a specific cache layer.
        
        Args:
            layer_name: Name of the cache layer
            
        Returns:
            CacheLayerMetrics instance for the layer
        """
        with self._lock:
            if layer_name not in self._metrics:
                self._metrics[layer_name] = CacheLayerMetrics(name=layer_name)
            return self._metrics[layer_name]
    
    def record_cache_hit(self, layer_name: str, time_saved_ms: float = 0.0, 
                        response_time_ms: float = 0.0) -> None:
        """Record a cache hit for a specific layer"""
        with self._lock:
            metrics = self.get_layer_metrics(layer_name)
            metrics.record_hit(time_saved_ms, response_time_ms)
            
            logger.debug(f"Cache hit recorded for {layer_name}: "
                        f"saved {time_saved_ms:.1f}ms, response {response_time_ms:.1f}ms")
    
    def record_cache_miss(self, layer_name: str, response_time_ms: float = 0.0) -> None:
        """Record a cache miss for a specific layer"""
        with self._lock:
            metrics = self.get_layer_metrics(layer_name)
            metrics.record_miss(response_time_ms)
            
            logger.debug(f"Cache miss recorded for {layer_name}: "
                        f"response {response_time_ms:.1f}ms")
    
    def record_cache_invalidation(self, layer_name: str) -> None:
        """Record a cache invalidation for a specific layer"""
        with self._lock:
            metrics = self.get_layer_metrics(layer_name)
            metrics.record_invalidation()
            
            logger.debug(f"Cache invalidation recorded for {layer_name}")
    
    def record_cache_error(self, layer_name: str) -> None:
        """Record a cache error for a specific layer"""
        with self._lock:
            metrics = self.get_layer_metrics(layer_name)
            metrics.record_error()
            
            logger.warning(f"Cache error recorded for {layer_name}")
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated metrics across all cache layers.
        
        Returns:
            Comprehensive cache system metrics
        """
        with self._lock:
            total_hits = sum(m.hits for m in self._metrics.values())
            total_misses = sum(m.misses for m in self._metrics.values())
            total_invalidations = sum(m.invalidations for m in self._metrics.values())
            total_errors = sum(m.errors for m in self._metrics.values())
            total_requests = total_hits + total_misses
            total_time_saved = sum(m.total_time_saved_ms for m in self._metrics.values())
            
            overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0.0
            session_duration = time.time() - self._session_start_time
            
            # Calculate effectiveness by layer
            layer_effectiveness = {
                name: metrics.effectiveness_score
                for name, metrics in self._metrics.items()
                if metrics.total_requests > 0
            }
            
            return {
                'overall': {
                    'total_hits': total_hits,
                    'total_misses': total_misses,
                    'total_invalidations': total_invalidations,
                    'total_errors': total_errors,
                    'total_requests': total_requests,
                    'overall_hit_rate_percent': round(overall_hit_rate, 2),
                    'total_time_saved_ms': round(total_time_saved, 2),
                    'session_duration_seconds': round(session_duration, 2),
                    'requests_per_second': round(total_requests / session_duration, 2) if session_duration > 0 else 0.0
                },
                'by_layer': {
                    name: metrics.to_dict() 
                    for name, metrics in self._metrics.items()
                },
                'effectiveness': {
                    'by_layer': layer_effectiveness,
                    'average_effectiveness': round(
                        sum(layer_effectiveness.values()) / len(layer_effectiveness), 2
                    ) if layer_effectiveness else 0.0
                }
            }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Analyze cache performance and provide optimization recommendations.
        
        Returns:
            List of optimization recommendations with priority and actions
        """
        recommendations = []
        
        with self._lock:
            for name, metrics in self._metrics.items():
                if metrics.total_requests == 0:
                    continue
                
                # Low hit rate recommendation
                if metrics.hit_rate_percent < 30:
                    recommendations.append({
                        'priority': 'HIGH',
                        'layer': name,
                        'issue': 'Low hit rate',
                        'current_rate': metrics.hit_rate_percent,
                        'recommendation': f'Investigate cache key strategy for {name}. '
                                        f'Current hit rate of {metrics.hit_rate_percent:.1f}% '
                                        'indicates poor cache effectiveness.',
                        'expected_improvement': 'Hit rate increase to >60%'
                    })
                
                # High error rate recommendation
                error_rate = (metrics.errors / metrics.total_requests) * 100
                if error_rate > 5:
                    recommendations.append({
                        'priority': 'HIGH',
                        'layer': name,
                        'issue': 'High error rate',
                        'current_rate': error_rate,
                        'recommendation': f'Cache layer {name} has {error_rate:.1f}% error rate. '
                                        'Investigate cache implementation stability.',
                        'expected_improvement': 'Error rate reduction to <1%'
                    })
                
                # Slow cache response recommendation
                if metrics.average_hit_time_ms > 100:
                    recommendations.append({
                        'priority': 'MEDIUM',
                        'layer': name,
                        'issue': 'Slow cache responses',
                        'current_time': metrics.average_hit_time_ms,
                        'recommendation': f'Cache hits for {name} taking {metrics.average_hit_time_ms:.1f}ms. '
                                        'Consider optimizing cache storage or lookup strategy.',
                        'expected_improvement': 'Cache hit time reduction to <50ms'
                    })
                
                # High invalidation rate recommendation
                if metrics.total_requests > 0:
                    invalidation_rate = (metrics.invalidations / metrics.total_requests) * 100
                    if invalidation_rate > 20:
                        recommendations.append({
                            'priority': 'MEDIUM',
                            'layer': name,
                            'issue': 'High invalidation rate',
                            'current_rate': invalidation_rate,
                            'recommendation': f'Cache layer {name} has {invalidation_rate:.1f}% invalidation rate. '
                                            'Consider longer TTL or more stable cache keys.',
                            'expected_improvement': 'Invalidation rate reduction to <10%'
                        })
        
        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations
    
    def reset_metrics(self, layer_name: Optional[str] = None) -> None:
        """
        Reset metrics for a specific layer or all layers.
        
        Args:
            layer_name: Specific layer to reset, or None for all layers
        """
        with self._lock:
            if layer_name:
                if layer_name in self._metrics:
                    self._metrics[layer_name] = CacheLayerMetrics(name=layer_name)
                    logger.info(f"Reset metrics for cache layer: {layer_name}")
            else:
                self._metrics.clear()
                self._initialize_cache_layers()
                self._session_start_time = time.time()
                logger.info("Reset all cache metrics")
    
    def generate_performance_report(self) -> str:
        """
        Generate a comprehensive cache performance report.
        
        Returns:
            Formatted performance report string
        """
        metrics = self.get_aggregated_metrics()
        recommendations = self.get_optimization_recommendations()
        
        report_lines = [
            "=" * 60,
            "CACHE PERFORMANCE REPORT",
            "=" * 60,
            "",
            "OVERALL PERFORMANCE:",
            f"  Total Requests: {metrics['overall']['total_requests']}",
            f"  Hit Rate: {metrics['overall']['overall_hit_rate_percent']:.1f}%",
            f"  Total Time Saved: {metrics['overall']['total_time_saved_ms']:.1f}ms",
            f"  Session Duration: {metrics['overall']['session_duration_seconds']:.1f}s",
            f"  Requests/Second: {metrics['overall']['requests_per_second']:.2f}",
            "",
            "BY CACHE LAYER:",
        ]
        
        for layer_name, layer_metrics in metrics['by_layer'].items():
            if layer_metrics['total_requests'] > 0:
                report_lines.extend([
                    f"  {layer_name}:",
                    f"    Requests: {layer_metrics['total_requests']}",
                    f"    Hit Rate: {layer_metrics['hit_rate_percent']:.1f}%",
                    f"    Effectiveness: {layer_metrics['effectiveness_score']:.1f}/100",
                    f"    Avg Hit Time: {layer_metrics['average_hit_time_ms']:.1f}ms",
                    ""
                ])
        
        if recommendations:
            report_lines.extend([
                "OPTIMIZATION RECOMMENDATIONS:",
                ""
            ])
            
            for i, rec in enumerate(recommendations[:5], 1):  # Top 5 recommendations
                report_lines.extend([
                    f"  {i}. [{rec['priority']}] {rec['layer']} - {rec['issue']}",
                    f"     {rec['recommendation']}",
                    ""
                ])
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)


# Global instance for unified cache metrics
_global_metrics_aggregator: Optional[CacheMetricsAggregator] = None


def get_global_cache_metrics() -> CacheMetricsAggregator:
    """Get the global cache metrics aggregator instance"""
    global _global_metrics_aggregator
    if _global_metrics_aggregator is None:
        _global_metrics_aggregator = CacheMetricsAggregator()
    return _global_metrics_aggregator