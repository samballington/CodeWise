"""
Cache Performance Monitor

Comprehensive monitoring and optimization system for all cache layers in CodeWise.
Provides real-time effectiveness monitoring, automatic cache tuning, and performance
optimization recommendations based on usage patterns.

Architectural Design:
- Real-time cache effectiveness monitoring across all layers
- LRU eviction and memory usage monitoring
- Automatic cache tuning based on hit rate patterns
- Performance dashboard with optimization recommendations
- Problem detection for cache misses that should be hits
"""

import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import collections

# Integration with all cache systems
try:
    from .cache_metrics import get_global_cache_metrics
    from .discovery_cache import get_global_discovery_cache
    from .embedding_cache import get_global_embedding_cache
    from .chunk_cache import get_global_chunk_cache
except ImportError:
    # Fallback for when module is imported directly
    try:
        from cache.cache_metrics import get_global_cache_metrics
        from cache.discovery_cache import get_global_discovery_cache
        from cache.embedding_cache import get_global_embedding_cache
        from cache.chunk_cache import get_global_chunk_cache
    except ImportError:
        def get_global_cache_metrics():
            return None
        def get_global_discovery_cache():
            return None
        def get_global_embedding_cache():
            return None
        def get_global_chunk_cache():
            return None

logger = logging.getLogger(__name__)


@dataclass
class CacheHealthMetrics:
    """Health metrics for a specific cache layer"""
    layer_name: str
    current_hit_rate: float
    average_hit_rate: float
    trend_direction: str  # 'improving', 'degrading', 'stable'
    memory_usage_mb: float
    cache_size: int
    last_optimization: Optional[datetime] = None
    recommendations: List[str] = field(default_factory=list)
    health_score: float = 0.0  # 0-100
    
    def calculate_health_score(self) -> float:
        """Calculate overall health score (0-100)"""
        hit_rate_score = min(self.current_hit_rate, 100.0)
        trend_score = {'improving': 20, 'stable': 10, 'degrading': -10}.get(self.trend_direction, 0)
        memory_penalty = max(0, (self.memory_usage_mb - 100) / 10)  # Penalty for >100MB usage
        
        self.health_score = max(0, min(100, hit_rate_score + trend_score - memory_penalty))
        return self.health_score


@dataclass
class PerformanceTrend:
    """Performance trend tracking for cache optimization"""
    timestamps: List[float] = field(default_factory=list)
    hit_rates: List[float] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)
    max_history: int = 100
    
    def add_measurement(self, hit_rate: float, response_time: float):
        """Add a new performance measurement"""
        current_time = time.time()
        self.timestamps.append(current_time)
        self.hit_rates.append(hit_rate)
        self.response_times.append(response_time)
        
        # Maintain history limit
        if len(self.timestamps) > self.max_history:
            self.timestamps.pop(0)
            self.hit_rates.pop(0)
            self.response_times.pop(0)
    
    def get_trend_direction(self) -> str:
        """Analyze trend direction over recent measurements"""
        if len(self.hit_rates) < 5:
            return 'unknown'
        
        recent = self.hit_rates[-5:]
        if recent[-1] > recent[0] + 5:
            return 'improving'
        elif recent[-1] < recent[0] - 5:
            return 'degrading'
        else:
            return 'stable'
    
    def get_average_hit_rate(self) -> float:
        """Get average hit rate over all measurements"""
        return sum(self.hit_rates) / len(self.hit_rates) if self.hit_rates else 0.0
    
    def get_average_response_time(self) -> float:
        """Get average response time over all measurements"""
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0


class CachePerformanceMonitor:
    """
    Comprehensive cache performance monitoring and optimization system.
    
    This monitor tracks all cache layers in real-time, provides performance
    analysis, and offers optimization recommendations. It includes automatic
    cache tuning based on usage patterns and memory management.
    
    Key Features:
    - Real-time performance monitoring across all cache layers
    - Automatic trend analysis and health scoring
    - Memory usage monitoring with LRU eviction recommendations
    - Performance optimization recommendations
    - Automatic cache tuning based on patterns
    """
    
    def __init__(self, monitor_interval: int = 30, history_retention_hours: int = 24):
        """
        Initialize cache performance monitor.
        
        Args:
            monitor_interval: Monitoring interval in seconds
            history_retention_hours: How long to retain performance history
        """
        self.monitor_interval = monitor_interval
        self.history_retention_hours = history_retention_hours
        
        # Performance tracking
        self.trends: Dict[str, PerformanceTrend] = {}
        self.health_metrics: Dict[str, CacheHealthMetrics] = {}
        self.last_optimization_check = time.time()
        
        # Memory management
        self.memory_limits = {
            'discovery_cross_session': 50,  # MB
            'bge_embeddings_query': 100,    # MB
            'bge_embeddings_document': 200, # MB
            'hierarchical_chunks': 75,      # MB
            'total_system': 500             # MB
        }
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.RLock()
        
        # Cache instances
        self.cache_metrics = None
        self.discovery_cache = None
        self.embedding_cache = None
        self.chunk_cache = None
        
        # Initialize cache connections
        self._initialize_cache_connections()
        
        logger.info(f"CachePerformanceMonitor initialized (interval: {monitor_interval}s)")
    
    def _initialize_cache_connections(self):
        """Initialize connections to all cache systems"""
        try:
            self.cache_metrics = get_global_cache_metrics()
            self.discovery_cache = get_global_discovery_cache()
            self.embedding_cache = get_global_embedding_cache()
            self.chunk_cache = get_global_chunk_cache()
            
            logger.info("Cache performance monitor connected to all cache systems")
        except Exception as e:
            logger.warning(f"Failed to connect to some cache systems: {e}")
    
    def start_monitoring(self):
        """Start real-time cache performance monitoring"""
        with self._lock:
            if self._monitoring:
                logger.warning("Cache monitoring is already running")
                return
            
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitor_thread.start()
            
            logger.info("Cache performance monitoring started")
    
    def stop_monitoring(self):
        """Stop cache performance monitoring"""
        with self._lock:
            self._monitoring = False
            
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5)
            
            logger.info("Cache performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread"""
        while self._monitoring:
            try:
                self._collect_performance_metrics()
                self._analyze_trends()
                self._check_memory_usage()
                self._generate_recommendations()
                
                # Check if optimization is needed
                if time.time() - self.last_optimization_check > 300:  # Every 5 minutes
                    self._perform_automatic_optimization()
                    self.last_optimization_check = time.time()
                
            except Exception as e:
                logger.error(f"Error in cache monitoring loop: {e}")
            
            time.sleep(self.monitor_interval)
    
    def _collect_performance_metrics(self):
        """Collect current performance metrics from all cache layers"""
        if not self.cache_metrics:
            return
        
        try:
            aggregated_metrics = self.cache_metrics.get_aggregated_metrics()
            
            for layer_name, layer_data in aggregated_metrics['by_layer'].items():
                if layer_data['total_requests'] > 0:
                    hit_rate = layer_data['hit_rate_percent']
                    avg_response_time = (layer_data['average_hit_time_ms'] + 
                                       layer_data['average_miss_time_ms']) / 2
                    
                    # Initialize trend tracking if needed
                    if layer_name not in self.trends:
                        self.trends[layer_name] = PerformanceTrend()
                    
                    # Add measurement
                    self.trends[layer_name].add_measurement(hit_rate, avg_response_time)
                    
                    # Update health metrics
                    if layer_name not in self.health_metrics:
                        self.health_metrics[layer_name] = CacheHealthMetrics(
                            layer_name=layer_name,
                            current_hit_rate=hit_rate,
                            average_hit_rate=hit_rate,
                            trend_direction='unknown',
                            memory_usage_mb=0.0,
                            cache_size=0
                        )
                    
                    health = self.health_metrics[layer_name]
                    health.current_hit_rate = hit_rate
                    health.average_hit_rate = self.trends[layer_name].get_average_hit_rate()
                    health.trend_direction = self.trends[layer_name].get_trend_direction()
                    
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
    
    def _analyze_trends(self):
        """Analyze performance trends and identify patterns"""
        for layer_name, trend in self.trends.items():
            if layer_name in self.health_metrics:
                health = self.health_metrics[layer_name]
                
                # Clear previous recommendations
                health.recommendations.clear()
                
                # Analyze hit rate trends
                if health.current_hit_rate < 30:
                    health.recommendations.append(
                        f"LOW HIT RATE: {layer_name} has {health.current_hit_rate:.1f}% hit rate. "
                        "Consider reviewing cache key strategy or increasing cache size."
                    )
                
                if health.trend_direction == 'degrading':
                    health.recommendations.append(
                        f"DEGRADING PERFORMANCE: {layer_name} hit rate is declining. "
                        "Check for memory pressure or cache invalidation issues."
                    )
                
                # Calculate and update health score
                health.calculate_health_score()
    
    def _check_memory_usage(self):
        """Check memory usage across all cache layers"""
        try:
            total_memory = 0.0
            
            # Check discovery cache
            if self.discovery_cache:
                stats = self.discovery_cache.get_stats()
                memory_mb = stats.get('cache_file_size_mb', 0)
                total_memory += memory_mb
                
                if 'discovery_cross_session' in self.health_metrics:
                    self.health_metrics['discovery_cross_session'].memory_usage_mb = memory_mb
                    self.health_metrics['discovery_cross_session'].cache_size = stats.get('total_requests', 0)
            
            # Check embedding cache
            if self.embedding_cache:
                stats = self.embedding_cache.get_cache_stats()
                memory_mb = stats.get('cache_file_size_mb', 0)
                total_memory += memory_mb
                
                for layer in ['bge_embeddings_query', 'bge_embeddings_document']:
                    if layer in self.health_metrics:
                        self.health_metrics[layer].memory_usage_mb = memory_mb / 2  # Split between query/doc
                        self.health_metrics[layer].cache_size = stats.get('memory_cache_size', 0)
            
            # Check chunk cache
            if self.chunk_cache:
                stats = self.chunk_cache.get_cache_stats()
                memory_mb = stats.get('total_cache_size_mb', 0)
                total_memory += memory_mb
                
                if 'hierarchical_chunks' in self.health_metrics:
                    self.health_metrics['hierarchical_chunks'].memory_usage_mb = memory_mb
                    self.health_metrics['hierarchical_chunks'].cache_size = stats.get('cached_files', 0)
            
            # Check if total memory exceeds limits
            if total_memory > self.memory_limits['total_system']:
                logger.warning(f"Total cache memory usage ({total_memory:.1f}MB) exceeds limit "
                             f"({self.memory_limits['total_system']}MB)")
                
                # Add recommendations for memory optimization
                for layer_name, health in self.health_metrics.items():
                    if health.memory_usage_mb > self.memory_limits.get(layer_name, 50):
                        health.recommendations.append(
                            f"HIGH MEMORY USAGE: {layer_name} using {health.memory_usage_mb:.1f}MB. "
                            "Consider implementing LRU eviction or reducing cache size."
                        )
            
        except Exception as e:
            logger.error(f"Failed to check memory usage: {e}")
    
    def _generate_recommendations(self):
        """Generate optimization recommendations based on analysis"""
        for layer_name, health in self.health_metrics.items():
            # Performance-based recommendations
            if health.current_hit_rate > 80 and health.trend_direction == 'improving':
                if not any('EXCELLENT' in rec for rec in health.recommendations):
                    health.recommendations.append(
                        f"EXCELLENT PERFORMANCE: {layer_name} is performing optimally "
                        f"({health.current_hit_rate:.1f}% hit rate, {health.trend_direction} trend)."
                    )
            
            elif health.current_hit_rate < 50 and health.trend_direction != 'improving':
                if not any('OPTIMIZATION NEEDED' in rec for rec in health.recommendations):
                    health.recommendations.append(
                        f"OPTIMIZATION NEEDED: {layer_name} requires attention "
                        f"({health.current_hit_rate:.1f}% hit rate, {health.trend_direction} trend). "
                        "Consider cache warming, key optimization, or size adjustments."
                    )
    
    def _perform_automatic_optimization(self):
        """Perform automatic cache optimization based on patterns"""
        try:
            optimizations_made = []
            
            for layer_name, health in self.health_metrics.items():
                # Only optimize layers with sufficient data
                if layer_name not in self.trends or len(self.trends[layer_name].hit_rates) < 10:
                    continue
                
                # Cleanup expired entries for low-performing caches
                if health.current_hit_rate < 40 and health.trend_direction == 'degrading':
                    if layer_name == 'discovery_cross_session' and self.discovery_cache:
                        cleaned = self.discovery_cache.cleanup_expired()
                        if cleaned > 0:
                            optimizations_made.append(f"Cleaned {cleaned} expired discovery cache entries")
                    
                    elif layer_name == 'hierarchical_chunks' and self.chunk_cache:
                        cleaned = self.chunk_cache.cleanup_expired_entries()
                        if cleaned > 0:
                            optimizations_made.append(f"Cleaned {cleaned} expired chunk cache entries")
                
                # Update last optimization time
                health.last_optimization = datetime.now()
            
            if optimizations_made:
                logger.info(f"Automatic cache optimization completed: {', '.join(optimizations_made)}")
            
        except Exception as e:
            logger.error(f"Failed to perform automatic optimization: {e}")
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance dashboard data.
        
        Returns:
            Performance dashboard with all metrics and recommendations
        """
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_active': self._monitoring,
            'overall_health': self._calculate_overall_health(),
            'layer_health': {},
            'system_recommendations': [],
            'memory_usage': self._get_memory_summary(),
            'trends': self._get_trend_summary()
        }
        
        # Add per-layer health metrics
        for layer_name, health in self.health_metrics.items():
            dashboard['layer_health'][layer_name] = {
                'health_score': health.health_score,
                'hit_rate_current': health.current_hit_rate,
                'hit_rate_average': health.average_hit_rate,
                'trend': health.trend_direction,
                'memory_mb': health.memory_usage_mb,
                'cache_size': health.cache_size,
                'recommendations': health.recommendations,
                'last_optimization': health.last_optimization.isoformat() if health.last_optimization else None
            }
        
        # Generate system-wide recommendations
        dashboard['system_recommendations'] = self._get_system_recommendations()
        
        return dashboard
    
    def _calculate_overall_health(self) -> float:
        """Calculate overall system health score"""
        if not self.health_metrics:
            return 0.0
        
        scores = [health.health_score for health in self.health_metrics.values()]
        return sum(scores) / len(scores) if scores else 0.0
    
    def _get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary"""
        total_memory = sum(health.memory_usage_mb for health in self.health_metrics.values())
        
        return {
            'total_mb': round(total_memory, 2),
            'limit_mb': self.memory_limits['total_system'],
            'utilization_percent': round((total_memory / self.memory_limits['total_system']) * 100, 1),
            'by_layer': {
                layer: health.memory_usage_mb 
                for layer, health in self.health_metrics.items()
            }
        }
    
    def _get_trend_summary(self) -> Dict[str, Any]:
        """Get performance trend summary"""
        trends_summary = {}
        
        for layer_name, trend in self.trends.items():
            if trend.hit_rates:
                trends_summary[layer_name] = {
                    'direction': trend.get_trend_direction(),
                    'average_hit_rate': round(trend.get_average_hit_rate(), 2),
                    'average_response_time': round(trend.get_average_response_time(), 2),
                    'measurements': len(trend.hit_rates),
                    'latest_hit_rate': trend.hit_rates[-1] if trend.hit_rates else 0
                }
        
        return trends_summary
    
    def _get_system_recommendations(self) -> List[str]:
        """Generate system-wide optimization recommendations"""
        recommendations = []
        
        overall_health = self._calculate_overall_health()
        if overall_health < 60:
            recommendations.append(
                f"SYSTEM HEALTH ALERT: Overall cache health is {overall_health:.1f}/100. "
                "Multiple cache layers need optimization."
            )
        
        total_memory = sum(health.memory_usage_mb for health in self.health_metrics.values())
        if total_memory > self.memory_limits['total_system'] * 0.8:
            recommendations.append(
                f"MEMORY WARNING: Cache system using {total_memory:.1f}MB "
                f"({(total_memory/self.memory_limits['total_system']*100):.1f}% of limit). "
                "Consider implementing more aggressive eviction policies."
            )
        
        # Check for consistently low-performing layers
        low_performers = [
            layer for layer, health in self.health_metrics.items()
            if health.current_hit_rate < 40 and health.trend_direction != 'improving'
        ]
        
        if len(low_performers) > 1:
            recommendations.append(
                f"MULTIPLE PERFORMANCE ISSUES: Layers {', '.join(low_performers)} "
                "are underperforming. Consider reviewing cache architecture."
            )
        
        return recommendations
    
    def force_optimization(self):
        """Force immediate cache optimization across all layers"""
        logger.info("Starting forced cache optimization")
        self._perform_automatic_optimization()
        logger.info("Forced cache optimization completed")
    
    def reset_monitoring_data(self):
        """Reset all monitoring data and trends"""
        with self._lock:
            self.trends.clear()
            self.health_metrics.clear()
            self.last_optimization_check = time.time()
            
            logger.info("Cache monitoring data reset")


# Global instance for cache performance monitoring
_global_performance_monitor: Optional[CachePerformanceMonitor] = None


def get_global_performance_monitor() -> CachePerformanceMonitor:
    """Get the global cache performance monitor instance"""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = CachePerformanceMonitor()
    return _global_performance_monitor


def start_cache_monitoring():
    """Start global cache performance monitoring"""
    monitor = get_global_performance_monitor()
    monitor.start_monitoring()


def stop_cache_monitoring():
    """Stop global cache performance monitoring"""
    monitor = get_global_performance_monitor()
    monitor.stop_monitoring()