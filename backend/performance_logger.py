#!/usr/bin/env python3
"""
Performance Logger Utility for CodeWise
Provides comprehensive timing and bottleneck identification across the query processing pipeline.
"""

import time
import logging
import functools
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import contextmanager
import asyncio

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Centralized performance metrics collection"""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.current_request_timings: Dict[str, float] = {}
        self.request_start_time: Optional[float] = None
        
    def start_request(self):
        """Mark the start of a new request"""
        self.request_start_time = time.time()
        self.current_request_timings.clear()
        
    def record_timing(self, operation: str, duration: float):
        """Record timing for a specific operation"""
        if operation not in self.timings:
            self.timings[operation] = []
        self.timings[operation].append(duration)
        self.current_request_timings[operation] = duration
        
    def get_total_request_time(self) -> Optional[float]:
        """Get total time for current request"""
        if self.request_start_time:
            return time.time() - self.request_start_time
        return None
        
    def log_request_summary(self):
        """Log comprehensive summary of current request performance"""
        total_time = self.get_total_request_time()
        if not total_time:
            return
            
        logger.info("🚀 PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📊 Total Request Time: {total_time:.3f}s")
        
        # Sort timings by duration (longest first)
        sorted_timings = sorted(
            self.current_request_timings.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        logger.info("🔍 Operation Breakdown:")
        for operation, duration in sorted_timings:
            percentage = (duration / total_time) * 100
            logger.info(f"   {operation:.<40} {duration:>8.3f}s ({percentage:>5.1f}%)")
            
        # Identify bottlenecks (operations taking >10% of total time)
        bottlenecks = [(op, dur) for op, dur in sorted_timings if (dur / total_time) > 0.1]
        if bottlenecks:
            logger.info("⚠️  BOTTLENECKS DETECTED (>10% of total time):")
            for operation, duration in bottlenecks:
                percentage = (duration / total_time) * 100
                logger.info(f"   🚨 {operation}: {duration:.3f}s ({percentage:.1f}%)")
        
        logger.info("=" * 60)
        
    def get_average_timings(self) -> Dict[str, float]:
        """Get average timings across all requests"""
        return {
            operation: sum(times) / len(times)
            for operation, times in self.timings.items()
            if times
        }

# Global metrics instance
_metrics = PerformanceMetrics()

def get_metrics() -> PerformanceMetrics:
    """Get the global metrics instance"""
    return _metrics

@contextmanager
def time_operation(operation_name: str):
    """Context manager for timing operations"""
    start_time = time.time()
    logger.info(f"🔄 Starting: {operation_name}")
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        _metrics.record_timing(operation_name, duration)
        logger.info(f"✅ Completed: {operation_name} ({duration:.3f}s)")

def time_function(operation_name: Optional[str] = None):
    """Decorator for timing function execution"""
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"
            
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with time_operation(operation_name):
                return func(*args, **kwargs)
                
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with time_operation(operation_name):
                return await func(*args, **kwargs)
                
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

@contextmanager
def time_block(block_name: str):
    """Context manager for timing code blocks"""
    start_time = time.time()
    logger.debug(f"🔄 Block start: {block_name}")
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        _metrics.record_timing(f"Block: {block_name}", duration)
        logger.debug(f"✅ Block end: {block_name} ({duration:.3f}s)")

def log_memory_usage():
    """Log current memory usage if psutil is available"""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(f"💾 Memory Usage: {memory_info.rss / 1024 / 1024:.1f}MB RSS, {memory_info.vms / 1024 / 1024:.1f}MB VMS")
    except ImportError:
        # psutil not available
        pass
    except Exception as e:
        logger.debug(f"Failed to get memory usage: {e}")

def start_request_timing():
    """Start timing a new request"""
    _metrics.start_request()
    logger.info("⏱️  REQUEST TIMING STARTED")
    log_memory_usage()

def end_request_timing():
    """End timing for current request and log summary"""
    _metrics.log_request_summary()
    log_memory_usage()

def log_operation_start(operation: str, details: Optional[Dict[str, Any]] = None):
    """Log the start of a major operation with optional details"""
    timestamp = datetime.now().isoformat()
    logger.info(f"🔄 OPERATION START: {operation} at {timestamp}")
    if details:
        for key, value in details.items():
            logger.info(f"   {key}: {value}")

def log_operation_end(operation: str, duration: float, details: Optional[Dict[str, Any]] = None):
    """Log the end of a major operation with timing"""
    timestamp = datetime.now().isoformat()
    logger.info(f"✅ OPERATION END: {operation} ({duration:.3f}s) at {timestamp}")
    if details:
        for key, value in details.items():
            logger.info(f"   {key}: {value}")

# Additional utility functions for specific bottleneck monitoring

def log_search_performance(query: str, results_count: int, duration: float, search_type: str = "hybrid"):
    """Log search performance details"""
    logger.info(f"🔍 SEARCH PERFORMANCE ({search_type}):")
    logger.info(f"   Query: {query[:100]}{'...' if len(query) > 100 else ''}")
    logger.info(f"   Results: {results_count} chunks")
    logger.info(f"   Duration: {duration:.3f}s")
    logger.info(f"   Rate: {results_count/duration:.1f} results/sec" if duration > 0 else "   Rate: N/A")

def log_llm_performance(provider: str, model: str, input_tokens: int, output_tokens: int, duration: float):
    """Log LLM API call performance"""
    logger.info(f"🤖 LLM PERFORMANCE ({provider}/{model}):")
    logger.info(f"   Input tokens: {input_tokens}")
    logger.info(f"   Output tokens: {output_tokens}")
    logger.info(f"   Duration: {duration:.3f}s")
    logger.info(f"   Input rate: {input_tokens/duration:.1f} tokens/sec" if duration > 0 else "   Input rate: N/A")
    logger.info(f"   Output rate: {output_tokens/duration:.1f} tokens/sec" if duration > 0 else "   Output rate: N/A")

def log_tool_performance(tool_name: str, duration: float, success: bool, details: Optional[Dict[str, Any]] = None):
    """Log tool execution performance"""
    status = "✅ SUCCESS" if success else "❌ FAILED"
    logger.info(f"🔧 TOOL PERFORMANCE: {tool_name} {status} ({duration:.3f}s)")
    if details:
        for key, value in details.items():
            logger.info(f"   {key}: {value}")