"""
Query Context Manager for CodeWise Performance Optimization.

This module provides lifecycle management for QueryExecutionContext instances,
ensuring proper creation, sharing, and cleanup of query contexts across
the entire application lifecycle.

Key Features:
- Centralized context lifecycle management
- Automatic resource cleanup and timeout handling
- Context monitoring and health checking
- Thread-safe context registry
- Graceful error handling and recovery

Author: CodeWise Performance Team
Created: 2025-08-06
Purpose: Manage query execution contexts to prevent memory leaks and ensure proper cleanup
"""

from typing import Dict, List, Optional, AsyncContextManager, Set
from contextlib import asynccontextmanager
import asyncio
import logging
import time
import weakref
from datetime import datetime, timedelta
import threading

from .query_context import (
    QueryExecutionContext, 
    QueryContextError,
    QueryTimeoutError,
    DEFAULT_QUERY_TIMEOUT_SECONDS,
    DEFAULT_CONTEXT_CLEANUP_INTERVAL
)

# Configuration constants (following NO HARDCODING principle)
MAX_CONCURRENT_CONTEXTS = 50  # Prevent resource exhaustion
CONTEXT_REGISTRY_CLEANUP_INTERVAL = 300  # 5 minutes
STALE_CONTEXT_THRESHOLD_MINUTES = 15  # Consider context stale after 15 minutes
HEALTH_CHECK_INTERVAL_SECONDS = 30  # Health check every 30 seconds

logger = logging.getLogger(__name__)

class ContextManagerError(Exception):
    """Base exception for context manager operations."""
    pass

class ContextLimitExceededError(ContextManagerError):
    """Raised when maximum concurrent contexts limit is exceeded."""
    pass

class ContextRegistryError(ContextManagerError):
    """Raised when context registry operations fail."""
    pass

class QueryContextManager:
    """
    Manages query execution contexts with proper lifecycle management.
    
    This class ensures that QueryExecutionContext instances are:
    - Created with proper initialization
    - Shared safely across concurrent operations
    - Cleaned up automatically to prevent memory leaks
    - Monitored for health and performance
    
    Thread Safety:
        All operations are thread-safe using async locks and proper synchronization.
        
    Performance:
        - Efficient context lookup using dictionary
        - Automatic cleanup of stale contexts
        - Resource usage monitoring and limiting
        
    Error Handling:
        - Graceful degradation on context failures
        - Comprehensive error logging and reporting
        - Automatic recovery from transient errors
    """
    
    def __init__(
        self,
        max_concurrent_contexts: int = MAX_CONCURRENT_CONTEXTS,
        cleanup_interval_seconds: int = DEFAULT_CONTEXT_CLEANUP_INTERVAL,
        enable_health_monitoring: bool = True
    ):
        """
        Initialize the query context manager.
        
        Args:
            max_concurrent_contexts: Maximum number of concurrent contexts
            cleanup_interval_seconds: Interval between cleanup operations
            enable_health_monitoring: Whether to enable health monitoring
        """
        # Validate configuration (defensive programming)
        if max_concurrent_contexts <= 0:
            raise ValueError(f"max_concurrent_contexts must be positive, got {max_concurrent_contexts}")
        
        if cleanup_interval_seconds <= 0:
            raise ValueError(f"cleanup_interval_seconds must be positive, got {cleanup_interval_seconds}")
        
        self.max_concurrent_contexts = max_concurrent_contexts
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.enable_health_monitoring = enable_health_monitoring
        
        # Context registry (thread-safe)
        self._active_contexts: Dict[str, QueryExecutionContext] = {}
        self._context_lock = asyncio.Lock()
        
        # Cleanup and monitoring state
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._is_shutdown = False
        
        # Performance metrics
        self._total_contexts_created = 0
        self._total_contexts_completed = 0
        self._total_contexts_failed = 0
        self._total_contexts_timeout = 0
        self._manager_start_time = datetime.now()
        
        # Thread-safe set for tracking context IDs being cleaned up
        self._cleanup_in_progress: Set[str] = set()
        self._cleanup_lock = threading.Lock()
        
        logger.info(f"🎯 CONTEXT MANAGER: Initialized with max_contexts={max_concurrent_contexts}")
        
        # Start background tasks if monitoring is enabled
        if self.enable_health_monitoring:
            self._start_background_tasks()
    
    def _start_background_tasks(self) -> None:
        """Start background cleanup and health monitoring tasks."""
        try:
            # Note: Tasks will be created when first context is created in async context
            logger.info("📊 CONTEXT MANAGER: Background monitoring enabled")
        except Exception as e:
            logger.error(f"❌ CONTEXT MANAGER: Failed to start background tasks: {e}")
            # Don't raise - manager should still work without background tasks
    
    async def _ensure_background_tasks(self) -> None:
        """Ensure background tasks are running (called from async context)."""
        if not self.enable_health_monitoring or self._is_shutdown:
            return
        
        try:
            # Start cleanup task if not running
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
                logger.debug("🧹 CONTEXT MANAGER: Started cleanup task")
            
            # Start health monitoring task if not running  
            if self._health_task is None or self._health_task.done():
                self._health_task = asyncio.create_task(self._health_monitor_loop())
                logger.debug("🏥 CONTEXT MANAGER: Started health monitoring task")
                
        except Exception as e:
            logger.error(f"❌ CONTEXT MANAGER: Failed to ensure background tasks: {e}")
    
    @asynccontextmanager
    async def create_query_context(
        self, 
        query: str, 
        project: str = "",
        timeout_seconds: float = DEFAULT_QUERY_TIMEOUT_SECONDS
    ) -> AsyncContextManager[QueryExecutionContext]:
        """
        Create and manage a query execution context with proper lifecycle.
        
        This is the main entry point for creating contexts. It ensures:
        - Context is properly initialized
        - Resource limits are enforced
        - Automatic cleanup on completion or failure
        - Comprehensive error handling
        
        Usage:
            async with context_manager.create_query_context("search query") as ctx:
                # All tool executions share this context
                result1 = await tool1.execute_with_context(ctx)
                result2 = await tool2.execute_with_context(ctx)
                
        Args:
            query: The user query string
            project: Project context (optional)
            timeout_seconds: Query timeout in seconds
            
        Yields:
            QueryExecutionContext: The created context
            
        Raises:
            ContextLimitExceededError: If max concurrent contexts exceeded
            QueryContextError: If context creation fails
            ValueError: If invalid parameters provided
        """
        # Validate inputs (defensive programming)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        
        if timeout_seconds <= 0:
            raise ValueError(f"Timeout must be positive, got {timeout_seconds}")
        
        # Check concurrent context limit
        async with self._context_lock:
            if len(self._active_contexts) >= self.max_concurrent_contexts:
                self._total_contexts_failed += 1
                raise ContextLimitExceededError(
                    f"Maximum concurrent contexts exceeded "
                    f"({len(self._active_contexts)}/{self.max_concurrent_contexts})"
                )
        
        # Create context
        context = None
        context_created = False
        
        try:
            context = QueryExecutionContext(
                query=query.strip(),
                project=project.strip() if project else "",
                timeout_seconds=timeout_seconds
            )
            
            await context.initialize_locks()
            context_created = True
            
            # Register context
            async with self._context_lock:
                self._active_contexts[context.query_id] = context
                self._total_contexts_created += 1
            
            # Ensure background tasks are running
            await self._ensure_background_tasks()
            
            logger.info(
                f"🎯 CREATED CONTEXT: {context.query_id[:8]} "
                f"(active: {len(self._active_contexts)}/{self.max_concurrent_contexts})"
            )
            
            # Track context creation in performance metrics
            start_time = time.perf_counter()
            
            try:
                yield context
                
                # Context completed successfully
                execution_time = time.perf_counter() - start_time
                logger.info(
                    f"✅ CONTEXT SUCCESS: {context.query_id[:8]} in {execution_time:.2f}s"
                )
                self._total_contexts_completed += 1
                
            except QueryTimeoutError as e:
                logger.error(f"⏱️ CONTEXT TIMEOUT: {context.query_id[:8]}: {e}")
                self._total_contexts_timeout += 1
                raise
                
            except Exception as e:
                logger.error(f"❌ CONTEXT FAILED: {context.query_id[:8]}: {e}")
                self._total_contexts_failed += 1
                raise
                
        except Exception as e:
            if not context_created:
                logger.error(f"❌ CONTEXT CREATION FAILED: {e}")
                self._total_contexts_failed += 1
            raise
            
        finally:
            # Cleanup context (always executed)
            if context is not None:
                await self._cleanup_context(context)
    
    async def _cleanup_context(self, context: QueryExecutionContext) -> None:
        """
        Clean up a single context with comprehensive resource cleanup.
        
        Args:
            context: The context to clean up
        """
        context_id_short = context.query_id[:8]
        
        # Prevent duplicate cleanup
        with self._cleanup_lock:
            if context.query_id in self._cleanup_in_progress:
                logger.debug(f"🔄 CONTEXT {context_id_short}: Cleanup already in progress")
                return
            self._cleanup_in_progress.add(context.query_id)
        
        try:
            logger.info(f"🧹 CLEANING CONTEXT: {context_id_short}")
            
            # Remove from active contexts
            async with self._context_lock:
                if context.query_id in self._active_contexts:
                    del self._active_contexts[context.query_id]
            
            # Execute context cleanup
            await context.cleanup()
            
            # Log execution summary for monitoring
            summary = context.get_execution_summary()
            logger.info(f"🏁 COMPLETED CONTEXT: {context_id_short} - {self._format_summary(summary)}")
            
            # Log performance metrics if discovery was run
            if summary['discovery_executed']:
                logger.info(
                    f"📊 DISCOVERY PERFORMANCE: {context_id_short} - "
                    f"Duration: {summary['discovery_duration_ms']:.1f}ms, "
                    f"Cache hit rate: {summary['cache_hit_rate_percent']:.1f}%"
                )
            
        except Exception as e:
            logger.error(f"❌ CONTEXT CLEANUP FAILED: {context_id_short}: {e}")
            # Don't re-raise - we don't want cleanup failures to propagate
            
        finally:
            # Always remove from cleanup tracking
            with self._cleanup_lock:
                self._cleanup_in_progress.discard(context.query_id)
    
    def _format_summary(self, summary: Dict) -> str:
        """Format context summary for logging."""
        return (
            f"Time: {summary['elapsed_seconds']}s, "
            f"Tools: {summary['tools_executed']}, "
            f"Files: {summary['files_examined']}, "
            f"Cache: {summary['cache_hit_rate_percent']:.1f}%"
        )
    
    async def get_active_contexts(self) -> List[Dict[str, any]]:
        """
        Get information about active contexts for monitoring.
        
        Returns:
            List of active context summaries
        """
        async with self._context_lock:
            active_summaries = []
            
            for context in self._active_contexts.values():
                try:
                    summary = context.get_execution_summary()
                    active_summaries.append(summary)
                except Exception as e:
                    logger.error(f"❌ Failed to get summary for context {context.query_id[:8]}: {e}")
                    # Include minimal info even if summary fails
                    active_summaries.append({
                        'query_id': context.query_id,
                        'status': 'error',
                        'error': str(e)
                    })
            
            return active_summaries
    
    async def get_manager_stats(self) -> Dict[str, any]:
        """
        Get comprehensive manager statistics for monitoring.
        
        Returns:
            Dictionary containing manager performance metrics
        """
        uptime = datetime.now() - self._manager_start_time
        
        async with self._context_lock:
            active_count = len(self._active_contexts)
        
        return {
            # Manager status
            'uptime_seconds': uptime.total_seconds(),
            'is_shutdown': self._is_shutdown,
            'health_monitoring_enabled': self.enable_health_monitoring,
            
            # Context statistics
            'active_contexts': active_count,
            'max_concurrent_contexts': self.max_concurrent_contexts,
            'utilization_percent': (active_count / self.max_concurrent_contexts) * 100,
            
            # Lifecycle counters
            'total_created': self._total_contexts_created,
            'total_completed': self._total_contexts_completed,
            'total_failed': self._total_contexts_failed,
            'total_timeout': self._total_contexts_timeout,
            
            # Performance metrics
            'success_rate_percent': (
                (self._total_contexts_completed / max(self._total_contexts_created, 1)) * 100
            ),
            'failure_rate_percent': (
                (self._total_contexts_failed / max(self._total_contexts_created, 1)) * 100
            ),
            'timeout_rate_percent': (
                (self._total_contexts_timeout / max(self._total_contexts_created, 1)) * 100
            ),
            
            # Background task status
            'cleanup_task_running': self._cleanup_task is not None and not self._cleanup_task.done(),
            'health_task_running': self._health_task is not None and not self._health_task.done(),
        }
    
    async def _cleanup_loop(self) -> None:
        """Background task to clean up stale contexts."""
        logger.info(f"🧹 CONTEXT MANAGER: Started cleanup loop (interval: {self.cleanup_interval_seconds}s)")
        
        while not self._is_shutdown:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)
                
                if self._is_shutdown:
                    break
                
                await self._cleanup_stale_contexts()
                
            except asyncio.CancelledError:
                logger.info("🧹 CONTEXT MANAGER: Cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"❌ CONTEXT MANAGER: Cleanup loop error: {e}")
                # Continue loop even on error
                await asyncio.sleep(5)  # Short delay before retry
        
        logger.info("🧹 CONTEXT MANAGER: Cleanup loop stopped")
    
    async def _cleanup_stale_contexts(self) -> None:
        """Clean up stale or timed-out contexts."""
        now = datetime.now()
        stale_contexts = []
        
        async with self._context_lock:
            for context in list(self._active_contexts.values()):
                elapsed = now - context.start_time
                
                # Check if context is stale or timed out
                if (elapsed.total_seconds() > context.timeout_seconds or 
                    elapsed.total_seconds() > STALE_CONTEXT_THRESHOLD_MINUTES * 60):
                    stale_contexts.append(context)
        
        # Clean up stale contexts
        for context in stale_contexts:
            logger.warning(
                f"🗑️ CONTEXT MANAGER: Cleaning stale context {context.query_id[:8]} "
                f"(age: {(now - context.start_time).total_seconds():.1f}s)"
            )
            await self._cleanup_context(context)
        
        if stale_contexts:
            logger.info(f"🧹 CONTEXT MANAGER: Cleaned up {len(stale_contexts)} stale contexts")
    
    async def _health_monitor_loop(self) -> None:
        """Background task to monitor context manager health."""
        logger.info(f"🏥 CONTEXT MANAGER: Started health monitoring (interval: {HEALTH_CHECK_INTERVAL_SECONDS}s)")
        
        while not self._is_shutdown:
            try:
                await asyncio.sleep(HEALTH_CHECK_INTERVAL_SECONDS)
                
                if self._is_shutdown:
                    break
                
                await self._perform_health_check()
                
            except asyncio.CancelledError:
                logger.info("🏥 CONTEXT MANAGER: Health monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"❌ CONTEXT MANAGER: Health monitoring error: {e}")
                # Continue monitoring even on error
                await asyncio.sleep(10)  # Longer delay on error
        
        logger.info("🏥 CONTEXT MANAGER: Health monitoring stopped")
    
    async def _perform_health_check(self) -> None:
        """Perform health check and log status."""
        try:
            stats = await self.get_manager_stats()
            active_contexts = await self.get_active_contexts()
            
            # Log health status periodically
            if self._total_contexts_created % 10 == 0 or len(active_contexts) > self.max_concurrent_contexts * 0.8:
                logger.info(
                    f"🏥 CONTEXT MANAGER HEALTH: "
                    f"Active: {stats['active_contexts']}/{stats['max_concurrent_contexts']}, "
                    f"Success rate: {stats['success_rate_percent']:.1f}%, "
                    f"Uptime: {stats['uptime_seconds']:.0f}s"
                )
            
            # Warn if approaching capacity
            if stats['utilization_percent'] > 90:
                logger.warning(
                    f"⚠️ CONTEXT MANAGER: High utilization "
                    f"({stats['utilization_percent']:.1f}%)"
                )
            
            # Warn if high failure rate
            if stats['failure_rate_percent'] > 10 and self._total_contexts_created > 10:
                logger.warning(
                    f"⚠️ CONTEXT MANAGER: High failure rate "
                    f"({stats['failure_rate_percent']:.1f}%)"
                )
                
        except Exception as e:
            logger.error(f"❌ CONTEXT MANAGER: Health check failed: {e}")
    
    async def shutdown(self) -> None:
        """
        Gracefully shut down the context manager.
        
        This method should be called when the application is shutting down
        to ensure all contexts are properly cleaned up.
        """
        logger.info("🔄 CONTEXT MANAGER: Starting graceful shutdown")
        
        self._is_shutdown = True
        
        try:
            # Cancel background tasks
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            if self._health_task and not self._health_task.done():
                self._health_task.cancel()
                try:
                    await self._health_task
                except asyncio.CancelledError:
                    pass
            
            # Clean up all active contexts
            async with self._context_lock:
                active_contexts = list(self._active_contexts.values())
            
            for context in active_contexts:
                try:
                    await self._cleanup_context(context)
                except Exception as e:
                    logger.error(f"❌ Failed to cleanup context {context.query_id[:8]} during shutdown: {e}")
            
            # Final statistics
            stats = await self.get_manager_stats()
            logger.info(
                f"✅ CONTEXT MANAGER SHUTDOWN: "
                f"Created: {stats['total_created']}, "
                f"Completed: {stats['total_completed']}, "
                f"Failed: {stats['total_failed']}, "
                f"Success rate: {stats['success_rate_percent']:.1f}%"
            )
            
        except Exception as e:
            logger.error(f"❌ CONTEXT MANAGER: Shutdown error: {e}")
        
        logger.info("🔚 CONTEXT MANAGER: Shutdown complete")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if not self._is_shutdown:
            logger.warning("⚠️ CONTEXT MANAGER: Destructor called without proper shutdown")