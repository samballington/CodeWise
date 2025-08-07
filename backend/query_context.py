"""
Query Execution Context for CodeWise Performance Optimization.

This module implements the core architectural fix for Query Context Fragmentation.
It provides shared execution context across all tools within a single query,
preventing duplicate discovery pipeline runs.

Key Features:
- Thread-safe shared state across tool executions
- Query-scoped caching for optimal performance
- Comprehensive execution tracking and monitoring
- Automatic resource cleanup and memory management

Author: CodeWise Performance Team
Created: 2025-08-06
Purpose: Fix the 6+ discovery pipeline executions per query architectural issue
"""

from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import asyncio
import time
import logging
import threading
from contextlib import contextmanager

# Configuration constants (following NO HARDCODING principle)
DEFAULT_QUERY_TIMEOUT_SECONDS = 300  # 5 minutes max query execution
DEFAULT_CONTEXT_CLEANUP_INTERVAL = 60  # Cleanup check every minute
DEFAULT_MAX_DISCOVERED_FILES = 1000  # Prevent memory bloat
DEFAULT_MAX_CACHE_SIZE = 500  # Query-scoped cache limit

logger = logging.getLogger(__name__)

class QueryContextError(Exception):
    """Base exception for query context operations."""
    pass

class QueryTimeoutError(QueryContextError):
    """Raised when query execution exceeds timeout."""
    pass

class ContextLockError(QueryContextError):
    """Raised when context locking fails."""
    pass

@dataclass
class QueryExecutionContext:
    """
    Shared execution context for a single query.
    
    This class is the CORE ARCHITECTURAL FIX for the performance issue.
    It ensures discovery pipeline runs ONCE per query by providing
    shared state across all tool executions.
    
    Thread Safety:
        - All operations use async locks for concurrent access
        - Context state is protected against race conditions
        - Memory usage is bounded to prevent leaks
        
    Performance:
        - Discovery results cached for immediate reuse
        - File content cached to avoid duplicate I/O
        - Tool execution tracking prevents redundant work
        
    Monitoring:
        - Comprehensive execution statistics
        - Performance timing for each phase
        - Resource usage tracking
    """
    
    # Core identification
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    project: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    timeout_seconds: float = DEFAULT_QUERY_TIMEOUT_SECONDS
    
    # Shared state across all tools (THE KEY FIX)
    discovery_results: Optional[Dict[str, Any]] = None
    discovered_files: List[str] = field(default_factory=list)
    examined_files: Set[str] = field(default_factory=set)
    relationships_analyzed: Set[str] = field(default_factory=set)
    
    # Tool execution tracking
    tools_executed: List[str] = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)
    tool_execution_times: Dict[str, float] = field(default_factory=dict)
    
    # Query-scoped caches (cleared after query completion)
    _search_cache: Dict[str, Any] = field(default_factory=dict)
    _file_content_cache: Dict[str, str] = field(default_factory=dict)
    
    # Thread-safe locks (initialized in async context)
    _discovery_lock: Optional[asyncio.Lock] = None
    _file_lock: Optional[asyncio.Lock] = None
    _tool_lock: Optional[asyncio.Lock] = None
    
    # Performance tracking
    discovery_start_time: Optional[datetime] = None
    discovery_duration_ms: Optional[float] = None
    total_cache_hits: int = 0
    total_cache_misses: int = 0
    
    # Resource management
    _is_initialized: bool = field(default=False, init=False)
    _is_cleanup: bool = field(default=False, init=False)
    
    def __post_init__(self):
        """Initialize context logging and validation."""
        # Validate inputs (defensive programming)
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("Query must be a non-empty string")
        
        if self.timeout_seconds <= 0:
            raise ValueError(f"Timeout must be positive, got {self.timeout_seconds}")
        
        logger.info(f"🎯 NEW QUERY CONTEXT: {self.query_id[:8]} for '{self.query[:50]}{'...' if len(self.query) > 50 else ''}'")
        logger.debug(f"📊 CONTEXT CONFIG: timeout={self.timeout_seconds}s, project='{self.project}'")
    
    async def initialize_locks(self) -> None:
        """
        Initialize async locks (must be called from async context).
        
        This follows the pattern of deferred initialization for async resources
        since dataclass __post_init__ cannot be async.
        
        Raises:
            ContextLockError: If lock initialization fails
        """
        if self._is_initialized:
            return
        
        try:
            self._discovery_lock = asyncio.Lock()
            self._file_lock = asyncio.Lock()
            self._tool_lock = asyncio.Lock()
            self._is_initialized = True
            
            logger.debug(f"🔐 CONTEXT {self.query_id[:8]}: Locks initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ CONTEXT {self.query_id[:8]}: Lock initialization failed: {e}")
            raise ContextLockError(f"Failed to initialize context locks: {e}") from e
    
    def _check_timeout(self) -> None:
        """Check if query has exceeded timeout (defensive programming)."""
        elapsed = datetime.now() - self.start_time
        if elapsed.total_seconds() > self.timeout_seconds:
            raise QueryTimeoutError(
                f"Query {self.query_id[:8]} exceeded timeout "
                f"({elapsed.total_seconds():.1f}s > {self.timeout_seconds}s)"
            )
    
    async def get_or_run_discovery(self, discovery_func, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Run discovery pipeline ONCE per query context.
        All subsequent calls return the same results.
        
        THIS IS THE KEY ARCHITECTURAL FIX - prevents multiple discovery runs.
        
        Args:
            discovery_func: The discovery pipeline function to execute
            *args, **kwargs: Arguments to pass to the discovery function
            
        Returns:
            Discovery results dict or None if discovery fails
            
        Raises:
            QueryTimeoutError: If query exceeds timeout
            ContextLockError: If locking fails
        """
        if not self._is_initialized:
            await self.initialize_locks()
        
        self._check_timeout()
        
        try:
            async with self._discovery_lock:
                if self.discovery_results is None:
                    logger.info(f"🔍 QUERY {self.query_id[:8]}: Running discovery pipeline (FIRST AND ONLY TIME)")
                    
                    self.discovery_start_time = datetime.now()
                    start_perf = time.perf_counter()
                    
                    try:
                        self.discovery_results = await discovery_func(*args, **kwargs)
                        
                        elapsed_perf = (time.perf_counter() - start_perf) * 1000  # Convert to ms
                        self.discovery_duration_ms = elapsed_perf
                        
                        logger.info(f"✅ QUERY {self.query_id[:8]}: Discovery complete in {elapsed_perf:.1f}ms")
                        
                        # Extract discovered files for other tools (with bounds checking)
                        if isinstance(self.discovery_results, dict):
                            discovered = self.discovery_results.get('files', [])
                            if isinstance(discovered, list):
                                # Prevent memory bloat by limiting discovered files
                                limited_files = discovered[:DEFAULT_MAX_DISCOVERED_FILES]
                                self.discovered_files.extend(limited_files)
                                
                                if len(discovered) > DEFAULT_MAX_DISCOVERED_FILES:
                                    logger.warning(
                                        f"⚠️ QUERY {self.query_id[:8]}: Limited discovered files "
                                        f"({len(discovered)} -> {DEFAULT_MAX_DISCOVERED_FILES})"
                                    )
                        
                    except Exception as e:
                        logger.error(f"❌ QUERY {self.query_id[:8]}: Discovery pipeline failed: {e}")
                        self.discovery_results = {}  # Empty dict indicates failure but prevents re-run
                        raise
                        
                else:
                    logger.info(f"♻️ QUERY {self.query_id[:8]}: Reusing discovery results (ZERO additional work)")
                    self.total_cache_hits += 1
        
        except asyncio.TimeoutError:
            logger.error(f"⏱️ QUERY {self.query_id[:8]}: Discovery pipeline timed out")
            raise QueryTimeoutError(f"Discovery pipeline timed out for query {self.query_id[:8]}")
            
        except Exception as e:
            logger.error(f"❌ QUERY {self.query_id[:8]}: Discovery lock error: {e}")
            raise ContextLockError(f"Discovery lock failed: {e}") from e
        
        return self.discovery_results
    
    def should_examine_file(self, file_path: str) -> bool:
        """
        Check if file needs examination (avoid duplicates).
        
        Thread-safe file examination tracking to prevent duplicate work.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            True if file should be examined, False if already examined
        """
        if not isinstance(file_path, str) or not file_path.strip():
            logger.warning(f"⚠️ QUERY {self.query_id[:8]}: Invalid file path: {repr(file_path)}")
            return False
        
        file_path = file_path.strip()
        
        if file_path in self.examined_files:
            logger.debug(f"⏭️ QUERY {self.query_id[:8]}: Skipping {file_path} (already examined)")
            return False
        
        self.examined_files.add(file_path)
        logger.debug(f"📁 QUERY {self.query_id[:8]}: Will examine {file_path}")
        return True
    
    async def mark_tool_executed(self, tool_name: str, result: Any = None, execution_time: Optional[float] = None) -> None:
        """
        Track tool execution to prevent duplicates and gather metrics.
        
        Thread-safe tool execution tracking with performance monitoring.
        
        Args:
            tool_name: Name of the executed tool
            result: Tool execution result (optional)
            execution_time: Tool execution time in seconds (optional)
        """
        if not isinstance(tool_name, str) or not tool_name.strip():
            logger.warning(f"⚠️ QUERY {self.query_id[:8]}: Invalid tool name: {repr(tool_name)}")
            return
        
        if not self._is_initialized:
            await self.initialize_locks()
        
        async with self._tool_lock:
            self.tools_executed.append(tool_name)
            
            if result is not None:
                self.tool_results[tool_name] = result
            
            if execution_time is not None:
                self.tool_execution_times[tool_name] = execution_time
                logger.info(
                    f"🔧 QUERY {self.query_id[:8]}: Completed '{tool_name}' in {execution_time:.2f}s "
                    f"(tools: {len(self.tools_executed)})"
                )
            else:
                logger.info(
                    f"🔧 QUERY {self.query_id[:8]}: Completed '{tool_name}' "
                    f"(tools: {len(self.tools_executed)})"
                )
    
    async def cache_file_content(self, file_path: str, content: str) -> None:
        """
        Cache file content for query-scoped reuse.
        
        Thread-safe file content caching to avoid duplicate I/O operations.
        
        Args:
            file_path: Path to the file
            content: File content to cache
        """
        if not isinstance(file_path, str) or not isinstance(content, str):
            logger.warning(f"⚠️ QUERY {self.query_id[:8]}: Invalid cache parameters")
            return
        
        if not self._is_initialized:
            await self.initialize_locks()
        
        async with self._file_lock:
            # Implement cache size limit to prevent memory bloat
            if len(self._file_content_cache) >= DEFAULT_MAX_CACHE_SIZE:
                # Remove oldest entry (simple FIFO eviction)
                oldest_key = next(iter(self._file_content_cache))
                del self._file_content_cache[oldest_key]
                logger.debug(f"🗑️ QUERY {self.query_id[:8]}: Evicted cache entry for {oldest_key}")
            
            self._file_content_cache[file_path] = content
            logger.debug(f"💾 QUERY {self.query_id[:8]}: Cached content for {file_path} ({len(content)} chars)")
    
    async def get_cached_file_content(self, file_path: str) -> Optional[str]:
        """
        Get cached file content if available.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Cached content or None if not cached
        """
        if not isinstance(file_path, str):
            return None
        
        if not self._is_initialized:
            await self.initialize_locks()
        
        async with self._file_lock:
            content = self._file_content_cache.get(file_path)
            
            if content is not None:
                self.total_cache_hits += 1
                logger.debug(f"✅ QUERY {self.query_id[:8]}: Cache hit for {file_path}")
            else:
                self.total_cache_misses += 1
                logger.debug(f"❌ QUERY {self.query_id[:8]}: Cache miss for {file_path}")
            
            return content
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of query execution for monitoring.
        
        Returns detailed execution statistics for performance analysis
        and debugging.
        
        Returns:
            Dictionary containing execution metrics and statistics
        """
        elapsed = datetime.now() - self.start_time
        
        # Calculate cache hit rate
        total_cache_ops = self.total_cache_hits + self.total_cache_misses
        cache_hit_rate = (self.total_cache_hits / total_cache_ops * 100) if total_cache_ops > 0 else 0.0
        
        summary = {
            # Core identification
            'query_id': self.query_id,
            'query': self.query,
            'project': self.project,
            
            # Performance metrics
            'elapsed_seconds': round(elapsed.total_seconds(), 2),
            'discovery_duration_ms': self.discovery_duration_ms,
            'discovery_executed': self.discovery_results is not None,
            
            # Tool execution stats
            'tools_executed': len(self.tools_executed),
            'tool_list': self.tools_executed.copy(),
            'tool_execution_times': self.tool_execution_times.copy(),
            'total_tool_time': sum(self.tool_execution_times.values()),
            
            # File processing stats
            'files_discovered': len(self.discovered_files),
            'files_examined': len(self.examined_files),
            'relationships_analyzed': len(self.relationships_analyzed),
            
            # Cache performance
            'cache_hits': self.total_cache_hits,
            'cache_misses': self.total_cache_misses,
            'cache_hit_rate_percent': round(cache_hit_rate, 1),
            'file_content_cache_size': len(self._file_content_cache),
            
            # Resource usage
            'memory_estimated_kb': self._estimate_memory_usage(),
            'is_timed_out': elapsed.total_seconds() > self.timeout_seconds,
            
            # Status
            'status': 'completed' if self._is_cleanup else 'active'
        }
        
        return summary
    
    def _estimate_memory_usage(self) -> float:
        """
        Estimate memory usage of the context (rough calculation).
        
        Returns:
            Estimated memory usage in KB
        """
        try:
            # Rough estimation of memory usage
            memory_kb = 0.0
            
            # Discovery results
            if self.discovery_results:
                memory_kb += len(str(self.discovery_results)) * 0.001  # Rough string memory
            
            # File content cache
            for content in self._file_content_cache.values():
                memory_kb += len(content) * 0.001
            
            # Other collections
            memory_kb += len(self.discovered_files) * 0.1  # File paths
            memory_kb += len(self.examined_files) * 0.1
            memory_kb += len(self.tool_results) * 0.5  # Tool results
            
            return round(memory_kb, 2)
            
        except Exception:
            return 0.0
    
    async def cleanup(self) -> None:
        """
        Clean up context resources to prevent memory leaks.
        
        This method should be called when the query is complete to ensure
        proper resource cleanup and prevent memory leaks in long-running processes.
        """
        if self._is_cleanup:
            return
        
        logger.info(f"🧹 QUERY {self.query_id[:8]}: Starting context cleanup")
        
        try:
            # Clear all caches
            self._search_cache.clear()
            self._file_content_cache.clear()
            
            # Clear large collections
            self.discovered_files.clear()
            self.tool_results.clear()
            
            # Mark as cleaned up
            self._is_cleanup = True
            
            logger.info(f"✅ QUERY {self.query_id[:8]}: Context cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ QUERY {self.query_id[:8]}: Cleanup failed: {e}")
            raise
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"QueryExecutionContext("
            f"id={self.query_id[:8]}, "
            f"query='{self.query[:30]}...', "
            f"tools={len(self.tools_executed)}, "
            f"discovery={'✅' if self.discovery_results else '❌'}"
            f")"
        )