"""
Unified Query Tool for Phase 3.3.2

Eliminates LLM decision paralysis by creating a single, intelligent entry point 
that internally routes to the optimal retrieval strategy. This moves complex 
decision-making from the probabilistic LLM into deterministic, reliable Python code.

This replaces the complexity of choosing between smart_search, kg_find_symbol,
kg_explore_neighborhood, and kg_find_callers with one intelligent router.
"""

from typing import Dict, List, Optional, Any
import logging
import re
from pathlib import Path

# Phase 3 imports
try:
    from search.query_classifier import QueryClassifier, QueryIntent, QueryAnalysis
    from smart_search import get_smart_search_engine  
    from hybrid_search import HybridSearchEngine
    PHASE3_IMPORTS_AVAILABLE = True
except ImportError:
    # Fallback for when imports fail
    QueryClassifier = None
    QueryIntent = None
    QueryAnalysis = None
    get_smart_search_engine = None
    HybridSearchEngine = None
    PHASE3_IMPORTS_AVAILABLE = False

# Phase 2 KG imports with fallbacks
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'storage'))
    from storage.database_manager import DatabaseManager
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False
    DatabaseManager = None

logger = logging.getLogger(__name__)


class QueryRouter:
    """
    Unified query routing system that eliminates tool choice complexity.
    
    The LLM no longer needs to choose between smart_search, kg_find_symbol,
    kg_explore_neighborhood, and kg_find_callers. This router intelligently
    determines the optimal strategy and executes it.
    """
    
    def __init__(self):
        """Initialize router with all available components"""
        # Check if Phase 3 imports are available
        if not PHASE3_IMPORTS_AVAILABLE:
            logger.warning("⚠️ Phase 3 imports not available - QueryRouter running in degraded mode")
            self.smart_search_engine = None
            self.hybrid_search = None
            self.phase3_available = False
            self.classifier = None
            self.kg_available = False
            self.kg = None
            return
            
        # Core components
        self.smart_search_engine = get_smart_search_engine()
        self.hybrid_search = HybridSearchEngine()
        
        # Phase 3 components
        self.phase3_available = getattr(self.smart_search_engine, 'phase3_available', False)
        
        if self.phase3_available:
            self.classifier = self.smart_search_engine.query_classifier
            logger.info("✅ QueryRouter initialized with Phase 3 intelligence")
        else:
            self.classifier = None
            logger.warning("⚠️ QueryRouter initialized with legacy fallback")
        
        # Phase 2 KG components
        self.kg_available = KG_AVAILABLE and self.phase3_available
        if self.kg_available:
            self.kg = self.smart_search_engine.db_manager
            logger.info("✅ Knowledge Graph available for structural queries")
        else:
            self.kg = None
            logger.warning("⚠️ Knowledge Graph not available - structural queries will use fallback")
    
    async def execute(self, query: str, filters: dict = None, analysis_mode: str = 'auto') -> dict:
        """
        The main entry point for the unified query_codebase tool.
        
        Args:
            query: Natural language query
            filters: Optional filters (file_type, directory, etc.)
            analysis_mode: 'auto', 'structural_kg', 'semantic_rag', 'specific_symbol'
            
        Returns:
            Unified result format with both CodeChunks and GraphNodes
        """
        try:
            logger.info(f"UNIFIED QUERY: '{query[:50]}...' (mode: {analysis_mode})")
            
            # Handle degraded mode
            if not PHASE3_IMPORTS_AVAILABLE:
                return {
                    'results': [],
                    'strategy': 'degraded_mode',
                    'error': 'Phase 3 components not available - imports failed',
                    'total_results': 0,
                    'unified_query': {
                        'original_query': query,
                        'detected_intent': 'unavailable',
                        'confidence': 0.0,
                        'filters_applied': filters or {},
                        'analysis_mode': analysis_mode
                    }
                }
            
            # Step 1: Classify query intent (unless overridden)
            if analysis_mode == 'auto' and self.phase3_available:
                analysis = self.classifier.classify_query(query)
                intent = analysis.intent
                confidence = analysis.confidence
                logger.info(f"Query classified as {intent.value} (confidence: {confidence:.2f})")
            else:
                intent = self._map_mode_to_intent(analysis_mode)
                confidence = 1.0
                analysis = None
                logger.info(f"Using manual mode: {intent.value}")
            
            # Step 2: Route to appropriate internal strategy
            if intent == QueryIntent.STRUCTURAL and self.kg_available:
                # Direct KG queries for structural relationships
                result = await self._execute_kg_direct_query(query, filters)
                
            elif intent == QueryIntent.SPECIFIC_SYMBOL and self.kg_available:
                # Retrieve-and-Expand pattern for symbol queries
                result = await self._execute_symbol_expand_query(query, filters)
                
            else:
                # Hybrid semantic + keyword search for conceptual queries
                result = await self._execute_hybrid_search(query, filters, analysis)
            
            # Add metadata
            result['unified_query'] = {
                'original_query': query,
                'detected_intent': intent.value if intent else 'unknown',
                'confidence': confidence,
                'filters_applied': filters or {},
                'analysis_mode': analysis_mode
            }
            
            logger.info(f"✅ Query completed: {result['strategy']} returned {result.get('total_results', 0)} results")
            return result
        
        except Exception as e:
            logger.error(f"Query routing failed: {e}")
            return {
                'results': [],
                'strategy': 'error',
                'error': str(e),
                'query_analysis': None,
                'total_results': 0,
                'unified_query': {
                    'original_query': query,
                    'detected_intent': 'error',
                    'confidence': 0.0,
                    'filters_applied': filters or {},
                    'analysis_mode': analysis_mode
                }
            }
    
    async def _execute_kg_direct_query(self, query: str, filters: dict = None) -> dict:
        """
        Direct Knowledge Graph queries for structural questions.
        
        Examples: "what calls authenticate_user", "show UserManager inheritance"
        """
        try:
            # Parse query to extract structural intent
            kg_operation = self._parse_structural_query(query)
            
            if kg_operation['type'] == 'find_callers':
                symbol_name = kg_operation['symbol']
                results = self._kg_find_callers(symbol_name)
                
            elif kg_operation['type'] == 'explore_neighborhood':
                symbol_name = kg_operation['symbol']
                max_depth = kg_operation.get('depth', 2)
                results = self._kg_explore_neighborhood(symbol_name, max_depth)
                
            elif kg_operation['type'] == 'find_symbol':
                symbol_name = kg_operation['symbol']
                results = self._kg_find_symbol(symbol_name)
                
            else:
                # Fallback to hybrid search
                logger.info("No specific KG operation detected, falling back to hybrid search")
                return await self._execute_hybrid_search(query, filters, None)
            
            return {
                'results': results,
                'strategy': f"kg_direct_{kg_operation['type']}",
                'total_results': len(results),
                'query_analysis': {'intent': 'structural_kg', 'confidence': 0.9}
            }
            
        except Exception as e:
            logger.warning(f"KG direct query failed, falling back to hybrid: {e}")
            return await self._execute_hybrid_search(query, filters, None)
    
    async def _execute_symbol_expand_query(self, query: str, filters: dict = None) -> dict:
        """
        Find specific symbol and expand with neighborhood context.
        
        Examples: "ProcessPayment function", "UserManager class definition"
        """
        try:
            # Extract symbol name from query
            symbol_name = self._extract_symbol_name(query)
            
            if not symbol_name:
                # Fall back to hybrid search if no clear symbol
                logger.info("No clear symbol found in query, falling back to hybrid search")
                return await self._execute_hybrid_search(query, filters, None)
            
            # Find the symbol in KG
            initial_results = self._kg_find_symbol(symbol_name)
            
            if initial_results:
                # Expand with neighborhood for richer context
                expanded_results = []
                for result in initial_results[:3]:  # Expand top 3 matches
                    try:
                        neighborhood = self._kg_explore_neighborhood(result.get('name', symbol_name), max_depth=1)
                        expanded_results.extend(neighborhood)
                    except Exception as e:
                        logger.debug(f"Failed to expand neighborhood for {result}: {e}")
                
                # Combine and deduplicate
                all_results = initial_results + expanded_results
                unique_results = self._deduplicate_results(all_results)
                
                return {
                    'results': unique_results,
                    'strategy': 'symbol_expand',
                    'total_results': len(unique_results),
                    'query_analysis': {'intent': 'specific_symbol', 'confidence': 0.85}
                }
            else:
                # Symbol not found in KG, fall back to hybrid search
                logger.info(f"Symbol '{symbol_name}' not found in KG, using hybrid search")
                return await self._execute_hybrid_search(query, filters, None)
                
        except Exception as e:
            logger.warning(f"Symbol expand query failed, falling back to hybrid: {e}")
            return await self._execute_hybrid_search(query, filters, None)
    
    async def _execute_hybrid_search(self, query: str, filters: dict = None, analysis: QueryAnalysis = None) -> dict:
        """
        Intelligent hybrid search using existing smart_search capabilities.
        
        This is the enhanced version of the original smart_search tool.
        """
        try:
            # Use the enhanced smart search engine
            smart_results = await self.smart_search_engine.search_phase3(query, k=15) if self.phase3_available else await self.smart_search_engine.search(query, k=15)
            
            # Apply filters if provided
            results = smart_results.get('results', [])
            if filters:
                results = self._apply_filters(results, filters)
            
            return {
                'results': results,
                'strategy': 'hybrid_intelligent' if self.phase3_available else 'hybrid_legacy',
                'total_results': len(results),
                'query_analysis': smart_results.get('query_analysis', {'intent': 'general', 'confidence': 0.5}),
                'execution_time': smart_results.get('execution_time', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return {
                'results': [],
                'strategy': 'hybrid_error',
                'total_results': 0,
                'query_analysis': {'intent': 'error', 'confidence': 0.0},
                'error': str(e)
            }
    
    def _map_mode_to_intent(self, analysis_mode: str) -> QueryIntent:
        """Map analysis mode to QueryIntent"""
        mode_mapping = {
            'structural_kg': QueryIntent.STRUCTURAL,
            'semantic_rag': QueryIntent.CONCEPTUAL,
            'specific_symbol': QueryIntent.SPECIFIC_SYMBOL,
            'auto': QueryIntent.EXPLORATORY
        }
        return mode_mapping.get(analysis_mode, QueryIntent.EXPLORATORY)
    
    def _parse_structural_query(self, query: str) -> dict:
        """Parse query to extract structural operation and symbol"""
        query_lower = query.lower()
        
        # Patterns for different KG operations
        caller_patterns = [
            r'what calls? (\w+)',
            r'who calls? (\w+)',
            r'callers? of (\w+)',
            r'(\w+) callers?'
        ]
        
        neighborhood_patterns = [
            r'explore (\w+)',
            r'neighborhood of (\w+)',
            r'around (\w+)',
            r'related to (\w+)'
        ]
        
        symbol_patterns = [
            r'find (\w+)',
            r'(\w+) definition',
            r'(\w+) function',
            r'(\w+) class',
            r'locate (\w+)'
        ]
        
        # Check for caller patterns
        for pattern in caller_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return {'type': 'find_callers', 'symbol': match.group(1)}
        
        # Check for neighborhood patterns
        for pattern in neighborhood_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return {'type': 'explore_neighborhood', 'symbol': match.group(1)}
        
        # Check for symbol patterns
        for pattern in symbol_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return {'type': 'find_symbol', 'symbol': match.group(1)}
        
        # Default fallback
        return {'type': 'unknown', 'symbol': None}
    
    def _extract_symbol_name(self, query: str) -> Optional[str]:
        """Extract symbol name from query"""
        # Look for common symbol patterns
        patterns = [
            r'(\w+)\s+function',
            r'(\w+)\s+class',
            r'(\w+)\s+method',
            r'function\s+(\w+)',
            r'class\s+(\w+)',
            r'method\s+(\w+)',
            r'(\w+)\(\)',  # function call pattern
            r'(\w+)\.(\w+)',  # method call pattern
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # If no specific pattern, try to find camelCase or snake_case identifiers
        identifier_pattern = r'\b([a-z]+[A-Z][a-zA-Z]*|[a-z]+_[a-z_]+)\b'
        match = re.search(identifier_pattern, query)
        if match:
            return match.group(1)
        
        return None
    
    def _kg_find_callers(self, symbol_name: str) -> List[dict]:
        """Find functions that call the given symbol"""
        if not self.kg:
            return []
        
        try:
            callers = self.kg.find_callers(symbol_name, max_depth=3)
            return [{'name': caller['name'], 'type': 'caller', 'file_path': caller['file_path'], 'depth': caller['depth']} for caller in callers]
        except Exception as e:
            logger.error(f"KG find_callers failed: {e}")
            return []
    
    def _kg_explore_neighborhood(self, symbol_name: str, max_depth: int = 2) -> List[dict]:
        """Explore neighborhood around a symbol"""
        if not self.kg:
            return []
        
        try:
            dependencies = self.kg.find_dependencies(symbol_name, max_depth=max_depth)
            return [{'name': dep['name'], 'type': 'dependency', 'file_path': dep['file_path'], 'relationship': dep['relationship']} for dep in dependencies]
        except Exception as e:
            logger.error(f"KG explore_neighborhood failed: {e}")
            return []
    
    def _kg_find_symbol(self, symbol_name: str) -> List[dict]:
        """Find symbol definition in KG"""
        if not self.kg:
            return []
        
        try:
            symbols = self.kg.get_nodes_by_name(symbol_name, exact_match=True)
            return [{'name': symbol['name'], 'type': symbol['type'], 'file_path': symbol['file_path'], 'line_start': symbol['line_start']} for symbol in symbols]
        except Exception as e:
            logger.error(f"KG find_symbol failed: {e}")
            return []
    
    def _deduplicate_results(self, results: List[dict]) -> List[dict]:
        """Remove duplicate results based on name and file_path"""
        seen = set()
        unique_results = []
        
        for result in results:
            key = (result.get('name', ''), result.get('file_path', ''))
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return unique_results
    
    def _apply_filters(self, results: List[dict], filters: dict) -> List[dict]:
        """Apply filters to results"""
        filtered_results = results
        
        if 'file_type' in filters:
            file_type = filters['file_type']
            filtered_results = [r for r in filtered_results if Path(r.get('file_path', '')).suffix == file_type]
        
        if 'directory' in filters:
            directory = filters['directory']
            filtered_results = [r for r in filtered_results if directory in r.get('file_path', '')]
        
        if 'symbol_type' in filters:
            symbol_type = filters['symbol_type']
            filtered_results = [r for r in filtered_results if r.get('type', '') == symbol_type]
        
        return filtered_results


# Singleton instance
_query_router = None

def get_query_router() -> QueryRouter:
    """Get singleton query router instance"""
    global _query_router
    if _query_router is None:
        _query_router = QueryRouter()
    return _query_router


async def query_codebase(query: str, filters: dict = None, analysis_mode: str = 'auto') -> dict:
    """
    Main unified query function - the single entry point for all codebase queries.
    
    This replaces the need to choose between smart_search, kg_find_symbol,
    kg_explore_neighborhood, and kg_find_callers.
    
    Args:
        query: Natural language query
        filters: Optional filters (file_type, directory, symbol_type, etc.)
        analysis_mode: 'auto', 'structural_kg', 'semantic_rag', 'specific_symbol'
        
    Returns:
        Unified result dictionary with results, strategy, and metadata
    """
    router = get_query_router()
    return await router.execute(query, filters, analysis_mode)