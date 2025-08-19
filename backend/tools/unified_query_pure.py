"""
Pure Unified Query Tool - Zero Custom Logic

This is a pure function version of unified_query that eliminates all custom 
reasoning in favor of SDK-native intelligence. The SDK handles all query 
classification and reasoning - this tool just executes searches.

Architecture:
- Input: Query string, filters, analysis mode
- Process: Execute search using Phase 1/2 infrastructure
- Output: Raw search results for SDK to reason about
- NO: Custom classification, reasoning, or decision making
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

async def query_codebase_pure(query: str, 
                             filters: Optional[Dict] = None, 
                             analysis_mode: str = "auto") -> Dict[str, Any]:
    """
    PURE codebase query function with zero custom logic.
    
    The SDK handles all reasoning - this function just executes searches
    and returns raw data for the SDK to analyze and present.
    
    Args:
        query: Natural language query string
        filters: Optional filters for file type, directory, etc.
        analysis_mode: Hint for search strategy ("auto", "structural_kg", "semantic_rag", "specific_symbol")
        
    Returns:
        Raw search results dictionary for SDK processing
    """
    try:
        logger.info(f"🔧 PURE QUERY EXECUTION: '{query[:50]}...' (mode: {analysis_mode})")
        
        # Import infrastructure components with fallbacks
        search_results = []
        total_results = 0
        strategy_used = "fallback"
        
        # Try different search strategies based on mode hint
        if analysis_mode == "structural_kg":
            search_results, strategy_used = await _execute_kg_search(query, filters)
        elif analysis_mode == "semantic_rag":
            search_results, strategy_used = await _execute_semantic_search(query, filters)
        elif analysis_mode == "specific_symbol":
            search_results, strategy_used = await _execute_symbol_search(query, filters)
        else:
            # Default: try hybrid search
            search_results, strategy_used = await _execute_hybrid_search(query, filters)
        
        total_results = len(search_results)
        
        # Return raw data for SDK to reason about
        response = {
            "success": True,
            "query": query,
            "analysis_mode": analysis_mode,
            "results": search_results,
            "total_results": total_results,
            "strategy": strategy_used,
            "unified_query": {
                "pure_execution": True,
                "sdk_native": True,
                "no_custom_logic": True
            },
            "filters_applied": filters or {}
        }
        
        logger.info(f"✅ Pure query executed: {total_results} results via {strategy_used}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Pure query execution failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "analysis_mode": analysis_mode,
            "results": [],
            "total_results": 0,
            "strategy": "error_fallback",
            "unified_query": {"error": True}
        }

async def _execute_kg_search(query: str, filters: Optional[Dict]) -> tuple[List[Dict], str]:
    """Pure KG search execution"""
    try:
        # Try to import and use KG
        from storage.database_manager import DatabaseManager
        db = DatabaseManager()
        
        # Simple KG query without custom logic
        results = []
        nodes = db.get_nodes_by_type("function")  # Basic search
        for node in nodes[:10]:  # Limit results
            if query.lower() in node.get("name", "").lower():
                results.append({
                    "type": "kg_node",
                    "data": node,
                    "source": "knowledge_graph"
                })
        
        return results, "kg_direct"
        
    except Exception as e:
        logger.warning(f"KG search failed: {e}")
        return [], "kg_fallback"

async def _execute_semantic_search(query: str, filters: Optional[Dict]) -> tuple[List[Dict], str]:
    """Pure semantic search execution"""
    try:
        # Try to use vector search
        from vector_store import get_vector_store
        vs = get_vector_store()
        
        # Basic vector search
        results = await vs.similarity_search(query, k=10)
        formatted_results = []
        
        for result in results:
            formatted_results.append({
                "type": "semantic_result",
                "content": result.get("content", ""),
                "file_path": result.get("file_path", ""),
                "score": result.get("score", 0.0),
                "source": "vector_search"
            })
        
        return formatted_results, "semantic_vector"
        
    except Exception as e:
        logger.warning(f"Semantic search failed: {e}")
        return [], "semantic_fallback"

async def _execute_symbol_search(query: str, filters: Optional[Dict]) -> tuple[List[Dict], str]:
    """Pure symbol search execution"""
    try:
        # Try to use KG for symbol search
        from storage.database_manager import DatabaseManager
        db = DatabaseManager()
        
        results = []
        
        # First try: Search for symbols by name
        symbols = db.get_nodes_by_name(query, exact_match=False)
        for symbol in symbols[:5]:
            results.append({
                "type": "symbol_match",
                "data": symbol,
                "content": f"{symbol.get('type', 'symbol')}: {symbol.get('name', 'unknown')}",
                "file_path": symbol.get('file_path', ''),
                "source": "knowledge_graph_symbols"
            })
        
        # Second try: If query looks like a project name (contains @ or _Project), search by file path
        if '@' in query or 'Project' in query:
            # Extract project name from query (e.g., "explain @SWE_Project architecture" -> "SWE_Project")
            import re
            project_match = re.search(r'@([A-Za-z_][A-Za-z0-9_]*)', query)
            if project_match:
                project_name = project_match.group(1)
            else:
                # Fallback: look for words containing "Project" 
                words = query.split()
                project_words = [w for w in words if 'Project' in w]
                project_name = project_words[0] if project_words else query.replace('@', '').strip()
            
            logger.info(f"Extracted project name: '{project_name}' from query: '{query[:50]}...'")
            try:
                cursor = db.connection.cursor()
                project_results = cursor.execute(
                    "SELECT * FROM nodes WHERE file_path LIKE ? LIMIT 10",
                    (f'%{project_name}%',)
                ).fetchall()
                
                for result in project_results:
                    node = dict(result)
                    # Parse JSON properties if present
                    if node.get('properties'):
                        import json
                        try:
                            node['properties'] = json.loads(node['properties'])
                        except:
                            pass
                    
                    results.append({
                        "type": "project_file",
                        "data": node,
                        "content": f"File: {node.get('file_path', '').split('/')[-1]} - {node.get('type', 'symbol')}: {node.get('name', 'unknown')}",
                        "file_path": node.get('file_path', ''),
                        "source": "knowledge_graph_project"
                    })
            except Exception as e:
                logger.warning(f"Project file search failed: {e}")
        
        return results[:15], "symbol_kg"  # Return max 15 results
        
    except Exception as e:
        logger.warning(f"Symbol search failed: {e}")
        return [], "symbol_fallback"

async def _execute_hybrid_search(query: str, filters: Optional[Dict]) -> tuple[List[Dict], str]:
    """Pure hybrid search execution"""
    try:
        # Try basic hybrid approach
        results = []
        
        # Combine semantic + symbol search
        semantic_results, _ = await _execute_semantic_search(query, filters)
        symbol_results, _ = await _execute_symbol_search(query, filters)
        
        # Simple combination (no custom scoring)
        results.extend(semantic_results[:5])
        results.extend(symbol_results[:5])
        
        return results, "hybrid_basic"
        
    except Exception as e:
        logger.warning(f"Hybrid search failed: {e}")
        return [{
            "type": "fallback_result",
            "content": f"Search infrastructure error for query: {query}",
            "error": str(e),
            "source": "error_fallback"
        }], "error_fallback"

# Legacy compatibility function name
async def query_codebase(query: str, filters: Optional[Dict] = None, analysis_mode: str = "auto") -> Dict[str, Any]:
    """Legacy compatibility wrapper"""
    return await query_codebase_pure(query, filters, analysis_mode)

# Export the main function
__all__ = ["query_codebase_pure", "query_codebase"]