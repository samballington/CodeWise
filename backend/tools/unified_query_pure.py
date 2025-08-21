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
    """
    Pure Knowledge Graph search execution using graph traversal and relationship analysis.
    
    This implements proper KG search patterns:
    1. Semantic term extraction from natural language
    2. Multi-type node discovery with relationship mapping
    3. Structural hierarchy construction from edges
    4. Context-aware result enrichment
    """
    try:
        from storage.database_manager import DatabaseManager
        db = DatabaseManager()
        cursor = db.connection.cursor()
        
        # Extract search terms from natural language query
        query_terms = _extract_semantic_terms(query)
        
        # Execute multi-phase KG search
        nodes = _discover_relevant_nodes(cursor, query_terms, filters)
        relationships = _discover_relationships(cursor, nodes)
        enriched_results = _build_structural_context(nodes, relationships)
        
        return enriched_results[:50], "kg_direct"
        
    except Exception as e:
        logger.warning(f"KG search failed: {e}")
        return [], "kg_fallback"

def _extract_semantic_terms(query: str) -> Dict[str, List[str]]:
    """Extract meaningful terms from natural language query"""
    query_lower = query.lower()
    
    # Entity type indicators
    type_indicators = {
        'class': ['class', 'classes', 'object', 'entity', 'model'],
        'function': ['function', 'method', 'operation', 'procedure'],
        'variable': ['variable', 'field', 'property', 'attribute'],
        'interface': ['interface', 'contract', 'protocol']
    }
    
    # Relationship indicators  
    relationship_indicators = {
        'hierarchy': ['hierarchy', 'inheritance', 'extends', 'inherits', 'parent', 'child'],
        'dependency': ['depends', 'uses', 'calls', 'imports', 'requires'],
        'composition': ['contains', 'has', 'owns', 'composed'],
        'association': ['relates', 'connects', 'links', 'associated']
    }
    
    # Structure indicators
    structure_indicators = {
        'diagram': ['diagram', 'chart', 'visualization', 'graph'],
        'overview': ['overview', 'summary', 'structure', 'architecture'],
        'detail': ['detail', 'implementation', 'code', 'specific']
    }
    
    # Extract domain terms (anything not a stop word)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'show', 'get', 'find', 'list'}
    words = query_lower.split()
    domain_terms = [w for w in words if len(w) > 2 and w not in stop_words]
    
    return {
        'types': [t for t, indicators in type_indicators.items() if any(ind in query_lower for ind in indicators)],
        'relationships': [r for r, indicators in relationship_indicators.items() if any(ind in query_lower for ind in indicators)],
        'structure': [s for s, indicators in structure_indicators.items() if any(ind in query_lower for ind in indicators)],
        'domain': domain_terms
    }

def _discover_relevant_nodes(cursor, terms: Dict, filters: Optional[Dict]) -> List[Dict]:
    """Discover nodes using semantic term matching across multiple dimensions"""
    
    # Build base query with proper node selection
    base_query = """
        SELECT DISTINCT id, type, name, file_path, line_start, line_end, signature, docstring
        FROM nodes 
        WHERE 1=1
    """
    params = []
    conditions = []
    
    # Apply type filters from semantic analysis
    if terms['types']:
        type_placeholders = ','.join(['?' for _ in terms['types']])
        conditions.append(f"type IN ({type_placeholders})")
        params.extend(terms['types'])
    
    # Apply domain term matching across name, file_path, and signature
    if terms['domain']:
        domain_conditions = []
        for term in terms['domain']:
            domain_conditions.append("(name LIKE ? OR file_path LIKE ? OR signature LIKE ?)")
            pattern = f"%{term}%"
            params.extend([pattern, pattern, pattern])
        
        if domain_conditions:
            conditions.append(f"({' OR '.join(domain_conditions)})")
    
    # Apply user-provided filters
    if filters:
        if filters.get('file_type'):
            conditions.append("file_path LIKE ?")
            params.append(f"%{filters['file_type']}")
        if filters.get('directory'):
            conditions.append("file_path LIKE ?") 
            params.append(f"%{filters['directory']}%")
        if filters.get('symbol_type'):
            conditions.append("type = ?")
            params.append(filters['symbol_type'])
    
    # Construct final query
    if conditions:
        query = base_query + " AND " + " AND ".join(conditions)
    else:
        query = base_query + " LIMIT 100"  # Fallback: get top nodes
    
    query += " ORDER BY type, name LIMIT 100"
    
    # Execute and format results
    rows = cursor.execute(query, params).fetchall()
    
    nodes = []
    for row in rows:
        node_id, node_type, name, file_path, line_start, line_end, signature, docstring = row
        nodes.append({
            'id': node_id,
            'type': node_type,
            'name': name,
            'file_path': file_path,
            'line_start': line_start,
            'line_end': line_end,
            'signature': signature,
            'docstring': docstring
        })
    
    return nodes

def _discover_relationships(cursor, nodes: List[Dict]) -> List[Dict]:
    """Discover relationships between nodes using edges table"""
    if not nodes:
        return []
    
    # Get node IDs for relationship lookup
    node_ids = [node['id'] for node in nodes]
    
    # Query edges table for relationships between discovered nodes
    if len(node_ids) > 0:
        placeholders = ','.join(['?' for _ in node_ids])
        edges_query = f"""
            SELECT source_id, target_id, type, properties
            FROM edges 
            WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
        """
        params = node_ids + node_ids  # For both source and target lookups
        
        try:
            edges = cursor.execute(edges_query, params).fetchall()
            
            relationships = []
            for edge in edges:
                source_id, target_id, rel_type, properties = edge
                relationships.append({
                    'source_id': source_id,
                    'target_id': target_id,
                    'type': rel_type,
                    'properties': properties
                })
            
            return relationships
        except Exception as e:
            logger.warning(f"Edge discovery failed: {e}")
            return []
    
    return []

def _build_structural_context(nodes: List[Dict], relationships: List[Dict]) -> List[Dict]:
    """Build rich structural context by combining nodes with their relationships"""
    
    # Create node lookup for fast access
    node_lookup = {node['id']: node for node in nodes}
    
    # Group relationships by source node
    relationships_by_source = {}
    relationships_by_target = {}
    
    for rel in relationships:
        source_id = rel['source_id']
        target_id = rel['target_id']
        
        if source_id not in relationships_by_source:
            relationships_by_source[source_id] = []
        relationships_by_source[source_id].append(rel)
        
        if target_id not in relationships_by_target:
            relationships_by_target[target_id] = []
        relationships_by_target[target_id].append(rel)
    
    # Enrich each node with its relationship context
    enriched_results = []
    
    for node in nodes:
        node_id = node['id']
        
        # Build enhanced node representation
        enhanced_node = {
            'type': 'kg_structural_node',
            'node_data': node,
            'outgoing_relationships': relationships_by_source.get(node_id, []),
            'incoming_relationships': relationships_by_target.get(node_id, []),
            'connected_nodes': [],
            'source': 'knowledge_graph'
        }
        
        # Add connected node information
        connected_ids = set()
        for rel in enhanced_node['outgoing_relationships']:
            connected_ids.add(rel['target_id'])
        for rel in enhanced_node['incoming_relationships']:
            connected_ids.add(rel['source_id'])
        
        # Lookup connected node details
        for connected_id in connected_ids:
            if connected_id in node_lookup:
                enhanced_node['connected_nodes'].append(node_lookup[connected_id])
        
        enriched_results.append(enhanced_node)
    
    return enriched_results

async def _execute_semantic_search(query: str, filters: Optional[Dict]) -> tuple[List[Dict], str]:
    """Pure semantic search execution"""
    try:
        # Try to use vector search
        from vector_store import get_vector_store
        vs = get_vector_store()
        
        # Basic vector search with project filtering
        results = await vs.similarity_search(query, k=10, filters=filters)
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