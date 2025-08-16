"""
Enhanced Smart Search with Knowledge Graph Integration

Phase 2 Enhanced: Adds Knowledge Graph expansion capabilities to the existing
smart search while maintaining backward compatibility and familiar interface.

Key Enhancement: KG-aware retrieve-and-expand provides relationship context
for queries that benefit from structural understanding.
"""

import logging
import re
import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

# Import existing smart search components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from backend.smart_search import SmartSearchEngine as BaseSmartSearchEngine, QueryIntent, SmartSearchResult
from backend.hybrid_search import HybridSearchEngine, SearchResult

# Import Phase 2 Knowledge Graph components
from storage.database_manager import DatabaseManager
from knowledge_graph.kg_aware_rag import KGAwareRAG
from indexer.enhanced_vector_store import EnhancedVectorStore

logger = logging.getLogger(__name__)


@dataclass
class EnhancedSearchResult(SmartSearchResult):
    """Enhanced search result with KG relationship context."""
    related_symbols: List[str] = None
    relationship_context: List[Dict] = None
    kg_expansion_used: bool = False
    expansion_depth: int = 0
    relationship_types: List[str] = None


class KGQueryIntentAnalyzer:
    """
    Enhanced query intent analyzer that identifies KG-beneficial queries.
    
    Adds intelligence to determine when Knowledge Graph expansion would
    improve search results over pure vector/keyword search.
    """
    
    def __init__(self):
        # KG-beneficial query patterns
        self.kg_beneficial_keywords = {
            # Relationship queries
            'calls', 'uses', 'depends', 'depends on', 'inherits', 'extends', 'implements',
            'related', 'connects', 'linked', 'references', 'imported by', 'exports',
            
            # Architectural queries
            'architecture', 'structure', 'flow', 'dependencies', 'hierarchy', 'components',
            'modules', 'relationships', 'connections', 'interacts', 'communicates',
            
            # Debugging/analysis queries
            'where is used', 'who calls', 'what calls', 'find callers', 'find usage',
            'trace', 'follow', 'impact', 'affected by', 'affects'
        }
        
        # Relationship query patterns
        self.relationship_patterns = [
            r'\bwho\s+(calls|uses|imports)\b',          # "who calls this function"
            r'\bwhat\s+(calls|uses|depends)\b',         # "what calls authenticate_user"
            r'\bfind\s+(callers|usage|dependencies)\b', # "find callers of login"
            r'\bshow\s+(hierarchy|structure|flow)\b',   # "show inheritance hierarchy"
            r'\btrace\s+(calls|dependencies)\b',        # "trace function calls"
            r'\b\w+\s+(inherits|extends)\s+from\b',     # "classes that inherit from"
            r'\brelated\s+to\s+\w+\b',                  # "functions related to auth"
            r'\bimpact\s+of\s+\w+\b',                   # "impact of changing this"
        ]
        
        # Symbol reference patterns (likely to benefit from KG expansion)
        self.symbol_patterns = [
            r'\b[A-Z][a-zA-Z0-9_]*\b',          # PascalCase (likely class names)
            r'\b[a-z_][a-zA-Z0-9_]*\(\)',       # function_name() calls
            r'\b[a-z_][a-zA-Z0-9_]*\.[a-z_]',   # object.method patterns
            r'\bclass\s+\w+\b',                 # class references
            r'\bfunction\s+\w+\b',              # function references
        ]
    
    def should_use_kg_expansion(self, query: str, intent: QueryIntent, 
                               confidence: float) -> Dict[str, Any]:
        """
        Determine if KG expansion would benefit this query.
        
        Args:
            query: Search query
            intent: Detected query intent
            confidence: Intent confidence score
            
        Returns:
            Dictionary with KG recommendation and reasoning
        """
        query_lower = query.lower()
        
        kg_score = 0.0
        reasoning = []
        
        # 1. Intent-based scoring
        intent_scores = {
            QueryIntent.ARCHITECTURE: 0.8,  # Architecture queries benefit greatly
            QueryIntent.ENTITY: 0.6,        # Entity relationships are valuable
            QueryIntent.GENERAL: 0.3,       # General queries may benefit
            QueryIntent.FILE: 0.1           # File queries rarely benefit
        }
        
        intent_score = intent_scores.get(intent, 0.3)
        kg_score += intent_score
        reasoning.append(f"Intent {intent.value} suggests KG benefit: {intent_score}")
        
        # 2. Keyword-based scoring
        keyword_matches = 0
        for keyword in self.kg_beneficial_keywords:
            if keyword in query_lower:
                keyword_matches += 1
                kg_score += 0.2
        
        if keyword_matches > 0:
            reasoning.append(f"Found {keyword_matches} KG-beneficial keywords")
        
        # 3. Pattern-based scoring
        pattern_matches = 0
        for pattern in self.relationship_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                pattern_matches += 1
                kg_score += 0.3
        
        if pattern_matches > 0:
            reasoning.append(f"Found {pattern_matches} relationship query patterns")
        
        # 4. Symbol reference scoring
        symbol_matches = []
        for pattern in self.symbol_patterns:
            matches = re.findall(pattern, query)
            symbol_matches.extend(matches)
        
        if symbol_matches:
            kg_score += min(len(symbol_matches) * 0.1, 0.5)
            reasoning.append(f"Found {len(symbol_matches)} potential symbol references")
        
        # 5. Query complexity scoring (longer queries often benefit from context)
        word_count = len(query.split())
        if word_count > 5:
            complexity_bonus = min((word_count - 5) * 0.05, 0.2)
            kg_score += complexity_bonus
            reasoning.append(f"Complex query ({word_count} words) may benefit from expansion")
        
        # Normalize score
        kg_score = min(kg_score, 1.0)
        
        # Decision threshold
        use_kg = kg_score >= 0.5
        
        return {
            'use_kg_expansion': use_kg,
            'kg_score': kg_score,
            'reasoning': reasoning,
            'extracted_symbols': symbol_matches,
            'relationship_patterns_found': pattern_matches,
            'beneficial_keywords_found': keyword_matches
        }


class KGEnhancedSmartSearchEngine:
    """
    Phase 2 Enhanced Smart Search with Knowledge Graph capabilities.
    
    Maintains backward compatibility with existing smart search while adding
    powerful relationship-aware expansion through the Knowledge Graph.
    
    Architecture: Hybrid approach that combines vector search, BM25, and KG
    expansion based on intelligent query analysis.
    """
    
    def __init__(self, db_path: str = "codewise.db"):
        """
        Initialize enhanced search engine with KG capabilities.
        
        Args:
            db_path: Path to SQLite Knowledge Graph database
        """
        # Initialize base smart search (maintains existing functionality)
        try:
            self.base_search_engine = BaseSmartSearchEngine()
        except Exception as e:
            logger.warning(f"Base search engine unavailable: {e}")
            self.base_search_engine = None
        
        # Initialize Phase 2 KG components
        try:
            self.db_manager = DatabaseManager(db_path)
            self.vector_store = EnhancedVectorStore()  # BGE-powered
            self.kg_aware_rag = KGAwareRAG(self.vector_store, self.db_manager)
            self.kg_available = True
        except Exception as e:
            logger.warning(f"KG components unavailable: {e}. Using base search only.")
            self.kg_available = False
            self.db_manager = None
            self.kg_aware_rag = None
        
        # Enhanced query analysis
        self.kg_intent_analyzer = KGQueryIntentAnalyzer()
        
        # Search statistics
        self.search_stats = {
            'total_queries': 0,
            'kg_enhanced_queries': 0,
            'kg_expansion_improvements': 0,
            'avg_kg_expansion_time': 0.0
        }
        
        logger.info(f"KG Enhanced Smart Search initialized (KG available: {self.kg_available})")
    
    async def search(self, query: str, limit: int = 10, 
                    enable_kg_expansion: bool = True,
                    context: Dict[str, Any] = None) -> List[EnhancedSearchResult]:
        """
        Enhanced search with optional Knowledge Graph expansion.
        
        Process:
        1. Analyze query intent and KG benefit potential
        2. Execute base search (vector + BM25) if available
        3. If beneficial, perform KG expansion
        4. Merge and rank results intelligently
        5. Return enhanced results with relationship context
        
        Args:
            query: Search query
            limit: Maximum number of results
            enable_kg_expansion: Whether to use KG expansion
            context: Additional search context
            
        Returns:
            List of enhanced search results with KG context
        """
        import time
        search_start = time.time()
        
        self.search_stats['total_queries'] += 1
        
        logger.debug(f"Enhanced search for query: '{query}' (limit: {limit})")
        
        # Phase 1: Execute base search if available
        base_results = []
        if self.base_search_engine:
            try:
                base_results = await self.base_search_engine.search(query, limit * 2)
            except Exception as e:
                logger.error(f"Base search failed: {e}")
        
        # Convert base results to enhanced format
        enhanced_results = self._convert_to_enhanced_results(base_results, 'base_search')
        
        # Phase 2: Determine if KG expansion would be beneficial
        if not enable_kg_expansion or not self.kg_available:
            logger.debug("KG expansion disabled or unavailable, returning base results")
            return enhanced_results[:limit]
        
        # Analyze query for KG benefit (use mock intent if base analyzer unavailable)
        if self.base_search_engine and hasattr(self.base_search_engine, 'intent_analyzer'):
            base_intent_analysis = self.base_search_engine.intent_analyzer.analyze_query(query)
            intent = base_intent_analysis['intent']
            confidence = base_intent_analysis['confidence']
        else:
            # Mock analysis for testing
            intent = QueryIntent.GENERAL
            confidence = 0.5
        
        kg_analysis = self.kg_intent_analyzer.should_use_kg_expansion(query, intent, confidence)
        
        if not kg_analysis['use_kg_expansion']:
            logger.debug(f"KG expansion not beneficial (score: {kg_analysis['kg_score']:.2f})")
            return enhanced_results[:limit]
        
        # Phase 3: Perform KG-aware expansion
        logger.debug(f"Performing KG expansion (score: {kg_analysis['kg_score']:.2f})")
        self.search_stats['kg_enhanced_queries'] += 1
        
        try:
            kg_start = time.time()
            
            # Use KG-aware RAG for enhanced retrieval
            kg_results = self.kg_aware_rag.retrieve_and_expand(
                query=query,
                initial_k=min(5, limit),
                expanded_k=limit * 2,
                enable_kg_expansion=True
            )
            
            kg_time = time.time() - kg_start
            self._update_kg_timing_stats(kg_time)
            
            # Convert KG results to enhanced search results
            kg_enhanced_results = self._convert_kg_results_to_enhanced(kg_results, kg_analysis)
            
            # Phase 4: Merge and rank results
            final_results = self._merge_and_rank_results(
                enhanced_results, kg_enhanced_results, query, limit
            )
            
            logger.debug(f"KG expansion completed in {kg_time:.3f}s, "
                        f"returned {len(final_results)} results")
            
            return final_results
            
        except Exception as e:
            logger.error(f"KG expansion failed: {e}")
            # Fallback to base results
            return enhanced_results[:limit]
    
    def _convert_to_enhanced_results(self, base_results: List[SmartSearchResult], 
                                   source_type: str) -> List[EnhancedSearchResult]:
        """Convert base search results to enhanced format."""
        enhanced_results = []
        
        for result in base_results:
            enhanced_result = EnhancedSearchResult(
                chunk_id=result.chunk_id,
                file_path=result.file_path,
                snippet=result.snippet,
                relevance_score=result.relevance_score,
                query_intent=result.query_intent,
                search_strategy=result.search_strategy + [source_type],
                matched_terms=result.matched_terms,
                metadata=result.metadata,
                confidence=result.confidence,
                related_symbols=[],
                relationship_context=[],
                kg_expansion_used=False,
                expansion_depth=0,
                relationship_types=[]
            )
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _convert_kg_results_to_enhanced(self, kg_results: List[Any], 
                                      kg_analysis: Dict) -> List[EnhancedSearchResult]:
        """Convert KG-aware RAG results to enhanced search results."""
        enhanced_results = []
        
        for kg_result in kg_results:
            # Extract relationship context from KG result
            relationship_context = getattr(kg_result, 'relationship_context', [])
            related_symbols = getattr(kg_result, 'related_symbols', [])
            
            # Determine expansion depth
            expansion_depth = 0
            relationship_types = []
            
            if relationship_context:
                for rel_ctx in relationship_context:
                    expansion_depth = max(expansion_depth, rel_ctx.get('depth', 0))
                    rel_type = rel_ctx.get('relationship_type', 'unknown')
                    if rel_type not in relationship_types:
                        relationship_types.append(rel_type)
            
            enhanced_result = EnhancedSearchResult(
                chunk_id=getattr(kg_result, 'chunk_id', 'kg_result'),
                file_path=kg_result.metadata.get('file_path', 'unknown'),
                snippet=kg_result.content[:200] + '...' if len(kg_result.content) > 200 else kg_result.content,
                relevance_score=kg_result.relevance_score,
                query_intent=QueryIntent.GENERAL,  # Could be enhanced with intent detection
                search_strategy=[kg_result.source_type, 'kg_expansion'],
                matched_terms=kg_analysis.get('extracted_symbols', []),
                metadata=kg_result.metadata,
                confidence=kg_result.relevance_score,
                related_symbols=related_symbols,
                relationship_context=relationship_context,
                kg_expansion_used=True,
                expansion_depth=expansion_depth,
                relationship_types=relationship_types
            )
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _merge_and_rank_results(self, base_results: List[EnhancedSearchResult],
                               kg_results: List[EnhancedSearchResult],
                               query: str, limit: int) -> List[EnhancedSearchResult]:
        """Merge and intelligently rank base and KG-expanded results."""
        
        # Combine all results
        all_results = base_results + kg_results
        
        # Deduplicate by file path and content similarity
        unique_results = self._deduplicate_results(all_results)
        
        # Re-rank results considering KG context
        ranked_results = self._rank_with_kg_context(unique_results, query)
        
        return ranked_results[:limit]
    
    def _deduplicate_results(self, results: List[EnhancedSearchResult]) -> List[EnhancedSearchResult]:
        """Remove duplicate results based on content similarity."""
        unique_results = []
        seen_content = set()
        
        for result in results:
            # Use snippet + file path as deduplication key
            content_key = f"{result.file_path}:{result.snippet[:50]}"
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)
        
        return unique_results
    
    def _rank_with_kg_context(self, results: List[EnhancedSearchResult], 
                             query: str) -> List[EnhancedSearchResult]:
        """Re-rank results considering KG relationship context."""
        
        for result in results:
            # Base score
            final_score = result.relevance_score
            
            # KG expansion bonus
            if result.kg_expansion_used:
                kg_bonus = 0.1  # Small bonus for KG context
                
                # Relationship type bonuses
                valuable_relationships = {'calls', 'inherits', 'imports', 'self'}
                for rel_type in result.relationship_types:
                    if rel_type in valuable_relationships:
                        kg_bonus += 0.05
                
                # Depth penalty (closer relationships are more relevant)
                depth_penalty = result.expansion_depth * 0.02
                
                final_score += kg_bonus - depth_penalty
            
            # Update relevance score
            result.relevance_score = min(final_score, 1.0)
        
        # Sort by final score
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
    
    def _update_kg_timing_stats(self, kg_time: float):
        """Update KG expansion timing statistics."""
        current_avg = self.search_stats['avg_kg_expansion_time']
        kg_queries = self.search_stats['kg_enhanced_queries']
        
        # Calculate new average
        new_avg = ((current_avg * (kg_queries - 1)) + kg_time) / kg_queries
        self.search_stats['avg_kg_expansion_time'] = new_avg
    
    # ==================== NEW KG-SPECIFIC METHODS ====================
    
    def find_related_symbols(self, symbol_name: str, 
                           relationship_type: str = None,
                           max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Find symbols related to the given symbol through KG relationships.
        
        Args:
            symbol_name: Name of symbol to find relationships for
            relationship_type: Type of relationship ('calls', 'inherits', etc.)
            max_depth: Maximum relationship depth to traverse
            
        Returns:
            List of related symbols with relationship context
        """
        if not self.kg_available:
            return []
        
        try:
            # Find symbol nodes in KG
            symbol_nodes = self.db_manager.get_nodes_by_name(symbol_name, exact_match=True)
            if not symbol_nodes:
                return []
            
            related_symbols = []
            
            for node in symbol_nodes:
                node_id = node['id']
                
                if relationship_type == 'callers':
                    related = self.db_manager.find_callers(node_id, max_depth)
                    for rel in related:
                        rel['relationship_type'] = 'calls'
                        rel['direction'] = 'incoming'
                        related_symbols.append(rel)
                
                elif relationship_type == 'dependencies':
                    related = self.db_manager.find_dependencies(node_id, max_depth)
                    for rel in related:
                        rel['relationship_type'] = rel.get('relationship', 'dependency')
                        rel['direction'] = 'outgoing'
                        related_symbols.append(rel)
                
                else:
                    # Get all relationships
                    callers = self.db_manager.find_callers(node_id, max_depth)
                    for rel in callers:
                        rel['relationship_type'] = 'calls'
                        rel['direction'] = 'incoming'
                        related_symbols.append(rel)
                    
                    deps = self.db_manager.find_dependencies(node_id, max_depth)
                    for rel in deps:
                        rel['relationship_type'] = rel.get('relationship', 'dependency')
                        rel['direction'] = 'outgoing'
                        related_symbols.append(rel)
            
            return related_symbols
            
        except Exception as e:
            logger.error(f"Failed to find related symbols for '{symbol_name}': {e}")
            return []
    
    def explore_symbol_neighborhood(self, symbol_name: str, 
                                  max_depth: int = 2) -> Dict[str, Any]:
        """
        Explore the 'neighborhood' of relationships around a symbol.
        
        Args:
            symbol_name: Symbol to explore
            max_depth: Maximum relationship depth
            
        Returns:
            Dictionary with comprehensive relationship context
        """
        if not self.kg_available:
            return {}
        
        try:
            symbol_nodes = self.db_manager.get_nodes_by_name(symbol_name, exact_match=True)
            if not symbol_nodes:
                return {'error': f"Symbol '{symbol_name}' not found in Knowledge Graph"}
            
            neighborhood_map = {}
            
            for node in symbol_nodes:
                node_id = node['id']
                
                # Get comprehensive relationship context
                neighborhood = {
                    'center_symbol': node,
                    'callers': self.db_manager.find_callers(node_id, max_depth),
                    'dependencies': self.db_manager.find_dependencies(node_id, max_depth),
                    'file_siblings': self._find_file_siblings(node),
                    'related_files': self._find_related_files(node)
                }
                
                neighborhood_map[node_id] = neighborhood
            
            return {
                'symbol_name': symbol_name,
                'neighborhoods': neighborhood_map,
                'total_relationships': sum(
                    len(n['callers']) + len(n['dependencies']) 
                    for n in neighborhood_map.values()
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to explore neighborhood for '{symbol_name}': {e}")
            return {'error': str(e)}
    
    def _find_file_siblings(self, node: Dict) -> List[Dict]:
        """Find other symbols in the same file."""
        try:
            file_path = node['file_path']
            file_symbols = self.db_manager.get_nodes_by_file(file_path)
            
            # Filter out the target node itself
            siblings = [s for s in file_symbols if s['id'] != node['id']]
            return siblings[:10]  # Limit results
            
        except Exception as e:
            logger.debug(f"Failed to find file siblings: {e}")
            return []
    
    def _find_related_files(self, node: Dict) -> List[str]:
        """Find files related through imports or usage."""
        try:
            related_files = set()
            node_id = node['id']
            
            # Files that import this symbol's file
            import_edges = self.db_manager.get_incoming_edges(node_id, 'imports')
            for edge in import_edges:
                source_node = self.db_manager.get_node(edge['source_id'])
                if source_node:
                    related_files.add(source_node['file_path'])
            
            # Files that this symbol calls into
            call_edges = self.db_manager.get_outgoing_edges(node_id, 'calls')
            for edge in call_edges:
                target_node = self.db_manager.get_node(edge['target_id'])
                if target_node:
                    related_files.add(target_node['file_path'])
            
            return list(related_files)[:5]  # Limit results
            
        except Exception as e:
            logger.debug(f"Failed to find related files: {e}")
            return []
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics including KG usage."""
        kg_stats = {}
        if self.kg_available and self.kg_aware_rag:
            kg_stats = self.kg_aware_rag.get_retrieval_statistics()
        
        return {
            **self.search_stats,
            'kg_enhancement_rate': (
                self.search_stats['kg_enhanced_queries'] /
                max(self.search_stats['total_queries'], 1)
            ),
            'kg_rag_statistics': kg_stats
        }
    
    def close(self):
        """Clean up resources."""
        if self.db_manager:
            self.db_manager.close()


if __name__ == "__main__":
    # CLI interface for testing enhanced search
    import argparse
    
    parser = argparse.ArgumentParser(description="Test KG Enhanced Smart Search")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--db-path", default="codewise.db", help="Database file path")
    parser.add_argument("--limit", type=int, default=5, help="Number of results")
    parser.add_argument("--no-kg", action="store_true", help="Disable KG expansion")
    parser.add_argument("--symbol", help="Find related symbols for this symbol")
    parser.add_argument("--neighborhood", help="Explore symbol neighborhood")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    async def test_search():
        try:
            # Create enhanced search engine
            search_engine = KGEnhancedSmartSearchEngine(db_path=args.db_path)
            
            if args.symbol:
                # Test symbol relationship finding
                print(f"\\nFinding related symbols for: {args.symbol}")
                related = search_engine.find_related_symbols(args.symbol)
                
                if related:
                    for rel in related[:5]:
                        print(f"  {rel['name']} ({rel['relationship_type']}) in {rel['file_path']}")
                else:
                    print("  No related symbols found")
            
            elif args.neighborhood:
                # Test neighborhood exploration
                print(f"\\nExploring neighborhood for: {args.neighborhood}")
                neighborhood = search_engine.explore_symbol_neighborhood(args.neighborhood)
                
                if 'error' in neighborhood:
                    print(f"  Error: {neighborhood['error']}")
                else:
                    print(f"  Total relationships: {neighborhood['total_relationships']}")
                    for node_id, data in neighborhood['neighborhoods'].items():
                        center = data['center_symbol']
                        print(f"  {center['name']} ({center['type']}) in {center['file_path']}")
                        print(f"    Callers: {len(data['callers'])}")
                        print(f"    Dependencies: {len(data['dependencies'])}")
            
            else:
                # Test enhanced search
                print(f"\\nSearching for: {args.query}")
                results = await search_engine.search(
                    query=args.query,
                    limit=args.limit,
                    enable_kg_expansion=not args.no_kg
                )
                
                print(f"\\nFound {len(results)} results:")
                for i, result in enumerate(results, 1):
                    kg_info = ""
                    if result.kg_expansion_used:
                        kg_info = f" [KG: {len(result.relationship_types)} rel types, depth {result.expansion_depth}]"
                    
                    print(f"{i}. {result.file_path} (score: {result.relevance_score:.3f}){kg_info}")
                    print(f"   {result.snippet[:100]}...")
                    
                    if result.related_symbols:
                        print(f"   Related symbols: {', '.join(result.related_symbols[:3])}")
            
            # Show statistics
            stats = search_engine.get_search_statistics()
            print(f"\\nSearch Statistics:")
            print(f"  Total queries: {stats['total_queries']}")
            print(f"  KG enhanced: {stats['kg_enhanced_queries']}")
            print(f"  KG enhancement rate: {stats['kg_enhancement_rate']:.2%}")
            
            search_engine.close()
            
        except Exception as e:
            print(f"Enhanced search test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run async test
    asyncio.run(test_search())