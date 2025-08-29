"""
KG-Aware RAG with High-Fidelity Embeddings

Enhanced Retrieve-and-Expand RAG that combines BGE embeddings with 
Knowledge Graph relationships for superior search quality and context expansion.

Architectural Innovation: Leverages Phase 1 BGE improvements to provide higher
quality initial retrieval, which enables more accurate symbol extraction for
Knowledge Graph expansion.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
import logging
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# Import Phase 1 and Phase 2 components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from storage.database_manager import DatabaseManager
from indexer.enhanced_vector_store import EnhancedVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with KG context."""
    chunk_id: str
    content: str
    relevance_score: float
    source_type: str  # 'vector', 'kg_expansion', 'hybrid'
    metadata: Dict[str, Any]
    related_symbols: List[str] = None
    relationship_context: List[Dict] = None


@dataclass 
class SymbolContext:
    """Context about a symbol extracted from retrieval results."""
    symbol_name: str
    symbol_type: str
    confidence: float
    source_chunk: str
    related_concepts: List[str] = None


class SymbolExtractor:
    """
    Extracts key symbols from high-quality BGE retrieval context.
    
    Design Decision: BGE's superior semantic understanding provides better
    initial context, leading to more accurate symbol extraction for KG queries.
    """
    
    def __init__(self):
        self.extraction_patterns = {
            # Function calls and definitions
            'function_patterns': [
                r'\b(\w+)\s*\(',                # function_name(
                r'def\s+(\w+)',                 # def function_name
                r'function\s+(\w+)',            # function function_name
                r'async\s+def\s+(\w+)',         # async def function_name
            ],
            
            # Class definitions and usage
            'class_patterns': [
                r'class\s+(\w+)',               # class ClassName
                r'new\s+(\w+)',                 # new ClassName
                r'(\w+)\.prototype',            # ClassName.prototype
                r'instanceof\s+(\w+)',          # instanceof ClassName
            ],
            
            # Variable and attribute access
            'variable_patterns': [
                r'(\w+)\.\w+',                  # object.attribute
                r'(\w+)\[\w+\]',                # object[key]
                r'self\.(\w+)',                 # self.attribute
            ],
            
            # Import and module references
            'import_patterns': [
                r'import\s+(\w+)',              # import module
                r'from\s+(\w+)',                # from module
                r'require\([\'"](\w+)[\'"]\)',  # require('module')
            ]
        }
    
    def extract_symbols_from_context(self, chunks: List[Dict], query: str) -> List[SymbolContext]:
        """
        Extract key symbols for KG expansion with improved accuracy from BGE context.
        
        The higher semantic quality of BGE-retrieved initial context significantly 
        improves symbol identification accuracy and reduces hallucination.
        
        Args:
            chunks: High-quality chunks from BGE retrieval
            query: Original search query
            
        Returns:
            List of symbol contexts for KG expansion
        """
        symbol_contexts = []
        
        # Combine all chunk content for analysis
        combined_content = "\\n".join([chunk.get('content', '') for chunk in chunks])
        
        # Extract symbols using pattern matching
        pattern_symbols = self._extract_with_patterns(combined_content)
        
        # Use LLM for semantic symbol extraction (leveraging high-quality context)
        semantic_symbols = self._extract_with_llm(combined_content, query)
        
        # Combine and deduplicate
        all_symbols = self._merge_symbol_extractions(pattern_symbols, semantic_symbols)
        
        # Score symbols based on relevance to query and context quality
        scored_symbols = self._score_symbols(all_symbols, query, chunks)
        
        # Convert to SymbolContext objects
        for symbol_name, score in scored_symbols:
            symbol_type = self._infer_symbol_type(symbol_name, combined_content)
            
            symbol_context = SymbolContext(
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                confidence=score,
                source_chunk=self._find_symbol_source_chunk(symbol_name, chunks),
                related_concepts=self._extract_related_concepts(symbol_name, combined_content)
            )
            
            symbol_contexts.append(symbol_context)
        
        # Sort by confidence and return top candidates
        symbol_contexts.sort(key=lambda x: x.confidence, reverse=True)
        return symbol_contexts[:10]  # Top 10 symbols for expansion
    
    def _extract_with_patterns(self, content: str) -> Set[str]:
        """Extract symbols using regex patterns."""
        import re
        
        symbols = set()
        
        for pattern_category, patterns in self.extraction_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                symbols.update(matches)
        
        # Filter out common noise words
        noise_words = {
            'self', 'this', 'super', 'class', 'def', 'function', 'var', 'let', 'const',
            'if', 'else', 'for', 'while', 'try', 'catch', 'return', 'print', 'len',
            'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple', 'None',
            'True', 'False', 'null', 'undefined', 'console', 'log'
        }
        
        return {s for s in symbols if len(s) > 2 and s.lower() not in noise_words}
    
    def _extract_with_llm(self, content: str, query: str) -> Set[str]:
        """
        Extract symbols using LLM with high-quality context.
        
        Note: This would integrate with the LLM used by the system.
        For now, we'll use a simplified heuristic approach.
        """
        # Simplified implementation - in production, this would call the LLM
        # with a carefully crafted prompt for symbol extraction
        
        symbols = set()
        
        # Extract identifiers that appear multiple times (likely important)
        import re
        identifiers = re.findall(r'\\b[a-zA-Z_][a-zA-Z0-9_]*\\b', content)
        identifier_counts = {}
        
        for identifier in identifiers:
            if len(identifier) > 2:
                identifier_counts[identifier] = identifier_counts.get(identifier, 0) + 1
        
        # Include identifiers that appear multiple times
        for identifier, count in identifier_counts.items():
            if count >= 2:
                symbols.add(identifier)
        
        # Extract identifiers that appear in the query
        query_words = set(re.findall(r'\\b[a-zA-Z_][a-zA-Z0-9_]*\\b', query.lower()))
        for identifier in identifiers:
            if identifier.lower() in query_words:
                symbols.add(identifier)
        
        return symbols
    
    def _merge_symbol_extractions(self, pattern_symbols: Set[str], 
                                semantic_symbols: Set[str]) -> Set[str]:
        """Merge and deduplicate symbol extractions."""
        # Combine both sets
        all_symbols = pattern_symbols.union(semantic_symbols)
        
        # Additional filtering could be applied here
        return all_symbols
    
    def _score_symbols(self, symbols: Set[str], query: str, chunks: List[Dict]) -> List[Tuple[str, float]]:
        """Score symbols based on relevance to query and context quality."""
        scored_symbols = []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for symbol in symbols:
            score = 0.0
            symbol_lower = symbol.lower()
            
            # Score based on query relevance
            if symbol_lower in query_lower:
                score += 1.0
            
            if any(word in symbol_lower for word in query_words):
                score += 0.5
            
            # Score based on frequency in chunks
            total_occurrences = 0
            for chunk in chunks:
                content = chunk.get('content', '').lower()
                total_occurrences += content.count(symbol_lower)
            
            # Normalize frequency score
            frequency_score = min(total_occurrences / 10.0, 1.0)
            score += frequency_score
            
            # Score based on symbol characteristics
            if symbol[0].isupper():  # Likely class name
                score += 0.2
            
            if '_' in symbol or symbol.lower().endswith('_function'):
                score += 0.1
            
            scored_symbols.append((symbol, score))
        
        return sorted(scored_symbols, key=lambda x: x[1], reverse=True)
    
    def _infer_symbol_type(self, symbol_name: str, content: str) -> str:
        """Infer the type of symbol based on context."""
        import re
        
        # Check for class patterns
        if re.search(f'class\\s+{symbol_name}', content, re.IGNORECASE):
            return 'class'
        
        # Check for function patterns  
        if re.search(f'def\\s+{symbol_name}|function\\s+{symbol_name}', content, re.IGNORECASE):
            return 'function'
        
        # Check for method patterns
        if re.search(f'\\.{symbol_name}\\s*\\(', content):
            return 'method'
        
        # Check for variable patterns
        if re.search(f'{symbol_name}\\s*=', content):
            return 'variable'
        
        return 'unknown'
    
    def _find_symbol_source_chunk(self, symbol_name: str, chunks: List[Dict]) -> str:
        """Find which chunk contains the most relevant mention of the symbol."""
        best_chunk = ""
        best_score = 0
        
        for chunk in chunks:
            content = chunk.get('content', '')
            occurrences = content.lower().count(symbol_name.lower())
            
            if occurrences > best_score:
                best_score = occurrences
                best_chunk = chunk.get('id', '')
        
        return best_chunk
    
    def _extract_related_concepts(self, symbol_name: str, content: str) -> List[str]:
        """Extract concepts related to the symbol."""
        import re
        
        related = []
        
        # Find lines containing the symbol
        lines = content.split('\\n')
        for line in lines:
            if symbol_name.lower() in line.lower():
                # Extract other identifiers from the same line
                identifiers = re.findall(r'\\b[a-zA-Z_][a-zA-Z0-9_]*\\b', line)
                for identifier in identifiers:
                    if identifier != symbol_name and len(identifier) > 2:
                        related.append(identifier)
        
        # Return unique related concepts
        return list(set(related))[:5]  # Top 5 related concepts


class KGAwareRAG:
    """
    Enhanced Retrieve-and-Expand RAG using high-fidelity BGE embeddings 
    for improved initial retrieval and more accurate KG symbol extraction.
    
    Core Innovation: Leverages Phase 1 BGE improvements to provide better
    initial context, which dramatically improves symbol extraction accuracy
    for Knowledge Graph expansion.
    """
    
    def __init__(self, enhanced_vector_store: EnhancedVectorStore, 
                 db_manager: DatabaseManager):
        self.vector_store = enhanced_vector_store
        self.db_manager = db_manager
        self.symbol_extractor = SymbolExtractor()
        
        # Performance tracking
        self.retrieval_stats = {
            'total_queries': 0,
            'kg_expansions': 0,
            'avg_initial_results': 0,
            'avg_expanded_results': 0,
            'symbol_extraction_accuracy': 0,
            'expansion_effectiveness': 0
        }
    
    def retrieve_and_expand(self, query: str, initial_k: int = 5, 
                           expanded_k: int = 15, enable_kg_expansion: bool = True) -> List[RetrievalResult]:
        """
        Enhanced retrieve-and-expand using high-fidelity semantic understanding.
        
        Process:
        1. High-quality initial retrieval with BGE embeddings and instruction prefixes
        2. Extract symbols with improved accuracy from better context
        3. Knowledge Graph expansion using accurately identified symbols
        4. Combine and rank results with BGE similarity scoring
        
        Args:
            query: Search query
            initial_k: Number of initial results to retrieve
            expanded_k: Total number of results to return after expansion
            enable_kg_expansion: Whether to perform KG expansion
            
        Returns:
            List of enhanced retrieval results with KG context
        """
        self.retrieval_stats['total_queries'] += 1
        
        # PHASE 1: High-fidelity initial retrieval with BGE embeddings
        logger.debug(f"Phase 1: Initial BGE retrieval for query: {query}")
        initial_results = self._semantic_retrieval(query, initial_k)
        
        self.retrieval_stats['avg_initial_results'] = (
            (self.retrieval_stats['avg_initial_results'] * (self.retrieval_stats['total_queries'] - 1) + 
             len(initial_results)) / self.retrieval_stats['total_queries']
        )
        
        if not enable_kg_expansion or not initial_results:
            return self._convert_to_retrieval_results(initial_results, 'vector')
        
        # PHASE 2: Extract symbols with improved accuracy from BGE context
        logger.debug("Phase 2: Symbol extraction from high-quality BGE context")
        symbol_contexts = self.symbol_extractor.extract_symbols_from_context(initial_results, query)
        
        if not symbol_contexts:
            logger.debug("No symbols extracted for KG expansion")
            return self._convert_to_retrieval_results(initial_results, 'vector')
        
        # PHASE 3: Knowledge Graph expansion using identified symbols
        logger.debug(f"Phase 3: KG expansion with {len(symbol_contexts)} symbols")
        expanded_results = self._expand_via_knowledge_graph(symbol_contexts, query)
        
        self.retrieval_stats['kg_expansions'] += 1
        
        # PHASE 4: Combine and rank with BGE similarity scoring
        logger.debug("Phase 4: Combining and ranking results")
        all_results = self._combine_and_rank_results(
            initial_results, expanded_results, query, expanded_k
        )
        
        self.retrieval_stats['avg_expanded_results'] = (
            (self.retrieval_stats['avg_expanded_results'] * (self.retrieval_stats['kg_expansions'] - 1) + 
             len(all_results)) / self.retrieval_stats['kg_expansions']
        )
        
        return all_results
    
    def _semantic_retrieval(self, query: str, k: int) -> List[Dict]:
        """Perform high-quality semantic retrieval using BGE embeddings."""
        try:
            # Use BGE with instruction prefix for optimal retrieval
            similarities, indices = self.vector_store.search(query, k * 2)  # Get more for filtering
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities, indices)):
                # Get chunk content from database
                chunk_data = self._get_chunk_by_index(idx)
                if chunk_data:
                    result = {
                        'chunk_id': chunk_data.get('id', f'chunk_{idx}'),
                        'content': chunk_data.get('content', ''),
                        'relevance_score': float(similarity),
                        'metadata': chunk_data.get('metadata', {}),
                        'file_path': chunk_data.get('file_path', ''),
                        'node_id': chunk_data.get('node_id')
                    }
                    results.append(result)
            
            # Return top k after filtering
            return results[:k]
            
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
            return []
    
    def _get_chunk_by_index(self, index: int) -> Optional[Dict]:
        """Get chunk data from database by index."""
        try:
            # This would need to be implemented based on how chunks are stored
            # For now, return a mock chunk
            return {
                'id': f'chunk_{index}',
                'content': f'Mock content for chunk {index}',
                'metadata': {},
                'file_path': 'mock.py',
                'node_id': None
            }
        except Exception as e:
            logger.error(f"Failed to get chunk by index {index}: {e}")
            return None
    
    def _expand_via_knowledge_graph(self, symbol_contexts: List[SymbolContext], 
                                  query: str) -> List[Dict]:
        """Expand using Knowledge Graph relationships for identified symbols."""
        expanded_chunks = []
        
        for symbol_context in symbol_contexts:
            try:
                # Find symbol nodes in KG
                symbol_nodes = self.db_manager.get_nodes_by_name(
                    symbol_context.symbol_name, exact_match=True
                )
                
                for node in symbol_nodes:
                    # Get related nodes through various relationships
                    related_chunks = self._get_related_chunks_for_node(node, query)
                    expanded_chunks.extend(related_chunks)
                
            except Exception as e:
                logger.error(f"KG expansion failed for symbol {symbol_context.symbol_name}: {e}")
                continue
        
        # Deduplicate by chunk ID
        seen_chunks = set()
        unique_chunks = []
        
        for chunk in expanded_chunks:
            chunk_id = chunk.get('chunk_id')
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _get_related_chunks_for_node(self, node: Dict, query: str) -> List[Dict]:
        """Get chunks related to a node through KG relationships."""
        related_chunks = []
        
        try:
            node_id = node['id']
            
            # Get callers (who calls this symbol)
            callers = self.db_manager.find_callers(node_id, max_depth=2)
            for caller in callers:
                chunks = self.db_manager.get_chunks_by_node(caller['id'])
                for chunk in chunks:
                    chunk_dict = dict(chunk)
                    chunk_dict['kg_relationship'] = 'caller'
                    chunk_dict['relationship_depth'] = caller.get('depth', 0)
                    related_chunks.append(chunk_dict)
            
            # Get dependencies (what this symbol calls/uses)
            dependencies = self.db_manager.find_dependencies(node_id, max_depth=2)
            for dep in dependencies:
                chunks = self.db_manager.get_chunks_by_node(dep['id'])
                for chunk in chunks:
                    chunk_dict = dict(chunk)
                    chunk_dict['kg_relationship'] = dep.get('relationship', 'dependency')
                    chunk_dict['relationship_depth'] = dep.get('depth', 0)
                    related_chunks.append(chunk_dict)
            
            # Get chunks for the node itself
            node_chunks = self.db_manager.get_chunks_by_node(node_id)
            for chunk in node_chunks:
                chunk_dict = dict(chunk)
                chunk_dict['kg_relationship'] = 'self'
                chunk_dict['relationship_depth'] = 0
                related_chunks.append(chunk_dict)
            
        except Exception as e:
            logger.error(f"Failed to get related chunks for node {node.get('id')}: {e}")
        
        return related_chunks
    
    def _combine_and_rank_results(self, initial_results: List[Dict], 
                                expanded_results: List[Dict], query: str, 
                                target_k: int) -> List[RetrievalResult]:
        """Combine initial and expanded results with intelligent ranking."""
        
        # Convert to RetrievalResult objects
        all_results = []
        
        # Add initial results
        for result in initial_results:
            retrieval_result = RetrievalResult(
                chunk_id=result['chunk_id'],
                content=result['content'],
                relevance_score=result['relevance_score'],
                source_type='vector',
                metadata=result.get('metadata', {}),
                related_symbols=[],
                relationship_context=[]
            )
            all_results.append(retrieval_result)
        
        # Add expanded results
        for result in expanded_results:
            # Calculate relevance score for KG-expanded results
            content = result.get('content', '')
            kg_score = self._calculate_kg_relevance_score(content, query, result)
            
            retrieval_result = RetrievalResult(
                chunk_id=result.get('id', f"kg_{len(all_results)}"),
                content=content,
                relevance_score=kg_score,
                source_type='kg_expansion',
                metadata=result.get('metadata', {}),
                related_symbols=[],
                relationship_context=[{
                    'relationship_type': result.get('kg_relationship', 'unknown'),
                    'depth': result.get('relationship_depth', 0)
                }]
            )
            all_results.append(retrieval_result)
        
        # Deduplicate by content similarity
        unique_results = self._deduplicate_results(all_results)
        
        # Rank by combined score
        ranked_results = self._rank_combined_results(unique_results, query)
        
        return ranked_results[:target_k]
    
    def _calculate_kg_relevance_score(self, content: str, query: str, kg_result: Dict) -> float:
        """Calculate relevance score for KG-expanded results."""
        base_score = 0.5  # Base score for KG results
        
        # Text similarity component
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        overlap = len(query_words.intersection(content_words))
        text_score = overlap / max(len(query_words), 1)
        
        # KG relationship bonus
        relationship_bonuses = {
            'self': 0.3,
            'caller': 0.2,
            'calls': 0.2,
            'inherits': 0.15,
            'imports': 0.1,
            'uses': 0.1
        }
        
        relationship_type = kg_result.get('kg_relationship', 'unknown')
        relationship_bonus = relationship_bonuses.get(relationship_type, 0.05)
        
        # Depth penalty (closer relationships are more relevant)
        depth = kg_result.get('relationship_depth', 0)
        depth_penalty = 0.1 * depth
        
        final_score = base_score + text_score + relationship_bonus - depth_penalty
        return max(0.0, min(1.0, final_score))  # Clamp between 0 and 1
    
    def _deduplicate_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Remove duplicate results based on content similarity."""
        unique_results = []
        seen_content = set()
        
        for result in results:
            # Use first 100 characters as deduplication key
            content_key = result.content[:100].strip()
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)
        
        return unique_results
    
    def _rank_combined_results(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """Rank combined results using multiple signals."""
        
        # Calculate final scores
        for result in results:
            final_score = result.relevance_score
            
            # Boost for vector results (they had good initial similarity)
            if result.source_type == 'vector':
                final_score *= 1.1
            
            # Boost for results with relationship context
            if result.relationship_context:
                final_score *= 1.05
            
            # Update the relevance score
            result.relevance_score = final_score
        
        # Sort by final score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results
    
    def _convert_to_retrieval_results(self, chunks: List[Dict], source_type: str) -> List[RetrievalResult]:
        """Convert chunk dictionaries to RetrievalResult objects."""
        results = []
        
        for chunk in chunks:
            result = RetrievalResult(
                chunk_id=chunk.get('chunk_id', chunk.get('id', '')),
                content=chunk.get('content', ''),
                relevance_score=chunk.get('relevance_score', 0.0),
                source_type=source_type,
                metadata=chunk.get('metadata', {}),
                related_symbols=[],
                relationship_context=[]
            )
            results.append(result)
        
        return results
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get comprehensive retrieval statistics."""
        return {
            **self.retrieval_stats,
            'kg_expansion_rate': (
                self.retrieval_stats['kg_expansions'] / 
                max(self.retrieval_stats['total_queries'], 1)
            ),
            'avg_expansion_factor': (
                self.retrieval_stats['avg_expanded_results'] /
                max(self.retrieval_stats['avg_initial_results'], 1)
            )
        }


if __name__ == "__main__":
    # CLI interface for testing KG-Aware RAG
    import argparse
    
    parser = argparse.ArgumentParser(description="Test KG-Aware RAG")
    parser.add_argument("--db-path", default="test_kg_rag.db", help="Database file path")
    parser.add_argument("--query", required=True, help="Test query")
    parser.add_argument("--stats", action="store_true", help="Show retrieval statistics")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        from storage.database_setup import setup_codewise_database
        
        # Set up database and components
        setup = setup_codewise_database(args.db_path)
        db_manager = DatabaseManager(args.db_path)
        
        # Create mock vector store (in practice, this would be the real EnhancedVectorStore)
        vector_store = None  # Would need actual implementation
        
        # Create KG-Aware RAG
        # kg_rag = KGAwareRAG(vector_store, db_manager)
        
        print(f"KG-Aware RAG test setup completed for query: '{args.query}'")
        print("Note: Full testing requires populated database and vector store")
        
        db_manager.close()
        
    except Exception as e:
        print(f"KG-Aware RAG test failed: {e}")
        import traceback
        traceback.print_exc()