"""
Hybrid Search Engine combining Vector Similarity and BM25 Keyword Search

This module provides a unified search interface that combines semantic vector search
with keyword-based BM25 search for comprehensive code retrieval.
"""

import logging
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from bm25_index import BM25Index, BM25Result
from vector_store import get_vector_store

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Unified search result combining vector and BM25 results"""
    chunk_id: int
    file_path: str
    snippet: str
    relevance_score: float
    vector_score: float
    bm25_score: float
    search_type: str  # 'vector', 'bm25', 'hybrid'
    matched_terms: List[str]
    metadata: Dict


class QueryProcessor:
    """Process and analyze search queries for optimal retrieval"""
    
    def __init__(self):
        # Technical terms that should boost BM25 search
        self.technical_indicators = {
            'function', 'class', 'method', 'variable', 'import', 'export',
            'async', 'await', 'return', 'def', 'const', 'let', 'var',
            'interface', 'type', 'enum', 'struct', 'public', 'private',
            'static', 'abstract', 'extends', 'implements'
        }
        
        # File type indicators
        self.file_type_patterns = {
            'python': ['py', 'python', '.py'],
            'javascript': ['js', 'javascript', '.js', 'node'],
            'typescript': ['ts', 'typescript', '.ts'],
            'react': ['jsx', 'tsx', 'react', 'component'],
            'config': ['config', 'configuration', 'settings', 'env'],
            'markdown': ['md', 'markdown', 'readme', 'doc']
        }
    
    def analyze_query(self, query: str) -> Dict:
        """
        Analyze query to determine optimal search strategy
        
        Returns:
            Dict with analysis results including:
            - has_technical_terms: bool
            - file_type_hints: List[str]
            - is_exact_match_query: bool
            - suggested_file_filters: List[str]
        """
        query_lower = query.lower()
        words = re.findall(r'\b\w+\b', query_lower)
        
        # Check for technical terms
        has_technical_terms = bool(set(words) & self.technical_indicators)
        
        # Check for file type hints
        file_type_hints = []
        for file_type, patterns in self.file_type_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                file_type_hints.append(file_type)
        
        # Check for exact match indicators (quotes, specific function names)
        is_exact_match_query = (
            '"' in query or 
            "'" in query or
            any(word.endswith('()') for word in words) or
            any(word.startswith('_') or word.endswith('_') for word in words)
        )
        
        # Suggest file filters based on query
        suggested_file_filters = []
        if file_type_hints:
            suggested_file_filters.extend(file_type_hints)
        
        return {
            'has_technical_terms': has_technical_terms,
            'file_type_hints': file_type_hints,
            'is_exact_match_query': is_exact_match_query,
            'suggested_file_filters': suggested_file_filters,
            'query_terms': words
        }


class ResultFusion:
    """Combine and rank results from multiple search methods"""
    
    def __init__(self, vector_weight: float = 0.6, bm25_weight: float = 0.4):
        """
        Initialize result fusion with weights
        
        Args:
            vector_weight: Weight for vector similarity scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
        """
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
    
    def fuse_results(self, vector_results: List[Tuple[str, str]], 
                    bm25_results: List[BM25Result],
                    query_analysis: Dict) -> List[SearchResult]:
        """
        Fuse vector and BM25 results into unified ranking
        
        Args:
            vector_results: List of (file_path, snippet) tuples from vector search
            bm25_results: List of BM25Result objects
            query_analysis: Query analysis from QueryProcessor
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        # Normalize scores to 0-1 range
        normalized_vector = self._normalize_vector_results(vector_results)
        normalized_bm25 = self._normalize_bm25_results(bm25_results)
        
        # Create unified result set
        unified_results = {}
        
        # Add vector results
        for i, (file_path, snippet, norm_score) in enumerate(normalized_vector):
            key = f"{file_path}:{snippet[:50]}"  # Use file + snippet start as key
            unified_results[key] = SearchResult(
                chunk_id=i,
                file_path=file_path,
                snippet=snippet,
                relevance_score=norm_score * self.vector_weight,
                vector_score=norm_score,
                bm25_score=0.0,
                search_type='vector',
                matched_terms=[],
                metadata={}
            )
        
        # Add or update with BM25 results
        for result in normalized_bm25:
            key = f"{result.file_path}:{result.snippet[:50]}"
            
            if key in unified_results:
                # Combine scores for results that appear in both
                existing = unified_results[key]
                existing.relevance_score = (
                    existing.vector_score * self.vector_weight +
                    result.score * self.bm25_weight
                )
                existing.bm25_score = result.score
                existing.search_type = 'hybrid'
                existing.matched_terms = result.matched_terms
            else:
                # Add BM25-only result
                unified_results[key] = SearchResult(
                    chunk_id=result.chunk_id,
                    file_path=result.file_path,
                    snippet=result.snippet,
                    relevance_score=result.score * self.bm25_weight,
                    vector_score=0.0,
                    bm25_score=result.score,
                    search_type='bm25',
                    matched_terms=result.matched_terms,
                    metadata={}
                )
        
        # Apply query-specific boosts
        self._apply_query_boosts(unified_results, query_analysis)
        
        # Sort by relevance score and return
        sorted_results = sorted(unified_results.values(), 
                              key=lambda x: x.relevance_score, reverse=True)
        
        return sorted_results
    
    def _normalize_vector_results(self, results: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
        """Normalize vector search results (assuming they come pre-scored)"""
        if not results:
            return []
        
        # For now, assign decreasing scores based on order (vector store should provide scores)
        normalized = []
        for i, (file_path, snippet) in enumerate(results):
            # Simulate score based on position (higher is better)
            score = max(0.1, 1.0 - (i * 0.2))
            normalized.append((file_path, snippet, score))
        
        return normalized
    
    def _normalize_bm25_results(self, results: List[BM25Result]) -> List[BM25Result]:
        """Normalize BM25 scores to 0-1 range"""
        if not results:
            return []
        
        # Find max score for normalization
        max_score = max(result.score for result in results)
        if max_score == 0:
            return results
        
        # Normalize scores
        normalized = []
        for result in results:
            normalized_result = BM25Result(
                chunk_id=result.chunk_id,
                score=result.score / max_score,
                file_path=result.file_path,
                snippet=result.snippet,
                matched_terms=result.matched_terms
            )
            normalized.append(normalized_result)
        
        return normalized
    
    def _apply_query_boosts(self, results: Dict[str, SearchResult], query_analysis: Dict):
        """Apply query-specific boosts to results"""
        # Boost exact match queries for BM25 results
        if query_analysis.get('is_exact_match_query', False):
            for result in results.values():
                if result.search_type in ['bm25', 'hybrid'] and result.matched_terms:
                    result.relevance_score *= 1.2
        
        # Boost results matching file type hints
        file_type_hints = query_analysis.get('file_type_hints', [])
        if file_type_hints:
            for result in results.values():
                file_ext = result.file_path.split('.')[-1].lower()
                if any(hint in file_ext or hint in result.file_path.lower() 
                       for hint in file_type_hints):
                    result.relevance_score *= 1.15
        
        # Boost technical term matches
        if query_analysis.get('has_technical_terms', False):
            for result in results.values():
                if result.search_type in ['bm25', 'hybrid']:
                    # Check if matched terms include technical terms
                    technical_matches = len([term for term in result.matched_terms 
                                           if term in query_analysis.get('query_terms', [])])
                    if technical_matches > 0:
                        result.relevance_score *= (1.0 + 0.1 * technical_matches)


class HybridSearchEngine:
    """Main hybrid search engine combining vector and BM25 search"""
    
    def __init__(self, vector_store=None, bm25_index: Optional[BM25Index] = None):
        """
        Initialize hybrid search engine
        
        Args:
            vector_store: Vector store instance (uses singleton if None)
            bm25_index: BM25 index instance (creates new if None)
        """
        self.vector_store = vector_store or get_vector_store()
        self.bm25_index = bm25_index or BM25Index()
        self.query_processor = QueryProcessor()
        self.result_fusion = ResultFusion()
        
        # Search parameters
        self.min_relevance_threshold = 0.25
        self.max_results = 20
        
    def build_bm25_index(self, documents: List[Dict]) -> bool:
        """
        Build BM25 index from documents
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.bm25_index.add_documents(documents)
            logger.info(f"Built BM25 index with {len(documents)} documents")
            return True
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            return False
    
    def search(self, query: str, k: int = 5, min_relevance: float = None) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and BM25 results
        
        Args:
            query: Search query string
            k: Maximum number of results to return
            min_relevance: Minimum relevance threshold (uses default if None)
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        if min_relevance is None:
            min_relevance = self.min_relevance_threshold
        
        logger.info(f"Hybrid search for: '{query}' (k={k}, min_relevance={min_relevance})")
        
        # Analyze query to determine search strategy
        query_analysis = self.query_processor.analyze_query(query)
        logger.debug(f"Query analysis: {query_analysis}")
        
        # Perform vector search
        vector_results = []
        try:
            vector_results = self.vector_store.query(query, k=self.max_results)
            logger.debug(f"Vector search returned {len(vector_results)} results")
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
        
        # Perform BM25 search
        bm25_results = []
        try:
            bm25_results = self.bm25_index.search(query, k=self.max_results)
            logger.debug(f"BM25 search returned {len(bm25_results)} results")
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
        
        # Fuse results
        unified_results = self.result_fusion.fuse_results(
            vector_results, bm25_results, query_analysis
        )
        
        # Filter by relevance threshold and limit results
        filtered_results = [
            result for result in unified_results 
            if result.relevance_score >= min_relevance
        ][:k]
        
        logger.info(f"Hybrid search returned {len(filtered_results)} results above threshold")
        
        # Log result details
        for i, result in enumerate(filtered_results[:3]):  # Log top 3
            logger.debug(f"Result {i+1}: {result.file_path} "
                        f"(relevance: {result.relevance_score:.3f}, "
                        f"type: {result.search_type})")
        
        return filtered_results
    
    def get_search_statistics(self) -> Dict:
        """Get search engine statistics"""
        vector_stats = {}
        try:
            # Get vector store stats if available
            if hasattr(self.vector_store, 'meta'):
                vector_stats = {
                    'total_chunks': len(self.vector_store.meta),
                    'index_available': self.vector_store.index is not None
                }
        except:
            pass
        
        bm25_stats = self.bm25_index.get_statistics()
        
        return {
            'vector_store': vector_stats,
            'bm25_index': bm25_stats,
            'fusion_weights': {
                'vector_weight': self.result_fusion.vector_weight,
                'bm25_weight': self.result_fusion.bm25_weight
            },
            'search_parameters': {
                'min_relevance_threshold': self.min_relevance_threshold,
                'max_results': self.max_results
            }
        }
    
    def update_fusion_weights(self, vector_weight: float, bm25_weight: float):
        """Update result fusion weights"""
        self.result_fusion.vector_weight = vector_weight
        self.result_fusion.bm25_weight = bm25_weight
        logger.info(f"Updated fusion weights: vector={vector_weight}, bm25={bm25_weight}")
    
    def set_relevance_threshold(self, threshold: float):
        """Set minimum relevance threshold"""
        self.min_relevance_threshold = threshold
        logger.info(f"Updated relevance threshold to {threshold}")