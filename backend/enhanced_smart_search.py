"""
Enhanced Smart Search with Phase 1 Integrations

Integrates BGE embeddings, hierarchical chunking, and semantic theming
while maintaining backward compatibility with existing smart_search interface.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Phase 1 imports
from ..indexer.enhanced_vector_store import EnhancedVectorStore
from ..indexer.storage.enhanced_metadata_store import EnhancedMetadataStore
from ..indexer.storage.context_reconstructor import ContextReconstructor
from ..benchmarks.evaluator import RetrievalEvaluator

# Existing imports
from .smart_search import (
    SmartSearchResult, QueryIntent, QueryIntentAnalyzer,
    SmartSearchEngine as OriginalSmartSearchEngine
)
from .hybrid_search import HybridSearchEngine
from .bm25_index import BM25Index

logger = logging.getLogger(__name__)


@dataclass
class EnhancedSearchResult(SmartSearchResult):
    """Enhanced search result with hierarchical context."""
    context_metadata: Dict[str, Any] = None
    hierarchical_context: Optional[Dict] = None
    semantic_role: Optional[str] = None
    chunk_type: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = None


class EnhancedSmartSearchEngine:
    """
    Enhanced smart search engine leveraging Phase 1 improvements:
    - BGE model embeddings for superior semantic understanding
    - Hierarchical chunking for better context
    - Multi-signal query classification for adaptive search
    """
    
    def __init__(self, storage_dir: str = ".vector_cache"):
        """
        Initialize enhanced search engine.
        
        Args:
            storage_dir: Directory for vector and metadata storage
        """
        # Phase 1 Enhanced Components
        self.vector_store = EnhancedVectorStore(storage_dir=storage_dir)
        self.metadata_store = EnhancedMetadataStore(storage_dir)
        self.context_reconstructor = ContextReconstructor(self.metadata_store)
        
        # Load existing vector index if available
        if not self.vector_store.load_index():
            logger.warning("No enhanced vector index found. Run reindexing to build BGE index.")
        
        # BM25 for keyword search
        self.bm25_index = BM25Index()
        
        # Hybrid search combining vector + BM25
        self.hybrid_search = HybridSearchEngine(self.vector_store, self.bm25_index)
        
        # Query analysis
        self.intent_analyzer = QueryIntentAnalyzer()
        
        # Performance monitoring
        self.search_stats = {
            'total_searches': 0,
            'bgb_searches': 0,
            'hierarchical_context_used': 0,
            'avg_results_per_query': 0
        }
        
        logger.info("Enhanced smart search engine initialized with Phase 1 improvements")
    
    def search(self, query: str, query_intent: str = None, limit: int = 10,
              use_hierarchical_context: bool = True) -> List[EnhancedSearchResult]:
        """
        Enhanced search using Phase 1 improvements while maintaining existing interface.
        
        Args:
            query: Search query
            query_intent: Optional explicit intent override
            limit: Maximum number of results
            use_hierarchical_context: Whether to include hierarchical context
            
        Returns:
            List of enhanced search results with hierarchical context
        """
        self.search_stats['total_searches'] += 1
        
        logger.debug(f"Enhanced search: '{query}' (limit={limit})")
        
        # Analyze query intent if not provided
        if query_intent is None:
            query_analysis = self.intent_analyzer.analyze_query(query)
            detected_intent = query_analysis['intent']
            confidence = query_analysis['confidence']
        else:
            detected_intent = QueryIntent(query_intent)
            confidence = 1.0
        
        # Use Phase 1 enhanced hybrid search
        try:
            search_results, query_analysis = self.hybrid_search.search(query, limit)
            self.search_stats['bgb_searches'] += 1
        except Exception as e:
            logger.error(f"Enhanced search failed, falling back to basic search: {e}")
            search_results = []
        
        # Convert to enhanced search results with hierarchical context
        enhanced_results = []
        for i, result in enumerate(search_results):
            try:
                enhanced_result = self._create_enhanced_result(
                    result, query, detected_intent, confidence, 
                    use_hierarchical_context
                )
                enhanced_results.append(enhanced_result)
                
            except Exception as e:
                logger.warning(f"Failed to enhance result {i}: {e}")
                # Fallback to basic result
                enhanced_results.append(self._create_fallback_result(result, detected_intent))
        
        # Update stats
        self.search_stats['avg_results_per_query'] = (
            (self.search_stats['avg_results_per_query'] * (self.search_stats['total_searches'] - 1) + 
             len(enhanced_results)) / self.search_stats['total_searches']
        )
        
        logger.debug(f"Enhanced search completed: {len(enhanced_results)} results")
        return enhanced_results
    
    def search_with_context(self, query: str, context_chunk_id: str, 
                           context_radius: int = 2) -> List[EnhancedSearchResult]:
        """
        Search within hierarchical context of a specific chunk.
        
        Args:
            query: Search query
            context_chunk_id: Chunk ID to center context search around
            context_radius: Number of hierarchy levels to include
            
        Returns:
            List of search results within hierarchical context
        """
        logger.debug(f"Context search around chunk {context_chunk_id}")
        
        # Get hierarchical context
        context = self.context_reconstructor.get_full_context(
            context_chunk_id, max_depth=context_radius
        )
        
        if not context:
            logger.warning(f"No context found for chunk {context_chunk_id}")
            return self.search(query)
        
        # Search within context chunks
        context_results = self.context_reconstructor.search_in_context(
            context_chunk_id, query, context_radius
        )
        
        # Convert to enhanced results
        enhanced_results = []
        for i, match_info in enumerate(context_results):
            chunk = match_info['chunk']
            
            result = EnhancedSearchResult(
                chunk_id=chunk['chunk_id'],
                file_path=chunk.get('file_path', ''),
                snippet=chunk.get('chunk_data', {}).get('content', '')[:200],
                relevance_score=1.0 - (match_info['context_distance'] * 0.1),
                query_intent=QueryIntent.GENERAL,
                search_strategy=['hierarchical_context'],
                matched_terms=[query],
                metadata=chunk.get('chunk_data', {}),
                confidence=0.8,
                context_metadata={'context_distance': match_info['context_distance']},
                hierarchical_context=context.target_chunk,
                chunk_type=chunk.get('chunk_type'),
                parent_chunk_id=chunk.get('chunk_data', {}).get('parent_chunk_id'),
                child_chunk_ids=chunk.get('chunk_data', {}).get('child_chunk_ids', [])
            )
            enhanced_results.append(result)
        
        self.search_stats['hierarchical_context_used'] += 1
        return enhanced_results
    
    def get_related_symbols(self, chunk_id: str, relation_types: List[str] = None) -> List[EnhancedSearchResult]:
        """
        Get symbols related to a chunk through hierarchical relationships.
        
        Args:
            chunk_id: Target chunk identifier
            relation_types: Types of relations to consider
            
        Returns:
            List of related symbol results
        """
        related_symbols = self.context_reconstructor.get_related_symbols(
            chunk_id, relation_types
        )
        
        enhanced_results = []
        for symbol in related_symbols:
            result = EnhancedSearchResult(
                chunk_id=symbol['chunk_id'],
                file_path=symbol.get('file_path', ''),
                snippet=symbol.get('chunk_data', {}).get('content', '')[:200],
                relevance_score=0.9,  # High relevance for direct relationships
                query_intent=QueryIntent.ENTITY,
                search_strategy=['hierarchical_relationship'],
                matched_terms=[],
                metadata=symbol.get('chunk_data', {}),
                confidence=0.9,
                context_metadata={'relation_type': 'symbol'},
                chunk_type=symbol.get('chunk_type'),
                semantic_role=symbol.get('chunk_data', {}).get('semantic_role'),
                parent_chunk_id=symbol.get('chunk_data', {}).get('parent_chunk_id'),
                child_chunk_ids=symbol.get('chunk_data', {}).get('child_chunk_ids', [])
            )
            enhanced_results.append(result)
        
        return enhanced_results
    
    def get_file_overview(self, file_path: str) -> Dict[str, Any]:
        """
        Get hierarchical overview of a file using Phase 1 chunking.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file's hierarchical structure
        """
        return self.context_reconstructor.get_file_context(file_path)
    
    def _create_enhanced_result(self, search_result, query: str, intent: QueryIntent,
                               confidence: float, use_hierarchical_context: bool) -> EnhancedSearchResult:
        """Create enhanced search result with hierarchical context."""
        
        # Get chunk metadata
        chunk_data = None
        hierarchical_context = None
        
        if hasattr(search_result, 'chunk_id'):
            chunk_id = search_result.chunk_id
            chunk_data = self.metadata_store.get_chunk_by_id(chunk_id)
            
            if chunk_data and use_hierarchical_context:
                hierarchical_context = self.context_reconstructor.get_full_context(chunk_id)
        
        # Extract metadata
        metadata = getattr(search_result, 'metadata', {})
        if chunk_data:
            metadata.update(chunk_data.get('chunk_data', {}))
        
        # Determine search strategy
        search_strategy = ['enhanced_vector']
        if hasattr(search_result, 'search_type'):
            search_strategy.append(search_result.search_type)
        
        return EnhancedSearchResult(
            chunk_id=getattr(search_result, 'chunk_id', f"result_{id(search_result)}"),
            file_path=getattr(search_result, 'file_path', ''),
            snippet=getattr(search_result, 'snippet', '')[:200],
            relevance_score=getattr(search_result, 'relevance_score', 0.5),
            query_intent=intent,
            search_strategy=search_strategy,
            matched_terms=self._extract_matched_terms(query),
            metadata=metadata,
            confidence=confidence,
            context_metadata=self._extract_hierarchical_context_metadata(search_result),
            hierarchical_context=hierarchical_context.target_chunk if hierarchical_context else None,
            semantic_role=metadata.get('semantic_role'),
            chunk_type=metadata.get('chunk_type'),
            parent_chunk_id=metadata.get('parent_chunk_id'),
            child_chunk_ids=metadata.get('child_chunk_ids', [])
        )
    
    def _create_fallback_result(self, search_result, intent: QueryIntent) -> EnhancedSearchResult:
        """Create fallback result when enhancement fails."""
        return EnhancedSearchResult(
            chunk_id=getattr(search_result, 'chunk_id', f"fallback_{id(search_result)}"),
            file_path=getattr(search_result, 'file_path', ''),
            snippet=getattr(search_result, 'snippet', ''),
            relevance_score=getattr(search_result, 'relevance_score', 0.0),
            query_intent=intent,
            search_strategy=['fallback'],
            matched_terms=[],
            metadata={},
            confidence=0.3,
            context_metadata={},
            hierarchical_context=None,
            chunk_type='unknown'
        )
    
    def _extract_hierarchical_context_metadata(self, search_result) -> Dict[str, Any]:
        """Extract hierarchical context metadata from search result."""
        if hasattr(search_result, 'metadata') and 'chunk_data' in search_result.metadata:
            chunk_data = search_result.metadata['chunk_data']
            
            context = {}
            if 'parent_chunk_id' in chunk_data:
                context['parent'] = chunk_data['parent_chunk_id']
            if 'child_chunk_ids' in chunk_data:
                context['children'] = chunk_data['child_chunk_ids']
            if 'type' in chunk_data:
                context['chunk_type'] = chunk_data['type']
            if 'semantic_role' in chunk_data:
                context['semantic_role'] = chunk_data['semantic_role']
                
            return context
        
        return {}
    
    def _extract_matched_terms(self, query: str) -> List[str]:
        """Extract potential matched terms from query."""
        # Simple word extraction - could be enhanced with NLP
        words = query.lower().split()
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search performance statistics."""
        return {
            **self.search_stats,
            'vector_store_stats': self.vector_store.get_performance_stats(),
            'metadata_store_stats': self.metadata_store.get_storage_stats()
        }
    
    def clear_search_stats(self):
        """Clear search performance statistics."""
        self.search_stats = {
            'total_searches': 0,
            'bgb_searches': 0,
            'hierarchical_context_used': 0,
            'avg_results_per_query': 0
        }
        self.vector_store.clear_performance_stats()


# Backward compatibility alias
SmartSearchEngineEnhanced = EnhancedSmartSearchEngine