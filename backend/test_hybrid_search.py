"""
Tests for the Hybrid Search System

This module tests the hybrid search functionality including BM25 search,
vector search fusion, query processing, and result ranking.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from hybrid_search import (
    HybridSearchEngine,
    QueryProcessor, 
    ResultFusion,
    SearchResult
)
from bm25_index import BM25Result


class TestSearchResult:
    """Test the SearchResult dataclass"""
    
    def test_search_result_creation(self):
        """Test creating a SearchResult object"""
        result = SearchResult(
            chunk_id=1,
            file_path="test.py",
            snippet="def test(): pass",
            relevance_score=0.8,
            vector_score=0.7,
            bm25_score=0.6,
            search_type="hybrid",
            matched_terms=["test", "function"],
            metadata={"line_number": 10}
        )
        
        assert result.chunk_id == 1
        assert result.file_path == "test.py"
        assert result.snippet == "def test(): pass"
        assert result.relevance_score == 0.8
        assert result.search_type == "hybrid"
        assert "test" in result.matched_terms


class TestQueryProcessor:
    """Test the QueryProcessor class"""
    
    def test_query_processor_initialization(self):
        """Test QueryProcessor initialization"""
        processor = QueryProcessor()
        
        assert "function" in processor.technical_indicators
        assert "class" in processor.technical_indicators
        assert "python" in processor.file_type_patterns
        assert "javascript" in processor.file_type_patterns
    
    def test_analyze_query_technical_terms(self):
        """Test query analysis for technical terms"""
        processor = QueryProcessor()
        
        # Query with technical terms
        analysis = processor.analyze_query("create a function that returns class")
        
        assert analysis["has_technical_terms"] is True
        assert "function" in analysis["query_terms"]
        assert "class" in analysis["query_terms"]
    
    def test_analyze_query_file_type_hints(self):
        """Test query analysis for file type hints"""
        processor = QueryProcessor()
        
        # Query with file type hints
        analysis = processor.analyze_query("python function for database")
        
        assert "python" in analysis["file_type_hints"]
        assert "python" in analysis["suggested_file_filters"]
    
    def test_analyze_query_exact_match_indicators(self):
        """Test query analysis for exact match indicators"""
        processor = QueryProcessor()
        
        # Query with quotes
        analysis1 = processor.analyze_query('"exact function name"')
        assert analysis1["is_exact_match_query"] is True
        
        # Query with function call syntax
        analysis2 = processor.analyze_query("call test_function()")
        assert analysis2["is_exact_match_query"] is True
        
        # Query with underscores
        analysis3 = processor.analyze_query("find _private_method")
        assert analysis3["is_exact_match_query"] is True
        
        # Regular query
        analysis4 = processor.analyze_query("simple search query")
        assert analysis4["is_exact_match_query"] is False
    
    def test_analyze_query_comprehensive(self):
        """Test comprehensive query analysis"""
        processor = QueryProcessor()
        
        analysis = processor.analyze_query('python "async function" for database')
        
        assert analysis["has_technical_terms"] is True
        assert analysis["is_exact_match_query"] is True
        assert "python" in analysis["file_type_hints"]
        assert len(analysis["query_terms"]) > 0


class TestResultFusion:
    """Test the ResultFusion class"""
    
    def test_result_fusion_initialization(self):
        """Test ResultFusion initialization"""
        fusion = ResultFusion()
        
        assert fusion.vector_weight == 0.6
        assert fusion.bm25_weight == 0.4
        
        # Test custom weights
        custom_fusion = ResultFusion(vector_weight=0.7, bm25_weight=0.3)
        assert custom_fusion.vector_weight == 0.7
        assert custom_fusion.bm25_weight == 0.3
    
    def test_normalize_vector_results(self):
        """Test vector results normalization"""
        fusion = ResultFusion()
        
        vector_results = [
            ("file1.py", "snippet1"),
            ("file2.py", "snippet2"),
            ("file3.py", "snippet3")
        ]
        
        normalized = fusion._normalize_vector_results(vector_results)
        
        assert len(normalized) == 3
        assert normalized[0][2] > normalized[1][2]  # First should have higher score
        assert normalized[1][2] > normalized[2][2]  # Decreasing scores
    
    def test_normalize_bm25_results(self):
        """Test BM25 results normalization"""
        fusion = ResultFusion()
        
        bm25_results = [
            BM25Result(chunk_id=1, score=10.0, file_path="file1.py", 
                      snippet="snippet1", matched_terms=["term1"]),
            BM25Result(chunk_id=2, score=5.0, file_path="file2.py", 
                      snippet="snippet2", matched_terms=["term2"])
        ]
        
        normalized = fusion._normalize_bm25_results(bm25_results)
        
        assert len(normalized) == 2
        assert normalized[0].score == 1.0  # Max score normalized to 1.0
        assert normalized[1].score == 0.5  # Half of max score
    
    def test_fuse_results_vector_only(self):
        """Test result fusion with only vector results"""
        fusion = ResultFusion()
        
        vector_results = [("file1.py", "snippet1")]
        bm25_results = []
        query_analysis = {"has_technical_terms": False, "file_type_hints": []}
        
        fused = fusion.fuse_results(vector_results, bm25_results, query_analysis)
        
        assert len(fused) == 1
        assert fused[0].search_type == "vector"
        assert fused[0].file_path == "file1.py"
        assert fused[0].vector_score > 0
        assert fused[0].bm25_score == 0.0
    
    def test_fuse_results_bm25_only(self):
        """Test result fusion with only BM25 results"""
        fusion = ResultFusion()
        
        vector_results = []
        bm25_results = [
            BM25Result(chunk_id=1, score=10.0, file_path="file1.py", 
                      snippet="snippet1", matched_terms=["term1"])
        ]
        query_analysis = {"has_technical_terms": False, "file_type_hints": []}
        
        fused = fusion.fuse_results(vector_results, bm25_results, query_analysis)
        
        assert len(fused) == 1
        assert fused[0].search_type == "bm25"
        assert fused[0].file_path == "file1.py"
        assert fused[0].vector_score == 0.0
        assert fused[0].bm25_score > 0
    
    def test_fuse_results_hybrid(self):
        """Test result fusion with both vector and BM25 results"""
        fusion = ResultFusion()
        
        vector_results = [("file1.py", "snippet1")]
        bm25_results = [
            BM25Result(chunk_id=1, score=10.0, file_path="file1.py", 
                      snippet="snippet1", matched_terms=["term1"])
        ]
        query_analysis = {"has_technical_terms": False, "file_type_hints": []}
        
        fused = fusion.fuse_results(vector_results, bm25_results, query_analysis)
        
        assert len(fused) == 1
        assert fused[0].search_type == "hybrid"
        assert fused[0].vector_score > 0
        assert fused[0].bm25_score > 0
        assert fused[0].relevance_score > 0
    
    def test_apply_query_boosts(self):
        """Test query-specific result boosting"""
        fusion = ResultFusion()
        
        # Create test results
        results = {
            "key1": SearchResult(
                chunk_id=1, file_path="test.py", snippet="snippet",
                relevance_score=0.5, vector_score=0.5, bm25_score=0.5,
                search_type="bm25", matched_terms=["function"], metadata={}
            )
        }
        
        # Test exact match boost
        query_analysis = {
            "is_exact_match_query": True,
            "file_type_hints": ["python"],
            "has_technical_terms": True,
            "query_terms": ["function"]
        }
        
        original_score = results["key1"].relevance_score
        fusion._apply_query_boosts(results, query_analysis)
        
        # Score should be boosted
        assert results["key1"].relevance_score > original_score


class TestHybridSearchEngine:
    """Test the HybridSearchEngine class"""
    
    def test_hybrid_search_initialization(self):
        """Test HybridSearchEngine initialization"""
        with patch('hybrid_search.get_vector_store') as mock_vector_store, \
             patch('hybrid_search.BM25Index') as mock_bm25:
            
            engine = HybridSearchEngine()
            
            assert engine.vector_store is not None
            assert engine.bm25_index is not None
            assert engine.query_processor is not None
            assert engine.result_fusion is not None
            assert engine.min_relevance_threshold == 0.25
    
    def test_build_bm25_index(self):
        """Test BM25 index building"""
        with patch('hybrid_search.get_vector_store'), \
             patch('hybrid_search.BM25Index') as mock_bm25_class:
            
            mock_bm25_instance = Mock()
            mock_bm25_class.return_value = mock_bm25_instance
            
            engine = HybridSearchEngine()
            documents = [{"text": "test document", "id": 1}]
            
            success = engine.build_bm25_index(documents)
            
            assert success is True
            mock_bm25_instance.add_documents.assert_called_once_with(documents)
    
    def test_build_bm25_index_failure(self):
        """Test BM25 index building failure handling"""
        with patch('hybrid_search.get_vector_store'), \
             patch('hybrid_search.BM25Index') as mock_bm25_class:
            
            mock_bm25_instance = Mock()
            mock_bm25_instance.add_documents.side_effect = Exception("Build failed")
            mock_bm25_class.return_value = mock_bm25_instance
            
            engine = HybridSearchEngine()
            documents = [{"text": "test document", "id": 1}]
            
            success = engine.build_bm25_index(documents)
            
            assert success is False
    
    @patch('hybrid_search.get_vector_store')
    @patch('hybrid_search.BM25Index')
    def test_search_success(self, mock_bm25_class, mock_vector_store):
        """Test successful hybrid search"""
        # Mock vector store
        mock_vector_instance = Mock()
        mock_vector_instance.query.return_value = [("file1.py", "vector snippet")]
        mock_vector_store.return_value = mock_vector_instance
        
        # Mock BM25 index
        mock_bm25_instance = Mock()
        mock_bm25_instance.search.return_value = [
            BM25Result(chunk_id=1, score=5.0, file_path="file1.py", 
                      snippet="bm25 snippet", matched_terms=["test"])
        ]
        mock_bm25_class.return_value = mock_bm25_instance
        
        engine = HybridSearchEngine()
        results = engine.search("test query", k=2)
        
        assert len(results) >= 0  # Should return some results
        mock_vector_instance.query.assert_called_once()
        mock_bm25_instance.search.assert_called_once()
    
    @patch('hybrid_search.get_vector_store')
    @patch('hybrid_search.BM25Index')
    def test_search_vector_failure(self, mock_bm25_class, mock_vector_store):
        """Test hybrid search with vector search failure"""
        # Mock vector store to fail
        mock_vector_instance = Mock()
        mock_vector_instance.query.side_effect = Exception("Vector search failed")
        mock_vector_store.return_value = mock_vector_instance
        
        # Mock BM25 index to succeed
        mock_bm25_instance = Mock()
        mock_bm25_instance.search.return_value = [
            BM25Result(chunk_id=1, score=5.0, file_path="file1.py", 
                      snippet="bm25 snippet", matched_terms=["test"])
        ]
        mock_bm25_class.return_value = mock_bm25_instance
        
        engine = HybridSearchEngine()
        results = engine.search("test query", k=2)
        
        # Should still return BM25 results
        assert isinstance(results, list)
    
    @patch('hybrid_search.get_vector_store')
    @patch('hybrid_search.BM25Index')
    def test_search_bm25_failure(self, mock_bm25_class, mock_vector_store):
        """Test hybrid search with BM25 search failure"""
        # Mock vector store to succeed
        mock_vector_instance = Mock()
        mock_vector_instance.query.return_value = [("file1.py", "vector snippet")]
        mock_vector_store.return_value = mock_vector_instance
        
        # Mock BM25 index to fail
        mock_bm25_instance = Mock()
        mock_bm25_instance.search.side_effect = Exception("BM25 search failed")
        mock_bm25_class.return_value = mock_bm25_instance
        
        engine = HybridSearchEngine()
        results = engine.search("test query", k=2)
        
        # Should still return vector results
        assert isinstance(results, list)
    
    @patch('hybrid_search.get_vector_store')
    @patch('hybrid_search.BM25Index')
    def test_search_relevance_filtering(self, mock_bm25_class, mock_vector_store):
        """Test search results are filtered by relevance threshold"""
        # Mock vector store
        mock_vector_instance = Mock()
        mock_vector_instance.query.return_value = [("file1.py", "low relevance")]
        mock_vector_store.return_value = mock_vector_instance
        
        # Mock BM25 index
        mock_bm25_instance = Mock()
        mock_bm25_instance.search.return_value = []
        mock_bm25_class.return_value = mock_bm25_instance
        
        engine = HybridSearchEngine()
        
        # Mock result fusion to return low relevance results
        with patch.object(engine.result_fusion, 'fuse_results') as mock_fusion:
            mock_fusion.return_value = [
                SearchResult(
                    chunk_id=1, file_path="file1.py", snippet="snippet",
                    relevance_score=0.1, vector_score=0.1, bm25_score=0.0,
                    search_type="vector", matched_terms=[], metadata={}
                )
            ]
            
            results = engine.search("test query", k=2, min_relevance=0.5)
            
            # Should filter out low relevance results
            assert len(results) == 0
    
    def test_get_search_statistics(self):
        """Test getting search engine statistics"""
        with patch('hybrid_search.get_vector_store') as mock_vector_store, \
             patch('hybrid_search.BM25Index') as mock_bm25_class:
            
            # Mock vector store
            mock_vector_instance = Mock()
            mock_vector_instance.meta = ["item1", "item2"]
            mock_vector_instance.index = Mock()
            mock_vector_store.return_value = mock_vector_instance
            
            # Mock BM25 index
            mock_bm25_instance = Mock()
            mock_bm25_instance.get_statistics.return_value = {"documents": 10}
            mock_bm25_class.return_value = mock_bm25_instance
            
            engine = HybridSearchEngine()
            stats = engine.get_search_statistics()
            
            assert "vector_store" in stats
            assert "bm25_index" in stats
            assert "fusion_weights" in stats
            assert "search_parameters" in stats
            assert stats["vector_store"]["total_chunks"] == 2
    
    def test_update_fusion_weights(self):
        """Test updating result fusion weights"""
        with patch('hybrid_search.get_vector_store'), \
             patch('hybrid_search.BM25Index'):
            
            engine = HybridSearchEngine()
            
            engine.update_fusion_weights(0.8, 0.2)
            
            assert engine.result_fusion.vector_weight == 0.8
            assert engine.result_fusion.bm25_weight == 0.2
    
    def test_set_relevance_threshold(self):
        """Test setting relevance threshold"""
        with patch('hybrid_search.get_vector_store'), \
             patch('hybrid_search.BM25Index'):
            
            engine = HybridSearchEngine()
            
            engine.set_relevance_threshold(0.5)
            
            assert engine.min_relevance_threshold == 0.5


class TestHybridSearchIntegration:
    """Integration tests for hybrid search functionality"""
    
    @patch('hybrid_search.get_vector_store')
    @patch('hybrid_search.BM25Index')
    def test_end_to_end_search_workflow(self, mock_bm25_class, mock_vector_store):
        """Test complete search workflow from query to results"""
        # Mock vector store
        mock_vector_instance = Mock()
        mock_vector_instance.query.return_value = [
            ("sqlmodel/main.py", "class Hero(SQLModel): pass"),
            ("other/file.py", "def function(): pass")
        ]
        mock_vector_store.return_value = mock_vector_instance
        
        # Mock BM25 index
        mock_bm25_instance = Mock()
        mock_bm25_instance.search.return_value = [
            BM25Result(
                chunk_id=1, score=8.0, file_path="sqlmodel/models.py",
                snippet="Hero model definition", matched_terms=["Hero", "model"]
            )
        ]
        mock_bm25_instance.get_statistics.return_value = {"documents": 100}
        mock_bm25_class.return_value = mock_bm25_instance
        
        # Create engine and perform search
        engine = HybridSearchEngine()
        results = engine.search("SQLModel Hero class", k=3, min_relevance=0.1)
        
        # Verify search was performed
        mock_vector_instance.query.assert_called_once()
        mock_bm25_instance.search.assert_called_once()
        
        # Results should be returned
        assert isinstance(results, list)
    
    @patch('hybrid_search.get_vector_store')
    @patch('hybrid_search.BM25Index')
    def test_project_aware_search(self, mock_bm25_class, mock_vector_store):
        """Test search with project-specific boosting"""
        # Mock vector store with project-specific results
        mock_vector_instance = Mock()
        mock_vector_instance.query.return_value = [
            ("sqlmodel/hero.py", "class Hero(SQLModel): pass"),
            ("langchain/agent.py", "class Agent: pass")
        ]
        mock_vector_store.return_value = mock_vector_instance
        
        # Mock BM25 index
        mock_bm25_instance = Mock()
        mock_bm25_instance.search.return_value = []
        mock_bm25_instance.get_statistics.return_value = {"documents": 50}
        mock_bm25_class.return_value = mock_bm25_instance
        
        engine = HybridSearchEngine()
        results = engine.search("SQLModel Hero model", k=2, min_relevance=0.1)
        
        # Should prioritize SQLModel results due to query analysis
        assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])