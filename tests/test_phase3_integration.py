"""
Phase 3 Integration Tests

Comprehensive test suite for Phase 3.1 Query Intent Classifier
to ensure dynamic search weighting works correctly in Docker environment.
"""

import os
import sys
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPhase3QueryClassifier:
    """Test Phase 3.1: Query Intent Classifier"""
    
    def test_query_classifier_import(self):
        """Test that QueryClassifier can be imported."""
        try:
            from backend.search.query_classifier import QueryClassifier, QueryIntent, QueryAnalysis
            assert True, "QueryClassifier imported successfully"
        except ImportError as e:
            pytest.fail(f"Failed to import QueryClassifier: {e}")
    
    def test_query_classifier_basic_functionality(self):
        """Test basic QueryClassifier functionality."""
        from backend.search.query_classifier import QueryClassifier, QueryIntent
        
        classifier = QueryClassifier()
        
        # Test different query types
        test_cases = [
            ("authenticate_user()", "Should detect specific symbol"),
            ("how does authentication work", "Should detect conceptual query"),
            ("find all functions that call login", "Should detect exploratory/debugging"),
            ("explain the system architecture", "Should detect conceptual/structural"),
            ("UserManager.validate", "Should detect specific symbol")
        ]
        
        for query, description in test_cases:
            analysis = classifier.classify_query(query)
            
            # Verify analysis structure
            assert hasattr(analysis, 'intent'), f"Analysis should have intent for: {query}"
            assert hasattr(analysis, 'confidence'), f"Analysis should have confidence for: {query}"
            assert hasattr(analysis, 'vector_weight'), f"Analysis should have vector_weight for: {query}"
            assert hasattr(analysis, 'bm25_weight'), f"Analysis should have bm25_weight for: {query}"
            
            # Verify weights sum to 1.0
            weight_sum = analysis.vector_weight + analysis.bm25_weight
            assert abs(weight_sum - 1.0) < 0.01, f"Weights should sum to 1.0 for: {query}"
            
            # Verify confidence is in valid range
            assert 0.0 <= analysis.confidence <= 1.0, f"Confidence should be 0-1 for: {query}"
    
    def test_intent_classification_accuracy(self):
        """Test that intent classification works as expected."""
        from backend.search.query_classifier import QueryClassifier, QueryIntent
        
        classifier = QueryClassifier()
        
        # Test specific symbol detection
        specific_queries = [
            "authenticate_user()",
            "UserManager.validate",
            "login_function",
            "MyClass.method()"
        ]
        
        for query in specific_queries:
            analysis = classifier.classify_query(query)
            # Should favor BM25 for specific symbols
            assert analysis.bm25_weight >= 0.5, f"Should favor BM25 for specific symbol: {query}"
        
        # Test conceptual query detection
        conceptual_queries = [
            "how does authentication work",
            "explain the login process",
            "what is the architecture",
            "show me how users are managed"
        ]
        
        for query in conceptual_queries:
            analysis = classifier.classify_query(query)
            # Should favor vector for conceptual queries
            assert analysis.vector_weight >= 0.5, f"Should favor vector for conceptual query: {query}"


class TestPhase3HybridSearchIntegration:
    """Test Phase 3.1: HybridSearchEngine Integration"""
    
    def test_hybrid_search_classifier_integration(self):
        """Test that HybridSearchEngine properly integrates with QueryClassifier."""
        try:
            from backend.hybrid_search import HybridSearchEngine, QUERY_CLASSIFIER_AVAILABLE
            
            # Should be available after our implementation
            assert QUERY_CLASSIFIER_AVAILABLE, "QueryClassifier should be available"
            
            # Test initialization
            search_engine = HybridSearchEngine()
            
            # Should have classifier
            assert hasattr(search_engine, 'query_classifier'), "Should have query_classifier attribute"
            assert search_engine.query_classifier is not None, "QueryClassifier should be initialized"
            
        except ImportError as e:
            pytest.fail(f"Failed to import/initialize HybridSearchEngine: {e}")
    
    def test_dynamic_weight_adjustment(self):
        """Test that weights are dynamically adjusted based on query."""
        from backend.hybrid_search import HybridSearchEngine
        
        search_engine = HybridSearchEngine()
        
        if search_engine.query_classifier:
            # Test different queries and verify weight changes
            specific_query = "authenticate_user()"
            conceptual_query = "how does authentication work"
            
            # Get analyses
            specific_analysis = search_engine.query_classifier.classify_query(specific_query)
            conceptual_analysis = search_engine.query_classifier.classify_query(conceptual_query)
            
            # Specific queries should favor BM25 more than conceptual queries
            assert (specific_analysis.bm25_weight > conceptual_analysis.bm25_weight), \
                "Specific queries should favor BM25 more than conceptual queries"
            
            # Conceptual queries should favor vector more than specific queries
            assert (conceptual_analysis.vector_weight > specific_analysis.vector_weight), \
                "Conceptual queries should favor vector more than specific queries"
    
    def test_search_statistics_tracking(self):
        """Test that search statistics properly track dynamic weighting."""
        from backend.hybrid_search import HybridSearchEngine
        
        search_engine = HybridSearchEngine()
        
        # Get initial statistics
        stats = search_engine.get_search_statistics()
        
        # Should have dynamic weighting section
        assert 'dynamic_weighting' in stats, "Statistics should include dynamic weighting info"
        
        dynamic_stats = stats['dynamic_weighting']
        
        if search_engine.query_classifier:
            assert dynamic_stats['classifier_enabled'] is True, "Classifier should be enabled"
            assert 'total_searches' in dynamic_stats, "Should track total searches"
            assert 'avg_vector_weight' in dynamic_stats, "Should track average vector weight"
            assert 'avg_bm25_weight' in dynamic_stats, "Should track average BM25 weight"
            assert 'intent_distribution' in dynamic_stats, "Should track intent distribution"
        else:
            assert dynamic_stats['classifier_enabled'] is False, "Should indicate classifier disabled"
            assert 'reason' in dynamic_stats, "Should provide reason for classifier being disabled"


class TestPhase3FallbackBehavior:
    """Test Phase 3.1: Fallback Behavior"""
    
    def test_graceful_fallback_when_classifier_unavailable(self):
        """Test that system works when QueryClassifier is not available."""
        # This test simulates the case where the classifier import fails
        
        # Test that HybridSearchEngine can still work without classifier
        # We'll test this by checking the fallback logic in hybrid_search.py
        
        from backend.hybrid_search import HybridSearchEngine
        
        # Even if classifier is available, test the fallback code path
        search_engine = HybridSearchEngine()
        
        # Temporarily disable classifier to test fallback
        original_classifier = search_engine.query_classifier
        search_engine.query_classifier = None
        
        try:
            # Get statistics - should show fallback mode
            stats = search_engine.get_search_statistics()
            dynamic_stats = stats['dynamic_weighting']
            
            assert dynamic_stats['classifier_enabled'] is False, "Should show classifier disabled"
            assert 'reason' in dynamic_stats, "Should provide fallback reason"
            
        finally:
            # Restore original classifier
            search_engine.query_classifier = original_classifier


class TestPhase3EndToEnd:
    """Test Phase 3.1: End-to-End Functionality"""
    
    def test_complete_query_processing_pipeline(self):
        """Test the complete query processing from classification to weight adjustment."""
        from backend.hybrid_search import HybridSearchEngine
        
        search_engine = HybridSearchEngine()
        
        if not search_engine.query_classifier:
            pytest.skip("QueryClassifier not available for end-to-end test")
        
        # Test complete pipeline for different query types
        test_queries = [
            "authenticate_user function",
            "how does user authentication work",
            "find all callers of login method",
            "explain the system architecture"
        ]
        
        for query in test_queries:
            # Classify query
            analysis = search_engine.query_classifier.classify_query(query)
            
            # Simulate weight update (what happens during search)
            original_vector_weight = search_engine.result_fusion.vector_weight
            original_bm25_weight = search_engine.result_fusion.bm25_weight
            
            # Update weights
            search_engine.result_fusion.update_weights(analysis.vector_weight, analysis.bm25_weight)
            
            # Verify weights were updated
            assert search_engine.result_fusion.vector_weight == analysis.vector_weight, \
                f"Vector weight should be updated for query: {query}"
            assert search_engine.result_fusion.bm25_weight == analysis.bm25_weight, \
                f"BM25 weight should be updated for query: {query}"
            
            # Verify the analysis makes sense
            assert analysis.confidence > 0, f"Should have confidence > 0 for query: {query}"
            assert len(analysis.reasoning) > 0, f"Should have reasoning for query: {query}"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])