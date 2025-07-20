#!/usr/bin/env python3
"""
Test script for Context Delivery System
"""

import sys
from pathlib import Path
# Import components individually to avoid dependencies
import sys
import os
from dataclasses import dataclass
from typing import List, Dict

# Mock SearchResult class
@dataclass
class SearchResult:
    chunk_id: int
    file_path: str
    snippet: str
    relevance_score: float
    vector_score: float
    bm25_score: float
    search_type: str
    matched_terms: List[str]
    metadata: Dict

# Import context delivery components without hybrid search dependency
sys.path.append(os.path.dirname(__file__))

# Mock the hybrid_search import
import types
mock_hybrid_search = types.ModuleType('hybrid_search')
mock_hybrid_search.HybridSearchEngine = object
mock_hybrid_search.SearchResult = SearchResult
sys.modules['hybrid_search'] = mock_hybrid_search

# Now import the context delivery components
from context_delivery import ContextDeliverySystem, TokenCounter, ContextOptimizer, ContextChunk

# Mock hybrid search engine for testing
class MockHybridSearchEngine:
    def __init__(self):
        pass
    
    def search(self, query: str, k: int = 5, min_relevance: float = 0.25):
        """Mock search that returns relevant results"""
        # Simulate search results based on query
        mock_results = []
        
        if "process data" in query.lower():
            mock_results.append(SearchResult(
                chunk_id=1,
                file_path="utils.py",
                snippet="def process_data(input_data):\n    \"\"\"Process input data and return results\"\"\"\n    return [item.upper() for item in input_data]",
                relevance_score=0.9,
                vector_score=0.8,
                bm25_score=0.7,
                search_type="hybrid",
                matched_terms=["process", "data"],
                metadata={}
            ))
        
        if "class" in query.lower():
            mock_results.append(SearchResult(
                chunk_id=2,
                file_path="processor.py",
                snippet="class DataProcessor:\n    \"\"\"A class for processing data\"\"\"\n    def __init__(self):\n        self.data = []\n    \n    def process(self, items):\n        return [item.strip() for item in items]",
                relevance_score=0.85,
                vector_score=0.9,
                bm25_score=0.6,
                search_type="hybrid",
                matched_terms=["class", "processor"],
                metadata={}
            ))
        
        if "async" in query.lower():
            mock_results.append(SearchResult(
                chunk_id=3,
                file_path="api.js",
                snippet="async function fetchData(url) {\n    const response = await fetch(url);\n    return response.json();\n}",
                relevance_score=0.8,
                vector_score=0.7,
                bm25_score=0.8,
                search_type="hybrid",
                matched_terms=["async", "function"],
                metadata={}
            ))
        
        # Filter by relevance and return top k
        filtered_results = [r for r in mock_results if r.relevance_score >= min_relevance]
        return filtered_results[:k]
    
    def get_search_statistics(self):
        return {
            'vector_store': {'total_chunks': 10},
            'bm25_index': {'total_documents': 10, 'vocabulary_size': 50}
        }

def test_token_counter():
    """Test token counting functionality"""
    print("Testing Token Counter...")
    
    # Test with different models
    counter_gpt4 = TokenCounter("gpt-4")
    counter_gpt35 = TokenCounter("gpt-3.5-turbo")
    
    test_texts = [
        "def hello_world():\n    print('Hello, World!')",
        "class MyClass:\n    def __init__(self):\n        self.value = 42",
        "async function fetchData(url) { return await fetch(url); }",
        "# This is a comment\nprint('Hello')"
    ]
    
    print("Token counts for different texts:")
    for i, text in enumerate(test_texts):
        tokens_gpt4 = counter_gpt4.count_tokens(text)
        tokens_gpt35 = counter_gpt35.count_tokens(text)
        estimated = counter_gpt4.estimate_tokens(text)
        
        print(f"  Text {i+1}: GPT-4={tokens_gpt4}, GPT-3.5={tokens_gpt35}, Estimated={estimated}")
        print(f"    Content: {text[:50]}...")
        
        # Basic validation
        if tokens_gpt4 <= 0 or tokens_gpt35 <= 0:
            print(f"❌ Invalid token count for text {i+1}")
            return False
    
    print("✅ Token counting works correctly")
    return True

def test_context_optimizer():
    """Test context optimization functionality"""
    print("\nTesting Context Optimizer...")
    
    token_counter = TokenCounter("gpt-4")
    optimizer = ContextOptimizer(token_counter)
    
    # Create mock search results
    search_results = [
        SearchResult(
            chunk_id=1,
            file_path="utils.py",
            snippet="def process_data(input_data):\n    return [item.upper() for item in input_data]",
            relevance_score=0.9,
            vector_score=0.8,
            bm25_score=0.7,
            search_type="hybrid",
            matched_terms=["process", "data"],
            metadata={}
        ),
        SearchResult(
            chunk_id=2,
            file_path="processor.py",
            snippet="class DataProcessor:\n    def __init__(self):\n        self.data = []",
            relevance_score=0.7,
            vector_score=0.6,
            bm25_score=0.8,
            search_type="hybrid",
            matched_terms=["class"],
            metadata={}
        ),
        SearchResult(
            chunk_id=3,
            file_path="config.py",
            snippet="# Configuration settings\nDEBUG = True\nDATABASE_URL = 'sqlite:///app.db'",
            relevance_score=0.5,
            vector_score=0.4,
            bm25_score=0.6,
            search_type="bm25",
            matched_terms=["config"],
            metadata={}
        )
    ]
    
    # Test context optimization
    context_package = optimizer.optimize_context(
        search_results, 
        max_tokens=500,  # Small budget to test optimization
        min_relevance=0.6
    )
    
    print(f"Context optimization results:")
    print(f"  Total tokens: {context_package.total_tokens}")
    print(f"  Chunks included: {context_package.chunks_included}")
    print(f"  Files covered: {context_package.files_covered}")
    print(f"  Context summary: {context_package.context_summary}")
    print(f"  Attribution: {context_package.attribution}")
    
    # Validate results
    if context_package.total_tokens <= 0:
        print("❌ No tokens in context package")
        return False
    
    if context_package.chunks_included <= 0:
        print("❌ No chunks included in context package")
        return False
    
    if context_package.total_tokens > 600:  # Allow some overhead
        print(f"❌ Context exceeds token budget: {context_package.total_tokens} > 600")
        return False
    
    # Check that high-relevance chunks are included
    if context_package.chunks_included < 2:  # Should include at least the top 2 relevant chunks
        print(f"❌ Expected at least 2 chunks, got {context_package.chunks_included}")
        return False
    
    print("✅ Context optimization works correctly")
    return True

def test_context_delivery_system():
    """Test the complete context delivery system"""
    print("\nTesting Context Delivery System...")
    
    # Create mock hybrid search engine
    mock_search = MockHybridSearchEngine()
    
    # Create context delivery system
    context_system = ContextDeliverySystem(mock_search, model_name="gpt-4")
    
    # Test different queries
    test_queries = [
        ("process data function", 1000),
        ("DataProcessor class", 800),
        ("async function javascript", 600),
        ("nonexistent query", 500)
    ]
    
    all_passed = True
    
    for query, max_tokens in test_queries:
        print(f"\n--- Testing query: '{query}' (max_tokens={max_tokens}) ---")
        
        try:
            context_package = context_system.prepare_context(
                query, 
                max_tokens=max_tokens,
                min_relevance=0.3
            )
            
            print(f"Results:")
            print(f"  Total tokens: {context_package.total_tokens}")
            print(f"  Chunks included: {context_package.chunks_included}")
            print(f"  Files covered: {context_package.files_covered}")
            print(f"  Relevance threshold: {context_package.relevance_threshold}")
            
            # Show formatted context preview
            preview = context_package.formatted_context[:200] + "..." if len(context_package.formatted_context) > 200 else context_package.formatted_context
            print(f"  Context preview: {preview}")
            
            # Validate results
            if context_package.total_tokens > max_tokens * 1.1:  # Allow 10% overhead
                print(f"❌ Context exceeds token budget: {context_package.total_tokens} > {max_tokens}")
                all_passed = False
            
            # For queries that should find results
            if query != "nonexistent query":
                if context_package.chunks_included == 0:
                    print(f"❌ Expected to find chunks for query: {query}")
                    all_passed = False
                else:
                    print(f"✅ Found relevant context for query")
            else:
                # For nonexistent query, should handle gracefully
                if context_package.chunks_included == 0:
                    print(f"✅ Correctly handled nonexistent query")
                else:
                    print(f"⚠️  Unexpected results for nonexistent query")
        
        except Exception as e:
            print(f"❌ Error processing query '{query}': {e}")
            all_passed = False
    
    return all_passed

def test_context_statistics():
    """Test context delivery system statistics"""
    print("\nTesting Context Statistics...")
    
    mock_search = MockHybridSearchEngine()
    context_system = ContextDeliverySystem(mock_search, model_name="gpt-4")
    
    # Get statistics
    stats = context_system.get_context_statistics()
    
    print("Context Delivery System Statistics:")
    print(f"  Token counter model: {stats['token_counter']['model_name']}")
    print(f"  Token encoding: {stats['token_counter']['encoding_name']}")
    print(f"  Default max tokens: {stats['default_parameters']['max_tokens']}")
    print(f"  Default min relevance: {stats['default_parameters']['min_relevance']}")
    print(f"  Default max results: {stats['default_parameters']['max_results']}")
    print(f"  Hybrid search stats: {stats['hybrid_search_stats']}")
    
    # Test parameter updates
    context_system.update_parameters(max_tokens=5000, min_relevance=0.4)
    updated_stats = context_system.get_context_statistics()
    
    if updated_stats['default_parameters']['max_tokens'] != 5000:
        print("❌ Parameter update failed")
        return False
    
    print("✅ Statistics and parameter updates work correctly")
    return True

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\nTesting Edge Cases...")
    
    token_counter = TokenCounter("gpt-4")
    optimizer = ContextOptimizer(token_counter)
    
    # Test with empty search results
    empty_context = optimizer.optimize_context([], max_tokens=1000)
    if empty_context.chunks_included != 0:
        print("❌ Empty search results should produce empty context")
        return False
    
    # Test with very small token budget
    search_results = [
        SearchResult(
            chunk_id=1,
            file_path="test.py",
            snippet="def test(): pass",
            relevance_score=0.9,
            vector_score=0.8,
            bm25_score=0.7,
            search_type="hybrid",
            matched_terms=["test"],
            metadata={}
        )
    ]
    
    small_budget_context = optimizer.optimize_context(search_results, max_tokens=50)
    if small_budget_context.total_tokens > 100:  # Allow some overhead
        print(f"❌ Small budget context too large: {small_budget_context.total_tokens}")
        return False
    
    # Test with very low relevance threshold
    low_relevance_context = optimizer.optimize_context(search_results, min_relevance=0.1)
    if low_relevance_context.chunks_included == 0:
        print("❌ Low relevance threshold should include some results")
        return False
    
    print("✅ Edge cases handled correctly")
    return True

def main():
    """Run all context delivery tests"""
    print("=== Context Delivery System Tests ===")
    
    test1 = test_token_counter()
    test2 = test_context_optimizer()
    test3 = test_context_delivery_system()
    test4 = test_context_statistics()
    test5 = test_edge_cases()
    
    if test1 and test2 and test3 and test4 and test5:
        print("\n✅ All context delivery tests passed!")
        return True
    else:
        print("\n❌ Some context delivery tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)