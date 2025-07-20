#!/usr/bin/env python3
"""
Test script for BM25 indexing system
"""

import sys
from pathlib import Path
from bm25_index import BM25Index

# Sample documents for testing
SAMPLE_DOCUMENTS = [
    {
        'id': 1,
        'text': 'def process_data(input_data): return [item.upper() for item in input_data]',
        'file_path': 'utils.py',
        'metadata': {'function_name': 'process_data', 'file_type': 'python'}
    },
    {
        'id': 2,
        'text': 'class DataProcessor: def __init__(self): self.data = [] def process(self, items): return items',
        'file_path': 'processor.py',
        'metadata': {'class_name': 'DataProcessor', 'file_type': 'python'}
    },
    {
        'id': 3,
        'text': 'async function fetchData(url) { const response = await fetch(url); return response.json(); }',
        'file_path': 'api.js',
        'metadata': {'function_name': 'fetchData', 'file_type': 'javascript'}
    },
    {
        'id': 4,
        'text': 'import React from "react"; export default function App() { return <div>Hello World</div>; }',
        'file_path': 'App.jsx',
        'metadata': {'function_name': 'App', 'file_type': 'javascript'}
    },
    {
        'id': 5,
        'text': '# Configuration Settings\ndatabase_url = "postgresql://localhost/mydb"\napi_key = "secret123"',
        'file_path': 'config.py',
        'metadata': {'file_type': 'python', 'chunk_type': 'config'}
    }
]

def test_bm25_indexing():
    """Test BM25 index creation and basic functionality"""
    print("Testing BM25 Index Creation...")
    
    # Create and populate index
    bm25 = BM25Index()
    bm25.add_documents(SAMPLE_DOCUMENTS)
    
    # Check index statistics
    stats = bm25.get_statistics()
    print(f"Index Statistics:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Vocabulary size: {stats['vocabulary_size']}")
    print(f"  Average document length: {stats['average_document_length']:.2f}")
    
    return stats['total_documents'] == len(SAMPLE_DOCUMENTS)

def test_bm25_search():
    """Test BM25 search functionality"""
    print("\nTesting BM25 Search...")
    
    # Create and populate index
    bm25 = BM25Index()
    bm25.add_documents(SAMPLE_DOCUMENTS)
    
    # Test queries
    test_queries = [
        "process data",
        "function",
        "DataProcessor class",
        "async fetch",
        "React component",
        "database configuration",
        "python",
        "javascript"
    ]
    
    all_tests_passed = True
    
    for query in test_queries:
        results = bm25.search(query, k=3)
        print(f"\nQuery: '{query}'")
        print(f"Results: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.file_path} (score: {result.score:.3f})")
            print(f"     Matched terms: {result.matched_terms}")
            print(f"     Snippet: {result.snippet[:60]}...")
        
        # Basic validation
        if query == "process data" and len(results) > 0:
            # Should find the process_data function
            found_process_data = any("process_data" in result.snippet or "process" in result.matched_terms for result in results)
            if not found_process_data:
                print(f"❌ Failed to find relevant results for '{query}'")
                all_tests_passed = False
        
        if query == "DataProcessor class" and len(results) > 0:
            # Should find the DataProcessor class
            found_class = any("DataProcessor" in result.snippet for result in results)
            if not found_class:
                print(f"❌ Failed to find DataProcessor class for '{query}'")
                all_tests_passed = False
    
    return all_tests_passed

def test_bm25_edge_cases():
    """Test BM25 edge cases and error handling"""
    print("\nTesting BM25 Edge Cases...")
    
    bm25 = BM25Index()
    
    # Test empty index search
    results = bm25.search("test query")
    if len(results) != 0:
        print("❌ Empty index should return no results")
        return False
    
    # Add documents and test
    bm25.add_documents(SAMPLE_DOCUMENTS)
    
    # Test empty query
    results = bm25.search("")
    if len(results) != 0:
        print("❌ Empty query should return no results")
        return False
    
    # Test query with no matches
    results = bm25.search("nonexistent_term_xyz123")
    if len(results) != 0:
        print("❌ Query with no matches should return no results")
        return False
    
    # Test query with stop words only
    results = bm25.search("the and or")
    if len(results) != 0:
        print("❌ Stop words only query should return no results")
        return False
    
    print("✅ All edge cases handled correctly")
    return True

def test_bm25_persistence():
    """Test saving and loading BM25 index"""
    print("\nTesting BM25 Index Persistence...")
    
    # Create and populate index
    bm25_original = BM25Index()
    bm25_original.add_documents(SAMPLE_DOCUMENTS)
    
    # Save index
    index_file = Path("test_bm25_index.json")
    if not bm25_original.save_index(index_file):
        print("❌ Failed to save BM25 index")
        return False
    
    # Load index into new instance
    bm25_loaded = BM25Index()
    if not bm25_loaded.load_index(index_file):
        print("❌ Failed to load BM25 index")
        return False
    
    # Compare statistics
    original_stats = bm25_original.get_statistics()
    loaded_stats = bm25_loaded.get_statistics()
    
    if original_stats != loaded_stats:
        print("❌ Loaded index statistics don't match original")
        print(f"Original: {original_stats}")
        print(f"Loaded: {loaded_stats}")
        return False
    
    # Test search on loaded index
    results_original = bm25_original.search("process data")
    results_loaded = bm25_loaded.search("process data")
    
    if len(results_original) != len(results_loaded):
        print("❌ Search results differ between original and loaded index")
        return False
    
    # Clean up
    try:
        index_file.unlink()
    except:
        pass
    
    print("✅ Index persistence works correctly")
    return True

def main():
    """Run all BM25 tests"""
    print("=== BM25 Index Tests ===")
    
    test1 = test_bm25_indexing()
    test2 = test_bm25_search()
    test3 = test_bm25_edge_cases()
    test4 = test_bm25_persistence()
    
    if test1 and test2 and test3 and test4:
        print("\n✅ All BM25 tests passed!")
        return True
    else:
        print("\n❌ Some BM25 tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)