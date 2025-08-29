#!/usr/bin/env python3
"""
Comprehensive Test Runner for CodeWise

Runs all tests to validate Phase 2 and Phase 3 implementations
in Docker environment. Provides detailed reporting and diagnostics.
"""

import os
import sys
import time
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def print_banner(text):
    """Print a banner with the given text."""
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80)

def print_section(text):
    """Print a section header."""
    print(f"\n--- {text} ---")

def test_environment_setup():
    """Test basic environment setup."""
    print_section("Testing Environment Setup")
    
    tests = []
    
    # Test Python version
    try:
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info >= (3, 8):
            tests.append(("Python Version", f"{python_version}", "PASS"))
        else:
            tests.append(("Python Version", f"{python_version} (need 3.8+)", "FAIL"))
    except Exception as e:
        tests.append(("Python Version", str(e), "FAIL"))
    
    # Test required directories
    dirs_to_check = ['backend', 'storage', 'knowledge_graph', 'indexer', 'tests']
    for dir_name in dirs_to_check:
        dir_path = project_root / dir_name
        if dir_path.exists():
            tests.append((f"Directory {dir_name}", "exists", "PASS"))
        else:
            tests.append((f"Directory {dir_name}", "missing", "FAIL"))
    
    # Test key files
    files_to_check = [
        'backend/hybrid_search.py',
        'backend/search/query_classifier.py',
        'storage/database_setup.py',
        'knowledge_graph/symbol_collector.py'
    ]
    for file_path in files_to_check:
        full_path = project_root / file_path
        if full_path.exists():
            tests.append((f"File {file_path}", "exists", "PASS"))
        else:
            tests.append((f"File {file_path}", "missing", "FAIL"))
    
    # Print results
    for test_name, result, status in tests:
        status_symbol = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"  {status_symbol} {test_name}: {result}")
    
    return all(status == "PASS" for _, _, status in tests)

def test_imports():
    """Test that all components can be imported."""
    print_section("Testing Component Imports")
    
    import_tests = [
        # Phase 2 Components
        ("Database Setup", "from storage.database_setup import DatabaseSetup"),
        ("Database Manager", "from storage.database_manager import DatabaseManager"),
        ("Symbol Collector", "from knowledge_graph.symbol_collector import SymbolCollector"),
        ("Relationship Extractor", "from knowledge_graph.relationship_extractor import RelationshipExtractor"),
        ("KG Aware RAG", "from knowledge_graph.kg_aware_rag import KGAwareRAG"),
        ("Enhanced Vector Store", "from indexer.enhanced_vector_store import EnhancedVectorStore"),
        
        # Phase 2.3 Components
        ("KG Query Methods", "from backend.kg_query_methods import kg_find_symbol"),
        ("KG Enhanced Search", "from backend.kg_enhanced_smart_search import KGEnhancedSmartSearchEngine"),
        ("KG Analyze Relationships", "from backend.kg_enhanced_analyze_relationships import enhanced_analyze_relationships"),
        
        # Phase 3.1 Components
        ("Query Classifier", "from backend.search.query_classifier import QueryClassifier"),
        ("Enhanced Hybrid Search", "from backend.hybrid_search import HybridSearchEngine"),
    ]
    
    results = []
    for test_name, import_statement in import_tests:
        try:
            exec(import_statement)
            results.append((test_name, "imported successfully", "PASS"))
            print(f"  [PASS] {test_name}: imported successfully")
        except ImportError as e:
            results.append((test_name, f"import failed: {e}", "FAIL"))
            print(f"  [FAIL] {test_name}: import failed - {e}")
        except Exception as e:
            results.append((test_name, f"error: {e}", "FAIL"))
            print(f"  [FAIL] {test_name}: error - {e}")
    
    return all(status == "PASS" for _, _, status in results)

def test_phase2_functionality():
    """Test Phase 2 functionality."""
    print_section("Testing Phase 2 Functionality")
    
    try:
        # Test database setup
        print("  Testing database setup...")
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            temp_db = tmp.name
        
        try:
            from storage.database_setup import DatabaseSetup
            from storage.database_manager import DatabaseManager
            
            # Initialize database
            db_setup = DatabaseSetup(temp_db)
            success = db_setup.initialize_database()
            
            if success:
                print("    [PASS] Database initialization: successful")
            else:
                print("    [FAIL] Database initialization: failed")
                return False
            
            # Test database operations
            db_manager = DatabaseManager(temp_db)
            
            # Insert test node
            success = db_manager.insert_node(
                node_id='test_node',
                node_type='function', 
                name='test_function',
                file_path='/test/file.py'
            )
            
            if success:
                print("    [PASS] Node insertion: successful")
            else:
                print("    [FAIL] Node insertion: failed")
                return False
            
            # Query test node
            nodes = db_manager.get_nodes_by_name('test_function')
            if len(nodes) > 0:
                print("    [PASS] Node query: successful")
            else:
                print("    [FAIL] Node query: failed")
                return False
            
            db_manager.close()
            
        finally:
            try:
                import time
                time.sleep(0.1)  # Brief delay for Windows file handle cleanup
                if os.path.exists(temp_db):
                    os.unlink(temp_db)
            except PermissionError:
                print(f"    [WARNING] Could not delete temporary DB: {temp_db}")
                pass  # Ignore cleanup errors on Windows
        
        # Test KG query methods
        print("  Testing KG query methods...")
        try:
            from backend.kg_query_methods import kg_find_symbol, kg_explore_neighborhood
            
            # These should handle missing database gracefully
            result1 = kg_find_symbol('test_symbol', True, 'nonexistent.db')
            result2 = kg_explore_neighborhood('test_symbol', 2, 'nonexistent.db')
            
            if 'error' in result1.lower() or 'not found' in result1.lower():
                print("    [PASS] KG query methods: handle missing DB gracefully")
            else:
                print("    [FAIL] KG query methods: unexpected behavior with missing DB")
                return False
                
        except Exception as e:
            print(f"    [FAIL] KG query methods: exception - {e}")
            return False
        
        print("  [PASS] Phase 2 functionality tests passed")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Phase 2 functionality tests failed: {e}")
        traceback.print_exc()
        return False

def test_phase3_functionality():
    """Test Phase 3 functionality."""
    print_section("Testing Phase 3 Functionality")
    
    try:
        # Test QueryClassifier
        print("  Testing QueryClassifier...")
        
        from backend.search.query_classifier import QueryClassifier, QueryIntent
        
        classifier = QueryClassifier()
        
        # Test different query types
        test_queries = [
            ("authenticate_user()", "specific symbol"),
            ("how does authentication work", "conceptual query"),
            ("find all callers", "exploratory query")
        ]
        
        for query, description in test_queries:
            analysis = classifier.classify_query(query)
            
            # Verify analysis structure
            if not hasattr(analysis, 'intent'):
                print(f"    [FAIL] Query analysis missing intent for: {query}")
                return False
            
            if not hasattr(analysis, 'vector_weight') or not hasattr(analysis, 'bm25_weight'):
                print(f"    [FAIL] Query analysis missing weights for: {query}")
                return False
            
            # Verify weights sum to 1
            weight_sum = analysis.vector_weight + analysis.bm25_weight
            if abs(weight_sum - 1.0) > 0.01:
                print(f"    [FAIL] Weights don't sum to 1.0 for: {query} (sum: {weight_sum})")
                return False
        
        print("    [PASS] QueryClassifier: working correctly")
        
        # Test HybridSearchEngine integration
        print("  Testing HybridSearchEngine integration...")
        
        from backend.hybrid_search import HybridSearchEngine, QUERY_CLASSIFIER_AVAILABLE
        
        if not QUERY_CLASSIFIER_AVAILABLE:
            print("    [FAIL] QueryClassifier not available in HybridSearchEngine")
            return False
        
        search_engine = HybridSearchEngine()
        
        if not hasattr(search_engine, 'query_classifier'):
            print("    [FAIL] HybridSearchEngine missing query_classifier attribute")
            return False
        
        if search_engine.query_classifier is None:
            print("    [FAIL] QueryClassifier not initialized in HybridSearchEngine")
            return False
        
        print("    [PASS] HybridSearchEngine integration: working correctly")
        
        # Test statistics tracking
        print("  Testing statistics tracking...")
        
        stats = search_engine.get_search_statistics()
        
        if 'dynamic_weighting' not in stats:
            print("    [FAIL] Statistics missing dynamic_weighting section")
            return False
        
        dynamic_stats = stats['dynamic_weighting']
        
        if 'classifier_enabled' not in dynamic_stats:
            print("    [FAIL] Statistics missing classifier_enabled field")
            return False
        
        print("    [PASS] Statistics tracking: working correctly")
        
        print("  [PASS] Phase 3 functionality tests passed")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Phase 3 functionality tests failed: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """Test end-to-end integration."""
    print_section("Testing End-to-End Integration")
    
    try:
        print("  Testing complete pipeline...")
        
        from backend.hybrid_search import HybridSearchEngine
        
        search_engine = HybridSearchEngine()
        
        if search_engine.query_classifier:
            # Test weight adjustment for different query types
            specific_query = "authenticate_user()"
            conceptual_query = "how does authentication work"
            
            # Get analyses
            specific_analysis = search_engine.query_classifier.classify_query(specific_query)
            conceptual_analysis = search_engine.query_classifier.classify_query(conceptual_query)
            
            # Test weight update
            search_engine.result_fusion.update_weights(
                specific_analysis.vector_weight, specific_analysis.bm25_weight
            )
            
            if (search_engine.result_fusion.vector_weight == specific_analysis.vector_weight and
                search_engine.result_fusion.bm25_weight == specific_analysis.bm25_weight):
                print("    [PASS] Weight update mechanism: working correctly")
            else:
                print("    [FAIL] Weight update mechanism: failed")
                return False
            
            # Test that different query types get different weights
            if specific_analysis.bm25_weight != conceptual_analysis.bm25_weight:
                print("    [PASS] Dynamic weighting: different queries get different weights")
            else:
                print("    [FAIL] Dynamic weighting: queries getting same weights")
                return False
        
        print("  [PASS] End-to-end integration tests passed")
        return True
        
    except Exception as e:
        print(f"  [FAIL] End-to-end integration tests failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print_banner("CodeWise Comprehensive Test Suite")
    print(f"Testing in directory: {project_root}")
    print(f"Python version: {sys.version}")
    
    start_time = time.time()
    
    # Run test suites
    test_results = [
        ("Environment Setup", test_environment_setup),
        ("Component Imports", test_imports),
        ("Phase 2 Functionality", test_phase2_functionality),
        ("Phase 3 Functionality", test_phase3_functionality),
        ("End-to-End Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in test_results:
        try:
            success = test_func()
            results.append((test_name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\n[FAIL] {test_name} failed with exception: {e}")
            traceback.print_exc()
            results.append((test_name, "FAIL"))
    
    # Print summary
    print_banner("Test Results Summary")
    
    passed = 0
    total = len(results)
    
    for test_name, status in results:
        status_symbol = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"  {status_symbol} {test_name}: {status}")
        if status == "PASS":
            passed += 1
    
    elapsed_time = time.time() - start_time
    
    print(f"\nResults: {passed}/{total} test suites passed")
    print(f"Time: {elapsed_time:.2f} seconds")
    
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED! CodeWise is ready for deployment.")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test suite(s) failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)