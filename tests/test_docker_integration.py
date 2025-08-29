"""
Docker Integration Tests

Tests to ensure all components work correctly in Docker environment
with proper dependency resolution and environment setup.
"""

import os
import sys
import pytest
import subprocess
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDockerEnvironment:
    """Test Docker environment setup and dependencies"""
    
    def test_python_environment(self):
        """Test that Python environment is properly set up."""
        import sys
        assert sys.version_info >= (3, 8), "Python 3.8+ required"
    
    def test_required_packages_import(self):
        """Test that all required packages can be imported."""
        required_packages = [
            'sqlite3',
            'pathlib',
            'asyncio',
            'json',
            'logging',
            'typing',
            'dataclasses',
            'enum',
            're'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                pytest.fail(f"Required package {package} not available")
    
    def test_sentence_transformers_availability(self):
        """Test that sentence transformers is available for embeddings."""
        try:
            import sentence_transformers
            # Try to load a model
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            assert model is not None, "Should be able to load sentence transformer model"
        except ImportError:
            pytest.skip("sentence-transformers not available - vector search will be limited")
    
    def test_faiss_availability(self):
        """Test that FAISS is available for vector search."""
        try:
            import faiss
            # Test basic FAISS functionality
            index = faiss.IndexFlatL2(384)  # Dimension for MiniLM
            assert index is not None, "Should be able to create FAISS index"
        except ImportError:
            pytest.skip("FAISS not available - vector search will use alternative")


class TestDockerFileStructure:
    """Test that file structure is correct in Docker environment"""
    
    def test_project_structure(self):
        """Test that all expected directories and files exist."""
        project_root = Path(__file__).parent.parent
        
        expected_dirs = [
            'backend',
            'storage',
            'knowledge_graph',
            'indexer',
            'tests'
        ]
        
        for dir_name in expected_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Directory {dir_name} should exist"
            assert dir_path.is_dir(), f"{dir_name} should be a directory"
    
    def test_backend_files(self):
        """Test that backend files exist."""
        backend_dir = Path(__file__).parent.parent / 'backend'
        
        expected_files = [
            'hybrid_search.py',
            'kg_enhanced_smart_search.py',
            'kg_enhanced_analyze_relationships.py',
            'kg_query_methods.py',
            'search/query_classifier.py'
        ]
        
        for file_path in expected_files:
            full_path = backend_dir / file_path
            assert full_path.exists(), f"File {file_path} should exist in backend"
    
    def test_storage_files(self):
        """Test that storage files exist."""
        storage_dir = Path(__file__).parent.parent / 'storage'
        
        expected_files = [
            'database_setup.py',
            'database_manager.py'
        ]
        
        for file_name in expected_files:
            file_path = storage_dir / file_name
            assert file_path.exists(), f"File {file_name} should exist in storage"
    
    def test_knowledge_graph_files(self):
        """Test that knowledge graph files exist."""
        kg_dir = Path(__file__).parent.parent / 'knowledge_graph'
        
        expected_files = [
            'symbol_collector.py',
            'relationship_extractor.py',
            'kg_aware_rag.py',
            'unified_indexer.py'
        ]
        
        for file_name in expected_files:
            file_path = kg_dir / file_name
            assert file_path.exists(), f"File {file_name} should exist in knowledge_graph"


class TestDockerImports:
    """Test that all imports work in Docker environment"""
    
    def test_phase2_imports(self):
        """Test Phase 2 component imports."""
        try:
            # Storage components
            from storage.database_setup import DatabaseSetup
            from storage.database_manager import DatabaseManager
            
            # Knowledge Graph components
            from knowledge_graph.symbol_collector import SymbolCollector
            from knowledge_graph.relationship_extractor import RelationshipExtractor
            from knowledge_graph.kg_aware_rag import KGAwareRAG
            
            # Enhanced search components
            from backend.kg_enhanced_smart_search import KGEnhancedSmartSearchEngine
            from backend.kg_enhanced_analyze_relationships import enhanced_analyze_relationships
            from backend.kg_query_methods import kg_find_symbol, kg_explore_neighborhood
            
            assert True, "All Phase 2 imports successful"
            
        except ImportError as e:
            pytest.fail(f"Phase 2 import failed: {e}")
    
    def test_phase3_imports(self):
        """Test Phase 3 component imports."""
        try:
            # Query classifier
            from backend.search.query_classifier import QueryClassifier, QueryIntent, QueryAnalysis
            
            # Enhanced hybrid search
            from backend.hybrid_search import HybridSearchEngine, QUERY_CLASSIFIER_AVAILABLE
            
            assert True, "All Phase 3 imports successful"
            
        except ImportError as e:
            pytest.fail(f"Phase 3 import failed: {e}")
    
    def test_cross_component_imports(self):
        """Test that components can import each other correctly."""
        try:
            # Test that HybridSearchEngine can import QueryClassifier
            from backend.hybrid_search import HybridSearchEngine
            search_engine = HybridSearchEngine()
            
            # Should have classifier initialized
            if hasattr(search_engine, 'query_classifier'):
                assert search_engine.query_classifier is not None, "QueryClassifier should be initialized"
            
            assert True, "Cross-component imports working"
            
        except Exception as e:
            pytest.fail(f"Cross-component import failed: {e}")


class TestDockerFunctionality:
    """Test that core functionality works in Docker environment"""
    
    def test_database_operations_in_docker(self):
        """Test database operations work in Docker."""
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
            assert success, "Database should initialize in Docker"
            
            # Test database operations
            db_manager = DatabaseManager(temp_db)
            
            success = db_manager.insert_node(
                node_id='docker_test',
                node_type='function',
                name='docker_test_function',
                file_path='/test/docker.py'
            )
            assert success, "Should be able to insert node in Docker"
            
            nodes = db_manager.get_nodes_by_name('docker_test_function')
            assert len(nodes) > 0, "Should be able to query nodes in Docker"
            
            db_manager.close()
            
        finally:
            if os.path.exists(temp_db):
                os.unlink(temp_db)
    
    def test_query_classifier_in_docker(self):
        """Test QueryClassifier works in Docker."""
        try:
            from backend.search.query_classifier import QueryClassifier
            
            classifier = QueryClassifier()
            
            # Test classification
            analysis = classifier.classify_query("test function")
            
            assert hasattr(analysis, 'intent'), "Should have intent"
            assert hasattr(analysis, 'vector_weight'), "Should have vector weight"
            assert hasattr(analysis, 'bm25_weight'), "Should have BM25 weight"
            
            # Weights should be valid
            assert 0 <= analysis.vector_weight <= 1, "Vector weight should be 0-1"
            assert 0 <= analysis.bm25_weight <= 1, "BM25 weight should be 0-1"
            assert abs(analysis.vector_weight + analysis.bm25_weight - 1.0) < 0.01, "Weights should sum to 1"
            
        except Exception as e:
            pytest.fail(f"QueryClassifier failed in Docker: {e}")
    
    def test_hybrid_search_in_docker(self):
        """Test HybridSearchEngine works in Docker."""
        try:
            from backend.hybrid_search import HybridSearchEngine
            
            search_engine = HybridSearchEngine()
            
            # Test that it initializes without errors
            assert search_engine is not None, "HybridSearchEngine should initialize"
            
            # Test statistics
            stats = search_engine.get_search_statistics()
            assert 'dynamic_weighting' in stats, "Should have dynamic weighting stats"
            
        except Exception as e:
            pytest.fail(f"HybridSearchEngine failed in Docker: {e}")


class TestDockerMemoryAndPerformance:
    """Test memory usage and performance in Docker environment"""
    
    def test_memory_usage_reasonable(self):
        """Test that components don't use excessive memory."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load heavy components
        from backend.hybrid_search import HybridSearchEngine
        from backend.search.query_classifier import QueryClassifier
        
        search_engine = HybridSearchEngine()
        classifier = QueryClassifier()
        
        # Check memory usage
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Should not use more than 500MB for basic initialization
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f}MB increase"
    
    def test_import_performance(self):
        """Test that imports are reasonably fast."""
        import time
        
        start_time = time.time()
        
        # Import heavy components
        from backend.hybrid_search import HybridSearchEngine
        from backend.search.query_classifier import QueryClassifier
        from storage.database_manager import DatabaseManager
        
        import_time = time.time() - start_time
        
        # Should import in less than 10 seconds
        assert import_time < 10, f"Imports too slow: {import_time:.2f} seconds"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])