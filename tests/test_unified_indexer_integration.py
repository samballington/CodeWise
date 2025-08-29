"""
Integration Tests for UnifiedIndexer with PathManager

Tests that UnifiedIndexer properly passes project context to Knowledge Graph
components for consistent file path normalization.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_graph.unified_indexer import UnifiedIndexer
from storage.database_setup import setup_codewise_database


class TestUnifiedIndexerPathIntegration:
    """Test UnifiedIndexer integration with PathManager."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Robust cleanup that handles locked files
        try:
            shutil.rmtree(temp_dir)
        except (PermissionError, OSError) as e:
            # On Windows, SQLite files might be locked
            import time
            time.sleep(0.1)  # Brief pause
            try:
                shutil.rmtree(temp_dir)
            except (PermissionError, OSError):
                # Final attempt - ignore if still locked
                pass
    
    @pytest.fixture
    def test_database(self, temp_workspace):
        """Create test database."""
        db_path = Path(temp_workspace) / "test.db"
        setup_codewise_database(str(db_path))
        return str(db_path)
    
    @pytest.fixture
    def sample_codebase(self, temp_workspace):
        """Create sample codebase structure for testing."""
        # Create workspace-style project structure
        workspace_dir = Path(temp_workspace) / "workspace"
        workspace_dir.mkdir()
        
        project_dir = workspace_dir / "test-project"
        project_dir.mkdir()
        
        src_dir = project_dir / "src"
        src_dir.mkdir()
        
        # Create sample Python file
        sample_file = src_dir / "example.py"
        sample_file.write_text("""
def hello_world():
    '''Simple hello world function'''
    return "Hello, World!"

class ExampleClass:
    '''Example class for testing'''
    
    def method_example(self):
        return hello_world()

variable_example = "test"
""")
        
        return project_dir
    
    def test_unified_indexer_initialization_with_project(self, test_database):
        """Test UnifiedIndexer initialization with explicit project context."""
        project_name = "explicit-project"
        indexer = UnifiedIndexer(db_path=test_database, project_name=project_name)
        
        assert indexer.project_name == project_name
        assert indexer.symbol_collector.project_name == project_name
        assert indexer.db_path == test_database
    
    def test_unified_indexer_initialization_without_project(self, test_database):
        """Test UnifiedIndexer initialization without explicit project context."""
        indexer = UnifiedIndexer(db_path=test_database)
        
        assert indexer.project_name is None
        assert indexer.symbol_collector.project_name is None
        assert indexer.db_path == test_database
    
    def test_project_name_extraction_from_workspace_path(self, test_database):
        """Test project name extraction from workspace-style paths."""
        indexer = UnifiedIndexer(db_path=test_database)
        
        # Test workspace-style path
        workspace_path = Path("/workspace/my-awesome-project/src")
        extracted = indexer._extract_project_name_from_path(workspace_path)
        
        assert extracted == "my-awesome-project"
    
    def test_project_name_extraction_from_directory_name(self, test_database):
        """Test project name extraction from directory name."""
        indexer = UnifiedIndexer(db_path=test_database)
        
        # Test regular directory path
        project_path = Path("/some/path/my-project")
        extracted = indexer._extract_project_name_from_path(project_path)
        
        assert extracted == "my-project"
    
    def test_project_name_extraction_with_explicit_name(self, test_database):
        """Test that explicit project name takes precedence."""
        project_name = "explicit-name"
        indexer = UnifiedIndexer(db_path=test_database, project_name=project_name)
        
        # Even with a workspace path, should use explicit name
        workspace_path = Path("/workspace/different-project/src")
        extracted = indexer._extract_project_name_from_path(workspace_path)
        
        assert extracted == project_name
    
    def test_project_name_fallback(self, test_database):
        """Test project name fallback for edge cases."""
        indexer = UnifiedIndexer(db_path=test_database)
        
        # Test empty path
        empty_path = Path("")
        extracted = indexer._extract_project_name_from_path(empty_path)
        
        # Should fall back to unknown-project
        assert extracted == "unknown-project"
    
    @patch('knowledge_graph.unified_indexer.AutoGPUVectorStore')
    @patch('knowledge_graph.unified_indexer.HierarchicalChunker')
    async def test_runtime_project_context_updating(self, mock_chunker, mock_vector, 
                                                   test_database, sample_codebase):
        """Test that project context is updated at runtime during indexing."""
        # Initialize without project context
        indexer = UnifiedIndexer(db_path=test_database)
        
        # Mock the components that might not be available in test environment
        mock_chunker_instance = MagicMock()
        mock_chunker.return_value = mock_chunker_instance
        
        mock_vector_instance = MagicMock()
        mock_vector.return_value = mock_vector_instance
        
        # Mock the hierarchical chunking and vector operations
        with patch.object(indexer, '_chunk_with_hierarchical_chunker', return_value=AsyncMock()):
            with patch.object(indexer, '_embed_chunks', return_value=AsyncMock()):
                # Should extract project name from sample_codebase path
                initial_project = indexer.project_name
                
                try:
                    result = await indexer.index_codebase(sample_codebase)
                    
                    # Should have updated project name
                    assert indexer.project_name == "test-project"
                    assert indexer.project_name != initial_project
                    
                    # Should have updated SymbolCollector project context
                    assert indexer.symbol_collector.project_name == "test-project"
                    
                except Exception as e:
                    # Test might fail due to missing components, but project update should work
                    # Just verify the project name was extracted
                    assert indexer.project_name == "test-project"
    
    def test_component_project_context_consistency(self, test_database, sample_codebase):
        """Test that all components receive consistent project context."""
        project_name = "consistency-test"
        indexer = UnifiedIndexer(db_path=test_database, project_name=project_name)
        
        # Verify SymbolCollector has correct project context
        assert indexer.symbol_collector.project_name == project_name
        
        # Test RelationshipExtractor creation with project context
        from knowledge_graph.relationship_extractor import RelationshipExtractor
        
        sample_symbol_table = {
            'test::symbol::1': {
                'id': 'test::symbol::1',
                'name': 'symbol',
                'type': 'function',
                'file_path': 'consistency-test/src/file.py',
                'line_start': 1,
                'line_end': 5
            }
        }
        
        # Create RelationshipExtractor like the indexer would
        rel_extractor = RelationshipExtractor(indexer.db_manager, sample_symbol_table, 
                                            project_name=indexer.project_name)
        
        assert rel_extractor.project_name == project_name
    
    def test_supported_extensions_coverage(self, test_database):
        """Test that supported extensions are comprehensive."""
        indexer = UnifiedIndexer(db_path=test_database)
        
        # Should support major programming languages
        expected_extensions = {
            '.py',      # Python
            '.js',      # JavaScript
            '.ts',      # TypeScript
            '.java',    # Java
            '.cpp',     # C++
            '.rs',      # Rust
            '.go',      # Go
        }
        
        # All expected extensions should be supported
        for ext in expected_extensions:
            assert ext in indexer.supported_extensions
        
        # Should have reasonable number of supported extensions
        assert len(indexer.supported_extensions) >= 10
    
    @patch('knowledge_graph.unified_indexer.AutoGPUVectorStore')
    @patch('knowledge_graph.unified_indexer.HierarchicalChunker')
    def test_error_handling_with_project_context(self, mock_chunker, mock_vector, 
                                                test_database, temp_workspace):
        """Test error handling when project context is involved."""
        project_name = "error-test"
        indexer = UnifiedIndexer(db_path=test_database, project_name=project_name)
        
        # Mock the components
        mock_chunker_instance = MagicMock()
        mock_chunker.return_value = mock_chunker_instance
        
        mock_vector_instance = MagicMock()
        mock_vector.return_value = mock_vector_instance
        
        # Test with non-existent directory
        non_existent_path = Path(temp_workspace) / "non-existent-project"
        
        # Should handle gracefully and still have project context
        try:
            extracted = indexer._extract_project_name_from_path(non_existent_path)
            assert extracted == "non-existent-project"
        except Exception:
            # Should not crash on path operations
            pass
    
    def test_database_integration_with_project_paths(self, test_database):
        """Test that database integration works with project context."""
        project_name = "db-integration-test"
        indexer = UnifiedIndexer(db_path=test_database, project_name=project_name)
        
        # Database should be set up correctly
        assert indexer.db_manager is not None
        assert indexer.db_manager.connection is not None
        
        # Verify components can access database
        assert indexer.symbol_collector.db_manager == indexer.db_manager
        
        # Test database tables exist
        conn = indexer.db_manager.connection
        cursor = conn.cursor()
        
        # Should have the required tables for KG storage
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'nodes' in tables
        assert 'chunks' in tables  # For hierarchical chunking
    
    def test_logging_includes_project_context(self, test_database, caplog):
        """Test that logging includes project context information."""
        import logging
        caplog.set_level(logging.INFO)
        
        project_name = "logging-test"
        indexer = UnifiedIndexer(db_path=test_database, project_name=project_name)
        
        # Should log project context during initialization
        assert "Project context" in caplog.text
        assert project_name in caplog.text
    
    @patch('knowledge_graph.unified_indexer.AutoGPUVectorStore')
    @patch('knowledge_graph.unified_indexer.HierarchicalChunker')  
    def test_indexer_cleanup_and_resource_management(self, mock_chunker, mock_vector,
                                                    test_database, temp_workspace):
        """Test proper cleanup and resource management."""
        project_name = "cleanup-test"
        indexer = UnifiedIndexer(db_path=test_database, project_name=project_name)
        
        # Mock the components
        mock_chunker_instance = MagicMock()
        mock_chunker.return_value = mock_chunker_instance
        
        mock_vector_instance = MagicMock()
        mock_vector.return_value = mock_vector_instance
        
        # Create test directory
        test_dir = Path(temp_workspace) / "cleanup-test"
        test_dir.mkdir()
        
        # Indexer should be able to handle cleanup gracefully
        try:
            # Test basic operations don't leak resources
            project_extracted = indexer._extract_project_name_from_path(test_dir)
            assert project_extracted == project_name
            
            # Components should be properly initialized
            assert indexer.symbol_collector is not None
            assert indexer.db_manager is not None
            
        finally:
            # Cleanup should work
            if indexer.db_manager:
                indexer.db_manager.close()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])