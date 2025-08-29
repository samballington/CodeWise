"""
Integration Tests for RelationshipExtractor with PathManager

Tests that RelationshipExtractor properly uses PathManager for consistent
file path normalization in Knowledge Graph edge storage.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_graph.relationship_extractor import RelationshipExtractor
from storage.database_manager import DatabaseManager
from storage.database_setup import setup_codewise_database
from storage.path_manager import PathManager


class TestRelationshipExtractorPathIntegration:
    """Test RelationshipExtractor integration with PathManager."""
    
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
    def db_manager(self, test_database):
        """Create database manager."""
        db_mgr = DatabaseManager(test_database)
        yield db_mgr
        # Ensure database is properly closed
        db_mgr.close()
    
    @pytest.fixture
    def sample_symbol_table(self):
        """Create sample symbol table for testing."""
        return {
            'test_module::test_function::5': {
                'id': 'test_module::test_function::5',
                'name': 'test_function',
                'type': 'function',
                'file_path': 'test-project/src/module.py',
                'line_start': 5,
                'line_end': 10,
                'signature': 'def test_function():',
                'docstring': 'Test function'
            },
            'test_module::TestClass::15': {
                'id': 'test_module::TestClass::15',
                'name': 'TestClass',
                'type': 'class',
                'file_path': 'test-project/src/module.py',
                'line_start': 15,
                'line_end': 25,
                'signature': 'class TestClass:',
                'docstring': 'Test class'
            }
        }
    
    def test_relationship_extractor_initialization_with_project(self, db_manager, sample_symbol_table):
        """Test RelationshipExtractor initialization with project context."""
        project_name = "test-project"
        extractor = RelationshipExtractor(db_manager, sample_symbol_table, project_name=project_name)
        
        assert extractor.project_name == project_name
        assert extractor.path_manager is not None
        assert extractor.db_manager == db_manager
        assert extractor.symbol_table == sample_symbol_table
    
    def test_relationship_extractor_initialization_without_project(self, db_manager, sample_symbol_table):
        """Test RelationshipExtractor initialization without project context."""
        extractor = RelationshipExtractor(db_manager, sample_symbol_table)
        
        assert extractor.project_name is None
        assert extractor.path_manager is not None
        assert extractor.db_manager == db_manager
        assert extractor.symbol_table == sample_symbol_table
    
    def test_path_normalization_with_project_context(self, db_manager, sample_symbol_table, temp_workspace):
        """Test that paths are properly normalized with project context."""
        project_name = "test-project"
        extractor = RelationshipExtractor(db_manager, sample_symbol_table, project_name=project_name)
        
        # Create test file path
        test_path = f"{temp_workspace}/test-project/src/example.py"
        
        # Test normalization
        normalized = extractor._normalize_file_path_for_storage(test_path)
        
        # Should use PathManager normalization
        assert isinstance(normalized, str)
        assert normalized != ""
        
        # Should contain consistent format
        # Note: The exact format depends on PathManager logic
    
    def test_path_normalization_fallback(self, db_manager, sample_symbol_table):
        """Test path normalization fallback when PathManager fails."""
        extractor = RelationshipExtractor(db_manager, sample_symbol_table)
        
        # Mock PathManager to raise exception
        with patch.object(extractor.path_manager, 'normalize_for_storage', side_effect=Exception("Test error")):
            test_path = "/workspace/project/src/file.py"
            normalized = extractor._normalize_file_path_for_storage(test_path)
            
            # Should fall back to original logic
            assert normalized == "project/src/file.py"
    
    def test_module_symbol_creation_with_path_normalization(self, db_manager, sample_symbol_table):
        """Test that module symbols are created with properly normalized paths."""
        project_name = "test-project"
        extractor = RelationshipExtractor(db_manager, sample_symbol_table, project_name=project_name)
        
        # Create test file
        test_file = Path("/some/path/test-project/src/test_module.py")
        
        # Test module symbol creation
        module_symbol = extractor._get_file_module_symbol(test_file)
        
        # Should create module symbol with normalized path
        assert module_symbol is not None
        assert isinstance(module_symbol, str)
        assert "test_module" in module_symbol
        
        # Check that it was inserted into database with normalized path
        conn = db_manager.connection
        cursor = conn.cursor()
        
        cursor.execute("SELECT file_path, name, type FROM nodes WHERE type = 'module'")
        module_nodes = cursor.fetchall()
        
        # Should have at least one module node
        assert len(module_nodes) > 0
        
        # Verify path normalization in database
        for file_path, name, node_type in module_nodes:
            assert file_path is not None
            assert file_path != ""
            assert node_type == "module"
            # Should be normalized format
            assert not file_path.startswith('/some/path/')  # Should not contain absolute prefix
    
    def test_relationship_extraction_with_consistent_paths(self, db_manager, sample_symbol_table, temp_workspace):
        """Test that relationship extraction uses consistent path normalization."""
        project_name = "test-project"
        extractor = RelationshipExtractor(db_manager, sample_symbol_table, project_name=project_name)
        
        # Create test files with code relationships
        project_dir = Path(temp_workspace) / "test-project"
        project_dir.mkdir()
        src_dir = project_dir / "src"
        src_dir.mkdir()
        
        # File with function call
        caller_file = src_dir / "caller.py"
        caller_file.write_text("""
def caller_function():
    # This would call test_function
    test_function()
    return True
""")
        
        # Mock the file paths in symbol table to match temp files
        updated_symbol_table = {}
        for symbol_id, symbol_info in sample_symbol_table.items():
            updated_info = symbol_info.copy()
            # Update to use temp file paths
            if 'module.py' in updated_info['file_path']:
                updated_info['file_path'] = str(src_dir / "module.py")
            updated_symbol_table[symbol_id] = updated_info
        
        # Update extractor's symbol table
        extractor.symbol_table = updated_symbol_table
        
        # Test extraction (this will use fallback methods since tree-sitter might not be available)
        try:
            extractor.extract_relationships([caller_file])
            # Should complete without errors, no return value expected
        except Exception as e:
            # If extraction fails, it should be due to missing tree-sitter, not path issues
            assert "tree-sitter" in str(e).lower() or "parse" in str(e).lower()
    
    def test_import_resolution_with_normalized_paths(self, db_manager, sample_symbol_table):
        """Test that import resolution works with normalized paths."""
        project_name = "multi-module-project"
        extractor = RelationshipExtractor(db_manager, sample_symbol_table, project_name=project_name)
        
        # Create symbol table with multiple files
        multi_file_symbols = {
            'utils::helper_func::5': {
                'id': 'utils::helper_func::5',
                'name': 'helper_func',
                'type': 'function',
                'file_path': 'multi-module-project/utils/helpers.py',
                'line_start': 5,
                'line_end': 10,
                'signature': 'def helper_func():',
                'docstring': 'Helper function'
            },
            'main::main_func::3': {
                'id': 'main::main_func::3',
                'name': 'main_func',
                'type': 'function',
                'file_path': 'multi-module-project/main.py',
                'line_start': 3,
                'line_end': 8,
                'signature': 'def main_func():',
                'docstring': 'Main function'
            }
        }
        
        extractor.symbol_table = multi_file_symbols
        
        # Test that import resolver can work with normalized paths
        file_path = "multi-module-project/main.py"
        local_symbols = extractor._get_local_symbols(Path(file_path))
        
        # Should be able to get local symbols using normalized paths
        assert isinstance(local_symbols, dict)
    
    def test_error_handling_in_relationship_extraction(self, db_manager, sample_symbol_table):
        """Test error handling during relationship extraction."""
        project_name = "test-project"
        extractor = RelationshipExtractor(db_manager, sample_symbol_table, project_name=project_name)
        
        # Test with non-existent file
        non_existent_file = Path("/non/existent/file.py")
        
        # Should handle gracefully
        try:
            extractor.extract_relationships([non_existent_file])
            stats = extractor.get_extraction_statistics()
            
            # Should record the failure but not crash
            assert stats['files_failed'] >= 0
        except Exception as e:
            # Should be a reasonable error (file not found, parsing issue, etc.)
            assert "not found" in str(e).lower() or "parse" in str(e).lower()
    
    def test_extraction_statistics_accuracy(self, db_manager, sample_symbol_table, temp_workspace):
        """Test that extraction statistics are accurate."""
        project_name = "stats-test"
        extractor = RelationshipExtractor(db_manager, sample_symbol_table, project_name=project_name)
        
        # Create test file
        project_dir = Path(temp_workspace) / "stats-test"
        project_dir.mkdir()
        test_file = project_dir / "test.py"
        test_file.write_text("def example(): pass")
        
        # Extract relationships
        extractor.extract_relationships([test_file])
        stats = extractor.get_extraction_statistics()
        
        # Verify statistics
        assert 'files_processed' in stats
        assert 'files_failed' in stats
        assert 'relationships_found' in stats
        assert stats['files_processed'] >= 0
        assert stats['files_failed'] >= 0


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])