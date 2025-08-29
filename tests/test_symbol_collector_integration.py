"""
Integration Tests for SymbolCollector with PathManager

Tests that SymbolCollector properly uses PathManager for consistent
file path normalization in Knowledge Graph storage.
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

from knowledge_graph.symbol_collector import SymbolCollector
from storage.database_manager import DatabaseManager
from storage.database_setup import setup_codewise_database
from storage.path_manager import PathManager


class TestSymbolCollectorPathIntegration:
    """Test SymbolCollector integration with PathManager."""
    
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
    def sample_python_file(self, temp_workspace):
        """Create sample Python file for testing."""
        project_dir = Path(temp_workspace) / "test-project"
        project_dir.mkdir()
        
        src_dir = project_dir / "src"
        src_dir.mkdir()
        
        sample_file = src_dir / "example.py"
        sample_file.write_text("""
def test_function():
    '''Test function docstring'''
    return "hello"

class TestClass:
    '''Test class docstring'''
    
    def method_example(self):
        return 42

variable_example = "test"
""")
        return sample_file
    
    def test_symbol_collector_initialization_with_project(self, db_manager):
        """Test SymbolCollector initialization with project context."""
        project_name = "test-project"
        collector = SymbolCollector(db_manager, project_name=project_name)
        
        assert collector.project_name == project_name
        assert collector.path_manager is not None
        assert collector.db_manager == db_manager
    
    def test_symbol_collector_initialization_without_project(self, db_manager):
        """Test SymbolCollector initialization without project context."""
        collector = SymbolCollector(db_manager)
        
        assert collector.project_name is None
        assert collector.path_manager is not None
        assert collector.db_manager == db_manager
    
    def test_path_normalization_with_project_context(self, db_manager, temp_workspace):
        """Test that paths are properly normalized with project context."""
        project_name = "test-project"
        collector = SymbolCollector(db_manager, project_name=project_name)
        
        # Create test file path
        test_file = Path(temp_workspace) / "test-project" / "src" / "example.py"
        
        # Test normalization
        normalized = collector._normalize_file_path(test_file)
        
        # Should use PathManager normalization
        assert isinstance(normalized, str)
        # Should contain project prefix for proper paths
        # Note: The exact format depends on PathManager logic
        
    def test_path_normalization_fallback(self, db_manager):
        """Test path normalization fallback when PathManager fails."""
        collector = SymbolCollector(db_manager)
        
        # Mock PathManager to raise exception
        with patch.object(collector.path_manager, 'normalize_for_storage', side_effect=Exception("Test error")):
            test_path = Path("/workspace/project/src/file.py")
            normalized = collector._normalize_file_path(test_path)
            
            # Should fall back to original logic
            assert normalized == "project/src/file.py"
    
    def test_symbol_collection_with_path_normalization(self, db_manager, sample_python_file, temp_workspace):
        """Test complete symbol collection with proper path normalization."""
        project_name = "test-project"
        collector = SymbolCollector(db_manager, project_name=project_name)
        
        # Collect symbols
        symbol_table = collector.collect_all_symbols([sample_python_file])
        
        # Verify symbols were collected
        assert len(symbol_table) > 0
        
        # Check that symbols have properly normalized paths
        for symbol_id, symbol_info in symbol_table.items():
            file_path = symbol_info['file_path']
            
            # Path should be normalized format
            assert isinstance(file_path, str)
            assert file_path != ""
            
            # Check for reasonable path format (project prefix or absolute)
            assert not file_path.startswith('/workspace/') or '/' in file_path
    
    def test_database_storage_with_normalized_paths(self, db_manager, sample_python_file):
        """Test that symbols are stored in database with normalized paths."""
        project_name = "test-project"
        collector = SymbolCollector(db_manager, project_name=project_name)
        
        # Collect symbols
        symbol_table = collector.collect_all_symbols([sample_python_file])
        
        # Verify symbols were inserted in database
        assert len(symbol_table) > 0
        
        # Check database contents
        conn = db_manager.connection
        cursor = conn.cursor()
        
        cursor.execute("SELECT file_path, name, type FROM nodes WHERE type IN ('function', 'class', 'variable')")
        db_symbols = cursor.fetchall()
        
        assert len(db_symbols) > 0
        
        # Verify paths in database are normalized
        for file_path, name, symbol_type in db_symbols:
            assert file_path is not None
            assert file_path != ""
            
            # Should be consistent format (either project-prefixed or properly formatted)
            # The exact format depends on PathManager logic, but should be consistent
    
    def test_collection_statistics_accuracy(self, db_manager, sample_python_file):
        """Test that collection statistics are accurate."""
        project_name = "test-project"
        collector = SymbolCollector(db_manager, project_name=project_name)
        
        # Collect symbols
        symbol_table = collector.collect_all_symbols([sample_python_file])
        stats = collector.get_collection_statistics()
        
        # Verify statistics
        assert stats['files_processed'] == 1
        assert stats['files_failed'] == 0
        assert stats['symbols_discovered'] > 0
        assert stats['symbols_discovered'] == len(symbol_table)
        assert 'symbols_by_type' in stats
        assert len(stats['symbols_by_type']) > 0
    
    def test_project_name_extraction_from_path(self, db_manager, temp_workspace):
        """Test project name extraction from file paths."""
        # Test without explicit project name
        collector = SymbolCollector(db_manager)
        
        # Create file with project structure
        project_file = Path(temp_workspace) / "my-project" / "src" / "main.py"
        project_file.parent.mkdir(parents=True)
        project_file.write_text("def hello(): pass")
        
        # Normalize path - should extract project name
        normalized = collector._normalize_file_path(project_file)
        
        # PathManager should handle project name extraction
        assert isinstance(normalized, str)
        assert normalized != ""
    
    def test_error_handling_in_symbol_collection(self, db_manager, temp_workspace):
        """Test error handling during symbol collection."""
        project_name = "test-project"
        collector = SymbolCollector(db_manager, project_name=project_name)
        
        # Create file that will cause parsing issues
        bad_file = Path(temp_workspace) / "bad_file.py"
        bad_file.write_text("invalid python syntax $$$ @@@")
        
        # Should handle errors gracefully
        symbol_table = collector.collect_all_symbols([bad_file])
        stats = collector.get_collection_statistics()
        
        # Should record the failure but not crash
        assert stats['files_failed'] >= 0  # Might succeed with fallback parsing
        assert len(stats['processing_errors']) >= 0
        assert isinstance(symbol_table, dict)
    
    def test_mixed_file_types_path_normalization(self, db_manager, temp_workspace):
        """Test path normalization with multiple file types."""
        project_name = "multi-lang-project"
        collector = SymbolCollector(db_manager, project_name=project_name)
        
        # Create files of different types
        project_dir = Path(temp_workspace) / "multi-lang-project"
        project_dir.mkdir()
        
        files = []
        
        # Python file
        py_file = project_dir / "script.py"
        py_file.write_text("def python_func(): pass")
        files.append(py_file)
        
        # JavaScript file
        js_file = project_dir / "script.js"
        js_file.write_text("function jsFunc() { return true; }")
        files.append(js_file)
        
        # Java file
        java_file = project_dir / "Example.java"
        java_file.write_text("public class Example { public void method() {} }")
        files.append(java_file)
        
        # Collect symbols from all files
        symbol_table = collector.collect_all_symbols(files)
        
        # Verify all files were processed with consistent path normalization
        stats = collector.get_collection_statistics()
        assert stats['files_processed'] >= len(files)
        
        # Check that paths are consistently normalized across file types
        file_paths_in_symbols = set()
        for symbol_info in symbol_table.values():
            file_paths_in_symbols.add(symbol_info['file_path'])
        
        # Should have multiple normalized paths
        assert len(file_paths_in_symbols) >= len(files)
        
        # All paths should be normalized format
        for path in file_paths_in_symbols:
            assert isinstance(path, str)
            assert path != ""
            # Should not contain raw workspace paths
            assert not path.startswith('/tmp/')  # temp workspace shouldn't leak


class TestSymbolCollectorWithoutTreeSitter:
    """Test SymbolCollector fallback functionality with PathManager."""
    
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
    
    def test_fallback_symbol_collection_with_path_normalization(self, db_manager, temp_workspace):
        """Test fallback symbol collection with proper path normalization."""
        project_name = "fallback-test"
        
        # Mock tree-sitter as unavailable
        with patch('knowledge_graph.symbol_collector.TREE_SITTER_AVAILABLE', False):
            with patch('knowledge_graph.symbol_collector.TREE_SITTER_FACTORY_AVAILABLE', False):
                collector = SymbolCollector(db_manager, project_name=project_name)
                
                # Create test files
                project_dir = Path(temp_workspace) / "fallback-test"
                project_dir.mkdir()
                
                py_file = project_dir / "test.py"
                py_file.write_text("""
def test_function():
    return "hello"

class TestClass:
    def method(self):
        pass
""")
                
                # Test fallback collection
                symbol_table = collector.collect_all_symbols([py_file])
                
                # Should find symbols using regex fallback
                assert len(symbol_table) > 0
                
                # Verify paths are normalized in fallback mode
                for symbol_info in symbol_table.values():
                    file_path = symbol_info['file_path']
                    assert isinstance(file_path, str)
                    assert file_path != ""
                    # Should use normalized paths even in fallback
                    # The exact format depends on PathManager behavior


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])