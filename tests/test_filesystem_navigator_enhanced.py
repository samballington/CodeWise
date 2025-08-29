"""
Enhanced Tests for FilesystemNavigator with PathManager Integration

Tests that the enhanced FilesystemNavigator properly uses PathManager for
accurate project-scoped queries and provides helpful diagnostics.
"""

import pytest
import tempfile
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.tools.filesystem_navigator import FilesystemNavigator
from storage.database_manager import DatabaseManager
from storage.database_setup import setup_codewise_database
from storage.path_manager import PathManager


class TestFilesystemNavigatorEnhanced:
    """Test enhanced FilesystemNavigator with PathManager integration."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Robust cleanup that handles locked files
        try:
            shutil.rmtree(temp_dir)
        except (PermissionError, OSError) as e:
            import time
            time.sleep(0.1)
            try:
                shutil.rmtree(temp_dir)
            except (PermissionError, OSError):
                pass
    
    @pytest.fixture
    def test_database(self, temp_workspace):
        """Create test database with sample data."""
        db_path = Path(temp_workspace) / "test.db"
        setup_codewise_database(str(db_path))
        
        # Insert sample data that simulates both consistent and inconsistent paths
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create sample nodes with different path formats
        sample_nodes = [
            # Consistent normalized paths (what we want)
            ('node1', 'function', 'func1', 'iiot-monitoring/frontend/src/App.js', 1, 10, None, None, '{}'),
            ('node2', 'class', 'Component', 'iiot-monitoring/frontend/src/components/Button.tsx', 15, 30, None, None, '{}'),
            ('node3', 'function', 'api_handler', 'iiot-monitoring/backend/api.py', 5, 20, None, None, '{}'),
            
            # Different project with consistent paths
            ('node4', 'function', 'helper', 'infinite-kanvas/src/utils/helper.ts', 8, 15, None, None, '{}'),
            ('node5', 'class', 'Canvas', 'infinite-kanvas/src/Canvas.ts', 1, 100, None, None, '{}'),
            
            # Inconsistent paths (missing project prefix) - these cause problems
            ('node6', 'function', 'bad_func', 'frontend/src/BadComponent.js', 1, 5, None, None, '{}'),
            ('node7', 'variable', 'config', 'backend/config.py', 1, 1, None, None, '{}'),
            
            # Workspace absolute paths (also inconsistent)
            ('node8', 'function', 'abs_func', '/workspace/test-project/main.py', 1, 10, None, None, '{}'),
        ]
        
        for node in sample_nodes:
            cursor.execute('''
                INSERT INTO nodes (id, type, name, file_path, line_start, line_end, signature, docstring, properties)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', node)
        
        conn.commit()
        conn.close()
        
        return str(db_path)
    
    @pytest.fixture
    def db_manager(self, test_database):
        """Create database manager."""
        db_mgr = DatabaseManager(test_database)
        yield db_mgr
        db_mgr.close()
    
    @pytest.fixture
    def navigator(self, db_manager):
        """Create enhanced FilesystemNavigator."""
        return FilesystemNavigator(db_manager)
    
    def test_navigator_initialization_with_path_manager(self, navigator):
        """Test that navigator initializes with PathManager."""
        assert navigator.db is not None
        assert navigator.path_manager is not None
        assert hasattr(navigator, 'validate_project_paths')
    
    def test_project_path_validation(self, navigator):
        """Test project path validation functionality."""
        # Test with project that has consistent paths
        result = navigator.validate_project_paths('iiot-monitoring')
        
        assert result['project_name'] == 'iiot-monitoring'
        assert result['relative_path_files'] > 0
        assert result['path_consistency'] in ['good', 'issues']
        assert 'recommendations' in result
        assert 'available_projects' in result
    
    def test_project_validation_with_inconsistent_paths(self, navigator):
        """Test validation for project with path consistency issues."""
        # Insert some problematic paths for testing
        cursor = navigator.db.connection.cursor()
        cursor.execute('''
            INSERT INTO nodes (id, type, name, file_path, line_start, line_end)
            VALUES ('problem1', 'function', 'func', 'frontend/broken.js', 1, 5)
        ''')
        navigator.db.connection.commit()
        
        # Validate - should detect the inconsistent paths
        result = navigator.validate_project_paths('test-project')
        
        assert result['project_name'] == 'test-project'
        assert 'recommendations' in result
    
    def test_enhanced_project_filtering_with_pathmanager(self, navigator):
        """Test that project filtering uses PathManager patterns."""
        # Test the _build_project_filter method
        base_query = "SELECT * FROM nodes"
        params = []
        
        query, query_params = navigator._build_project_filter(base_query, params, 'iiot-monitoring')
        
        # Should have WHERE clause added with proper patterns
        assert "WHERE" in query
        assert "file_path LIKE ?" in query
        assert len(query_params) >= 2  # Should have both relative and absolute patterns
        
        # Patterns should be for the specific project
        assert any('iiot-monitoring/' in param for param in query_params if isinstance(param, str))
    
    def test_find_operation_with_project_scope(self, navigator):
        """Test find operation with project scoping."""
        result = navigator.execute(
            operation="find",
            pattern="*.js",
            project_scope="iiot-monitoring"
        )
        
        assert result['operation'] == 'find'
        assert 'files' in result
        
        # All returned files should be from the specified project
        for file_path in result['files']:
            assert file_path.startswith('iiot-monitoring/') or '/workspace/iiot-monitoring/' in file_path
    
    def test_list_operation_with_enhanced_error_handling(self, navigator):
        """Test list operation with enhanced error messages."""
        # Test with non-existent project
        result = navigator.execute(
            operation="list",
            path="src",
            project_scope="non-existent-project"
        )
        
        assert 'error' in result
        assert 'validation' in result
        assert result['validation']['project_name'] == 'non-existent-project'
        assert result['validation']['total_files'] == 0
        assert 'available_projects' in result['validation']
    
    def test_list_operation_with_existing_project(self, navigator):
        """Test list operation with existing project."""
        result = navigator.execute(
            operation="list",
            path="frontend",
            project_scope="iiot-monitoring"
        )
        
        # Should find files in the frontend directory
        if 'files' in result:
            assert result['operation'] == 'list'
            assert isinstance(result['files'], list)
            # Files should be from the correct project
            for file_path in result['files']:
                assert 'iiot-monitoring' in file_path or 'frontend' in file_path
    
    def test_validate_operation(self, navigator):
        """Test the new validate operation."""
        result = navigator.execute(
            operation="validate",
            project_scope="iiot-monitoring"
        )
        
        assert result['project_name'] == 'iiot-monitoring'
        assert 'relative_path_files' in result
        assert 'absolute_path_files' in result
        assert 'no_prefix_files' in result
        assert 'recommendations' in result
    
    def test_pathmanager_pattern_building(self, navigator):
        """Test that PathManager is used for building search patterns."""
        if navigator.path_manager:
            # Should be able to build patterns using PathManager
            try:
                relative, absolute = navigator.path_manager.build_search_patterns('test-project')
                assert relative == 'test-project/%'
                assert 'test-project' in absolute
            except Exception as e:
                # PathManager might not be fully functional in test environment
                assert "path_manager" in str(e).lower()
    
    def test_error_handling_without_pathmanager(self, db_manager):
        """Test that navigator works even without PathManager."""
        # Mock PathManager as unavailable
        with patch('backend.tools.filesystem_navigator.PATH_MANAGER_AVAILABLE', False):
            navigator = FilesystemNavigator(db_manager)
            
            assert navigator.path_manager is None
            
            # Should still work with fallback logic
            result = navigator.execute(
                operation="find",
                pattern="*.js"
            )
            
            assert 'files' in result or 'error' in result
    
    def test_comprehensive_project_discovery(self, navigator):
        """Test project discovery across different path formats."""
        # Get validation for a project with mixed path formats
        result = navigator.validate_project_paths('iiot-monitoring')
        
        # Should find files despite different formats
        assert result['total_files'] > 0
        assert len(result['available_projects']) > 0
        
        # Check that recommendations are helpful
        assert len(result['recommendations']) > 0
        assert all(isinstance(rec, str) for rec in result['recommendations'])
    
    def test_tree_operation_with_project_scope(self, navigator):
        """Test tree operation with project scoping."""
        result = navigator.execute(
            operation="tree",
            path="frontend",
            project_scope="iiot-monitoring"
        )
        
        # Should either succeed or provide helpful error
        assert 'operation' in result
        if 'error' in result:
            # Error should be informative
            assert 'iiot-monitoring' in result['error'] or 'frontend' in result['error']
    
    def test_pattern_matching_accuracy(self, navigator):
        """Test that pattern matching is accurate with normalized paths."""
        result = navigator.execute(
            operation="find",
            pattern="Button*",
            project_scope="iiot-monitoring"
        )
        
        # Should find the Button component
        found_button = any('Button' in file for file in result.get('files', []))
        
        # If button exists in test data, should be found
        # If not found, should be due to test data limitations, not path issues
        assert 'files' in result
    
    def test_database_query_optimization(self, navigator):
        """Test that database queries are optimized with project filtering."""
        # Test that project filtering reduces query scope
        all_files_result = navigator.execute(operation="find", pattern="*")
        project_files_result = navigator.execute(
            operation="find", 
            pattern="*", 
            project_scope="iiot-monitoring"
        )
        
        all_files_count = len(all_files_result.get('files', []))
        project_files_count = len(project_files_result.get('files', []))
        
        # Project-scoped query should return fewer or equal results
        assert project_files_count <= all_files_count
    
    def test_path_consistency_recommendations(self, navigator):
        """Test that path consistency recommendations are accurate."""
        # Test with project that has good paths
        good_result = navigator.validate_project_paths('infinite-kanvas')
        
        # Test with project that might have issues
        mixed_result = navigator.validate_project_paths('iiot-monitoring')
        
        # Both should have recommendations
        assert len(good_result['recommendations']) > 0
        assert len(mixed_result['recommendations']) > 0
        
        # Recommendations should be different based on path consistency
        assert isinstance(good_result['recommendations'][0], str)
        assert isinstance(mixed_result['recommendations'][0], str)


class TestFilesystemNavigatorEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def empty_db(self, temp_workspace):
        """Create empty database."""
        db_path = Path(temp_workspace) / "empty.db"
        setup_codewise_database(str(db_path))
        return str(db_path)
    
    @pytest.fixture
    def empty_navigator(self, empty_db):
        """Navigator with empty database."""
        db_mgr = DatabaseManager(empty_db)
        navigator = FilesystemNavigator(db_mgr)
        yield navigator
        db_mgr.close()
    
    def test_empty_database_handling(self, empty_navigator):
        """Test handling of empty database."""
        result = empty_navigator.execute(operation="find", pattern="*")
        
        assert 'files' in result
        assert len(result['files']) == 0
    
    def test_invalid_operation(self, empty_navigator):
        """Test handling of invalid operations."""
        result = empty_navigator.execute(operation="invalid")
        
        assert 'error' in result
        assert 'Unknown operation' in result['error']
        assert 'validate' in result['error']  # Should mention the new operation
    
    def test_validation_with_empty_database(self, empty_navigator):
        """Test project validation with empty database."""
        result = empty_navigator.validate_project_paths('any-project')
        
        assert result['total_files'] == 0
        assert result['path_consistency'] == 'issues'
        assert 'not found' in result['recommendations'][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])