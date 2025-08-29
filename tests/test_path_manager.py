"""
Unit Tests for PathManager Service

Tests all path normalization scenarios to ensure filesystem navigator compatibility.
REQ-PATH-001.1: Centralized Path Management Service testing
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import sys
import os

# Add parent directory to path to import storage modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.path_manager import PathManager, get_path_manager, normalize_path, get_project_from_path


class TestPathManager:
    """Test suite for PathManager class."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def path_manager(self, temp_workspace):
        """Create PathManager instance with temporary workspace."""
        return PathManager(workspace_root=temp_workspace)
    
    def test_init_default_workspace(self):
        """Test PathManager initialization with default workspace."""
        manager = PathManager()
        assert str(manager.workspace_root).endswith('workspace')
    
    def test_init_custom_workspace(self, temp_workspace):
        """Test PathManager initialization with custom workspace."""
        manager = PathManager(workspace_root=temp_workspace)
        assert manager.workspace_root == Path(temp_workspace).resolve()
    
    def test_normalize_absolute_workspace_path(self, path_manager, temp_workspace):
        """Test normalization of absolute path under workspace."""
        # Setup
        test_path = Path(temp_workspace) / "iiot-monitoring" / "frontend" / "src" / "App.js"
        
        # Execute
        result = path_manager.normalize_for_storage(test_path)
        
        # Validate
        assert result == "iiot-monitoring/frontend/src/App.js"
        assert "/" in result  # Ensure forward slashes
        assert not result.startswith("/")  # No leading slash
    
    def test_normalize_absolute_path_outside_workspace(self, path_manager):
        """Test normalization of absolute path outside workspace."""
        # Setup - path outside workspace
        external_path = "/some/other/directory/file.py"
        
        # Execute
        result = path_manager.normalize_for_storage(external_path, project_name="test-project")
        
        # Validate - should add project prefix
        assert result == "test-project/file.py"
    
    def test_normalize_relative_path_with_project(self, path_manager):
        """Test normalization of relative path with project context."""
        # Setup
        relative_path = "src/components/Button.tsx"
        
        # Execute
        result = path_manager.normalize_for_storage(relative_path, project_name="frontend-app")
        
        # Validate
        assert result == "frontend-app/src/components/Button.tsx"
    
    def test_normalize_windows_path_separators(self, path_manager, temp_workspace):
        """Test normalization converts Windows backslashes to forward slashes."""
        # Setup - Windows-style path
        if os.name == 'nt':
            test_path = Path(temp_workspace) / "project" / "src" / "file.js"
            windows_path = str(test_path).replace("/", "\\")
        else:
            # Simulate Windows path on Unix systems
            windows_path = str(Path(temp_workspace) / "project" / "src" / "file.js").replace("/", "\\")
        
        # Execute
        result = path_manager.normalize_for_storage(windows_path)
        
        # Validate - should use forward slashes
        assert "/" in result
        assert "\\" not in result
    
    def test_normalize_file_at_workspace_root(self, path_manager, temp_workspace):
        """Test normalization of file at workspace root."""
        # Setup
        root_file = Path(temp_workspace) / "README.md"
        
        # Execute
        result = path_manager.normalize_for_storage(root_file, project_name="root-project")
        
        # Validate
        assert result == "root-project/README.md"
    
    def test_extract_project_name_valid_path(self, path_manager):
        """Test project name extraction from valid normalized path."""
        # Setup
        test_path = "iiot-monitoring/backend/src/main.py"
        
        # Execute
        result = path_manager.extract_project_name(test_path)
        
        # Validate
        assert result == "iiot-monitoring"
    
    def test_extract_project_name_single_file(self, path_manager):
        """Test project name extraction from single file path."""
        # Setup
        test_path = "standalone-file.py"
        
        # Execute
        result = path_manager.extract_project_name(test_path)
        
        # Validate
        assert result == "standalone-file.py"
    
    def test_extract_project_name_empty_path(self, path_manager):
        """Test project name extraction from empty path."""
        # Execute
        result = path_manager.extract_project_name("")
        
        # Validate
        assert result is None
    
    def test_build_search_patterns(self, path_manager):
        """Test search pattern building for filesystem navigator."""
        # Execute
        relative_pattern, absolute_pattern = path_manager.build_search_patterns("test-project")
        
        # Validate
        assert relative_pattern == "test-project/%"
        assert "test-project" in absolute_pattern
        assert absolute_pattern.endswith("test-project/%")
    
    def test_normalize_edge_case_empty_string(self, path_manager):
        """Test normalization of empty string."""
        # Execute
        result = path_manager.normalize_for_storage("", project_name="fallback-project")
        
        # Validate
        assert result == "fallback-project/"
    
    def test_normalize_edge_case_dot_path(self, path_manager):
        """Test normalization of current directory path."""
        # Execute
        result = path_manager.normalize_for_storage(".", project_name="current-project")
        
        # Validate
        assert "current-project" in result
    
    def test_validate_path_consistency_placeholder(self, path_manager):
        """Test path consistency validation method."""
        # Execute
        result = path_manager.validate_path_consistency("test-project")
        
        # Validate - placeholder implementation
        assert result["project"] == "test-project"
        assert "consistent" in result
        assert "issues" in result


class TestGlobalFunctions:
    """Test suite for global convenience functions."""
    
    @patch('storage.path_manager._path_manager', None)
    def test_get_path_manager_singleton(self):
        """Test singleton pattern for global path manager."""
        # Execute - get manager twice
        manager1 = get_path_manager()
        manager2 = get_path_manager()
        
        # Validate - same instance
        assert manager1 is manager2
        assert isinstance(manager1, PathManager)
    
    def test_normalize_path_convenience_function(self):
        """Test global normalize_path convenience function."""
        # Execute
        result = normalize_path("/some/path/file.py", project_name="test")
        
        # Validate
        assert isinstance(result, str)
        assert "test" in result or "file.py" in result
    
    def test_get_project_from_path_convenience_function(self):
        """Test global get_project_from_path convenience function."""
        # Execute
        result = get_project_from_path("project-name/src/file.js")
        
        # Validate
        assert result == "project-name"


class TestPathManagerIntegration:
    """Integration tests for PathManager with real filesystem scenarios."""
    
    @pytest.fixture
    def mock_workspace(self, tmp_path):
        """Create mock workspace with project structure."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        
        # Create project structure
        project_dir = workspace / "test-project"
        project_dir.mkdir()
        
        src_dir = project_dir / "src"
        src_dir.mkdir()
        
        # Create test file
        test_file = src_dir / "app.py"
        test_file.write_text("# Test file content")
        
        return workspace
    
    def test_real_filesystem_normalization(self, mock_workspace):
        """Test normalization with real filesystem paths."""
        # Setup
        manager = PathManager(workspace_root=str(mock_workspace))
        test_file = mock_workspace / "test-project" / "src" / "app.py"
        
        # Execute
        result = manager.normalize_for_storage(test_file)
        
        # Validate
        assert result == "test-project/src/app.py"
        assert test_file.exists()  # Verify file actually exists
    
    def test_cross_platform_path_handling(self, mock_workspace):
        """Test cross-platform path handling."""
        # Setup
        manager = PathManager(workspace_root=str(mock_workspace))
        
        # Test both Unix and Windows style paths
        unix_path = "test-project/src/component.js"
        windows_path = "test-project\\src\\component.js"
        
        # Execute
        unix_result = manager.normalize_for_storage(unix_path)
        windows_result = manager.normalize_for_storage(windows_path)
        
        # Validate - both should normalize to same format
        assert "/" in unix_result
        assert "/" in windows_result
        assert "\\" not in unix_result
        assert "\\" not in windows_result


# Test performance with large path operations
class TestPathManagerPerformance:
    """Performance tests for PathManager operations."""
    
    def test_normalize_performance_many_paths(self):
        """Test normalization performance with many paths."""
        import time
        
        # Setup
        manager = PathManager()
        test_paths = [f"/workspace/project-{i}/src/file-{j}.py" 
                     for i in range(10) for j in range(100)]
        
        # Execute
        start_time = time.time()
        results = [manager.normalize_for_storage(path) for path in test_paths]
        end_time = time.time()
        
        # Validate
        assert len(results) == len(test_paths)
        assert all(isinstance(result, str) for result in results)
        
        # Performance check - should complete in reasonable time
        elapsed = end_time - start_time
        assert elapsed < 5.0, f"Normalization too slow: {elapsed}s for {len(test_paths)} paths"


# Integration with logging
class TestPathManagerLogging:
    """Test logging functionality of PathManager."""
    
    def test_initialization_logging(self, caplog):
        """Test that PathManager logs initialization properly."""
        import logging
        
        # Setup logging capture
        caplog.set_level(logging.INFO)
        
        # Execute
        PathManager("/test/workspace")
        
        # Validate
        assert "PathManager initialized" in caplog.text
        assert "test" in caplog.text and "workspace" in caplog.text  # Cross-platform compatible
    
    def test_normalization_debug_logging(self, caplog):
        """Test debug logging during normalization."""
        import logging
        
        # Setup
        caplog.set_level(logging.DEBUG)
        manager = PathManager()
        
        # Execute
        manager.normalize_for_storage("/workspace/test/file.py")
        
        # Note: Debug logging currently not implemented in normalize_for_storage
        # This test documents expected behavior for future enhancement


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])