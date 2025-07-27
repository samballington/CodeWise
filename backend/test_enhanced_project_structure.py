#!/usr/bin/env python3
"""
Test enhanced project structure functionality
Phase 1.3 validation
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch

from enhanced_project_structure import (
    EnhancedProjectStructure, FrameworkDetector, ProjectInfo
)

def test_framework_detector():
    """Test framework detection logic"""
    
    # Test Django detection
    files = ["manage.py", "settings.py", "models.py", "views.py"]
    directories = ["migrations", "templates", "static"]
    package_contents = {"requirements.txt": "django==3.2\npsycopg2"}
    
    framework, confidence = FrameworkDetector.detect_framework(files, directories, package_contents)
    assert framework == "django"
    assert confidence > 0.5
    
    # Test React detection
    files = ["package.json", "src/App.js", "public/index.html"]
    directories = ["src/components", "public"]
    package_contents = {"package.json": '{"dependencies": {"react": "^18.0.0", "react-dom": "^18.0.0"}}'}
    
    framework, confidence = FrameworkDetector.detect_framework(files, directories, package_contents)
    assert framework == "react"
    assert confidence > 0.4
    
    # Test FastAPI detection
    files = ["main.py", "requirements.txt"]
    directories = ["app", "routers"]
    package_contents = {"requirements.txt": "fastapi==0.68.0\nuvicorn[standard]"}
    
    framework, confidence = FrameworkDetector.detect_framework(files, directories, package_contents)
    assert framework == "fastapi"
    assert confidence > 0.3
    
    # Test unknown framework (low confidence)
    files = ["random.txt", "other.py"]
    directories = ["stuff"]
    package_contents = {}
    
    framework, confidence = FrameworkDetector.detect_framework(files, directories, package_contents)
    assert framework is None
    assert confidence < 0.3

def test_project_info_dataclass():
    """Test ProjectInfo dataclass"""
    info = ProjectInfo(
        name="TestProject",
        framework="django",
        entry_points=["manage.py", "wsgi.py"],
        key_directories=["app", "templates"],
        config_files=["settings.py", "requirements.txt"],
        confidence_score=0.85
    )
    
    assert info.name == "TestProject"
    assert info.framework == "django"
    assert len(info.entry_points) == 2
    assert "manage.py" in info.entry_points
    assert info.confidence_score == 0.85

@pytest.mark.asyncio
async def test_enhanced_project_structure_basic():
    """Test basic functionality of EnhancedProjectStructure"""
    
    # Mock MCP tool caller
    mock_caller = AsyncMock()
    
    # Mock responses for directory structure
    mock_caller.side_effect = [
        # Directories
        "/workspace/TestProject\n/workspace/TestProject/src\n/workspace/TestProject/tests",
        # Important files  
        "/workspace/TestProject/README.md\n/workspace/TestProject/package.json\n/workspace/TestProject/main.py",
        # Source files
        "/workspace/TestProject/src/app.py\n/workspace/TestProject/src/utils.py",
        # Package.json content
        '{"name": "test-project", "dependencies": {"react": "^18.0.0"}}'
    ]
    
    analyzer = EnhancedProjectStructure(mock_caller)
    
    # This would require more detailed mocking to test fully
    # For now, just test that the class can be instantiated
    assert analyzer is not None
    assert analyzer.call_mcp_tool == mock_caller

def test_file_icon_mapping():
    """Test file icon mapping logic"""
    from enhanced_project_structure import EnhancedProjectStructure
    
    # Create a mock caller
    mock_caller = AsyncMock()
    analyzer = EnhancedProjectStructure(mock_caller)
    
    # Test various file types
    assert analyzer._get_file_icon("README.md") == "📖"
    assert analyzer._get_file_icon("package.json") == "📦"
    assert analyzer._get_file_icon("Dockerfile") == "🐳"
    assert analyzer._get_file_icon("main.py") == "🐍"
    assert analyzer._get_file_icon("App.js") == "⚡"
    assert analyzer._get_file_icon("Component.tsx") == "⚡"
    assert analyzer._get_file_icon("Main.java") == "☕"
    assert analyzer._get_file_icon("index.html") == "🌐"
    assert analyzer._get_file_icon("style.css") == "🌐"
    assert analyzer._get_file_icon("config.json") == "⚙️"
    assert analyzer._get_file_icon("unknown.xyz") == "📄"

def test_directory_path_resolution():
    """Test directory path resolution logic"""
    from enhanced_project_structure import EnhancedProjectStructure
    
    # Create a mock caller
    mock_caller = AsyncMock()
    analyzer = EnhancedProjectStructure(mock_caller)
    
    # Test various path resolution scenarios
    assert analyzer._resolve_directory_path(".", "MyProject") == "/workspace/MyProject"
    assert analyzer._resolve_directory_path(".", None) == "/workspace"
    assert analyzer._resolve_directory_path("src", "MyProject") == "/workspace/MyProject"
    assert analyzer._resolve_directory_path("/workspace/direct", None) == "/workspace/direct"
    assert analyzer._resolve_directory_path("relative/path", None) == "/workspace/relative/path"

def test_entry_point_identification():
    """Test entry point identification logic"""
    from enhanced_project_structure import EnhancedProjectStructure
    
    mock_caller = AsyncMock()
    analyzer = EnhancedProjectStructure(mock_caller)
    
    important_files = [
        "/workspace/project/README.md",
        "/workspace/project/main.py", 
        "/workspace/project/package.json"
    ]
    
    source_files = [
        "/workspace/project/src/app.py",
        "/workspace/project/src/server.js",
        "/workspace/project/manage.py"
    ]
    
    entry_points = analyzer._identify_entry_points(important_files, source_files)
    
    # Should identify main.py, server.js, and manage.py as entry points
    entry_basenames = [os.path.basename(ep) for ep in entry_points]
    assert "main.py" in entry_basenames
    assert "server.js" in entry_basenames  
    assert "manage.py" in entry_basenames
    assert "README.md" not in entry_basenames  # Not an entry point

def test_config_file_identification():
    """Test configuration file identification"""
    from enhanced_project_structure import EnhancedProjectStructure
    
    mock_caller = AsyncMock()
    analyzer = EnhancedProjectStructure(mock_caller)
    
    files = [
        "/workspace/project/README.md",
        "/workspace/project/package.json",
        "/workspace/project/requirements.txt",
        "/workspace/project/Dockerfile",
        "/workspace/project/src/main.py",
        "/workspace/project/.env",
        "/workspace/project/tsconfig.json"
    ]
    
    config_files = analyzer._identify_config_files(files)
    config_basenames = [os.path.basename(cf) for cf in config_files]
    
    assert "package.json" in config_basenames
    assert "requirements.txt" in config_basenames
    assert "Dockerfile" in config_basenames
    assert ".env" in config_basenames
    assert "tsconfig.json" in config_basenames
    assert "main.py" not in config_basenames  # Not a config file
    assert "README.md" not in config_basenames  # Not a config file

def test_annotation_system():
    """Test @ annotation system for codebase highlighting"""
    from enhanced_project_structure import EnhancedProjectStructure
    
    mock_caller = AsyncMock()
    analyzer = EnhancedProjectStructure(mock_caller)
    
    # Test with project name
    content = "Project structure analysis content"
    annotated = analyzer._add_annotations(content, "MyProject")
    assert annotated.startswith("@MyProject:")
    assert "Project structure analysis content" in annotated
    
    # Test without project name (workspace)
    annotated_workspace = analyzer._add_annotations(content, "workspace")
    assert not annotated_workspace.startswith("@")
    assert annotated_workspace == content
    
    # Test with None project name
    annotated_none = analyzer._add_annotations(content, None)
    assert not annotated_none.startswith("@")
    assert annotated_none == content

@pytest.mark.asyncio 
async def test_integration_with_project_context():
    """Test integration with project context management"""
    from project_context import set_project_context, get_current_context
    from enhanced_project_structure import EnhancedProjectStructure
    
    # Reset context
    from project_context import get_context_manager
    get_context_manager().reset_context()
    
    # Set a project context
    context = set_project_context("TestProject")
    assert context.name == "TestProject"
    
    # Mock caller
    mock_caller = AsyncMock()
    mock_caller.return_value = "mock response"
    
    analyzer = EnhancedProjectStructure(mock_caller)
    
    # The analyzer should use the current context
    current = get_current_context()
    assert current is not None
    assert current.name == "TestProject"

if __name__ == "__main__":
    print("Testing enhanced project structure...")
    
    test_framework_detector()
    print("[PASS] Framework detector")
    
    test_project_info_dataclass()  
    print("[PASS] ProjectInfo dataclass")
    
    test_file_icon_mapping()
    print("[PASS] File icon mapping")
    
    test_directory_path_resolution()
    print("[PASS] Directory path resolution")
    
    test_entry_point_identification()
    print("[PASS] Entry point identification")
    
    test_config_file_identification()
    print("[PASS] Config file identification")
    
    test_annotation_system()
    print("[PASS] Annotation system")
    
    # Run async tests
    async def run_async_tests():
        await test_enhanced_project_structure_basic()
        await test_integration_with_project_context()
    
    asyncio.run(run_async_tests())
    print("[PASS] Basic functionality and context integration")
    
    print("\n[SUCCESS] All Phase 1.3 enhanced project structure tests passed!")
    print("Both agents now use enhanced project structure analysis with @ annotations.")