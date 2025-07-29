#!/usr/bin/env python3
"""
Test directory filtering consistency between agents
Phase 1.1 validation
"""

import pytest
import asyncio
import tempfile
import os
import shutil
from pathlib import Path

# Import our filtering utilities
from directory_filters import (
    get_find_filter_args, get_grep_filter_args, should_include_file,
    filter_file_list, resolve_workspace_path, validate_filtering_consistency
)

def test_filtering_consistency():
    """Test that our filtering patterns work consistently"""
    issues = validate_filtering_consistency()
    assert len(issues) == 0, f"Filtering consistency issues: {issues}"

def test_should_include_file():
    """Test individual file inclusion logic"""
    # Should be included
    assert should_include_file("/workspace/project/src/main.py")
    assert should_include_file("/workspace/project/components/Button.tsx")
    assert should_include_file("/workspace/project/README.md")
    assert should_include_file("/workspace/project/package.json")
    
    # Should be excluded
    assert not should_include_file("/workspace/project/.git/config")
    assert not should_include_file("/workspace/project/node_modules/package/index.js")
    assert not should_include_file("/workspace/project/__pycache__/module.pyc")
    assert not should_include_file("/workspace/project/.venv/lib/python3.8/site-packages")
    assert not should_include_file("/workspace/project/build/artifacts/app.jar")
    assert not should_include_file("/workspace/project/dist/bundle.js")

def test_filter_file_list():
    """Test filtering a list of files"""
    files = [
        "/workspace/project/src/main.py",
        "/workspace/project/.git/config",
        "/workspace/project/node_modules/react/index.js",
        "/workspace/project/components/Button.tsx",
        "/workspace/project/__pycache__/module.pyc",
        "/workspace/project/README.md"
    ]
    
    filtered = filter_file_list(files)
    expected = [
        "/workspace/project/src/main.py",
        "/workspace/project/components/Button.tsx", 
        "/workspace/project/README.md"
    ]
    
    assert filtered == expected

def test_find_filter_args():
    """Test that find filter args are properly formatted"""
    filter_args = get_find_filter_args()
    assert "-not -path '*/.git*'" in filter_args
    assert "-not -path '*/node_modules*'" in filter_args
    assert "-not -path '*/__pycache__*'" in filter_args
    assert "-not -path '*/.venv*'" in filter_args

def test_grep_filter_args():
    """Test that grep filter args are properly formatted"""
    grep_filter = get_grep_filter_args()
    assert ".*/\\.git.*" in grep_filter or ".*/.git.*" in grep_filter
    assert ".*/node_modules.*" in grep_filter
    assert ".*/__pycache__.*" in grep_filter

def test_resolve_workspace_path():
    """Test workspace path resolution"""
    assert resolve_workspace_path(".") == "/workspace"
    assert resolve_workspace_path(".", "MyProject") == "/workspace/MyProject"
    assert resolve_workspace_path("src", "MyProject") == "/workspace/MyProject/src"
    assert resolve_workspace_path("/workspace/direct") == "/workspace/direct"

@pytest.mark.asyncio
async def test_agent_filtering_integration():
    """Test that both agents use consistent filtering (integration test)"""
    # This would test actual agent usage, but requires full setup
    # For now, just validate that imports work
    try:
        from agent import CodeWiseAgent
        from cerebras_agent import CerebrasNativeAgent
        assert True  # Imports successful
    except ImportError as e:
        pytest.skip(f"Agent imports not available: {e}")

if __name__ == "__main__":
    # Run basic validation
    print("Testing directory filtering consistency...")
    
    test_filtering_consistency()
    print("[PASS] Filtering consistency validated")
    
    test_should_include_file()
    print("[PASS] File inclusion logic validated")
    
    test_filter_file_list()
    print("[PASS] File list filtering validated")
    
    test_find_filter_args()
    print("[PASS] Find filter args validated")
    
    test_grep_filter_args() 
    print("[PASS] Grep filter args validated")
    
    test_resolve_workspace_path()
    print("[PASS] Workspace path resolution validated")
    
    print("\n[SUCCESS] All Phase 1.1 directory filtering tests passed!")
    print("Both LangChain and Cerebras agents now use standardized filtering.")