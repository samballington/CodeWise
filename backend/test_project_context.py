#!/usr/bin/env python3
"""
Test project context isolation between agents
Phase 1.2 validation
"""

import pytest
import asyncio
from project_context import (
    ProjectContextManager, get_context_manager, set_project_context,
    get_current_context, filter_files_by_context
)

def test_project_context_manager_creation():
    """Test that project context manager can be created and initialized"""
    manager = ProjectContextManager()
    assert manager.current_project is None
    assert len(manager.contexts) == 0

def test_set_project_context():
    """Test setting project context"""
    # Reset global state
    manager = get_context_manager()
    manager.reset_context()
    
    # Set first project context
    context1 = set_project_context("Gymmy")
    assert context1.name == "Gymmy"
    assert context1.base_path == "/workspace/Gymmy"
    assert manager.current_project == "Gymmy"
    
    # Switch to second project context
    context2 = set_project_context("SWE_Project")
    assert context2.name == "SWE_Project"
    assert context2.base_path == "/workspace/SWE_Project"
    assert manager.current_project == "SWE_Project"
    
    # Ensure both contexts are stored
    assert len(manager.contexts) == 2
    assert "Gymmy" in manager.contexts
    assert "SWE_Project" in manager.contexts

def test_context_switch_detection():
    """Test that context switches are properly detected and logged"""
    manager = get_context_manager()
    manager.reset_context()
    
    # First context
    set_project_context("Project1")
    assert manager.current_project == "Project1"
    
    # Switch context - should trigger clearing
    set_project_context("Project2")
    assert manager.current_project == "Project2"
    
    # Same context - should not trigger clearing
    set_project_context("Project2")
    assert manager.current_project == "Project2"

def test_file_context_detection():
    """Test that file paths are correctly associated with project contexts"""
    manager = get_context_manager()
    
    # Test different file path formats
    test_cases = [
        ("/workspace/Gymmy/models.py", "Gymmy"),
        ("/workspace/SWE_Project/src/main.java", "SWE_Project"),
        ("/workspace/root_file.txt", "workspace"),
        ("/workspace/", "workspace"),
        ("not_workspace_path.py", None)
    ]
    
    for file_path, expected_project in test_cases:
        result = manager.get_context_for_file(file_path)
        assert result == expected_project, f"Failed for {file_path}: expected {expected_project}, got {result}"

def test_file_filtering_by_context():
    """Test that files are properly filtered by current context"""
    manager = get_context_manager()
    manager.reset_context()
    
    # Set context to Gymmy project
    set_project_context("Gymmy")
    
    # Test file list with mixed projects
    files = [
        "/workspace/Gymmy/models.py",           # Should include
        "/workspace/Gymmy/views.py",            # Should include
        "/workspace/SWE_Project/main.java",     # Should exclude
        "/workspace/codebase-digest/app.py",    # Should exclude
        "/workspace/Gymmy/templates/base.html", # Should include
        "/workspace/root_config.json"           # Should exclude
    ]
    
    # Filter by current context
    filtered = manager.filter_files_by_context(files)
    expected = [
        "/workspace/Gymmy/models.py",
        "/workspace/Gymmy/views.py", 
        "/workspace/Gymmy/templates/base.html"
    ]
    
    assert filtered == expected

def test_context_isolation_prevents_contamination():
    """Test that context isolation prevents cross-project contamination"""
    manager = get_context_manager()
    manager.reset_context()
    
    # Simulate Project A context
    set_project_context("ProjectA")
    manager.add_file_to_context("/workspace/ProjectA/file1.py")
    manager.add_search_to_context("search query A")
    
    context_a = get_current_context()
    assert len(context_a.mentioned_files) == 1
    assert len(context_a.search_history) == 1
    
    # Switch to Project B context
    set_project_context("ProjectB")
    manager.add_file_to_context("/workspace/ProjectB/file2.py")
    manager.add_search_to_context("search query B")
    
    context_b = get_current_context()
    assert len(context_b.mentioned_files) == 1
    assert len(context_b.search_history) == 1
    
    # Verify Project A context is preserved but separate
    context_a_check = manager.contexts["ProjectA"]
    assert len(context_a_check.mentioned_files) == 1
    assert "/workspace/ProjectA/file1.py" in context_a_check.mentioned_files
    assert "search query A" in context_a_check.search_history
    
    # Verify Project B context is separate
    assert "/workspace/ProjectB/file2.py" in context_b.mentioned_files
    assert "search query B" in context_b.search_history
    
    # Verify no cross-contamination
    assert "/workspace/ProjectB/file2.py" not in context_a_check.mentioned_files
    assert "search query B" not in context_a_check.search_history

def test_mentioned_projects_handling():
    """Test that mentioned projects are handled correctly"""
    manager = get_context_manager()
    manager.reset_context()
    
    # Test with multiple mentioned projects - should use first as primary
    context = set_project_context("default", ["Gymmy", "SWE_Project"])
    assert context.name == "Gymmy"  # First mentioned project becomes primary
    assert manager.current_project == "Gymmy"
    
    # Test with empty mentioned projects
    context = set_project_context("workspace", [])
    assert context.name == "workspace"
    assert manager.current_project == "workspace"

def test_context_cleanup():
    """Test that old contexts are cleaned up to prevent memory buildup"""
    manager = get_context_manager()
    manager.reset_context()
    manager.max_contexts = 3  # Set low limit for testing
    
    # Create more contexts than the limit
    for i in range(5):
        set_project_context(f"Project{i}")
    
    # Should only keep the most recent contexts
    assert len(manager.contexts) <= manager.max_contexts
    
    # Current project should still be accessible
    assert manager.current_project == "Project4"
    assert "Project4" in manager.contexts

def test_convenience_functions():
    """Test convenience functions for context management"""
    manager = get_context_manager()
    manager.reset_context()
    
    # Test convenience functions
    context = set_project_context("TestProject")
    assert context.name == "TestProject"
    
    current = get_current_context()
    assert current is not None
    assert current.name == "TestProject"
    
    # Test file filtering convenience function
    files = ["/workspace/TestProject/file1.py", "/workspace/OtherProject/file2.py"]
    filtered = filter_files_by_context(files)
    assert len(filtered) == 1
    assert filtered[0] == "/workspace/TestProject/file1.py"

if __name__ == "__main__":
    print("Testing project context isolation...")
    
    test_project_context_manager_creation()
    print("[PASS] Project context manager creation")
    
    test_set_project_context()
    print("[PASS] Set project context")
    
    test_context_switch_detection()
    print("[PASS] Context switch detection")
    
    test_file_context_detection()
    print("[PASS] File context detection")
    
    test_file_filtering_by_context()
    print("[PASS] File filtering by context")
    
    test_context_isolation_prevents_contamination()
    print("[PASS] Context isolation prevents contamination")
    
    test_mentioned_projects_handling()
    print("[PASS] Mentioned projects handling")
    
    test_context_cleanup()
    print("[PASS] Context cleanup")
    
    test_convenience_functions()
    print("[PASS] Convenience functions")
    
    print("\n[SUCCESS] All Phase 1.2 project context isolation tests passed!")
    print("Cross-contamination between project contexts is now prevented.")