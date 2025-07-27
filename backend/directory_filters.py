#!/usr/bin/env python3
"""
Shared directory filtering utilities for both LangChain and Cerebras agents
Ensures consistent filtering behavior across all file discovery operations
"""

import os
from typing import List

# Centralized ignore patterns for consistent filtering
IGNORE_PATTERNS = [
    '*/.git*', 
    '*/node_modules*', 
    '*/__pycache__*',
    '*/.venv*', 
    '*/.env*',
    '*/build*', 
    '*/dist*', 
    '*/.pytest_cache*',
    '*/.next*', 
    '*/coverage*', 
    '*/.nyc_output*',
    '*/.vscode*',
    '*/.idea*',
    '*/target*',  # Java/Rust build dirs
    '*/bin*',     # Binary dirs
    '*/obj*',     # C# build dirs
    '*/.gradle*', # Gradle cache
    '*/.m2*',     # Maven cache
    '*/vendor*',  # PHP/Go dependencies
    '*/.cache*',  # General cache dirs
    '*/tmp*',     # Temporary dirs
    '*/.DS_Store*' # macOS files
]

def get_find_filter_args() -> str:
    """
    Generate find command filter arguments for consistent directory exclusion
    Returns: String of -not -path arguments for find command
    """
    return ' '.join([f"-not -path '{pattern}'" for pattern in IGNORE_PATTERNS])

def get_grep_filter_args() -> str:
    """
    Generate grep exclusion pattern for consistent directory filtering
    Returns: Regex pattern for grep -v -E option
    """
    # Convert shell patterns to regex patterns for grep
    grep_patterns = []
    for pattern in IGNORE_PATTERNS:
        # Convert shell glob to regex: */pattern* -> .*/pattern.*
        regex_pattern = pattern.replace('*/', '.*/').replace('*', '.*')
        grep_patterns.append(regex_pattern)
    
    return f"({'|'.join(grep_patterns)})"

def should_include_file(file_path: str) -> bool:
    """
    Check if a file should be included based on our filtering rules
    Args:
        file_path: Path to check
    Returns:
        True if file should be included, False if filtered out
    """
    file_path_lower = file_path.lower()
    
    # Check against each ignore pattern
    for pattern in IGNORE_PATTERNS:
        # Remove leading */ and trailing * for simple contains check
        clean_pattern = pattern.strip('*').strip('/')
        if clean_pattern in file_path_lower:
            return False
    
    return True

def filter_file_list(file_paths: List[str]) -> List[str]:
    """
    Filter a list of file paths using our ignore patterns
    Args:
        file_paths: List of file paths to filter
    Returns:
        Filtered list of file paths
    """
    return [f for f in file_paths if should_include_file(f)]

def resolve_workspace_path(directory: str, project_name: str = None) -> str:
    """
    Resolve directory path within workspace context
    Args:
        directory: Directory path (can be relative)
        project_name: Optional project name for scoping
    Returns:
        Absolute workspace path
    """
    workspace_base = "/workspace"
    
    if project_name:
        # Project-specific path
        if directory == "." or directory == "":
            return f"{workspace_base}/{project_name}"
        else:
            # Ensure we're within the project scope
            clean_dir = directory.lstrip('./')
            return f"{workspace_base}/{project_name}/{clean_dir}"
    else:
        # General workspace path
        if directory == "." or directory == "":
            return workspace_base
        elif directory.startswith("/workspace"):
            return directory
        else:
            clean_dir = directory.lstrip('./')
            return f"{workspace_base}/{clean_dir}"

def get_project_from_path(file_path: str) -> str:
    """
    Extract project name from a workspace file path
    Args:
        file_path: File path within workspace
    Returns:
        Project name or 'workspace' if in root
    """
    if not file_path.startswith('/workspace/'):
        return 'workspace'
    
    path_parts = file_path.replace('/workspace/', '').split('/')
    if len(path_parts) > 0 and path_parts[0]:
        return path_parts[0]
    
    return 'workspace'

# Validation function for testing
def validate_filtering_consistency():
    """
    Validate that our filtering patterns work consistently
    Returns list of any issues found
    """
    issues = []
    
    # Test cases that should be filtered out
    test_paths_filtered = [
        "/workspace/project/.git/config",
        "/workspace/project/node_modules/package/index.js", 
        "/workspace/project/__pycache__/module.pyc",
        "/workspace/project/.venv/lib/python3.8/site-packages",
        "/workspace/project/build/artifacts/app.jar",
        "/workspace/project/dist/bundle.js",
        "/workspace/project/.pytest_cache/test.log"
    ]
    
    # Test cases that should be included
    test_paths_included = [
        "/workspace/project/src/main.py",
        "/workspace/project/components/Button.tsx",
        "/workspace/project/README.md",
        "/workspace/project/package.json",
        "/workspace/project/requirements.txt"
    ]
    
    for path in test_paths_filtered:
        if should_include_file(path):
            issues.append(f"Should filter out but didn't: {path}")
    
    for path in test_paths_included:
        if not should_include_file(path):
            issues.append(f"Should include but didn't: {path}")
    
    return issues