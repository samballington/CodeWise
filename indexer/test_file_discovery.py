#!/usr/bin/env python3
"""
Test script for the enhanced file discovery system
"""

import sys
from pathlib import Path
from indexer.file_discovery import FileDiscoveryEngine

def test_file_discovery():
    """Test the file discovery engine"""
    workspace = Path("/workspace")
    
    if not workspace.exists():
        print("Workspace directory not found, creating test directory...")
        workspace = Path(".")
    
    print(f"Testing file discovery in: {workspace}")
    
    # Create discovery engine
    discovery_engine = FileDiscoveryEngine(workspace)
    
    # Discover files
    discovered_files = discovery_engine.discover_files()
    
    print(f"\nDiscovered {len(discovered_files)} files:")
    
    # Show first 10 files as examples
    for i, file_info in enumerate(discovered_files[:10]):
        print(f"  {i+1}. {file_info.relative_path} ({file_info.file_type}, {file_info.detection_method})")
    
    if len(discovered_files) > 10:
        print(f"  ... and {len(discovered_files) - 10} more files")
    
    # Show statistics
    stats = discovery_engine.get_discovery_stats()
    print(f"\nCoverage: {stats.get_coverage_percentage():.2f}%")
    print(f"Files by type: {dict(stats.files_by_type)}")
    
    return len(discovered_files) > 0

if __name__ == "__main__":
    success = test_file_discovery()
    sys.exit(0 if success else 1)