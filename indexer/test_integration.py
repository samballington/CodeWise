#!/usr/bin/env python3
"""
Test integration of file discovery and AST chunking
"""

import sys
from pathlib import Path
from file_discovery import FileDiscoveryEngine
from ast_chunker import ASTChunker

def test_integration():
    """Test the integration of file discovery and AST chunking"""
    print("Testing integration of enhanced file discovery and AST chunking...")
    
    workspace = Path(".")
    
    # Discover files
    discovery_engine = FileDiscoveryEngine(workspace)
    discovered_files = discovery_engine.discover_files()
    
    print(f"Discovered {len(discovered_files)} files")
    
    # Initialize AST chunker
    ast_chunker = ASTChunker()
    
    total_chunks = 0
    processed_files = 0
    
    # Process each discovered file
    for file_info in discovered_files[:5]:  # Test first 5 files
        try:
            content = file_info.path.read_text(encoding="utf-8", errors="ignore")
            chunks = ast_chunker.chunk_content(content, file_info.path)
            
            print(f"\n--- {file_info.relative_path} ---")
            print(f"File type: {file_info.file_type}")
            print(f"Detection method: {file_info.detection_method}")
            print(f"Chunks created: {len(chunks)}")
            
            for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
                print(f"  Chunk {i+1}: {chunk.metadata.chunk_type} ({chunk.start_line}-{chunk.end_line})")
                if chunk.metadata.function_name:
                    print(f"    Function: {chunk.metadata.function_name}")
                if chunk.metadata.class_name:
                    print(f"    Class: {chunk.metadata.class_name}")
            
            if len(chunks) > 2:
                print(f"  ... and {len(chunks) - 2} more chunks")
            
            total_chunks += len(chunks)
            processed_files += 1
            
        except Exception as e:
            print(f"Error processing {file_info.relative_path}: {e}")
    
    print(f"\n=== Integration Test Summary ===")
    print(f"Files processed: {processed_files}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Average chunks per file: {total_chunks / processed_files if processed_files > 0 else 0:.2f}")
    
    # Show discovery statistics
    stats = discovery_engine.get_discovery_stats()
    print(f"File coverage: {stats.get_coverage_percentage():.2f}%")
    print(f"Files by type: {dict(stats.files_by_type)}")
    
    return total_chunks > 0

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)