#!/usr/bin/env python3
"""Test TreeSitterChunker with real SWE_Project Java file"""

import sys
sys.path.append('/app/indexer')

from parsers.tree_sitter_parser import TreeSitterFactory
from ast_chunker import TreeSitterChunker
from pathlib import Path

def main():
    print("=" * 60)
    print("REAL JAVA FILE TEST - Book.java from SWE_Project")
    print("=" * 60)
    
    # Read the real Java file
    java_file_path = '/workspace/SWE_Project/obs/src/main/java/com/example/obs/model/Book.java'
    
    try:
        with open(java_file_path, 'r', encoding='utf-8') as f:
            java_content = f.read()
        
        print(f"File: {java_file_path}")
        print(f"Size: {len(java_content)} characters")
        print(f"Lines: {len(java_content.split(chr(10)))} lines")
        print(f"Preview: {java_content[:200]}...")
        
        # Test TreeSitterChunker with real file
        chunker = TreeSitterChunker()
        chunks = chunker.chunk_content(java_content, Path(java_file_path))
        
        print(f"\nChunking Results:")
        print(f"  Total chunks: {len(chunks)}")
        
        # Analyze chunks
        function_chunks = []
        class_chunks = []
        
        for i, chunk in enumerate(chunks):
            metadata = getattr(chunk, 'metadata', None)
            if metadata:
                chunk_type = getattr(metadata, 'chunk_type', 'unknown')
                if chunk_type == 'function':
                    function_name = getattr(metadata, 'function_name', 'unnamed')
                    function_chunks.append(function_name)
                    print(f"  {i+1}. Function: {function_name} (lines {metadata.line_start}-{metadata.line_end})")
                elif chunk_type == 'class':
                    class_name = getattr(metadata, 'class_name', 'unnamed')
                    class_chunks.append(class_name)
                    print(f"  {i+1}. Class: {class_name} (lines {metadata.line_start}-{metadata.line_end})")
                else:
                    print(f"  {i+1}. Other: {chunk_type}")
        
        print(f"\nSummary:")
        print(f"  Classes found: {len(class_chunks)} - {class_chunks}")
        print(f"  Functions found: {len(function_chunks)} - {function_chunks}")
        
        # This is the key metric - if we get symbols here, Knowledge Graph should work
        total_symbols = len(class_chunks) + len(function_chunks)
        print(f"  TOTAL SYMBOLS: {total_symbols}")
        
        if total_symbols > 0:
            print(f"\n✅ SUCCESS: {total_symbols} symbols extracted from real Java file!")
            print(f"   This should generate {total_symbols} nodes in Knowledge Graph")
        else:
            print(f"\n❌ FAILURE: No symbols extracted from real Java file")
            
    except Exception as e:
        print(f"❌ Error processing {java_file_path}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()