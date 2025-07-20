#!/usr/bin/env python3
"""
Detailed test script for the AST chunker to verify specific functionality
"""

from ast_chunker import ASTChunker
from pathlib import Path

def test_python_detailed():
    """Test Python AST chunking with detailed verification"""
    chunker = ASTChunker()
    
    content = """import os
import sys
from typing import List

def simple_function():
    '''Simple function docstring.'''
    return "hello"

async def async_function():
    '''Async function docstring.'''
    await asyncio.sleep(1)
    return "async result"

@decorator
def decorated_function():
    '''Decorated function.'''
    pass

class SimpleClass:
    '''Simple class docstring.'''
    
    def __init__(self):
        self.value = 0
    
    def method(self):
        return self.value

class InheritedClass(BaseClass):
    '''Inherited class.'''
    pass
"""
    
    chunks = chunker.chunk_content(content, Path('test.py'))
    
    print("=== PYTHON DETAILED TEST ===")
    print(f"Total chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Type: {chunk.metadata.chunk_type}")
        print(f"  Function: {chunk.metadata.function_name}")
        print(f"  Class: {chunk.metadata.class_name}")
        print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"  Docstring: {chunk.metadata.docstring}")
        print(f"  Decorators: {chunk.metadata.decorators}")
        print(f"  Dependencies: {chunk.metadata.dependencies}")
        print(f"  Imports count: {len(chunk.metadata.imports)}")
        
        # Show first few lines of text
        lines = chunk.text.split('\n')[:3]
        print(f"  Text preview: {' | '.join(lines)}")

def test_javascript_detailed():
    """Test JavaScript chunking with detailed verification"""
    chunker = ASTChunker()
    
    content = """import React from 'react';

function regularFunction() {
    console.log('regular function');
    return true;
}

export function exportedFunction(param) {
    return param * 2;
}

async function asyncFunction() {
    const result = await fetch('/api');
    return result.json();
}

const arrowFunction = () => {
    return 'arrow';
};

const paramArrowFunction = (a, b) => {
    return a + b;
};

class RegularClass {
    constructor() {
        this.value = 0;
    }
    
    method() {
        return this.value;
    }
}

export class ExportedClass {
    render() {
        return null;
    }
}
"""
    
    chunks = chunker.chunk_content(content, Path('test.js'))
    
    print("\n=== JAVASCRIPT DETAILED TEST ===")
    print(f"Total chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Type: {chunk.metadata.chunk_type}")
        print(f"  Function: {chunk.metadata.function_name}")
        print(f"  Class: {chunk.metadata.class_name}")
        print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"  File type: {chunk.metadata.file_type}")
        
        # Show first few lines of text
        lines = chunk.text.split('\n')[:2]
        print(f"  Text preview: {' | '.join(lines)}")

def test_config_and_markdown():
    """Test config and markdown chunkers"""
    chunker = ASTChunker()
    
    # Test JSON config
    json_content = """{
    "name": "test-project",
    "version": "1.0.0",
    "scripts": {
        "start": "node index.js",
        "test": "jest"
    }
}"""
    
    json_chunks = chunker.chunk_content(json_content, Path('package.json'))
    
    # Test Markdown
    md_content = """# Main Title

Introduction paragraph.

## Section 1

Content for section 1.

### Subsection 1.1

Nested content.

## Section 2

Content for section 2.
"""
    
    md_chunks = chunker.chunk_content(md_content, Path('README.md'))
    
    print("\n=== CONFIG AND MARKDOWN TEST ===")
    print(f"JSON chunks: {len(json_chunks)}")
    for chunk in json_chunks:
        print(f"  JSON - Type: {chunk.metadata.chunk_type}, File type: {chunk.metadata.file_type}")
    
    print(f"Markdown chunks: {len(md_chunks)}")
    for chunk in md_chunks:
        print(f"  MD - Type: {chunk.metadata.chunk_type}, Context: {chunk.metadata.parent_context}")

if __name__ == '__main__':
    test_python_detailed()
    test_javascript_detailed()
    test_config_and_markdown()