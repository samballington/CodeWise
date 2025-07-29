"""
Unit tests for the AST-based chunking system

This module tests the AST chunker functionality including Python AST parsing,
JavaScript/TypeScript chunking, and various file type handling.
"""

import pytest
from pathlib import Path
from indexer.ast_chunker import (
    ASTChunker, 
    PythonASTChunker, 
    JavaScriptChunker,
    ConfigChunker,
    MarkdownChunker,
    SmallFileChunker,
    ChunkMetadata,
    CodeChunk
)


class TestChunkMetadata:
    """Test the ChunkMetadata dataclass"""
    
    def test_chunk_metadata_creation(self):
        """Test creating ChunkMetadata with various fields"""
        metadata = ChunkMetadata(
            file_type="python",
            chunk_type="function",
            function_name="test_func",
            class_name=None,
            imports=["import os", "from typing import List"],
            line_start=10,
            line_end=20,
            docstring="Test function docstring",
            decorators=["@pytest.fixture"]
        )
        
        assert metadata.file_type == "python"
        assert metadata.chunk_type == "function"
        assert metadata.function_name == "test_func"
        assert len(metadata.imports) == 2
        assert metadata.line_start == 10
        assert metadata.docstring == "Test function docstring"


class TestCodeChunk:
    """Test the CodeChunk dataclass"""
    
    def test_code_chunk_creation(self):
        """Test creating CodeChunk with metadata sync"""
        metadata = ChunkMetadata(
            file_type="python",
            chunk_type="function",
            line_start=0,  # Should be updated by __post_init__
            line_end=0
        )
        
        chunk = CodeChunk(
            text="def test(): pass",
            metadata=metadata,
            file_path="/test/file.py",
            start_line=5,
            end_line=6
        )
        
        assert chunk.metadata.line_start == 5
        assert chunk.metadata.line_end == 6
        assert chunk.text == "def test(): pass"


class TestPythonASTChunker:
    """Test the Python AST chunker"""
    
    def setup_method(self):
        """Set up test environment"""
        self.chunker = PythonASTChunker()
        self.test_file_path = Path("test_file.py")
    
    def test_simple_function_chunking(self):
        """Test chunking a simple Python function"""
        content = '''import os
from typing import List

def hello_world():
    """A simple greeting function."""
    print("Hello, World!")
    return "greeting"

def another_function(x: int) -> str:
    return str(x)
'''
        
        chunks = self.chunker.chunk_content(content, self.test_file_path)
        
        assert len(chunks) == 2
        
        # Check first function
        func1 = chunks[0]
        assert func1.metadata.chunk_type == "function"
        assert func1.metadata.function_name == "hello_world"
        assert func1.metadata.docstring == "A simple greeting function."
        assert "def hello_world():" in func1.text
        assert len(func1.metadata.imports) == 2
        
        # Check second function
        func2 = chunks[1]
        assert func2.metadata.function_name == "another_function"
        assert "def another_function(x: int) -> str:" in func2.text
    
    def test_class_chunking(self):
        """Test chunking Python classes"""
        content = '''from dataclasses import dataclass

@dataclass
class Person:
    """A person data class."""
    name: str
    age: int
    
    def greet(self):
        return f"Hello, I'm {self.name}"

class Animal(object):
    def __init__(self, species):
        self.species = species
'''
        
        chunks = self.chunker.chunk_content(content, self.test_file_path)
        
        assert len(chunks) == 2
        
        # Check first class
        class1 = chunks[0]
        assert class1.metadata.chunk_type == "class"
        assert class1.metadata.class_name == "Person"
        assert class1.metadata.docstring == "A person data class."
        assert "@dataclass" in class1.text
        assert "dataclass" in class1.metadata.decorators
        
        # Check second class
        class2 = chunks[1]
        assert class2.metadata.class_name == "Animal"
        assert "object" in class2.metadata.dependencies
    
    def test_async_function_chunking(self):
        """Test chunking async functions"""
        content = '''import asyncio

async def fetch_data():
    """Fetch data asynchronously."""
    await asyncio.sleep(1)
    return "data"
'''
        
        chunks = self.chunker.chunk_content(content, self.test_file_path)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.metadata.chunk_type == "async_function"
        assert chunk.metadata.function_name == "fetch_data"
        assert "async def fetch_data():" in chunk.text
    
    def test_syntax_error_fallback(self):
        """Test fallback chunking when syntax errors occur"""
        content = '''def broken_function(
    # Missing closing parenthesis and colon
    print("This will cause a syntax error")
'''
        
        chunks = self.chunker.chunk_content(content, self.test_file_path)
        
        # Should fallback to character-based chunking
        assert len(chunks) >= 1
        assert chunks[0].metadata.chunk_type == "fallback"
    
    def test_empty_file_handling(self):
        """Test handling of empty or minimal files"""
        content = '''# Just a comment
import os
'''
        
        chunks = self.chunker.chunk_content(content, self.test_file_path)
        
        # Should create a single module chunk
        assert len(chunks) == 1
        assert chunks[0].metadata.chunk_type == "module"
        assert len(chunks[0].metadata.imports) == 1


class TestJavaScriptChunker:
    """Test the JavaScript/TypeScript chunker"""
    
    def setup_method(self):
        """Set up test environment"""
        self.chunker = JavaScriptChunker()
        self.js_file_path = Path("test_file.js")
        self.ts_file_path = Path("test_file.ts")
    
    def test_function_chunking(self):
        """Test chunking JavaScript functions"""
        content = '''import { useState } from 'react';

function HelloWorld() {
    return <div>Hello World</div>;
}

export function AnotherFunction(props) {
    const [state, setState] = useState(0);
    return state;
}

const arrowFunction = () => {
    console.log("Arrow function");
};
'''
        
        chunks = self.chunker.chunk_content(content, self.js_file_path)
        
        assert len(chunks) >= 2  # Should find at least the regular functions
        
        # Check that functions are detected
        function_names = [chunk.metadata.function_name for chunk in chunks if chunk.metadata.function_name]
        assert "HelloWorld" in function_names
        assert "AnotherFunction" in function_names
    
    def test_class_chunking(self):
        """Test chunking JavaScript classes"""
        content = '''export class MyComponent {
    constructor(props) {
        this.props = props;
    }
    
    render() {
        return null;
    }
}

class AnotherClass extends BaseClass {
    method() {
        return "test";
    }
}
'''
        
        chunks = self.chunker.chunk_content(content, self.js_file_path)
        
        assert len(chunks) >= 2
        
        # Check that classes are detected
        class_names = [chunk.metadata.class_name for chunk in chunks if chunk.metadata.class_name]
        assert "MyComponent" in class_names
        assert "AnotherClass" in class_names
    
    def test_typescript_detection(self):
        """Test TypeScript file type detection"""
        content = '''interface User {
    name: string;
    age: number;
}

function greetUser(user: User): string {
    return `Hello, ${user.name}!`;
}
'''
        
        chunks = self.chunker.chunk_content(content, self.ts_file_path)
        
        # Should detect TypeScript file type
        for chunk in chunks:
            if chunk.metadata.function_name:
                assert chunk.metadata.file_type == "typescript"


class TestConfigChunker:
    """Test the configuration file chunker"""
    
    def setup_method(self):
        """Set up test environment"""
        self.chunker = ConfigChunker()
    
    def test_json_config_chunking(self):
        """Test chunking JSON configuration files"""
        content = '''{
    "name": "test-project",
    "version": "1.0.0",
    "dependencies": {
        "react": "^18.0.0"
    }
}'''
        
        file_path = Path("package.json")
        chunks = self.chunker.chunk_content(content, file_path)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.metadata.file_type == "json"
        assert chunk.metadata.chunk_type == "config_file"
        assert chunk.text == content
    
    def test_yaml_config_chunking(self):
        """Test chunking YAML configuration files"""
        content = '''name: test-project
version: 1.0.0
dependencies:
  - react
  - typescript
'''
        
        file_path = Path("config.yaml")
        chunks = self.chunker.chunk_content(content, file_path)
        
        assert len(chunks) == 1
        assert chunks[0].metadata.file_type == "yaml"


class TestMarkdownChunker:
    """Test the Markdown chunker"""
    
    def setup_method(self):
        """Set up test environment"""
        self.chunker = MarkdownChunker()
        self.md_file_path = Path("README.md")
    
    def test_headline_based_chunking(self):
        """Test chunking Markdown by headlines"""
        content = '''# Main Title

This is the introduction section.

## Section 1

Content for section 1.
More content here.

## Section 2

Content for section 2.

### Subsection 2.1

Nested content.
'''
        
        chunks = self.chunker.chunk_content(content, self.md_file_path)
        
        assert len(chunks) >= 3  # Should have multiple sections
        
        # Check that sections are properly identified
        section_titles = [chunk.metadata.parent_context for chunk in chunks]
        assert "Main Title" in section_titles
        assert "Section 1" in section_titles
        assert "Section 2" in section_titles
    
    def test_no_headlines_fallback(self):
        """Test fallback when no headlines are present"""
        content = '''This is just plain markdown text
without any headlines.

It should be treated as a single chunk.
'''
        
        chunks = self.chunker.chunk_content(content, self.md_file_path)
        
        assert len(chunks) == 1
        assert chunks[0].metadata.chunk_type == "markdown_file"


class TestSmallFileChunker:
    """Test the small file chunker"""
    
    def setup_method(self):
        """Set up test environment"""
        self.chunker = SmallFileChunker(max_size=100)
    
    def test_small_file_chunking(self):
        """Test chunking small files as single chunks"""
        content = "Small file content"
        file_path = Path("small.txt")
        
        chunks = self.chunker.chunk_content(content, file_path)
        
        assert len(chunks) == 1
        assert chunks[0].metadata.chunk_type == "small_file"
        assert chunks[0].text == content
    
    def test_large_file_rejection(self):
        """Test that large files are rejected"""
        content = "x" * 200  # Larger than max_size
        file_path = Path("large.txt")
        
        chunks = self.chunker.chunk_content(content, file_path)
        
        assert len(chunks) == 0  # Should reject large files


class TestASTChunker:
    """Test the main AST chunker coordinator"""
    
    def setup_method(self):
        """Set up test environment"""
        self.chunker = ASTChunker()
    
    def test_file_type_detection(self):
        """Test file type detection based on extensions"""
        assert self.chunker.get_file_type(Path("test.py")) == "python"
        assert self.chunker.get_file_type(Path("test.js")) == "javascript"
        assert self.chunker.get_file_type(Path("test.ts")) == "typescript"
        assert self.chunker.get_file_type(Path("README.md")) == "markdown"
        assert self.chunker.get_file_type(Path("config.json")) == "config"
        assert self.chunker.get_file_type(Path("unknown.xyz")) == "text"
    
    def test_small_file_priority(self):
        """Test that small files are handled by SmallFileChunker"""
        content = "Small content"
        file_path = Path("test.py")
        
        chunks = self.chunker.chunk_content(content, file_path)
        
        assert len(chunks) == 1
        assert chunks[0].metadata.chunk_type == "small_file"
    
    def test_python_file_chunking(self):
        """Test that Python files use PythonASTChunker"""
        content = '''def test_function():
    """Test function."""
    return "test"

class TestClass:
    pass
'''
        
        file_path = Path("test.py")
        chunks = self.chunker.chunk_content(content, file_path)
        
        assert len(chunks) == 2
        assert chunks[0].metadata.chunk_type == "function"
        assert chunks[1].metadata.chunk_type == "class"
    
    def test_fallback_chunking(self):
        """Test fallback to character-based chunking"""
        content = "x" * 1000  # Large content for unknown file type
        file_path = Path("unknown.xyz")
        
        chunks = self.chunker.chunk_content(content, file_path)
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        assert all(chunk.metadata.chunk_type == "fallback" for chunk in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])