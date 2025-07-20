"""
Tests for the Vector Store System

This module tests the vector store functionality including metadata handling,
search operations, and compatibility with both legacy and enhanced formats.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import faiss

from vector_store import VectorStore, get_vector_store


class TestVectorStore:
    """Test the VectorStore class"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_dir = str(Path(self.temp_dir) / "workspace")
        Path(self.workspace_dir).mkdir(parents=True)
        
        # Create test cache directory
        self.cache_dir = Path(self.workspace_dir) / ".vector_cache"
        self.cache_dir.mkdir(parents=True)
        
        self.index_file = self.cache_dir / "index.faiss"
        self.meta_file = self.cache_dir / "meta.json"
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_vector_store_initialization_empty(self):
        """Test VectorStore initialization with no existing index"""
        with patch('vector_store.INDEX_FILE', self.index_file), \
             patch('vector_store.META_FILE', self.meta_file):
            
            store = VectorStore(self.workspace_dir)
            
            assert store.index is not None
            assert isinstance(store.index, faiss.IndexFlatL2)
            assert len(store.meta) == 0
    
    def test_vector_store_load_legacy_metadata(self):
        """Test loading legacy tuple-based metadata"""
        # Create legacy metadata
        legacy_meta = [
            ["file1.py", "def hello(): pass"],
            ["file2.js", "function world() {}"]
        ]
        
        # Create dummy index
        index = faiss.IndexFlatL2(768)
        embeddings = np.random.random((2, 768)).astype('float32')
        index.add(embeddings)
        
        # Save to files
        faiss.write_index(index, str(self.index_file))
        with open(self.meta_file, 'w') as f:
            json.dump(legacy_meta, f)
        
        with patch('vector_store.INDEX_FILE', self.index_file), \
             patch('vector_store.META_FILE', self.meta_file):
            
            store = VectorStore(self.workspace_dir)
            
            assert len(store.meta) == 2
            # Should be converted to dict format
            assert isinstance(store.meta[0], dict)
            assert store.meta[0]["relative_path"] == "file1.py"
            assert store.meta[0]["chunk_text"] == "def hello(): pass"
            assert store.meta[0]["chunk_type"] == "legacy"
    
    def test_vector_store_load_enhanced_metadata(self):
        """Test loading enhanced dictionary-based metadata"""
        # Create enhanced metadata
        enhanced_meta = [
            {
                "file_path": "/workspace/file1.py",
                "relative_path": "file1.py",
                "chunk_text": "def hello(): pass",
                "start_line": 1,
                "end_line": 1,
                "file_type": "python",
                "chunk_type": "function",
                "function_name": "hello",
                "class_name": None,
                "imports": [],
                "parent_context": None,
                "dependencies": [],
                "docstring": None,
                "decorators": []
            },
            {
                "file_path": "/workspace/file2.js",
                "relative_path": "file2.js", 
                "chunk_text": "function world() {}",
                "start_line": 5,
                "end_line": 5,
                "file_type": "javascript",
                "chunk_type": "function",
                "function_name": "world",
                "class_name": None,
                "imports": [],
                "parent_context": None,
                "dependencies": [],
                "docstring": None,
                "decorators": []
            }
        ]
        
        # Create dummy index
        index = faiss.IndexFlatL2(768)
        embeddings = np.random.random((2, 768)).astype('float32')
        index.add(embeddings)
        
        # Save to files
        faiss.write_index(index, str(self.index_file))
        with open(self.meta_file, 'w') as f:
            json.dump(enhanced_meta, f)
        
        with patch('vector_store.INDEX_FILE', self.index_file), \
             patch('vector_store.META_FILE', self.meta_file):
            
            store = VectorStore(self.workspace_dir)
            
            assert len(store.meta) == 2
            assert isinstance(store.meta[0], dict)
            assert store.meta[0]["relative_path"] == "file1.py"
            assert store.meta[0]["function_name"] == "hello"
            assert store.meta[0]["file_type"] == "python"
    
    @patch('vector_store.openai.embeddings.create')
    def test_query_with_enhanced_metadata(self, mock_embeddings):
        """Test querying with enhanced metadata format"""
        # Mock embedding response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 768)]
        mock_embeddings.return_value = mock_response
        
        # Create enhanced metadata
        enhanced_meta = [
            {
                "relative_path": "test.py",
                "chunk_text": "def test_function(): pass",
                "file_type": "python",
                "chunk_type": "function",
                "function_name": "test_function"
            }
        ]
        
        # Create index with one embedding
        index = faiss.IndexFlatL2(768)
        embedding = np.random.random((1, 768)).astype('float32')
        index.add(embedding)
        
        with patch('vector_store.INDEX_FILE', self.index_file), \
             patch('vector_store.META_FILE', self.meta_file):
            
            store = VectorStore(self.workspace_dir)
            store.index = index
            store.meta = enhanced_meta
            
            results = store.query("test function", k=1, min_relevance=0.0)
            
            assert len(results) == 1
            assert results[0][0] == "test.py"  # file_path
            assert results[0][1] == "def test_function(): pass"  # snippet
    
    @patch('vector_store.openai.embeddings.create')
    def test_query_with_legacy_metadata(self, mock_embeddings):
        """Test querying with legacy tuple metadata format"""
        # Mock embedding response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 768)]
        mock_embeddings.return_value = mock_response
        
        # Create legacy metadata (as tuples)
        legacy_meta = [("test.py", "def test_function(): pass")]
        
        # Create index with one embedding
        index = faiss.IndexFlatL2(768)
        embedding = np.random.random((1, 768)).astype('float32')
        index.add(embedding)
        
        with patch('vector_store.INDEX_FILE', self.index_file), \
             patch('vector_store.META_FILE', self.meta_file):
            
            store = VectorStore(self.workspace_dir)
            store.index = index
            store.meta = legacy_meta
            
            results = store.query("test function", k=1, min_relevance=0.0)
            
            assert len(results) == 1
            assert results[0][0] == "test.py"  # file_path
            assert results[0][1] == "def test_function(): pass"  # snippet
    
    @patch('vector_store.openai.embeddings.create')
    def test_query_relevance_filtering(self, mock_embeddings):
        """Test query results are filtered by relevance threshold"""
        # Mock embedding response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 768)]
        mock_embeddings.return_value = mock_response
        
        # Create metadata
        meta = [
            {"relative_path": "test1.py", "chunk_text": "relevant code"},
            {"relative_path": "test2.py", "chunk_text": "irrelevant code"}
        ]
        
        # Create index with embeddings that will have different distances
        index = faiss.IndexFlatL2(768)
        embeddings = np.array([
            [0.1] * 768,  # Close to query
            [0.9] * 768   # Far from query
        ]).astype('float32')
        index.add(embeddings)
        
        with patch('vector_store.INDEX_FILE', self.index_file), \
             patch('vector_store.META_FILE', self.meta_file):
            
            store = VectorStore(self.workspace_dir)
            store.index = index
            store.meta = meta
            
            # Test with high relevance threshold
            results = store.query("test", k=2, min_relevance=0.8)
            
            # Should filter out low-relevance results
            assert len(results) <= 1
    
    def test_query_empty_index(self):
        """Test querying an empty index"""
        with patch('vector_store.INDEX_FILE', self.index_file), \
             patch('vector_store.META_FILE', self.meta_file):
            
            store = VectorStore(self.workspace_dir)
            results = store.query("test query")
            
            assert results == []
    
    @patch('vector_store.openai.embeddings.create')
    def test_query_error_handling(self, mock_embeddings):
        """Test query error handling"""
        # Mock embedding to raise an exception
        mock_embeddings.side_effect = Exception("API Error")
        
        with patch('vector_store.INDEX_FILE', self.index_file), \
             patch('vector_store.META_FILE', self.meta_file):
            
            store = VectorStore(self.workspace_dir)
            store.index = faiss.IndexFlatL2(768)
            store.meta = [{"relative_path": "test.py", "chunk_text": "test"}]
            
            results = store.query("test query")
            
            assert results == []
    
    def test_remove_project_embeddings_enhanced_format(self):
        """Test removing project embeddings with enhanced metadata"""
        enhanced_meta = [
            {"relative_path": "project1/file1.py", "chunk_text": "code1"},
            {"relative_path": "project2/file2.py", "chunk_text": "code2"},
            {"relative_path": "project1/file3.py", "chunk_text": "code3"}
        ]
        
        # Create index
        index = faiss.IndexFlatL2(768)
        embeddings = np.random.random((3, 768)).astype('float32')
        index.add(embeddings)
        
        with patch('vector_store.INDEX_FILE', self.index_file), \
             patch('vector_store.META_FILE', self.meta_file):
            
            store = VectorStore(self.workspace_dir)
            store.index = index
            store.meta = enhanced_meta
            
            # Remove project1
            success = store.remove_project_embeddings("project1")
            
            assert success is True
            assert len(store.meta) == 1
            assert store.meta[0]["relative_path"] == "project2/file2.py"
    
    def test_remove_project_embeddings_legacy_format(self):
        """Test removing project embeddings with legacy metadata"""
        legacy_meta = [
            ("project1/file1.py", "code1"),
            ("project2/file2.py", "code2"),
            ("project1/file3.py", "code3")
        ]
        
        # Create index
        index = faiss.IndexFlatL2(768)
        embeddings = np.random.random((3, 768)).astype('float32')
        index.add(embeddings)
        
        with patch('vector_store.INDEX_FILE', self.index_file), \
             patch('vector_store.META_FILE', self.meta_file):
            
            store = VectorStore(self.workspace_dir)
            store.index = index
            store.meta = legacy_meta
            
            # Remove project1
            success = store.remove_project_embeddings("project1")
            
            assert success is True
            assert len(store.meta) == 1
            assert store.meta[0][0] == "project2/file2.py"
    
    def test_calculate_relevance_score(self):
        """Test relevance score calculation"""
        with patch('vector_store.INDEX_FILE', self.index_file), \
             patch('vector_store.META_FILE', self.meta_file):
            
            store = VectorStore(self.workspace_dir)
            
            # Test base score calculation
            score1 = store._calculate_relevance_score(0.0, "test.py", "test query")
            score2 = store._calculate_relevance_score(1.0, "test.py", "test query")
            
            assert score1 > score2  # Lower distance should give higher score
            
            # Test project boost
            score_with_boost = store._calculate_relevance_score(0.5, "sqlmodel/test.py", "sqlmodel query")
            score_without_boost = store._calculate_relevance_score(0.5, "other/test.py", "sqlmodel query")
            
            assert score_with_boost > score_without_boost
            
            # Test file type boost
            score_py = store._calculate_relevance_score(0.5, "test.py", "test")
            score_txt = store._calculate_relevance_score(0.5, "test.txt", "test")
            
            assert score_py > score_txt


class TestVectorStoreSingleton:
    """Test the vector store singleton functionality"""
    
    def test_get_vector_store_singleton(self):
        """Test that get_vector_store returns the same instance"""
        store1 = get_vector_store()
        store2 = get_vector_store()
        
        assert store1 is store2
    
    def test_get_vector_store_type(self):
        """Test that get_vector_store returns VectorStore instance"""
        store = get_vector_store()
        
        assert isinstance(store, VectorStore)


class TestVectorStoreIntegration:
    """Integration tests for vector store functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_dir = str(Path(self.temp_dir) / "workspace")
        Path(self.workspace_dir).mkdir(parents=True)
        
        # Create test files
        test_files = {
            "test1.py": "def hello_world():\n    print('Hello, World!')",
            "test2.js": "function greetUser(name) {\n    console.log(`Hello, ${name}!`);\n}",
            "README.md": "# Test Project\n\nThis is a test project for vector store testing."
        }
        
        for filename, content in test_files.items():
            (Path(self.workspace_dir) / filename).write_text(content)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('vector_store.openai.embeddings.create')
    def test_end_to_end_indexing_and_search(self, mock_embeddings):
        """Test complete indexing and search workflow"""
        # Mock embedding responses
        def mock_embedding_response(model, input):
            response = Mock()
            response.data = []
            for text in input:
                # Create different embeddings for different content
                if "hello" in text.lower():
                    embedding = [0.8] + [0.1] * 767
                elif "greet" in text.lower():
                    embedding = [0.7] + [0.2] * 767
                else:
                    embedding = [0.1] + [0.1] * 767
                response.data.append(Mock(embedding=embedding))
            return response
        
        mock_embeddings.side_effect = mock_embedding_response
        
        with patch('vector_store.WORKSPACE_DIR', self.workspace_dir):
            store = VectorStore(self.workspace_dir)
            
            # Build index
            store._build()
            
            # Test search
            results = store.query("hello function", k=2, min_relevance=0.0)
            
            assert len(results) > 0
            # Should find relevant code snippets
            found_hello = any("hello" in result[1].lower() for result in results)
            assert found_hello


if __name__ == "__main__":
    pytest.main([__file__, "-v"])