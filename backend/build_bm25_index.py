#!/usr/bin/env python3
"""
Build BM25 index from existing vector store data
"""

import json
import logging
from pathlib import Path
from bm25_index import BM25Index
from vector_store import get_vector_store

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_bm25_from_vector_store():
    """Build BM25 index from existing vector store metadata"""
    logger.info("Building BM25 index from vector store data...")
    
    try:
        # Get vector store instance
        vector_store = get_vector_store()
        vector_store.force_refresh()  # Ensure we have the latest index from disk
        
        if not vector_store.meta:
            logger.error("No metadata found in vector store")
            return False
        
        # Create BM25 index
        bm25_index = BM25Index()
        
        # Convert vector store metadata to BM25 documents
        documents = []
        for i, meta in enumerate(vector_store.meta):
            # Extract text content
            text = meta.get('chunk_text', '')
            if not text.strip():
                continue
            
            # Create document for BM25
            doc = {
                'id': i,
                'text': text,
                'file_path': meta.get('file_path', ''),
                'relative_path': meta.get('relative_path', ''),
                'chunk_type': meta.get('chunk_type', 'unknown'),
                'function_name': meta.get('function_name'),
                'class_name': meta.get('class_name'),
                'file_type': meta.get('file_type', 'unknown')
            }
            documents.append(doc)
        
        logger.info(f"Converting {len(documents)} documents to BM25 index...")
        
        # Add documents to BM25 index
        bm25_index.add_documents(documents)
        
        # Save BM25 index
        cache_dir = Path("/workspace/.vector_cache")
        bm25_file = cache_dir / "bm25_index.json"
        
        bm25_index.save_index(str(bm25_file))
        
        logger.info(f"BM25 index built successfully with {len(documents)} documents")
        logger.info(f"Index saved to: {bm25_file}")
        
        # Test the index
        test_results = bm25_index.search("function", k=3)
        logger.info(f"Test search for 'function' returned {len(test_results)} results")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to build BM25 index: {e}")
        return False

if __name__ == "__main__":
    success = build_bm25_from_vector_store()
    if success:
        print("BM25 index built successfully!")
    else:
        print("Failed to build BM25 index")