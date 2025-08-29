"""
Enhanced Vector Store with BGE Model and Query Instructions

Provides significant improvement in semantic understanding for code retrieval
through state-of-the-art BGE embeddings with proper query instruction handling.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import time

logger = logging.getLogger(__name__)


class EnhancedVectorStore:
    """
    Upgraded vector store with bge-large-en-v1.5 model and query instructions.
    
    Key improvements over existing VectorStore:
    - BGE model with 1024 dimensions vs MiniLM 384 dimensions
    - Query instruction prefixes for improved retrieval
    - Proper normalization for consistent similarity calculations
    - Better error handling and performance monitoring
    """
    
    def __init__(self, model_name: str = 'BAAI/bge-large-en-v1.5', storage_dir: str = ".vector_cache"):
        """
        Initialize enhanced vector store.
        
        Args:
            model_name: BGE model name for embeddings
            storage_dir: Directory for storing vector index and metadata
        """
        self.model_name = model_name
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # BGE model specifications
        self.embedding_dim = 1024  # bge-large-en-v1.5 dimensions
        self.query_instruction = "Represent this sentence for searching relevant passages: {}"
        
        # Initialize model
        try:
            logger.info(f"Loading BGE model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info(f"BGE model loaded successfully with {self.embedding_dim} dimensions")
        except Exception as e:
            logger.error(f"Failed to load BGE model {model_name}: {e}")
            logger.info("Falling back to MiniLM model")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_dim = 384
            self.model_name = "all-MiniLM-L6-v2"
        
        # Vector index components
        self.index: Optional[faiss.Index] = None
        self.index_path = self.storage_dir / "enhanced_index.faiss"
        self.meta_path = self.storage_dir / "enhanced_meta.json"
        
        # Performance tracking
        self.embedding_times: List[float] = []
        self.search_times: List[float] = []
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed user query with BGE instruction prefix.
        
        CRITICAL: Only queries get the instruction prefix for optimal retrieval.
        
        Args:
            query: User query string
            
        Returns:
            Normalized query embedding
        """
        start_time = time.time()
        
        try:
            # Apply instruction prefix for BGE models
            if "bge" in self.model_name.lower():
                instructional_query = self.query_instruction.format(query)
            else:
                instructional_query = query
            
            embedding = self.model.encode(
                instructional_query, 
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Ensure correct dimensionality
            if embedding.shape[0] != self.embedding_dim:
                logger.warning(f"Embedding dimension mismatch: got {embedding.shape[0]}, expected {self.embedding_dim}")
            
            embedding_time = time.time() - start_time
            self.embedding_times.append(embedding_time)
            
            logger.debug(f"Query embedded in {embedding_time:.3f}s")
            return embedding
            
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def embed_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed code chunks without any prefix.
        
        CRITICAL: Documents are embedded as-is for best semantic representation.
        
        Args:
            documents: List of document strings to embed
            batch_size: Batch size for processing
            
        Returns:
            Array of normalized document embeddings
        """
        if not documents:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        
        logger.info(f"Embedding {len(documents)} documents in batches of {batch_size}")
        start_time = time.time()
        
        try:
            embeddings = self.model.encode(
                documents,
                normalize_embeddings=True, 
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=True
            )
            
            embedding_time = time.time() - start_time
            logger.info(f"Document embedding completed in {embedding_time:.2f}s")
            
            # Validate embeddings
            if embeddings.shape[1] != self.embedding_dim:
                logger.error(f"Embedding dimension mismatch: got {embeddings.shape[1]}, expected {self.embedding_dim}")
                return np.empty((0, self.embedding_dim), dtype=np.float32)
            
            logger.info(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Document embedding failed: {e}")
            return np.empty((0, self.embedding_dim), dtype=np.float32)
    
    def build_index(self, documents: List[str], metadata: List[Dict] = None) -> bool:
        """
        Build FAISS index from documents.
        
        Args:
            documents: List of document strings
            metadata: Optional metadata for each document
            
        Returns:
            True if index built successfully
        """
        if not documents:
            logger.warning("No documents provided for index building")
            return False
        
        logger.info(f"Building enhanced vector index for {len(documents)} documents")
        
        # Generate embeddings
        embeddings = self.embed_documents(documents)
        if embeddings.size == 0:
            logger.error("Failed to generate embeddings")
            return False
        
        # Create FAISS index
        try:
            # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(embeddings)
            
            logger.info(f"FAISS index created with {self.index.ntotal} vectors")
            
            # Save index
            faiss.write_index(self.index, str(self.index_path))
            logger.info(f"Index saved to {self.index_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Index building failed: {e}")
            return False
    
    def load_index(self) -> bool:
        """
        Load existing FAISS index.
        
        Returns:
            True if index loaded successfully
        """
        if not self.index_path.exists():
            logger.warning(f"Index file not found: {self.index_path}")
            return False
        
        try:
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def search(self, query: str, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar documents.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            Tuple of (similarities, indices)
        """
        if self.index is None:
            logger.error("No index loaded for search")
            return np.array([]), np.array([])
        
        start_time = time.time()
        
        try:
            # Embed query with instruction
            query_embedding = self.embed_query(query)
            
            # Ensure query embedding is 2D
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Search
            similarities, indices = self.index.search(query_embedding, k)
            
            search_time = time.time() - start_time
            self.search_times.append(search_time)
            
            logger.debug(f"Search completed in {search_time:.3f}s, found {len(indices[0])} results")
            
            return similarities[0], indices[0]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return np.array([]), np.array([])
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings.
        
        Args:
            query_embedding: Query embedding vector
            doc_embeddings: Document embedding matrix
            
        Returns:
            Similarity scores
        """
        try:
            # Both are already normalized, so dot product = cosine similarity
            if query_embedding.ndim == 1:
                similarities = np.dot(doc_embeddings, query_embedding)
            else:
                similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
            
            return similarities
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return np.array([])
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return model information for compatibility checking.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'instruction_template': self.query_instruction,
            'normalization_enabled': True,
            'index_type': 'IndexFlatIP',
            'total_vectors': self.index.ntotal if self.index else 0
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            'total_embeddings': len(self.embedding_times),
            'total_searches': len(self.search_times),
            'avg_embedding_time': np.mean(self.embedding_times) if self.embedding_times else 0,
            'avg_search_time': np.mean(self.search_times) if self.search_times else 0
        }
        
        if self.embedding_times:
            stats.update({
                'min_embedding_time': np.min(self.embedding_times),
                'max_embedding_time': np.max(self.embedding_times),
                'std_embedding_time': np.std(self.embedding_times)
            })
        
        if self.search_times:
            stats.update({
                'min_search_time': np.min(self.search_times),
                'max_search_time': np.max(self.search_times),
                'std_search_time': np.std(self.search_times)
            })
        
        return stats
    
    def clear_performance_stats(self):
        """Clear performance tracking data."""
        self.embedding_times.clear()
        self.search_times.clear()
    
    def validate_embeddings(self, embeddings: np.ndarray) -> bool:
        """
        Validate embedding array.
        
        Args:
            embeddings: Embedding array to validate
            
        Returns:
            True if embeddings are valid
        """
        if embeddings.size == 0:
            logger.error("Empty embeddings array")
            return False
        
        if embeddings.ndim != 2:
            logger.error(f"Expected 2D embeddings, got {embeddings.ndim}D")
            return False
        
        if embeddings.shape[1] != self.embedding_dim:
            logger.error(f"Embedding dimension mismatch: {embeddings.shape[1]} != {self.embedding_dim}")
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            logger.error("Embeddings contain NaN or infinite values")
            return False
        
        return True
    
    def optimize_index(self):
        """Optimize the FAISS index for better performance."""
        if self.index is None:
            logger.warning("No index to optimize")
            return
        
        try:
            # For larger indices, could use IndexIVFFlat or other optimized indices
            original_total = self.index.ntotal
            
            if original_total > 10000:
                logger.info("Converting to optimized index for large dataset")
                # Create IVF index for better performance on large datasets
                nlist = min(4096, int(np.sqrt(original_total)))
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                optimized_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                
                # Train and populate the optimized index
                if hasattr(self.index, 'reconstruct_n'):
                    # Extract vectors from existing index
                    vectors = np.array([self.index.reconstruct(i) for i in range(original_total)])
                    optimized_index.train(vectors)
                    optimized_index.add(vectors)
                    
                    # Replace the index
                    self.index = optimized_index
                    logger.info(f"Index optimized for {original_total} vectors")
                
        except Exception as e:
            logger.warning(f"Index optimization failed: {e}")


# Backwards compatibility alias
BGEVectorStore = EnhancedVectorStore