"""
Enhanced Metadata Storage for Hierarchical Chunks

Provides persistent storage for hierarchical chunk metadata with
bidirectional relationships, enabling cross-session context reconstruction.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from ..schemas.chunk_schemas import (
    AnyChunk, ChunkRelationship, SymbolChunk, BlockChunk, SummaryChunk
)

logger = logging.getLogger(__name__)


class EnhancedMetadataStore:
    """
    Enhanced metadata storage system with hierarchical relationship support.
    
    Stores complete bidirectional relationships for efficient context reconstruction
    while maintaining backward compatibility with existing vector store format.
    """
    
    def __init__(self, storage_dir: str = ".vector_cache"):
        """
        Initialize enhanced metadata store.
        
        Args:
            storage_dir: Directory for storing metadata files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # File paths
        self.meta_file = self.storage_dir / "meta.json"
        self.hierarchy_file = self.storage_dir / "hierarchy.json"
        self.relationships_file = self.storage_dir / "relationships.json"
        
        # In-memory caches
        self._chunk_cache: Dict[str, Dict] = {}
        self._relationship_cache: Dict[str, ChunkRelationship] = {}
        self._load_caches()
    
    def store_chunks(self, chunks: List[AnyChunk], vector_indices: List[int] = None) -> None:
        """
        Store hierarchical chunks with complete metadata.
        
        Args:
            chunks: List of hierarchical chunks to store
            vector_indices: Corresponding vector indices for each chunk
        """
        logger.info(f"Storing {len(chunks)} hierarchical chunks")
        
        # Prepare data structures
        meta_data = []
        hierarchy_data = {}
        relationships_data = {}
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            vector_index = vector_indices[i] if vector_indices else i
            
            # Basic metadata for vector store compatibility
            chunk_meta = {
                "chunk_id": chunk.id,
                "chunk_data": chunk.dict(),
                "vector_index": vector_index,
                "file_path": chunk.file_path,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "chunk_type": chunk.type.value,
                "created_at": datetime.utcnow().isoformat()
            }
            meta_data.append(chunk_meta)
            
            # Hierarchical structure for efficient lookup
            hierarchy_data[chunk.id] = {
                "type": chunk.type.value,
                "file_path": chunk.file_path,
                "line_range": [chunk.line_start, chunk.line_end],
                "parent_id": getattr(chunk, 'parent_chunk_id', None),
                "child_ids": getattr(chunk, 'child_chunk_ids', [])
            }
            
            # Relationship data for bidirectional traversal
            relationships_data[chunk.id] = self._build_relationship_data(chunk)
        
        # Store to files
        self._save_meta_data(meta_data)
        self._save_hierarchy_data(hierarchy_data)
        self._save_relationships_data(relationships_data)
        
        # Update caches
        self._update_caches(chunks)
        
        logger.info(f"Successfully stored {len(chunks)} chunks with relationships")
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Get chunk data by ID.
        
        Args:
            chunk_id: Unique chunk identifier
            
        Returns:
            Chunk data dictionary or None if not found
        """
        return self._chunk_cache.get(chunk_id)
    
    def get_chunks_by_file(self, file_path: str) -> List[Dict]:
        """
        Get all chunks for a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of chunk data dictionaries
        """
        file_chunks = []
        for chunk_data in self._chunk_cache.values():
            if chunk_data.get('file_path') == file_path:
                file_chunks.append(chunk_data)
        
        # Sort by line number
        file_chunks.sort(key=lambda x: x.get('line_start', 0))
        return file_chunks
    
    def get_chunks_by_type(self, chunk_type: str) -> List[Dict]:
        """
        Get all chunks of a specific type.
        
        Args:
            chunk_type: Type of chunks to retrieve (symbol, block, summary)
            
        Returns:
            List of chunk data dictionaries
        """
        type_chunks = []
        for chunk_data in self._chunk_cache.values():
            if chunk_data.get('chunk_type') == chunk_type:
                type_chunks.append(chunk_data)
        return type_chunks
    
    def get_relationship(self, chunk_id: str) -> Optional[ChunkRelationship]:
        """
        Get relationship data for a chunk.
        
        Args:
            chunk_id: Unique chunk identifier
            
        Returns:
            ChunkRelationship object or None if not found
        """
        return self._relationship_cache.get(chunk_id)
    
    def get_children(self, chunk_id: str) -> List[Dict]:
        """
        Get all child chunks for a given chunk.
        
        Args:
            chunk_id: Parent chunk identifier
            
        Returns:
            List of child chunk data dictionaries
        """
        relationship = self.get_relationship(chunk_id)
        if not relationship:
            return []
        
        children = []
        for child_id in relationship.child_ids:
            child_data = self.get_chunk_by_id(child_id)
            if child_data:
                children.append(child_data)
        
        return children
    
    def get_parent(self, chunk_id: str) -> Optional[Dict]:
        """
        Get parent chunk for a given chunk.
        
        Args:
            chunk_id: Child chunk identifier
            
        Returns:
            Parent chunk data dictionary or None if no parent
        """
        relationship = self.get_relationship(chunk_id)
        if not relationship or not relationship.parent_id:
            return None
        
        return self.get_chunk_by_id(relationship.parent_id)
    
    def get_siblings(self, chunk_id: str) -> List[Dict]:
        """
        Get sibling chunks for a given chunk.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            List of sibling chunk data dictionaries
        """
        # Get parent, then get all children of parent (excluding self)
        parent = self.get_parent(chunk_id)
        if not parent:
            return []
        
        siblings = []
        for child in self.get_children(parent['chunk_id']):
            if child['chunk_id'] != chunk_id:
                siblings.append(child)
        
        return siblings
    
    def update_chunk(self, chunk_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update chunk metadata.
        
        Args:
            chunk_id: Chunk identifier
            updates: Dictionary of fields to update
            
        Returns:
            True if update successful, False otherwise
        """
        if chunk_id not in self._chunk_cache:
            return False
        
        # Update cache
        self._chunk_cache[chunk_id].update(updates)
        
        # Persist changes
        self._save_current_caches()
        
        logger.debug(f"Updated chunk {chunk_id} with {len(updates)} fields")
        return True
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk and update relationships.
        
        Args:
            chunk_id: Chunk identifier to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        if chunk_id not in self._chunk_cache:
            return False
        
        # Remove from caches
        del self._chunk_cache[chunk_id]
        if chunk_id in self._relationship_cache:
            relationship = self._relationship_cache[chunk_id]
            
            # Update parent's child list
            if relationship.parent_id:
                parent_rel = self._relationship_cache.get(relationship.parent_id)
                if parent_rel and chunk_id in parent_rel.child_ids:
                    parent_rel.child_ids.remove(chunk_id)
            
            # Update children's parent reference
            for child_id in relationship.child_ids:
                child_rel = self._relationship_cache.get(child_id)
                if child_rel:
                    child_rel.parent_id = relationship.parent_id
            
            del self._relationship_cache[chunk_id]
        
        # Persist changes
        self._save_current_caches()
        
        logger.debug(f"Deleted chunk {chunk_id}")
        return True
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored chunks.
        
        Returns:
            Dictionary with storage statistics
        """
        type_counts = {}
        for chunk_data in self._chunk_cache.values():
            chunk_type = chunk_data.get('chunk_type', 'unknown')
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        return {
            'total_chunks': len(self._chunk_cache),
            'type_distribution': type_counts,
            'total_relationships': len(self._relationship_cache),
            'storage_files': {
                'meta_file': self.meta_file.exists(),
                'hierarchy_file': self.hierarchy_file.exists(),
                'relationships_file': self.relationships_file.exists()
            }
        }
    
    def _build_relationship_data(self, chunk: AnyChunk) -> ChunkRelationship:
        """Build relationship data from chunk."""
        parent_id = getattr(chunk, 'parent_chunk_id', None)
        child_ids = getattr(chunk, 'child_chunk_ids', [])
        
        return ChunkRelationship(
            chunk_id=chunk.id,
            parent_id=parent_id,
            child_ids=child_ids,
            sibling_ids=[],  # Will be computed during relationship building
            chunk_type=chunk.type
        )
    
    def _save_meta_data(self, meta_data: List[Dict]) -> None:
        """Save metadata in vector store compatible format."""
        with open(self.meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)
    
    def _save_hierarchy_data(self, hierarchy_data: Dict) -> None:
        """Save hierarchical structure data."""
        with open(self.hierarchy_file, 'w', encoding='utf-8') as f:
            json.dump(hierarchy_data, f, indent=2, ensure_ascii=False)
    
    def _save_relationships_data(self, relationships_data: Dict) -> None:
        """Save relationship data."""
        # Convert ChunkRelationship objects to dictionaries for JSON storage
        serializable_data = {}
        for chunk_id, relationship in relationships_data.items():
            if isinstance(relationship, ChunkRelationship):
                serializable_data[chunk_id] = relationship.dict()
            else:
                serializable_data[chunk_id] = relationship
        
        with open(self.relationships_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    def _load_caches(self) -> None:
        """Load data into memory caches."""
        # Load chunk cache from meta file
        if self.meta_file.exists():
            try:
                with open(self.meta_file, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                
                for item in meta_data:
                    if isinstance(item, dict) and 'chunk_id' in item:
                        self._chunk_cache[item['chunk_id']] = item
                    
                logger.debug(f"Loaded {len(self._chunk_cache)} chunks into cache")
            except Exception as e:
                logger.warning(f"Failed to load meta cache: {e}")
        
        # Load relationship cache
        if self.relationships_file.exists():
            try:
                with open(self.relationships_file, 'r', encoding='utf-8') as f:
                    relationships_data = json.load(f)
                
                for chunk_id, rel_data in relationships_data.items():
                    try:
                        self._relationship_cache[chunk_id] = ChunkRelationship(**rel_data)
                    except Exception as e:
                        logger.warning(f"Failed to load relationship for {chunk_id}: {e}")
                
                logger.debug(f"Loaded {len(self._relationship_cache)} relationships into cache")
            except Exception as e:
                logger.warning(f"Failed to load relationship cache: {e}")
    
    def _update_caches(self, chunks: List[AnyChunk]) -> None:
        """Update memory caches with new chunks."""
        for chunk in chunks:
            # Update chunk cache
            chunk_data = {
                "chunk_id": chunk.id,
                "chunk_data": chunk.dict(),
                "file_path": chunk.file_path,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "chunk_type": chunk.type.value
            }
            self._chunk_cache[chunk.id] = chunk_data
            
            # Update relationship cache
            self._relationship_cache[chunk.id] = self._build_relationship_data(chunk)
    
    def _save_current_caches(self) -> None:
        """Save current cache state to files."""
        # Save meta data
        meta_data = list(self._chunk_cache.values())
        self._save_meta_data(meta_data)
        
        # Save relationships
        relationships_data = {
            chunk_id: rel.dict() for chunk_id, rel in self._relationship_cache.items()
        }
        self._save_relationships_data(relationships_data)