"""
Context Reconstructor for Hierarchical Chunks

Provides utilities for reconstructing hierarchical context from stored
bidirectional relationships without re-parsing AST.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass

from .enhanced_metadata_store import EnhancedMetadataStore

logger = logging.getLogger(__name__)


@dataclass
class ContextPath:
    """Represents a path through the hierarchical context."""
    chunks: List[Dict]
    depth: int
    total_lines: int
    
    def __len__(self) -> int:
        return len(self.chunks)


@dataclass
class HierarchicalContext:
    """Complete hierarchical context for a chunk."""
    target_chunk: Dict
    ancestors: List[Dict]  # From root to immediate parent
    descendants: List[Dict]  # Direct children and their subtrees
    siblings: List[Dict]  # Chunks at same level
    full_hierarchy: List[Dict]  # Complete tree path
    context_summary: str
    
    @property
    def depth(self) -> int:
        """Get depth of target chunk in hierarchy."""
        return len(self.ancestors)
    
    @property
    def total_chunks(self) -> int:
        """Get total number of chunks in context."""
        return len(self.full_hierarchy)


class ContextReconstructor:
    """
    Utility for reconstructing hierarchical context from stored relationships.
    
    Enables instant context traversal without AST re-parsing by leveraging
    persistent bidirectional relationships.
    """
    
    def __init__(self, metadata_store: EnhancedMetadataStore):
        """
        Initialize context reconstructor.
        
        Args:
            metadata_store: Enhanced metadata store with relationship data
        """
        self.metadata_store = metadata_store
    
    def get_full_context(self, chunk_id: str, max_depth: int = 10) -> Optional[HierarchicalContext]:
        """
        Reconstruct complete hierarchical context for a chunk.
        
        Args:
            chunk_id: Target chunk identifier
            max_depth: Maximum depth to traverse (prevents infinite loops)
            
        Returns:
            HierarchicalContext object or None if chunk not found
        """
        target_chunk = self.metadata_store.get_chunk_by_id(chunk_id)
        if not target_chunk:
            logger.warning(f"Chunk {chunk_id} not found for context reconstruction")
            return None
        
        logger.debug(f"Reconstructing context for chunk {chunk_id}")
        
        # Traverse UP to root (using backward links)
        ancestors = self._get_ancestors(chunk_id, max_depth)
        
        # Traverse DOWN to children (using forward links)
        descendants = self._get_descendants(chunk_id, max_depth)
        
        # Get siblings at same level
        siblings = self.metadata_store.get_siblings(chunk_id)
        
        # Build full hierarchy path
        full_hierarchy = ancestors + [target_chunk] + descendants
        
        # Generate context summary
        context_summary = self._generate_context_summary(target_chunk, ancestors, descendants)
        
        return HierarchicalContext(
            target_chunk=target_chunk,
            ancestors=ancestors,
            descendants=descendants,
            siblings=siblings,
            full_hierarchy=full_hierarchy,
            context_summary=context_summary
        )
    
    def get_ancestors(self, chunk_id: str, max_depth: int = 10) -> List[Dict]:
        """
        Get all ancestor chunks from root to immediate parent.
        
        Args:
            chunk_id: Target chunk identifier
            max_depth: Maximum depth to traverse
            
        Returns:
            List of ancestor chunks (root to parent order)
        """
        return self._get_ancestors(chunk_id, max_depth)
    
    def get_descendants(self, chunk_id: str, max_depth: int = 5) -> List[Dict]:
        """
        Get all descendant chunks in depth-first order.
        
        Args:
            chunk_id: Target chunk identifier
            max_depth: Maximum depth to traverse
            
        Returns:
            List of descendant chunks
        """
        return self._get_descendants(chunk_id, max_depth)
    
    def get_context_path(self, chunk_id: str, target_chunk_id: str) -> Optional[ContextPath]:
        """
        Find the shortest path between two chunks in the hierarchy.
        
        Args:
            chunk_id: Starting chunk identifier
            target_chunk_id: Target chunk identifier
            
        Returns:
            ContextPath object or None if no path exists
        """
        # Get ancestors for both chunks
        chunk_ancestors = self._get_ancestors(chunk_id)
        target_ancestors = self._get_ancestors(target_chunk_id)
        
        # Find common ancestor
        common_ancestor = self._find_common_ancestor(chunk_ancestors, target_ancestors)
        if not common_ancestor:
            return None
        
        # Build path: chunk -> common ancestor -> target
        path_chunks = []
        
        # Path from chunk to common ancestor (reverse)
        chunk_to_common = []
        current = self.metadata_store.get_chunk_by_id(chunk_id)
        while current and current['chunk_id'] != common_ancestor['chunk_id']:
            chunk_to_common.append(current)
            parent = self.metadata_store.get_parent(current['chunk_id'])
            current = parent
        
        # Path from common ancestor to target
        common_to_target = []
        current = self.metadata_store.get_chunk_by_id(target_chunk_id)
        while current and current['chunk_id'] != common_ancestor['chunk_id']:
            common_to_target.insert(0, current)  # Insert at beginning
            parent = self.metadata_store.get_parent(current['chunk_id'])
            current = parent
        
        # Combine paths
        path_chunks = chunk_to_common + [common_ancestor] + common_to_target
        
        total_lines = sum(
            chunk.get('line_end', 0) - chunk.get('line_start', 0) + 1 
            for chunk in path_chunks
        )
        
        return ContextPath(
            chunks=path_chunks,
            depth=len(path_chunks),
            total_lines=total_lines
        )
    
    def get_related_symbols(self, chunk_id: str, relation_types: List[str] = None) -> List[Dict]:
        """
        Get symbols related to a chunk through various relationships.
        
        Args:
            chunk_id: Target chunk identifier
            relation_types: Types of relations to consider (parent, child, sibling)
            
        Returns:
            List of related symbol chunks
        """
        if relation_types is None:
            relation_types = ['parent', 'child', 'sibling']
        
        related_symbols = []
        
        # Get chunks based on requested relation types
        if 'parent' in relation_types:
            parent = self.metadata_store.get_parent(chunk_id)
            if parent and parent.get('chunk_type') == 'symbol':
                related_symbols.append(parent)
        
        if 'child' in relation_types:
            children = self.metadata_store.get_children(chunk_id)
            for child in children:
                if child.get('chunk_type') == 'symbol':
                    related_symbols.append(child)
        
        if 'sibling' in relation_types:
            siblings = self.metadata_store.get_siblings(chunk_id)
            for sibling in siblings:
                if sibling.get('chunk_type') == 'symbol':
                    related_symbols.append(sibling)
        
        # Deduplicate by chunk_id
        seen_ids = set()
        unique_symbols = []
        for symbol in related_symbols:
            if symbol['chunk_id'] not in seen_ids:
                unique_symbols.append(symbol)
                seen_ids.add(symbol['chunk_id'])
        
        return unique_symbols
    
    def get_file_context(self, file_path: str) -> Dict[str, Any]:
        """
        Get complete hierarchical context for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file's hierarchical structure
        """
        file_chunks = self.metadata_store.get_chunks_by_file(file_path)
        if not file_chunks:
            return {}
        
        # Find file summary chunk
        summary_chunk = None
        symbol_chunks = []
        block_chunks = []
        
        for chunk in file_chunks:
            chunk_type = chunk.get('chunk_type')
            if chunk_type == 'summary':
                summary_chunk = chunk
            elif chunk_type == 'symbol':
                symbol_chunks.append(chunk)
            elif chunk_type == 'block':
                block_chunks.append(chunk)
        
        return {
            'file_path': file_path,
            'summary': summary_chunk,
            'symbols': symbol_chunks,
            'blocks': block_chunks,
            'total_chunks': len(file_chunks),
            'hierarchy_depth': self._calculate_file_depth(file_chunks)
        }
    
    def search_in_context(self, chunk_id: str, search_term: str, 
                         context_radius: int = 2) -> List[Dict]:
        """
        Search for chunks containing a term within hierarchical context.
        
        Args:
            chunk_id: Center chunk for context search
            search_term: Term to search for
            context_radius: Number of levels up/down to search
            
        Returns:
            List of matching chunks with context information
        """
        # Get hierarchical context
        context = self.get_full_context(chunk_id, max_depth=context_radius)
        if not context:
            return []
        
        matches = []
        search_term_lower = search_term.lower()
        
        # Search in full hierarchy
        for chunk in context.full_hierarchy:
            chunk_data = chunk.get('chunk_data', {})
            content = chunk_data.get('content', '')
            
            if search_term_lower in content.lower():
                match_info = {
                    'chunk': chunk,
                    'match_type': 'content',
                    'context_distance': self._calculate_context_distance(
                        chunk['chunk_id'], chunk_id, context
                    )
                }
                matches.append(match_info)
            
            # Also search in symbol names
            symbol_name = chunk_data.get('symbol_name', '')
            if symbol_name and search_term_lower in symbol_name.lower():
                match_info = {
                    'chunk': chunk,
                    'match_type': 'symbol_name',
                    'context_distance': self._calculate_context_distance(
                        chunk['chunk_id'], chunk_id, context
                    )
                }
                matches.append(match_info)
        
        # Sort by context distance (closer chunks first)
        matches.sort(key=lambda x: x['context_distance'])
        return matches
    
    def _get_ancestors(self, chunk_id: str, max_depth: int = 10) -> List[Dict]:
        """Get all ancestor chunks from root to immediate parent."""
        ancestors = []
        current_id = chunk_id
        visited = set()
        
        for _ in range(max_depth):
            if current_id in visited:
                logger.warning(f"Circular reference detected in ancestors of {chunk_id}")
                break
            visited.add(current_id)
            
            parent = self.metadata_store.get_parent(current_id)
            if not parent:
                break
            
            ancestors.insert(0, parent)  # Insert at beginning for root-to-parent order
            current_id = parent['chunk_id']
        
        return ancestors
    
    def _get_descendants(self, chunk_id: str, max_depth: int = 5) -> List[Dict]:
        """Get all descendant chunks in depth-first order."""
        descendants = []
        visited = set()
        
        def _traverse_children(current_id: str, current_depth: int):
            if current_depth >= max_depth or current_id in visited:
                return
            visited.add(current_id)
            
            children = self.metadata_store.get_children(current_id)
            for child in children:
                descendants.append(child)
                _traverse_children(child['chunk_id'], current_depth + 1)
        
        _traverse_children(chunk_id, 0)
        return descendants
    
    def _find_common_ancestor(self, ancestors1: List[Dict], 
                             ancestors2: List[Dict]) -> Optional[Dict]:
        """Find the lowest common ancestor between two ancestor chains."""
        # Convert to sets of chunk IDs for efficient lookup
        ids1 = {chunk['chunk_id'] for chunk in ancestors1}
        ids2 = {chunk['chunk_id'] for chunk in ancestors2}
        
        # Find common ancestor IDs
        common_ids = ids1.intersection(ids2)
        if not common_ids:
            return None
        
        # Find the lowest (deepest) common ancestor
        for chunk in reversed(ancestors1):  # Start from deepest
            if chunk['chunk_id'] in common_ids:
                return chunk
        
        return None
    
    def _generate_context_summary(self, target_chunk: Dict, 
                                ancestors: List[Dict], 
                                descendants: List[Dict]) -> str:
        """Generate a textual summary of the hierarchical context."""
        summary_parts = []
        
        # Target chunk info
        chunk_type = target_chunk.get('chunk_type', 'unknown')
        chunk_name = target_chunk.get('chunk_data', {}).get('symbol_name', 'unnamed')
        summary_parts.append(f"Target: {chunk_type} '{chunk_name}'")
        
        # Ancestor context
        if ancestors:
            ancestor_names = [
                chunk.get('chunk_data', {}).get('symbol_name', 'unnamed')
                for chunk in ancestors
            ]
            summary_parts.append(f"Context path: {' > '.join(ancestor_names)}")
        
        # Descendant info
        if descendants:
            symbol_count = sum(1 for d in descendants if d.get('chunk_type') == 'symbol')
            block_count = sum(1 for d in descendants if d.get('chunk_type') == 'block')
            summary_parts.append(f"Contains: {symbol_count} symbols, {block_count} blocks")
        
        return "; ".join(summary_parts)
    
    def _calculate_file_depth(self, file_chunks: List[Dict]) -> int:
        """Calculate the maximum hierarchy depth for a file."""
        max_depth = 0
        
        for chunk in file_chunks:
            ancestors = self._get_ancestors(chunk['chunk_id'])
            depth = len(ancestors)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_context_distance(self, chunk_id: str, center_id: str, 
                                   context: HierarchicalContext) -> int:
        """Calculate distance between chunks in hierarchical context."""
        # Find positions in full hierarchy
        chunk_pos = None
        center_pos = None
        
        for i, chunk in enumerate(context.full_hierarchy):
            if chunk['chunk_id'] == chunk_id:
                chunk_pos = i
            if chunk['chunk_id'] == center_id:
                center_pos = i
        
        if chunk_pos is None or center_pos is None:
            return float('inf')
        
        return abs(chunk_pos - center_pos)