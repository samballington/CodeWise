"""
Database Manager for CodeWise Knowledge Graph

Provides high-level interface for graph operations, vector search, and 
transactional data management. Core operational layer for Phase 2 KG architecture.
"""

import sqlite3
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    High-level interface for Knowledge Graph database operations.
    
    Architectural Pattern: Repository pattern providing clean abstraction
    over SQLite + VSS for graph relationships and vector operations.
    """
    
    def __init__(self, db_path: str = "codewise.db"):
        self.db_path = Path(db_path)
        self.connection = None
        self.vss_enabled = False
        self._connect()
    
    def _connect(self):
        """Establish database connection with optimizations."""
        try:
            self.connection = sqlite3.connect(
                self.db_path, 
                check_same_thread=False,
                timeout=30.0  # 30 second timeout for busy database
            )
            self.connection.row_factory = sqlite3.Row
            
            # Enable foreign keys and optimizations
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA journal_mode = WAL")
            
            # Check if VSS is available
            try:
                self.connection.execute("SELECT vss_version()")
                self.vss_enabled = True
                logger.debug("VSS extension available")
            except:
                self.vss_enabled = False
                logger.debug("VSS extension not available")
                
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions with automatic rollback."""
        cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Transaction failed, rolled back: {e}")
            raise
    
    # ==================== NODE OPERATIONS ====================
    
    def insert_node(self, node_id: str, node_type: str, name: str, 
                   file_path: str, **kwargs) -> bool:
        """
        Insert or update a node in the Knowledge Graph.
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of node (function, class, module, variable, import)
            name: Display name of the symbol
            file_path: Path to source file
            **kwargs: Additional properties (line_start, line_end, signature, docstring, properties)
            
        Returns:
            True if operation successful
        """
        try:
            with self.transaction() as cursor:
                cursor.execute("""
                    INSERT OR REPLACE INTO nodes 
                    (id, type, name, file_path, line_start, line_end, signature, docstring, properties, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    node_id, 
                    node_type, 
                    name, 
                    file_path,
                    kwargs.get('line_start'), 
                    kwargs.get('line_end'),
                    kwargs.get('signature'), 
                    kwargs.get('docstring'),
                    json.dumps(kwargs.get('properties', {}))
                ))
            
            logger.debug(f"Inserted/updated node: {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert node {node_id}: {e}")
            return False
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        """
        Retrieve a node by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node dictionary or None if not found
        """
        try:
            cursor = self.connection.cursor()
            result = cursor.execute(
                "SELECT * FROM nodes WHERE id = ?", 
                (node_id,)
            ).fetchone()
            
            if result:
                node = dict(result)
                node['properties'] = json.loads(node['properties'] or '{}')
                return node
            return None
            
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            return None
    
    def get_nodes_by_type(self, node_type: str, limit: int = None) -> List[Dict]:
        """Get all nodes of a specific type."""
        try:
            cursor = self.connection.cursor()
            
            query = "SELECT * FROM nodes WHERE type = ?"
            params = [node_type]
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            results = cursor.execute(query, params).fetchall()
            
            nodes = []
            for result in results:
                node = dict(result)
                node['properties'] = json.loads(node['properties'] or '{}')
                nodes.append(node)
            
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to get nodes by type {node_type}: {e}")
            return []
    
    def get_nodes_by_name(self, name: str, exact_match: bool = True) -> List[Dict]:
        """
        Find nodes by name.
        
        Args:
            name: Symbol name to search for
            exact_match: If True, exact match; if False, partial match
            
        Returns:
            List of matching nodes
        """
        try:
            cursor = self.connection.cursor()
            
            if exact_match:
                query = "SELECT * FROM nodes WHERE name = ?"
                params = [name]
            else:
                query = "SELECT * FROM nodes WHERE name LIKE ?"
                params = [f"%{name}%"]
            
            results = cursor.execute(query, params).fetchall()
            
            nodes = []
            for result in results:
                node = dict(result)
                node['properties'] = json.loads(node['properties'] or '{}')
                nodes.append(node)
            
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to search nodes by name {name}: {e}")
            return []
    
    def get_nodes_by_file(self, file_path: str) -> List[Dict]:
        """Get all nodes in a specific file."""
        try:
            cursor = self.connection.cursor()
            results = cursor.execute(
                "SELECT * FROM nodes WHERE file_path = ? ORDER BY line_start", 
                (file_path,)
            ).fetchall()
            
            nodes = []
            for result in results:
                node = dict(result)
                node['properties'] = json.loads(node['properties'] or '{}')
                nodes.append(node)
            
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to get nodes by file {file_path}: {e}")
            return []
    
    def get_all_nodes(self, limit: int = None) -> List[Dict]:
        """Get all nodes in the database."""
        try:
            cursor = self.connection.cursor()
            
            query = "SELECT * FROM nodes ORDER BY file_path, line_start"
            if limit:
                query += f" LIMIT {limit}"
            
            results = cursor.execute(query).fetchall()
            
            nodes = []
            for result in results:
                node = dict(result)
                node['properties'] = json.loads(node['properties'] or '{}')
                nodes.append(node)
            
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to get all nodes: {e}")
            return []
    
    # ==================== EDGE OPERATIONS ====================
    
    def insert_edge(self, source_id: str, target_id: str, edge_type: str, 
                   file_path: str = None, **kwargs) -> bool:
        """
        Insert a relationship edge between nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID  
            edge_type: Relationship type (calls, inherits, imports, defines, contains)
            file_path: File where relationship is observed
            **kwargs: Additional properties (line_number, properties)
            
        Returns:
            True if operation successful
        """
        try:
            with self.transaction() as cursor:
                cursor.execute("""
                    INSERT OR IGNORE INTO edges 
                    (source_id, target_id, type, properties, file_path, line_number)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    source_id, 
                    target_id, 
                    edge_type,
                    json.dumps(kwargs.get('properties', {})),
                    file_path, 
                    kwargs.get('line_number')
                ))
            
            logger.debug(f"Inserted edge: {source_id} --{edge_type}--> {target_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert edge {source_id}->{target_id}: {e}")
            return False
    
    def get_outgoing_edges(self, node_id: str, edge_type: str = None) -> List[Dict]:
        """
        Get all outgoing edges from a node.
        
        Args:
            node_id: Source node ID
            edge_type: Optional filter by edge type
            
        Returns:
            List of edge dictionaries
        """
        try:
            cursor = self.connection.cursor()
            
            if edge_type:
                query = "SELECT * FROM edges WHERE source_id = ? AND type = ?"
                params = [node_id, edge_type]
            else:
                query = "SELECT * FROM edges WHERE source_id = ?"
                params = [node_id]
            
            results = cursor.execute(query, params).fetchall()
            
            edges = []
            for result in results:
                edge = dict(result)
                edge['properties'] = json.loads(edge['properties'] or '{}')
                edges.append(edge)
            
            return edges
            
        except Exception as e:
            logger.error(f"Failed to get outgoing edges for {node_id}: {e}")
            return []
    
    def get_incoming_edges(self, node_id: str, edge_type: str = None) -> List[Dict]:
        """Get all incoming edges to a node."""
        try:
            cursor = self.connection.cursor()
            
            if edge_type:
                query = "SELECT * FROM edges WHERE target_id = ? AND type = ?"
                params = [node_id, edge_type]
            else:
                query = "SELECT * FROM edges WHERE target_id = ?"
                params = [node_id]
            
            results = cursor.execute(query, params).fetchall()
            
            edges = []
            for result in results:
                edge = dict(result)
                edge['properties'] = json.loads(edge['properties'] or '{}')
                edges.append(edge)
            
            return edges
            
        except Exception as e:
            logger.error(f"Failed to get incoming edges for {node_id}: {e}")
            return []
    
    # ==================== GRAPH TRAVERSAL QUERIES ====================
    
    def find_callers(self, function_id: str, max_depth: int = 3) -> List[Dict]:
        """
        Find all functions that call this function using recursive SQL.
        
        Args:
            function_id: Target function node ID
            max_depth: Maximum traversal depth
            
        Returns:
            List of caller nodes with depth information
        """
        try:
            cursor = self.connection.cursor()
            results = cursor.execute("""
                WITH RECURSIVE callers(id, name, type, file_path, depth) AS (
                    -- Base case: the target function
                    SELECT n.id, n.name, n.type, n.file_path, 0
                    FROM nodes n WHERE n.id = ?
                    
                    UNION ALL
                    
                    -- Recursive case: functions that call nodes in our current set
                    SELECT n.id, n.name, n.type, n.file_path, c.depth + 1
                    FROM nodes n
                    JOIN edges e ON e.source_id = n.id
                    JOIN callers c ON e.target_id = c.id
                    WHERE e.type = 'calls' AND c.depth < ?
                )
                SELECT DISTINCT id, name, type, file_path, depth 
                FROM callers 
                WHERE depth > 0
                ORDER BY depth, name
            """, (function_id, max_depth)).fetchall()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to find callers for {function_id}: {e}")
            return []
    
    def find_dependencies(self, node_id: str, max_depth: int = 2) -> List[Dict]:
        """
        Find all nodes this node depends on through calls, imports, inheritance.
        
        Args:
            node_id: Source node ID
            max_depth: Maximum traversal depth
            
        Returns:
            List of dependency nodes with depth information
        """
        try:
            cursor = self.connection.cursor()
            results = cursor.execute("""
                WITH RECURSIVE deps(id, name, type, file_path, depth, relationship) AS (
                    -- Base case: the source node
                    SELECT n.id, n.name, n.type, n.file_path, 0, 'self'
                    FROM nodes n WHERE n.id = ?
                    
                    UNION ALL
                    
                    -- Recursive case: nodes that our current set depends on
                    SELECT n.id, n.name, n.type, n.file_path, d.depth + 1, e.type
                    FROM nodes n
                    JOIN edges e ON e.target_id = n.id
                    JOIN deps d ON e.source_id = d.id
                    WHERE e.type IN ('calls', 'imports', 'inherits') AND d.depth < ?
                )
                SELECT DISTINCT id, name, type, file_path, depth, relationship
                FROM deps
                WHERE depth > 0
                ORDER BY depth, relationship, name
            """, (node_id, max_depth)).fetchall()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to find dependencies for {node_id}: {e}")
            return []
    
    def find_related_symbols(self, symbol_name: str, relationship_types: List[str] = None) -> Dict[str, List[Dict]]:
        """
        Find all symbols related to the given symbol through various relationships.
        
        Args:
            symbol_name: Name of the symbol to analyze
            relationship_types: List of relationship types to consider
            
        Returns:
            Dictionary mapping relationship types to lists of related symbols
        """
        if not relationship_types:
            relationship_types = ['calls', 'inherits', 'imports']
        
        # First find all nodes with this name
        symbol_nodes = self.get_nodes_by_name(symbol_name, exact_match=True)
        if not symbol_nodes:
            return {}
        
        related_symbols = {}
        
        for node in symbol_nodes:
            node_id = node['id']
            
            for rel_type in relationship_types:
                if rel_type not in related_symbols:
                    related_symbols[rel_type] = []
                
                # Get outgoing relationships
                outgoing = self.get_outgoing_edges(node_id, rel_type)
                for edge in outgoing:
                    target_node = self.get_node(edge['target_id'])
                    if target_node:
                        target_node['relationship_direction'] = 'outgoing'
                        target_node['relationship_context'] = edge
                        related_symbols[rel_type].append(target_node)
                
                # Get incoming relationships
                incoming = self.get_incoming_edges(node_id, rel_type)
                for edge in incoming:
                    source_node = self.get_node(edge['source_id'])
                    if source_node:
                        source_node['relationship_direction'] = 'incoming'
                        source_node['relationship_context'] = edge
                        related_symbols[rel_type].append(source_node)
        
        return related_symbols
    
    # ==================== CHUNK OPERATIONS ====================
    
    def insert_chunk(self, chunk_id: str, content: str, file_path: str, 
                    chunk_type: str = None, node_id: str = None, **kwargs) -> bool:
        """
        Insert a text chunk for vector search.
        
        Args:
            chunk_id: Unique chunk identifier
            content: Text content
            file_path: Source file path
            chunk_type: Type of chunk (symbol, block, summary)
            node_id: Associated node ID if applicable
            **kwargs: Additional metadata (line_start, line_end, metadata, embedding_vector)
            
        Returns:
            True if operation successful
        """
        try:
            with self.transaction() as cursor:
                cursor.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (id, content, file_path, chunk_type, line_start, line_end, metadata, node_id, embedding_vector)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk_id,
                    content,
                    file_path,
                    chunk_type,
                    kwargs.get('line_start'),
                    kwargs.get('line_end'),
                    json.dumps(kwargs.get('metadata', {})),
                    node_id,
                    kwargs.get('embedding_vector')  # BLOB for fallback vector storage
                ))
            
            logger.debug(f"Inserted chunk: {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert chunk {chunk_id}: {e}")
            return False
    
    def get_chunks_by_node(self, node_id: str) -> List[Dict]:
        """Get all chunks associated with a node."""
        try:
            cursor = self.connection.cursor()
            results = cursor.execute(
                "SELECT * FROM chunks WHERE node_id = ?", 
                (node_id,)
            ).fetchall()
            
            chunks = []
            for result in results:
                chunk = dict(result)
                chunk['metadata'] = json.loads(chunk['metadata'] or '{}')
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to get chunks for node {node_id}: {e}")
            return []
    
    def get_chunks_by_file(self, file_path: str) -> List[Dict]:
        """Get all chunks in a file."""
        try:
            cursor = self.connection.cursor()
            results = cursor.execute(
                "SELECT * FROM chunks WHERE file_path = ? ORDER BY line_start", 
                (file_path,)
            ).fetchall()
            
            chunks = []
            for result in results:
                chunk = dict(result)
                chunk['metadata'] = json.loads(chunk['metadata'] or '{}')
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to get chunks for file {file_path}: {e}")
            return []
    
    # ==================== FILE TRACKING ====================
    
    def update_file_status(self, file_path: str, status: str, **kwargs) -> bool:
        """
        Update file processing status.
        
        Args:
            file_path: File path
            status: Processing status (pending, processing, completed, error)
            **kwargs: Additional fields (language, error_message, nodes_count, chunks_count)
            
        Returns:
            True if operation successful
        """
        try:
            with self.transaction() as cursor:
                cursor.execute("""
                    INSERT OR REPLACE INTO files 
                    (file_path, language, processing_status, error_message, nodes_count, chunks_count, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    file_path,
                    kwargs.get('language'),
                    status,
                    kwargs.get('error_message'),
                    kwargs.get('nodes_count', 0),
                    kwargs.get('chunks_count', 0)
                ))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update file status for {file_path}: {e}")
            return False
    
    def get_file_status(self, file_path: str) -> Optional[Dict]:
        """Get file processing status."""
        try:
            cursor = self.connection.cursor()
            result = cursor.execute(
                "SELECT * FROM files WHERE file_path = ?", 
                (file_path,)
            ).fetchone()
            
            return dict(result) if result else None
            
        except Exception as e:
            logger.error(f"Failed to get file status for {file_path}: {e}")
            return None
    
    def get_files_by_status(self, status: str) -> List[Dict]:
        """Get files by processing status."""
        try:
            cursor = self.connection.cursor()
            results = cursor.execute(
                "SELECT * FROM files WHERE processing_status = ?", 
                (status,)
            ).fetchall()
            
            return [dict(result) for result in results]
            
        except Exception as e:
            logger.error(f"Failed to get files by status {status}: {e}")
            return []
    
    # ==================== CLEANUP OPERATIONS ====================
    
    def delete_file_data(self, file_path: str) -> bool:
        """Delete all data associated with a file."""
        try:
            with self.transaction() as cursor:
                # Delete chunks
                cursor.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
                
                # Delete edges where the relationship was observed in this file
                cursor.execute("DELETE FROM edges WHERE file_path = ?", (file_path,))
                
                # Delete nodes in this file
                cursor.execute("DELETE FROM nodes WHERE file_path = ?", (file_path,))
                
                # Delete file record
                cursor.execute("DELETE FROM files WHERE file_path = ?", (file_path,))
            
            logger.info(f"Deleted all data for file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete data for file {file_path}: {e}")
            return False
    
    def cleanup_orphaned_edges(self) -> int:
        """Remove edges that reference non-existent nodes."""
        try:
            with self.transaction() as cursor:
                # Find and delete orphaned edges
                result = cursor.execute("""
                    DELETE FROM edges 
                    WHERE source_id NOT IN (SELECT id FROM nodes) 
                       OR target_id NOT IN (SELECT id FROM nodes)
                """)
                
                deleted_count = result.rowcount
                logger.info(f"Cleaned up {deleted_count} orphaned edges")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup orphaned edges: {e}")
            return 0
    
    # ==================== STATISTICS ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        try:
            cursor = self.connection.cursor()
            
            stats = {}
            
            # Basic counts
            stats['nodes_total'] = cursor.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            stats['edges_total'] = cursor.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            stats['chunks_total'] = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            stats['files_total'] = cursor.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            
            # Node type distribution
            node_types = cursor.execute("""
                SELECT type, COUNT(*) FROM nodes GROUP BY type ORDER BY COUNT(*) DESC
            """).fetchall()
            stats['node_types'] = {row[0]: row[1] for row in node_types}
            
            # Edge type distribution
            edge_types = cursor.execute("""
                SELECT type, COUNT(*) FROM edges GROUP BY type ORDER BY COUNT(*) DESC
            """).fetchall()
            stats['edge_types'] = {row[0]: row[1] for row in edge_types}
            
            # Language distribution
            languages = cursor.execute("""
                SELECT language, COUNT(*) FROM files WHERE language IS NOT NULL 
                GROUP BY language ORDER BY COUNT(*) DESC
            """).fetchall()
            stats['languages'] = {row[0]: row[1] for row in languages}
            
            # Processing status
            statuses = cursor.execute("""
                SELECT processing_status, COUNT(*) FROM files 
                GROUP BY processing_status ORDER BY COUNT(*) DESC
            """).fetchall()
            stats['processing_status'] = {row[0]: row[1] for row in statuses}
            
            # Database size
            if self.db_path.exists():
                stats['database_size_mb'] = round(self.db_path.stat().st_size / (1024 * 1024), 2)
            else:
                stats['database_size_mb'] = 0
            
            # VSS status
            stats['vss_enabled'] = self.vss_enabled
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database statistics: {e}")
            return {}
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.debug("Database connection closed")


# ==================== CONVENIENCE FUNCTIONS ====================

def get_database_manager(db_path: str = "codewise.db") -> DatabaseManager:
    """Get a DatabaseManager instance with connection verification."""
    manager = DatabaseManager(db_path)
    
    # Verify connection is working
    try:
        manager.connection.execute("SELECT 1")
        return manager
    except Exception as e:
        logger.error(f"Database connection verification failed: {e}")
        raise


if __name__ == "__main__":
    # CLI interface for database operations
    import argparse
    
    parser = argparse.ArgumentParser(description="CodeWise Database Manager")
    parser.add_argument("--db-path", default="codewise.db", help="Database file path")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup orphaned edges")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        manager = get_database_manager(args.db_path)
        
        if args.stats:
            print("\nDatabase Statistics:")
            stats = manager.get_statistics()
            
            for category, data in stats.items():
                if isinstance(data, dict):
                    print(f"\n{category.replace('_', ' ').title()}:")
                    for key, value in data.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"{category.replace('_', ' ').title()}: {data}")
        
        if args.cleanup:
            print("\nCleaning up orphaned edges...")
            deleted = manager.cleanup_orphaned_edges()
            print(f"Deleted {deleted} orphaned edges")
        
        manager.close()
        
    except Exception as e:
        print(f"Database operation failed: {e}")