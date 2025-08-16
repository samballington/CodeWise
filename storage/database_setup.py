"""
Database Setup for CodeWise Knowledge Graph

Sets up SQLite database with VSS extension for unified vector and graph storage.
Provides the foundation for local-first Knowledge Graph architecture.
"""

import sqlite3
import sys
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DatabaseSetup:
    """
    Initialize and configure SQLite database with VSS extension for Knowledge Graph.
    
    Architectural Decision: SQLite + VSS provides local-first, zero-dependency 
    solution combining ACID transactions, graph storage, and vector search.
    """
    
    def __init__(self, db_path: str = "codewise.db"):
        self.db_path = Path(db_path)
        self.connection = None
        self.vss_enabled = False
    
    def initialize_database(self) -> bool:
        """
        Initialize SQLite database with VSS extension.
        
        Returns:
            True if initialization successful
        """
        try:
            # Connect to database
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            
            # Enable foreign keys for referential integrity
            self.connection.execute("PRAGMA foreign_keys = ON")
            
            # Optimize SQLite for our workload
            self._optimize_sqlite_settings()
            
            # Load sqlite-vss extension
            self.vss_enabled = self._load_vss_extension()
            
            # Create schema
            self._create_schema()
            
            logger.info(f"Database initialized successfully at {self.db_path}")
            logger.info(f"VSS extension enabled: {self.vss_enabled}")
            
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            if self.connection:
                self.connection.close()
            raise
    
    def _optimize_sqlite_settings(self):
        """Optimize SQLite for Knowledge Graph workload."""
        optimizations = [
            "PRAGMA journal_mode = WAL",           # Write-Ahead Logging for better concurrency
            "PRAGMA synchronous = NORMAL",         # Balance between safety and performance
            "PRAGMA cache_size = -64000",          # 64MB cache for better query performance
            "PRAGMA temp_store = MEMORY",          # Store temporary tables in memory
            "PRAGMA mmap_size = 268435456",        # 256MB memory-mapped I/O
        ]
        
        for pragma in optimizations:
            try:
                self.connection.execute(pragma)
                logger.debug(f"Applied optimization: {pragma}")
            except Exception as e:
                logger.warning(f"Could not apply optimization '{pragma}': {e}")
    
    def _load_vss_extension(self) -> bool:
        """
        Load sqlite-vss extension for vector search capabilities.
        
        Returns:
            True if VSS extension loaded successfully
        """
        try:
            # Try common extension locations for different platforms
            extension_paths = [
                "vss.so",           # Linux
                "vss.dll",          # Windows
                "vss.dylib",        # macOS
                "sqlite-vss/vss.so",
                "sqlite-vss/vss.dll",
                "sqlite-vss/vss.dylib",
                "./vss.so",
                "./vss.dll",
                "./vss.dylib"
            ]
            
            self.connection.enable_load_extension(True)
            
            for ext_path in extension_paths:
                if Path(ext_path).exists():
                    try:
                        self.connection.load_extension(ext_path)
                        logger.info(f"Loaded VSS extension from {ext_path}")
                        return True
                    except Exception as e:
                        logger.debug(f"Failed to load {ext_path}: {e}")
                        continue
            
            # Test if VSS is available without explicit loading (system-wide install)
            try:
                self.connection.execute("SELECT vss_version()")
                logger.info("VSS extension available system-wide")
                return True
            except:
                pass
            
            # Fallback: VSS not available
            logger.warning("VSS extension not found. Vector search will be disabled.")
            logger.warning("To enable vector search, install sqlite-vss: pip install sqlite-vss")
            
            return False
            
        except Exception as e:
            logger.warning(f"VSS extension load failed: {e}. Vector search disabled.")
            return False
    
    def _create_schema(self):
        """Create database schema for Knowledge Graph storage."""
        cursor = self.connection.cursor()
        
        # Core schema for Knowledge Graph
        schema_sql = """
        -- Nodes table: stores all code entities (functions, classes, modules)
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL, -- function, class, module, variable, import
            name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            line_start INTEGER,
            line_end INTEGER,
            signature TEXT, -- function signatures, class definitions
            docstring TEXT,
            properties JSON, -- flexible storage for language-specific attributes
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Edges table: stores relationships between nodes (calls, inherits, imports)
        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            type TEXT NOT NULL, -- calls, inherits, imports, defines, contains
            properties JSON, -- additional relationship metadata
            file_path TEXT, -- where this relationship is observed
            line_number INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_id) REFERENCES nodes (id) ON DELETE CASCADE,
            FOREIGN KEY (target_id) REFERENCES nodes (id) ON DELETE CASCADE
        );
        
        -- Chunks table: stores text chunks for vector search
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            file_path TEXT NOT NULL,
            chunk_type TEXT, -- symbol, block, summary (from Phase 1)
            line_start INTEGER,
            line_end INTEGER,
            metadata JSON, -- hierarchical chunk metadata from Phase 1
            node_id TEXT, -- link to corresponding node if applicable
            embedding_vector BLOB, -- for vector search (when VSS unavailable)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (node_id) REFERENCES nodes (id) ON DELETE SET NULL
        );
        
        -- Files table: track file-level metadata and processing status
        CREATE TABLE IF NOT EXISTS files (
            file_path TEXT PRIMARY KEY,
            language TEXT,
            last_modified TIMESTAMP,
            processing_status TEXT DEFAULT 'pending', -- pending, processing, completed, error
            error_message TEXT,
            nodes_count INTEGER DEFAULT 0,
            chunks_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Indexes for performance optimization
        CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
        CREATE INDEX IF NOT EXISTS idx_nodes_file_path ON nodes(file_path);
        CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
        CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
        CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
        CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type);
        CREATE INDEX IF NOT EXISTS idx_edges_source_type ON edges(source_id, type);
        CREATE INDEX IF NOT EXISTS idx_edges_target_type ON edges(target_id, type);
        CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);
        CREATE INDEX IF NOT EXISTS idx_chunks_node_id ON chunks(node_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
        CREATE INDEX IF NOT EXISTS idx_files_status ON files(processing_status);
        CREATE INDEX IF NOT EXISTS idx_files_language ON files(language);
        """
        
        cursor.executescript(schema_sql)
        
        # Create VSS virtual table if extension is available
        if self.vss_enabled:
            try:
                # Updated for BGE embeddings (1024 dimensions)
                vss_sql = """
                CREATE VIRTUAL TABLE IF NOT EXISTS vss_chunks USING vss0(
                    embedding(1024), -- BGE-large-en-v1.5 dimensions
                    id UNINDEXED
                );
                """
                cursor.execute(vss_sql)
                logger.info("Created VSS virtual table for vector search")
            except Exception as e:
                logger.error(f"Failed to create VSS table: {e}")
                self.vss_enabled = False
        
        self.connection.commit()
        logger.info("Database schema created successfully")
    
    def test_database_functionality(self) -> dict:
        """
        Test database functionality and return status report.
        
        Returns:
            Dictionary with test results
        """
        test_results = {
            'database_connected': False,
            'schema_created': False,
            'vss_enabled': self.vss_enabled,
            'foreign_keys_enabled': False,
            'write_test_passed': False,
            'read_test_passed': False
        }
        
        try:
            # Test database connection
            if self.connection:
                test_results['database_connected'] = True
            
            # Test schema existence
            cursor = self.connection.cursor()
            tables = cursor.execute("""
                SELECT name FROM sqlite_master WHERE type='table'
            """).fetchall()
            
            required_tables = {'nodes', 'edges', 'chunks', 'files'}
            existing_tables = {row[0] for row in tables}
            
            if required_tables.issubset(existing_tables):
                test_results['schema_created'] = True
            
            # Test foreign keys
            fk_status = cursor.execute("PRAGMA foreign_keys").fetchone()
            if fk_status and fk_status[0] == 1:
                test_results['foreign_keys_enabled'] = True
            
            # Test write operation
            cursor.execute("""
                INSERT OR REPLACE INTO nodes (id, type, name, file_path) 
                VALUES ('test_node', 'function', 'test_function', 'test.py')
            """)
            self.connection.commit()
            test_results['write_test_passed'] = True
            
            # Test read operation
            result = cursor.execute("""
                SELECT * FROM nodes WHERE id = 'test_node'
            """).fetchone()
            
            if result:
                test_results['read_test_passed'] = True
            
            # Clean up test data
            cursor.execute("DELETE FROM nodes WHERE id = 'test_node'")
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Database functionality test failed: {e}")
        
        return test_results
    
    def get_database_stats(self) -> dict:
        """Get current database statistics."""
        if not self.connection:
            return {}
        
        cursor = self.connection.cursor()
        
        try:
            stats = {}
            
            # Table counts
            stats['nodes_count'] = cursor.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            stats['edges_count'] = cursor.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            stats['chunks_count'] = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            stats['files_count'] = cursor.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            
            # Node type distribution
            node_types = cursor.execute("""
                SELECT type, COUNT(*) FROM nodes GROUP BY type
            """).fetchall()
            stats['node_types'] = {row[0]: row[1] for row in node_types}
            
            # Edge type distribution
            edge_types = cursor.execute("""
                SELECT type, COUNT(*) FROM edges GROUP BY type
            """).fetchall()
            stats['edge_types'] = {row[0]: row[1] for row in edge_types}
            
            # Database file size
            stats['database_size_bytes'] = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.debug("Database connection closed")


def setup_codewise_database(db_path: str = "codewise.db") -> DatabaseSetup:
    """
    Convenience function to set up CodeWise database.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        Initialized DatabaseSetup instance
    """
    setup = DatabaseSetup(db_path)
    setup.initialize_database()
    return setup


if __name__ == "__main__":
    # CLI interface for database setup
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up CodeWise Knowledge Graph database")
    parser.add_argument("--db-path", default="codewise.db", help="Database file path")
    parser.add_argument("--test", action="store_true", help="Run functionality tests")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        setup = setup_codewise_database(args.db_path)
        
        if args.test:
            print("\nRunning database functionality tests...")
            test_results = setup.test_database_functionality()
            
            for test_name, passed in test_results.items():
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {test_name}: {status}")
        
        if args.stats:
            print("\nDatabase statistics:")
            stats = setup.get_database_stats()
            
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
        
        setup.close()
        print(f"\nDatabase setup completed successfully: {args.db_path}")
        
    except Exception as e:
        print(f"Database setup failed: {e}")
        sys.exit(1)