"""
Unified Indexer - Integration of Two-Pass KG Pipeline with Phase 1 Components

Orchestrates the complete indexing workflow combining:
- Phase 1: Hierarchical chunking with BGE embeddings
- Phase 2: Two-pass AST-to-Graph pipeline for Knowledge Graph construction

Architectural Decision: KG pipeline runs BEFORE hierarchical chunking to provide
relationship context for better chunk organization and linking.
"""

from typing import List, Dict, Optional, Any, Set
from pathlib import Path
import logging
import time
from dataclasses import dataclass
from datetime import datetime

# Import Phase 1 and Phase 2 components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from storage.database_manager import DatabaseManager
from storage.database_setup import setup_codewise_database
from knowledge_graph.symbol_collector import SymbolCollector
from knowledge_graph.relationship_extractor import RelationshipExtractor
from indexer.enhanced_vector_store import EnhancedVectorStore
from indexer.chunkers.hierarchical_chunker import HierarchicalChunker

logger = logging.getLogger(__name__)


def safe_get(obj, key: str, default=None):
    """
    Safely get attribute/key from object - handles both dict and Pydantic objects.
    REQ-3.7.2: Fix SummaryChunk attribute errors
    """
    if hasattr(obj, 'get'):
        # Dictionary-like object
        return obj.get(key, default)
    elif hasattr(obj, key):
        # Object with attributes (like Pydantic models)
        return getattr(obj, key, default)
    else:
        return default


@dataclass
class IndexingResult:
    """Results from the unified indexing process."""
    success: bool
    files_processed: int
    files_failed: int
    symbols_discovered: int
    relationships_found: int
    chunks_created: int
    processing_time: float
    error_details: List[str]


@dataclass
class FileProcessingStats:
    """Statistics for a single file processing."""
    file_path: str
    success: bool
    symbols_count: int
    relationships_count: int
    chunks_count: int
    processing_time: float
    error_message: Optional[str] = None


class UnifiedIndexer:
    """
    Complete indexing pipeline with KG + hierarchical chunks + BGE embeddings.
    
    Integration Strategy:
    1. Pass 1: Symbol Collection (populates nodes table)
    2. Pass 2: Relationship Extraction (populates edges table) 
    3. Phase 1 Integration: Hierarchical Chunking with KG context (populates chunks table)
    4. Vector Embedding: BGE embeddings for enhanced semantic search
    
    Design Decision: This ordering ensures relationship context is available
    for intelligent chunk linking and organization.
    """
    
    def __init__(self, db_path: str = "codewise.db", vector_model: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize unified indexer with all components.
        
        Args:
            db_path: Path to SQLite database
            vector_model: BGE model for embeddings
        """
        # Initialize database
        self.db_path = db_path
        self.db_setup = setup_codewise_database(db_path)
        self.db_manager = DatabaseManager(db_path)
        
        # Initialize Phase 2 components
        self.symbol_collector = SymbolCollector(self.db_manager)
        self.relationship_extractor = None  # Created after symbol collection
        
        # Initialize Phase 1 components
        self.hierarchical_chunker = HierarchicalChunker()
        self.vector_store = EnhancedVectorStore(model_name=vector_model)
        
        # Indexing statistics
        self.indexing_stats = {
            'total_files_processed': 0,
            'total_symbols_discovered': 0,
            'total_relationships_found': 0,
            'total_chunks_created': 0,
            'total_processing_time': 0.0,
            'file_stats': []
        }
        
        # Supported file extensions
        self.supported_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx',          # Python, JavaScript, TypeScript
            '.java', '.kt',                               # Java, Kotlin
            '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx',    # C/C++
            '.rs',                                        # Rust
            '.go',                                        # Go
            '.rb',                                        # Ruby
            '.php',                                       # PHP
            '.cs',                                        # C#
            '.swift',                                     # Swift
            '.scala',                                     # Scala
            '.clj', '.cljs',                             # Clojure
        }
        
        logger.info(f"Unified indexer initialized with database: {db_path}")
        logger.info(f"BGE model: {vector_model}")
    
    async def index_codebase(self, codebase_path: Path, 
                      force_reindex: bool = False,
                      file_patterns: List[str] = None) -> IndexingResult:
        """
        Complete indexing pipeline with KG + hierarchical chunks.
        
        Args:
            codebase_path: Root path of codebase to index
            force_reindex: Whether to reindex already processed files
            file_patterns: Optional list of glob patterns to match files
            
        Returns:
            IndexingResult with comprehensive statistics
        """
        start_time = time.time()
        
        logger.info(f"Starting unified indexing of codebase: {codebase_path}")
        
        # Discover files to process
        file_paths = self._discover_files(codebase_path, file_patterns)
        
        if not file_paths:
            logger.warning("No supported files found for indexing")
            return IndexingResult(
                success=False,
                files_processed=0,
                files_failed=0,
                symbols_discovered=0,
                relationships_found=0,
                chunks_created=0,
                processing_time=0.0,
                error_details=["No supported files found"]
            )
        
        logger.info(f"Discovered {len(file_paths)} files for indexing")
        
        # Filter files if not force reindexing
        if not force_reindex:
            file_paths = self._filter_modified_files(file_paths)
            logger.info(f"Processing {len(file_paths)} modified files")
        
        if not file_paths:
            logger.info("No files need reindexing")
            return IndexingResult(
                success=True,
                files_processed=0,
                files_failed=0,
                symbols_discovered=0,
                relationships_found=0,
                chunks_created=0,
                processing_time=time.time() - start_time,
                error_details=[]
            )
        
        try:
            # PASS 1: Symbol Collection (populates nodes table)
            logger.info("=== PASS 1: Symbol Collection ===")
            symbol_table = self.symbol_collector.collect_all_symbols(file_paths)
            symbol_stats = self.symbol_collector.get_collection_statistics()
            
            logger.info(f"Symbol collection completed:")
            logger.info(f"  Files processed: {symbol_stats['files_processed']}")
            logger.info(f"  Symbols discovered: {symbol_stats['symbols_discovered']}")
            
            # PASS 2: Universal Relationship Extraction (populates edges table)
            logger.info("=== PASS 2: Universal Relationship Extraction ===")
            relationship_stats = await self._extract_universal_relationships(file_paths, symbol_table)
            
            logger.info(f"Relationship extraction completed:")
            logger.info(f"  Files processed: {relationship_stats['files_processed']}")
            logger.info(f"  Relationships found: {relationship_stats['relationships_found']}")
            
            # PHASE 1 INTEGRATION: Hierarchical Chunking with KG context
            logger.info("=== PHASE 1 INTEGRATION: Hierarchical Chunking + BGE Embeddings ===")
            chunk_stats = self._process_hierarchical_chunks_with_kg_context(file_paths, symbol_table)
            
            logger.info(f"Hierarchical chunking completed:")
            logger.info(f"  Files processed: {chunk_stats['files_processed']}")
            logger.info(f"  Chunks created: {chunk_stats['chunks_created']}")
            
            # Calculate final statistics
            total_time = time.time() - start_time
            
            result = IndexingResult(
                success=True,
                files_processed=len(file_paths),
                files_failed=symbol_stats['files_failed'] + relationship_stats['files_failed'],
                symbols_discovered=symbol_stats['symbols_discovered'],
                relationships_found=relationship_stats['relationships_found'],
                chunks_created=chunk_stats['chunks_created'],
                processing_time=total_time,
                error_details=symbol_stats['processing_errors'] + relationship_stats['processing_errors']
            )
            
            # Update global statistics
            self._update_global_statistics(result)
            
            logger.info(f"=== UNIFIED INDEXING COMPLETED ===")
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            logger.info(f"Files processed: {result.files_processed}")
            logger.info(f"Symbols discovered: {result.symbols_discovered}")
            logger.info(f"Relationships found: {result.relationships_found}")
            logger.info(f"Chunks created: {result.chunks_created}")
            
            return result
            
        except Exception as e:
            logger.error(f"Unified indexing failed: {e}")
            import traceback
            traceback.print_exc()
            
            return IndexingResult(
                success=False,
                files_processed=0,
                files_failed=len(file_paths),
                symbols_discovered=0,
                relationships_found=0,
                chunks_created=0,
                processing_time=time.time() - start_time,
                error_details=[str(e)]
            )
    
    def _discover_files(self, codebase_path: Path, file_patterns: List[str] = None) -> List[Path]:
        """Discover all supported files in the codebase."""
        file_paths = []
        
        try:
            if file_patterns:
                # Use specific patterns
                for pattern in file_patterns:
                    file_paths.extend(list(codebase_path.rglob(pattern)))
            else:
                # Use supported extensions
                for file_path in codebase_path.rglob("*"):
                    if file_path.is_file() and file_path.suffix in self.supported_extensions:
                        file_paths.append(file_path)
            
            # Filter out common directories to ignore
            ignore_dirs = {
                'node_modules', '.git', '__pycache__', '.pytest_cache',
                'build', 'dist', 'target', 'venv', '.venv', 'env',
                '.next', '.nuxt', 'coverage'
            }
            
            filtered_paths = []
            for file_path in file_paths:
                # Check if any parent directory should be ignored
                if not any(part in ignore_dirs for part in file_path.parts):
                    filtered_paths.append(file_path)
            
            return filtered_paths
            
        except Exception as e:
            logger.error(f"File discovery failed: {e}")
            return []
    
    def _filter_modified_files(self, file_paths: List[Path]) -> List[Path]:
        """Filter files that need reindexing based on modification time."""
        files_to_process = []
        
        for file_path in file_paths:
            try:
                # Get file status from database
                file_status = self.db_manager.get_file_status(str(file_path))
                
                if not file_status:
                    # File not in database, needs processing
                    files_to_process.append(file_path)
                    continue
                
                # Check if file was modified since last processing
                file_mtime = file_path.stat().st_mtime
                last_processed = file_status.get('updated_at')
                
                if last_processed:
                    # Parse timestamp and compare
                    from datetime import datetime
                    last_processed_dt = datetime.fromisoformat(last_processed.replace('Z', '+00:00'))
                    file_modified_dt = datetime.fromtimestamp(file_mtime)
                    
                    if file_modified_dt > last_processed_dt:
                        files_to_process.append(file_path)
                else:
                    # No last processed time, needs processing
                    files_to_process.append(file_path)
                    
            except Exception as e:
                logger.debug(f"Error checking file modification time for {file_path}: {e}")
                # On error, include file for processing
                files_to_process.append(file_path)
        
        return files_to_process
    
    def _process_hierarchical_chunks_with_kg_context(self, file_paths: List[Path], 
                                                   symbol_table: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Process hierarchical chunks with Knowledge Graph context for enhanced linking.
        
        Integration Innovation: Uses KG symbol information to create better
        chunk-to-node relationships and improve chunk organization.
        """
        chunk_stats = {
            'files_processed': 0,
            'files_failed': 0,
            'chunks_created': 0,
            'kg_linked_chunks': 0,
            'processing_errors': []
        }
        
        for file_path in file_paths:
            try:
                file_start_time = time.time()
                
                # Read file content
                try:
                    content = file_path.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    content = file_path.read_text(encoding='latin-1')
                
                # Get KG symbols for this file to provide context
                file_symbols = self._get_file_symbols(file_path, symbol_table)
                
                # Generate hierarchical chunks with KG context
                hierarchical_chunks = self.hierarchical_chunker.chunk_file(content, file_path)
                
                # Link chunks to KG nodes where applicable
                linked_chunks = self._link_chunks_to_kg_nodes(hierarchical_chunks, file_symbols)
                
                # Store chunks with embeddings
                stored_chunks = self._store_chunks_with_embeddings(linked_chunks, file_path)
                
                chunk_stats['files_processed'] += 1
                chunk_stats['chunks_created'] += len(stored_chunks)
                chunk_stats['kg_linked_chunks'] += sum(1 for chunk in stored_chunks if safe_get(chunk, 'node_id'))
                
                # Update file processing stats
                processing_time = time.time() - file_start_time
                
                self.db_manager.update_file_status(
                    str(file_path),
                    'completed',
                    chunks_count=len(stored_chunks)
                )
                
                logger.debug(f"Processed chunks for {file_path}: {len(stored_chunks)} chunks in {processing_time:.2f}s")
                
            except Exception as e:
                error_msg = f"Chunk processing failed for {file_path}: {e}"
                logger.error(error_msg)
                chunk_stats['files_failed'] += 1
                chunk_stats['processing_errors'].append(error_msg)
                
                # Update file error status
                self.db_manager.update_file_status(
                    str(file_path),
                    'error',
                    error_message=str(e)
                )
        
        return chunk_stats
    
    def _get_file_symbols(self, file_path: Path, symbol_table: Dict[str, Dict]) -> Dict[str, Dict]:
        """Get all symbols defined in a specific file."""
        file_symbols = {}
        file_path_str = str(file_path)
        
        for symbol_id, symbol_info in symbol_table.items():
            if symbol_info['file_path'] == file_path_str:
                # Create a mapping from symbol name to symbol info for easy lookup
                symbol_name = symbol_info['name']
                file_symbols[symbol_name] = symbol_info
        
        return file_symbols
    
    def _link_chunks_to_kg_nodes(self, chunks: List[Dict], file_symbols: Dict[str, Dict]) -> List[Dict]:
        """
        Link hierarchical chunks to Knowledge Graph nodes.
        
        Enhancement: Creates intelligent chunk-to-node relationships using
        symbol location information from the KG.
        """
        linked_chunks = []
        
        for chunk in chunks:
            linked_chunk = chunk.copy()
            
            # Try to link chunk to KG node based on content and location
            node_id = self._find_matching_kg_node(chunk, file_symbols)
            if node_id:
                linked_chunk['node_id'] = node_id
                
                # Add KG metadata to chunk
                symbol_info = None
                for symbol in file_symbols.values():
                    if symbol['id'] == node_id:
                        symbol_info = symbol
                        break
                
                if symbol_info:
                    kg_metadata = {
                        'symbol_name': symbol_info['name'],
                        'symbol_type': symbol_info['type'],
                        'has_kg_context': True
                    }
                    
                    # Merge with existing metadata
                    existing_metadata = linked_chunk.get('metadata', {})
                    existing_metadata.update(kg_metadata)
                    linked_chunk['metadata'] = existing_metadata
            
            linked_chunks.append(linked_chunk)
        
        return linked_chunks
    
    def _find_matching_kg_node(self, chunk: Dict, file_symbols: Dict[str, Dict]) -> Optional[str]:
        """Find the best matching KG node for a chunk."""
        chunk_line_start = safe_get(chunk, 'line_start')
        chunk_line_end = safe_get(chunk, 'line_end')
        chunk_content = safe_get(chunk, 'content', '')
        
        if not chunk_line_start or not chunk_line_end:
            return None
        
        # Look for symbols that overlap with chunk location
        best_match = None
        best_overlap = 0
        
        for symbol_name, symbol_info in file_symbols.items():
            symbol_line_start = symbol_info.get('line_start')
            symbol_line_end = symbol_info.get('line_end')
            
            if not symbol_line_start or not symbol_line_end:
                continue
            
            # Calculate line overlap
            overlap_start = max(chunk_line_start, symbol_line_start)
            overlap_end = min(chunk_line_end, symbol_line_end)
            overlap = max(0, overlap_end - overlap_start + 1)
            
            # Check if symbol name appears in chunk content
            name_in_content = symbol_name.lower() in chunk_content.lower()
            
            # Calculate matching score
            score = overlap
            if name_in_content:
                score += 10  # Bonus for name match
            
            if score > best_overlap:
                best_overlap = score
                best_match = symbol_info['id']
        
        return best_match
    
    def _store_chunks_with_embeddings(self, chunks: List[Dict], file_path: Path) -> List[Dict]:
        """Store chunks in database with BGE embeddings."""
        stored_chunks = []
        
        for chunk in chunks:
            try:
                # Generate embedding for chunk content
                content = safe_get(chunk, 'content', '')
                if not content.strip():
                    continue
                
                # Create unique chunk ID
                chunk_id = self._generate_chunk_id(chunk, file_path)
                
                # Store chunk in database
                success = self.db_manager.insert_chunk(
                    chunk_id=chunk_id,
                    content=content,
                    file_path=str(file_path),
                    chunk_type=safe_get(chunk, 'type', 'unknown'),
                    node_id=safe_get(chunk, 'node_id'),
                    line_start=safe_get(chunk, 'line_start'),
                    line_end=safe_get(chunk, 'line_end'),
                    metadata=safe_get(chunk, 'metadata', {})
                )
                
                if success:
                    # Handle both dict and object formats
                    if hasattr(chunk, '__setitem__'):
                        chunk['id'] = chunk_id
                    elif hasattr(chunk, 'id'):
                        chunk.id = chunk_id
                    stored_chunks.append(chunk)
                    
                    # Store embedding in vector store (if available)
                    try:
                        if hasattr(self.vector_store, 'add_chunks'):
                            self.vector_store.add_chunks([content], [chunk_id])
                    except Exception as e:
                        logger.debug(f"Failed to store embedding for chunk {chunk_id}: {e}")
                
            except Exception as e:
                logger.error(f"Failed to store chunk: {e}")
                continue
        
        return stored_chunks
    
    def _generate_chunk_id(self, chunk: Dict, file_path: Path) -> str:
        """Generate unique chunk identifier."""
        import hashlib
        
        # Create ID based on file path, line numbers, and content hash
        content = safe_get(chunk, 'content', '')
        line_start = safe_get(chunk, 'line_start', 0)
        line_end = safe_get(chunk, 'line_end', 0)
        
        # Create content hash for uniqueness
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        
        # Sanitize file path
        file_stem = file_path.stem.replace(' ', '_')
        
        return f"{file_stem}_{line_start}_{line_end}_{content_hash}"
    
    async def _extract_universal_relationships(self, file_paths: List[Path], symbol_table: Dict) -> Dict[str, Any]:
        """
        Extract relationships using the Universal Dependency Indexer.
        
        This replaces the old RelationshipExtractor with our new universal system
        that can detect architectural patterns across all programming languages.
        """
        try:
            # Import the universal indexer
            from backend.indexer.universal_dependency_indexer import UniversalDependencyExtractor
            
            logger.info("🔗 Starting universal relationship extraction")
            
            # Initialize universal dependency indexer
            universal_indexer = UniversalDependencyExtractor()
            
            # Track statistics
            files_processed = 0
            files_failed = 0
            relationships_found = 0
            processing_errors = []
            
            for file_path in file_paths:
                try:
                    logger.debug(f"Processing relationships for: {file_path}")
                    
                    # Read file content
                    if not file_path.exists():
                        continue
                        
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    
                    # Extract dependencies using universal indexer
                    language = universal_indexer.detect_language(str(file_path))
                    if language:
                        imports = universal_indexer.extract_imports_for_language(content, language, str(file_path))
                        # Get available components from symbol table
                        available_components = list(symbol_table.keys()) if symbol_table else []
                        calls = universal_indexer.extract_call_relationships(content, language, str(file_path), available_components)
                        file_dependencies = {
                            'imports': imports,
                            'calls': calls
                        }
                    else:
                        file_dependencies = {'imports': [], 'calls': []}
                    
                    # Convert universal dependencies to KG format and store
                    file_relationships = self._store_universal_dependencies_as_kg_relationships(
                        file_dependencies, file_path, symbol_table)
                    
                    relationships_found += len(file_relationships)
                    files_processed += 1
                    
                    if files_processed % 10 == 0:
                        logger.info(f"Processed {files_processed}/{len(file_paths)} files, found {relationships_found} relationships")
                    
                except Exception as e:
                    files_failed += 1
                    error_msg = f"Failed to extract relationships from {file_path}: {e}"
                    logger.warning(error_msg)
                    processing_errors.append(error_msg)
            
            logger.info(f"✅ Universal relationship extraction completed")
            logger.info(f"   Files processed: {files_processed}")
            logger.info(f"   Files failed: {files_failed}")
            logger.info(f"   Relationships found: {relationships_found}")
            
            return {
                'files_processed': files_processed,
                'files_failed': files_failed,
                'relationships_found': relationships_found,
                'processing_errors': processing_errors
            }
            
        except Exception as e:
            logger.error(f"❌ Universal relationship extraction failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to old system if universal system fails
            logger.info("Falling back to legacy relationship extractor")
            return await self._fallback_relationship_extraction(file_paths, symbol_table)
    
    def _store_universal_dependencies_as_kg_relationships(self, dependencies: Dict, file_path: Path, symbol_table: Dict) -> List[Dict]:
        """
        Convert universal dependencies to KG relationships and store in database.
        """
        relationships = []
        
        try:
            cursor = self.db_manager.connection.cursor()
            
            # Process different types of dependencies
            for dep_type, deps in dependencies.items():
                for dep in deps:
                    if isinstance(dep, dict):
                        source_name = dep.get('source', file_path.stem)
                        target_name = dep.get('target', 'unknown')
                        relationship_type = dep.get('type', dep_type)
                        confidence = dep.get('confidence', 0.8)
                        
                        # Create relationship entry
                        relationship = {
                            'source_id': f"{source_name}::{source_name}::1",
                            'target_id': f"{target_name}::{target_name}::1", 
                            'type': relationship_type,
                            'properties': {
                                'confidence': confidence,
                                'source_file': str(file_path),
                                'detection_method': 'universal_dependency_indexer',
                                'language': dep.get('language', 'unknown')
                            }
                        }
                        
                        # Store in database
                        try:
                            cursor.execute("""
                                INSERT OR REPLACE INTO edges (source_id, target_id, type, properties)
                                VALUES (?, ?, ?, ?)
                            """, (
                                relationship['source_id'],
                                relationship['target_id'], 
                                relationship['type'],
                                str(relationship['properties'])
                            ))
                            
                            relationships.append(relationship)
                            
                        except Exception as e:
                            logger.debug(f"Failed to store relationship {source_name}->{target_name}: {e}")
            
            # Commit changes
            self.db_manager.connection.commit()
            
        except Exception as e:
            logger.warning(f"Failed to store universal dependencies for {file_path}: {e}")
        
        return relationships
    
    async def _fallback_relationship_extraction(self, file_paths: List[Path], symbol_table: Dict) -> Dict[str, Any]:
        """Fallback to legacy relationship extractor if universal system fails."""
        try:
            from knowledge_graph.relationship_extractor import RelationshipExtractor
            
            logger.info("Using legacy relationship extractor")
            self.relationship_extractor = RelationshipExtractor(self.db_manager, symbol_table)
            self.relationship_extractor.extract_relationships(file_paths)
            return self.relationship_extractor.get_extraction_statistics()
            
        except Exception as e:
            logger.error(f"Legacy relationship extraction also failed: {e}")
            return {
                'files_processed': 0,
                'files_failed': len(file_paths),
                'relationships_found': 0,
                'processing_errors': [f"All relationship extraction methods failed: {e}"]
            }
    
    def _update_global_statistics(self, result: IndexingResult):
        """Update global indexing statistics."""
        self.indexing_stats['total_files_processed'] += result.files_processed
        self.indexing_stats['total_symbols_discovered'] += result.symbols_discovered
        self.indexing_stats['total_relationships_found'] += result.relationships_found
        self.indexing_stats['total_chunks_created'] += result.chunks_created
        self.indexing_stats['total_processing_time'] += result.processing_time
    
    def get_indexing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive indexing statistics."""
        db_stats = self.db_manager.get_statistics()
        
        return {
            **self.indexing_stats,
            'database_statistics': db_stats,
            'database_size_mb': db_stats.get('database_size_mb', 0),
            'avg_processing_time_per_file': (
                self.indexing_stats['total_processing_time'] / 
                max(self.indexing_stats['total_files_processed'], 1)
            )
        }
    
    def cleanup_stale_data(self, older_than_days: int = 30) -> Dict[str, int]:
        """Clean up stale data from the database."""
        logger.info(f"Cleaning up data older than {older_than_days} days")
        
        cleanup_stats = {
            'orphaned_edges_removed': 0,
            'stale_files_removed': 0
        }
        
        try:
            # Clean up orphaned edges
            orphaned_edges = self.db_manager.cleanup_orphaned_edges()
            cleanup_stats['orphaned_edges_removed'] = orphaned_edges
            
            logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return cleanup_stats
    
    def close(self):
        """Clean up resources."""
        if self.db_manager:
            self.db_manager.close()
        
        if hasattr(self.db_setup, 'close'):
            self.db_setup.close()


if __name__ == "__main__":
    # CLI interface for unified indexing
    import argparse
    
    parser = argparse.ArgumentParser(description="CodeWise Unified Indexer")
    parser.add_argument("--path", required=True, help="Path to codebase to index")
    parser.add_argument("--db-path", default="codewise.db", help="Database file path")
    parser.add_argument("--force", action="store_true", help="Force reindex all files")
    parser.add_argument("--patterns", nargs="+", help="File patterns to match")
    parser.add_argument("--stats", action="store_true", help="Show detailed statistics")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup stale data")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Create unified indexer
        indexer = UnifiedIndexer(db_path=args.db_path)
        
        # Run cleanup if requested
        if args.cleanup:
            cleanup_stats = indexer.cleanup_stale_data()
            print(f"\\nCleanup Results:")
            for key, value in cleanup_stats.items():
                print(f"  {key}: {value}")
        
        # Run indexing
        codebase_path = Path(args.path)
        result = indexer.index_codebase(
            codebase_path=codebase_path,
            force_reindex=args.force,
            file_patterns=args.patterns
        )
        
        # Print results
        print(f"\\nIndexing Results:")
        print(f"  Success: {result.success}")
        print(f"  Files processed: {result.files_processed}")
        print(f"  Files failed: {result.files_failed}")
        print(f"  Symbols discovered: {result.symbols_discovered}")
        print(f"  Relationships found: {result.relationships_found}")
        print(f"  Chunks created: {result.chunks_created}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        
        if result.error_details:
            print(f"\\nErrors ({len(result.error_details)}):")
            for error in result.error_details[:5]:  # Show first 5 errors
                print(f"  - {error}")
        
        # Show detailed statistics if requested
        if args.stats:
            stats = indexer.get_indexing_statistics()
            print(f"\\nDetailed Statistics:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
        
        indexer.close()
        
    except Exception as e:
        print(f"Unified indexing failed: {e}")
        import traceback
        traceback.print_exc()