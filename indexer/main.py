import os, time, json, sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
# type: ignore
import openai
openai.log = "warning"
from sentence_transformers import SentenceTransformer
import faiss
from watchfiles import watch
from indexer.file_discovery import FileDiscoveryEngine, FileInfo
from indexer.ast_chunker import ASTChunker, CodeChunk
from indexer.complexity import choose_chunk_size

# BM25 integration - conditional import with graceful fallback
try:
    # Import BM25Index from local copy
    from bm25_index import BM25Index
    BM25_AVAILABLE = True
    print("[indexer] BM25Index successfully imported", flush=True)
except ImportError as e:
    print(f"[indexer] Warning: BM25Index not available: {e}", flush=True)
    BM25_AVAILABLE = False

# Load local embedder once
_local_embedder = SentenceTransformer("all-MiniLM-L6-v2")

WORKSPACE = Path(os.getenv("WORKSPACE_DIR", "/workspace"))
CACHE_DIR = WORKSPACE / ".vector_cache"
CACHE_DIR.mkdir(exist_ok=True)
INDEX_FILE = CACHE_DIR / "index.faiss"
META_FILE = CACHE_DIR / "meta.json"
STATS_FILE = CACHE_DIR / "discovery_stats.json"
CHUNK_SIZE = 400
EMBED_MODEL = "text-embedding-3-small"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Optional: disable automatic re-indexing in production to avoid expensive rebuilds
WATCH_ENABLED = os.getenv("INDEXER_WATCH", "0") == "1"

# Simple build mutex to prevent overlapping rebuilds (e.g., when many file events fire)
_build_lock = False
# Flag to remember if a rebuild was requested while a build is already running
_pending_rebuild = False

def chunk_text(text: str) -> List[str]:
    return [text[i:i+CHUNK_SIZE] for i in range(0,len(text),CHUNK_SIZE)]

def discover_files() -> List[FileInfo]:
    """Use enhanced file discovery engine"""
    discovery_engine = FileDiscoveryEngine(WORKSPACE)
    discovered_files = discovery_engine.discover_files()
    
    # Save discovery statistics
    stats = discovery_engine.get_discovery_stats()
    try:
        with STATS_FILE.open("w", encoding="utf-8") as f:
            json.dump({
                "total_files_scanned": stats.total_files_scanned,
                "files_indexed": stats.files_indexed,
                "files_skipped": stats.files_skipped,
                "coverage_percentage": stats.get_coverage_percentage(),
                "files_by_type": dict(stats.files_by_type),
                "skipped_reasons": dict(stats.skipped_reasons),
                "binary_files_skipped": stats.binary_files_skipped,
                "symlinks_followed": stats.symlinks_followed,
                "symlinks_skipped": stats.symlinks_skipped,
                "dotfiles_included": stats.dotfiles_included,
                "content_detected_files": stats.content_detected_files
            }, f, indent=2)
    except Exception as e:
        print(f"[indexer] Warning: Could not save discovery stats: {e}")
    
    return discovered_files

def embed_batch(texts: List[str], batch_size: int = 100):
    """Generate embeddings locally using MiniLM with batch processing; returns float32 numpy array."""
    import numpy as np
    
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    
    all_vectors = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    print(f"[indexer] Processing {len(texts)} chunks in {total_batches} batches of {batch_size}", flush=True)
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        try:
            print(f"[indexer] Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)", flush=True)
            vectors = _local_embedder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_vectors.append(np.array(vectors, dtype="float32"))
            
            # Add small delay to prevent memory pressure
            import time
            time.sleep(0.1)
            
        except Exception as e:
            print(f"[indexer] Batch {batch_num} embedding error: {e}, using zero vectors", flush=True)
            # Create zero vectors for failed batch
            zero_vectors = np.zeros((len(batch), 384), dtype="float32")
            all_vectors.append(zero_vectors)
    
    if all_vectors:
        result = np.vstack(all_vectors)
        print(f"[indexer] Successfully generated {result.shape[0]} embeddings", flush=True)
        return result
    else:
        print(f"[indexer] No embeddings generated, returning zero array", flush=True)
        return np.zeros((len(texts), 384), dtype="float32")

def build_bm25_index_from_chunks(texts: List[str], enhanced_meta: List[dict]) -> bool:
    """
    Build BM25 index from the same data used for vector embeddings
    
    Args:
        texts: List of text chunks (same as used for embeddings)
        enhanced_meta: List of metadata dictionaries for each chunk
        
    Returns:
        True if successful, False otherwise
    """
    if not BM25_AVAILABLE:
        print("[indexer] BM25Index not available, skipping BM25 index building", flush=True)
        return False
    
    if not texts or not enhanced_meta or len(texts) != len(enhanced_meta):
        print("[indexer] Warning: Invalid data for BM25 index building", flush=True)
        return False
    
    try:
        print(f"[indexer] Building BM25 index from {len(texts)} documents", flush=True)
        
        # Create BM25 documents from enhanced metadata
        bm25_documents = []
        for i, (text, meta) in enumerate(zip(texts, enhanced_meta)):
            doc = {
                'id': i,
                'text': text,
                'file_path': meta.get('file_path', ''),
                'relative_path': meta.get('relative_path', ''),
                'project': meta.get('project', ''),
                'chunk_type': meta.get('chunk_type', 'unknown'),
                'function_name': meta.get('function_name'),
                'class_name': meta.get('class_name'),
                'file_type': meta.get('file_type', 'unknown'),
                'metadata': {
                    'start_line': meta.get('start_line', 0),
                    'end_line': meta.get('end_line', 0),
                    'imports': meta.get('imports', []),
                    'parent_context': meta.get('parent_context'),
                    'dependencies': meta.get('dependencies', []),
                    'docstring': meta.get('docstring'),
                    'decorators': meta.get('decorators', [])
                }
            }
            bm25_documents.append(doc)
        
        # Build BM25 index
        bm25_index = BM25Index()
        bm25_index.add_documents(bm25_documents)
        
        # Get statistics before saving
        stats = bm25_index.get_statistics()
        
        # Save to cache directory (same location as vector index)
        bm25_file = CACHE_DIR / "bm25_index.json"
        success = bm25_index.save_index(bm25_file)
        
        if success:
            print(f"[indexer] BM25 index built successfully:", flush=True)
            print(f"  - Documents: {stats['total_documents']}", flush=True)
            print(f"  - Vocabulary: {stats['vocabulary_size']} unique terms", flush=True)
            print(f"  - Average doc length: {stats['average_document_length']:.1f} terms", flush=True)
            print(f"  - Saved to: {bm25_file}", flush=True)
            return True
        else:
            print("[indexer] Failed to save BM25 index", flush=True)
            return False
            
    except Exception as e:
        print(f"[indexer] Error building BM25 index: {e}", flush=True)
        return False

def build_knowledge_graph_from_chunks(enhanced_meta: List[dict], build_mode: str, project: str = None) -> bool:
    """
    Build Knowledge Graph from indexed metadata.
    
    Args:
        enhanced_meta: List of metadata dictionaries for each chunk
        build_mode: "full" or "incremental" 
        project: Project name for incremental builds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"[KG] ========== KNOWLEDGE GRAPH BUILDING START ==========", flush=True)
        print(f"[KG] Mode: {build_mode}, Project: {project}, Chunks: {len(enhanced_meta)}", flush=True)
        
        # Import KG components
        import sys
        sys.path.append('/app/backend')
        try:
            from storage.database_manager import DatabaseManager
            print(f"[KG] ✅ Successfully imported DatabaseManager", flush=True)
        except ImportError as e:
            print(f"[KG] ❌ Failed to import DatabaseManager: {e}", flush=True)
            return False
        
        # Initialize database in shared workspace (accessible by both containers)
        db_path = '/workspace/.vector_cache/codewise.db'
        print(f"[KG] Initializing database at: {db_path}", flush=True)
        
        try:
            db = DatabaseManager(db_path)
            print(f"[KG] ✅ Database connection established", flush=True)
        except Exception as e:
            print(f"[KG] ❌ Database connection failed: {e}", flush=True)
            return False
        
        # Check existing data
        try:
            cursor = db.connection.cursor()
            existing_nodes = cursor.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            existing_edges = cursor.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            print(f"[KG] Existing data: {existing_nodes} nodes, {existing_edges} edges", flush=True)
        except Exception as e:
            print(f"[KG] Warning: Could not check existing data: {e}", flush=True)
        
        # For incremental builds, remove existing nodes for this project
        if build_mode == "incremental" and project:
            try:
                print(f"[KG] Incremental build: cleaning existing data for project '{project}'", flush=True)
                # Get all nodes for this project
                cursor = db.connection.cursor()
                nodes_to_delete = cursor.execute(
                    "SELECT id FROM nodes WHERE file_path LIKE ?", 
                    (f"%{project}%",)
                ).fetchall()
                
                print(f"[KG] Found {len(nodes_to_delete)} existing nodes to delete", flush=True)
                
                # Delete nodes and related edges
                deleted_edges = 0
                for (node_id,) in nodes_to_delete:
                    edges_result = cursor.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))
                    deleted_edges += edges_result.rowcount
                    cursor.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
                
                db.connection.commit()
                print(f"[KG] ✅ Cleaned {len(nodes_to_delete)} nodes and {deleted_edges} edges for project '{project}'", flush=True)
                
            except Exception as e:
                print(f"[KG] ❌ Warning: Could not clean existing KG data for project '{project}': {e}", flush=True)
        
        # Process chunks and extract symbols
        print(f"[KG] Starting symbol extraction from {len(enhanced_meta)} chunks", flush=True)
        nodes_added = 0
        edges_added = 0
        chunks_processed = 0
        
        for meta in enhanced_meta:
            chunks_processed += 1
            chunk_type = meta.get('chunk_type', '')
            function_name = meta.get('function_name')
            class_name = meta.get('class_name')
            file_path = meta.get('relative_path', meta.get('file_path', ''))
            
            if chunks_processed % 1000 == 0:
                print(f"[KG] Progress: {chunks_processed}/{len(enhanced_meta)} chunks processed, {nodes_added} nodes added", flush=True)
            
            # Add function nodes
            if function_name and chunk_type == 'function':
                node_id = f"{file_path}::{function_name}::{meta.get('start_line', 0)}"
                if chunks_processed <= 10:  # Log first 10 for debugging
                    print(f"[KG] Adding function node: {function_name} in {file_path}", flush=True)
                
                success = db.insert_node(
                    node_id=node_id,
                    node_type='function',
                    name=function_name,
                    file_path=file_path,
                    line_start=meta.get('start_line'),
                    line_end=meta.get('end_line'),
                    signature=meta.get('chunk_text', '').split('\\n')[0] if meta.get('chunk_text') else '',
                    docstring=meta.get('docstring'),
                    properties={}
                )
                if success:
                    nodes_added += 1
                elif chunks_processed <= 10:
                    print(f"[KG] ❌ Failed to add function node: {function_name}", flush=True)
            
            # Add class nodes
            if class_name and chunk_type == 'class':
                node_id = f"{file_path}::{class_name}::{meta.get('start_line', 0)}"
                if chunks_processed <= 10:  # Log first 10 for debugging
                    print(f"[KG] Adding class node: {class_name} in {file_path}", flush=True)
                
                success = db.insert_node(
                    node_id=node_id,
                    node_type='class',
                    name=class_name,
                    file_path=file_path,
                    line_start=meta.get('start_line'),
                    line_end=meta.get('end_line'),
                    signature=meta.get('chunk_text', '').split('\\n')[0] if meta.get('chunk_text') else '',
                    docstring=meta.get('docstring'),
                    properties={}
                )
                if success:
                    nodes_added += 1
                elif chunks_processed <= 10:
                    print(f"[KG] ❌ Failed to add class node: {class_name}", flush=True)
            
            # Add module-level imports as nodes
            imports = meta.get('imports', [])
            if imports and chunks_processed <= 5:  # Log first 5 imports for debugging
                print(f"[KG] Processing {len(imports)} imports for {file_path}: {imports[:3]}...", flush=True)
            
            for import_name in imports:
                if import_name and isinstance(import_name, str):
                    node_id = f"{file_path}::import::{import_name}"
                    success = db.insert_node(
                        node_id=node_id,
                        node_type='import',
                        name=import_name,
                        file_path=file_path,
                        line_start=meta.get('start_line'),
                        line_end=meta.get('start_line'),
                        signature=f"import {import_name}",
                        properties={}
                    )
                    if success:
                        nodes_added += 1
        
        # Simple relationship extraction from dependencies
        for meta in enhanced_meta:
            function_name = meta.get('function_name')
            dependencies = meta.get('dependencies', [])
            file_path = meta.get('relative_path', meta.get('file_path', ''))
            
            if function_name and dependencies:
                source_id = f"{file_path}::{function_name}::{meta.get('start_line', 0)}"
                
                for dep in dependencies:
                    if dep and isinstance(dep, str):
                        # Simple heuristic: if dependency looks like a function call
                        if '(' in dep or dep.endswith('()'):
                            dep_clean = dep.replace('()', '').replace('(', '').strip()
                            target_id = f"{file_path}::{dep_clean}::0"  # Simplified target
                            
                            success = db.insert_edge(
                                source_id=source_id,
                                target_id=target_id,
                                edge_type='calls',
                                file_path=file_path,
                                line_number=meta.get('start_line')
                            )
                            if success:
                                edges_added += 1
        
        print(f"[KG] Symbol extraction complete: {nodes_added} nodes, {edges_added} edges", flush=True)
        
        # Get final statistics
        try:
            cursor = db.connection.cursor()
            final_nodes = cursor.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            final_edges = cursor.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            node_types = cursor.execute("SELECT type, COUNT(*) FROM nodes GROUP BY type").fetchall()
            
            print(f"[KG] Final statistics: {final_nodes} total nodes, {final_edges} total edges", flush=True)
            print(f"[KG] Node types: {dict(node_types)}", flush=True)
            
        except Exception as e:
            print(f"[KG] Warning: Could not get final statistics: {e}", flush=True)
        
        print(f"[KG] ========== KNOWLEDGE GRAPH BUILDING COMPLETE ==========", flush=True)
        return True
        
    except ImportError as e:
        print(f"[KG] ❌ Knowledge Graph components not available: {e}", flush=True)
        return False
    except Exception as e:
        print(f"[KG] ❌ Error building Knowledge Graph: {e}", flush=True)
        import traceback
        print(f"[KG] ❌ Traceback: {traceback.format_exc()}", flush=True)
        return False

def build_index(project: str | None = None):
    global _build_lock, _pending_rebuild
    if _build_lock:
        # A build is already running – queue another one so that new files are guaranteed to be indexed
        print("[indexer] build already in progress – queueing another rebuild trigger", flush=True)
        _pending_rebuild = True
        return
    _build_lock = True
    
    # Determine build mode
    if project is None:
        print("[indexer] building FULL vector index with AST chunking…", flush=True)
        build_mode = "full"
    else:
        print(f"[indexer] building INCREMENTAL vector index for project '{project}' with AST chunking…", flush=True)
        build_mode = "incremental"
    
    texts: List[str] = []
    enhanced_meta: List[dict] = []  # Enhanced metadata with chunk information
    
    # For incremental builds, load existing data first
    existing_texts = []
    existing_meta = []
    if build_mode == "incremental" and INDEX_FILE.exists() and META_FILE.exists():
        try:
            print(f"[indexer] Loading existing index for incremental build...", flush=True)
            with META_FILE.open("r", encoding="utf-8") as f:
                existing_meta = json.load(f)
            
            # Remove existing chunks for this project to avoid duplicates
            filtered_meta = []
            removed_count = 0
            for meta in existing_meta:
                meta_project = meta.get("project", "")
                if meta_project != project:
                    filtered_meta.append(meta)
                    existing_texts.append(meta.get("chunk_text", ""))
                else:
                    removed_count += 1
            
            existing_meta = filtered_meta
            print(f"[indexer] Loaded {len(existing_meta)} existing chunks, removed {removed_count} chunks from project '{project}'", flush=True)
            
        except Exception as e:
            print(f"[indexer] Warning: Could not load existing index for incremental build: {e}. Falling back to full rebuild.", flush=True)
            build_mode = "full"
            existing_texts = []
            existing_meta = []
    
    # Initialize AST chunker
    ast_chunker = ASTChunker()
    
    # If a project scope is provided, adjust the root dir
    root_dir: Path = WORKSPACE if project is None else WORKSPACE / project
    if project and not root_dir.exists():
        print(f"[indexer] Requested project '{project}' not found – falling back to whole workspace", flush=True)
        root_dir = WORKSPACE
        build_mode = "full"
    
    # Discover files within the requested scope
    if root_dir == WORKSPACE:
        discovered_files = discover_files()
    else:
        # Create scoped discovery engine for single project
        from indexer.file_discovery import FileDiscoveryEngine
        discovery_engine = FileDiscoveryEngine(root_dir)
        discovered_files = discovery_engine.discover_files()
    
    total_chunks = 0
    processed_files = 0
    
    failed_files = 0
    
    for file_info in discovered_files:
        try:
            # Add file existence check
            if not file_info.path.exists():
                print(f"[indexer] Warning: File no longer exists: {file_info.relative_path}")
                failed_files += 1
                continue
                
            content = file_info.path.read_text(encoding="utf-8", errors="ignore")
            
            # Add content validation
            if not content.strip():
                print(f"[indexer] Warning: Empty file skipped: {file_info.relative_path}")
                continue
                
        except Exception as e:
            print(f"[indexer] Warning: Could not read {file_info.relative_path}: {e}")
            failed_files += 1
            continue
        
        # Optionally add a file-level summary chunk for large files
        if len(content) > 2500:
            summary_len = 1200
            summary_text = content[:summary_len]

            try:
                full_rel_path = str(file_info.path.relative_to(WORKSPACE))
            except Exception:
                full_rel_path = file_info.relative_path
            project_name = full_rel_path.split("/", 1)[0].split("\\", 1)[0]

            texts.append(summary_text)
            summary_meta = {
                "file_path": str(file_info.path),
                "relative_path": full_rel_path,
                "project": project_name,
                "chunk_text": summary_text,
                "start_line": 1,
                "end_line": 0,
                "file_type": file_info.file_type,
                "chunk_type": "file_summary",
                "function_name": None,
                "class_name": None,
                "imports": [],
                "parent_context": None,
                "dependencies": [],
                "docstring": None,
                "decorators": []
            }
            enhanced_meta.append(summary_meta)
            total_chunks += 1
        
        # Use AST chunker to create intelligent chunks
        try:
            code_chunks = ast_chunker.chunk_content(content, file_info.path)
            
            for chunk in code_chunks:
                texts.append(chunk.text)
                
                # Compute workspace-relative path to preserve project folder
                try:
                    full_rel_path = str(Path(chunk.file_path).relative_to(WORKSPACE))
                except Exception:
                    full_rel_path = file_info.relative_path  # Fallback

                # Derive project name (first path component) - fixed to use WORKSPACE consistently
                project_name = full_rel_path.split("/", 1)[0].split("\\", 1)[0]

                # Create enhanced metadata (now includes project)
                chunk_meta = {
                    "file_path": chunk.file_path,
                    "relative_path": full_rel_path,
                    "project": project_name,
                    "chunk_text": chunk.text,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "file_type": chunk.metadata.file_type,
                    "chunk_type": chunk.metadata.chunk_type,
                    "function_name": chunk.metadata.function_name,
                    "class_name": chunk.metadata.class_name,
                    "imports": chunk.metadata.imports,
                    "parent_context": chunk.metadata.parent_context,
                    "dependencies": chunk.metadata.dependencies,
                    "docstring": chunk.metadata.docstring,
                    "decorators": chunk.metadata.decorators
                }
                enhanced_meta.append(chunk_meta)
                total_chunks += 1
            
            processed_files += 1
            
        except Exception as e:
            print(f"[indexer] Warning: AST chunking failed for {file_info.relative_path}, using fallback: {e}")
            # Fallback to simple chunking for this file
            dynamic_size = choose_chunk_size(content)
            def dynamic_chunks(txt: str):
                return [txt[i:i+dynamic_size] for i in range(0, len(txt), dynamic_size)]

            for chunk in dynamic_chunks(content):
                texts.append(chunk)

                try:
                    full_rel_path = str(file_info.path.relative_to(WORKSPACE))
                except Exception:
                    full_rel_path = file_info.relative_path
                project_name = full_rel_path.split("/", 1)[0].split("\\", 1)[0]

                chunk_meta = {
                    "file_path": str(file_info.path),
                    "relative_path": full_rel_path,
                    "project": project_name,
                    "chunk_text": chunk,
                    "start_line": 0,
                    "end_line": 0,
                    "file_type": file_info.file_type,
                    "chunk_type": "fallback",
                    "function_name": None,
                    "class_name": None,
                    "imports": [],
                    "parent_context": None,
                    "dependencies": [],
                    "docstring": None,
                    "decorators": []
                }
                enhanced_meta.append(chunk_meta)
                total_chunks += 1
    
    # Combine with existing data for incremental builds
    if build_mode == "incremental" and existing_texts:
        all_texts = existing_texts + texts
        all_meta = existing_meta + enhanced_meta
        print(f"[indexer] Incremental build: {len(existing_texts)} existing + {len(texts)} new = {len(all_texts)} total chunks", flush=True)
    else:
        all_texts = texts
        all_meta = enhanced_meta
        print(f"[indexer] Full build: {len(all_texts)} total chunks", flush=True)
    
    if not all_texts:
        print("[indexer] no text files found")
        return
        
    print(f"[indexer] processed {processed_files} files into {total_chunks} intelligent chunks", flush=True)
    
    if not all_texts:
        print("[indexer] no text chunks to embed")
        return
    
    # Check for existing checkpoint
    checkpoint_file = CACHE_DIR / "embedding_checkpoint.json"
    progress_file = CACHE_DIR / "embedding_progress.faiss"
    
    start_batch = 0
    existing_embeddings = []
    
    # Try to resume from checkpoint
    if checkpoint_file.exists() and progress_file.exists():
        try:
            with checkpoint_file.open("r") as f:
                checkpoint_data = json.load(f)
            
            # Verify checkpoint matches current data
            if (checkpoint_data.get("total_chunks") == len(all_texts) and 
                checkpoint_data.get("metadata_hash") == hash(str(all_meta))):
                
                start_batch = checkpoint_data.get("completed_batches", 0)
                if start_batch > 0:
                    # Load existing embeddings
                    existing_index = faiss.read_index(str(progress_file))
                    existing_embeddings = []
                    for i in range(existing_index.ntotal):
                        existing_embeddings.append(existing_index.reconstruct(i))
                    
                    print(f"[indexer] Resuming from checkpoint: {start_batch} batches completed ({existing_index.ntotal} embeddings)", flush=True)
            else:
                print("[indexer] Checkpoint data mismatch, starting fresh", flush=True)
                # Clean up invalid checkpoint
                checkpoint_file.unlink(missing_ok=True)
                progress_file.unlink(missing_ok=True)
        except Exception as e:
            print(f"[indexer] Failed to load checkpoint: {e}, starting fresh", flush=True)
            checkpoint_file.unlink(missing_ok=True)
            progress_file.unlink(missing_ok=True)
    
    print(f"[indexer] generating embeddings for {len(all_texts)} chunks with batch processing", flush=True)
    
    # Process embeddings with checkpointing
    batch_size = 100
    total_batches = (len(all_texts) + batch_size - 1) // batch_size
    all_embeddings = existing_embeddings.copy()
    
    for batch_idx in range(start_batch, total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_texts))
        batch_texts = all_texts[start_idx:end_idx]
        
        print(f"[indexer] Processing embedding batch {batch_idx + 1}/{total_batches} ({len(batch_texts)} chunks)", flush=True)
        
        try:
            batch_embeddings = embed_batch(batch_texts, batch_size=len(batch_texts))
            
            # Add batch embeddings to collection
            for embedding in batch_embeddings:
                all_embeddings.append(embedding)
            
            # Save checkpoint after each batch
            checkpoint_data = {
                "total_chunks": len(all_texts),
                "completed_batches": batch_idx + 1,
                "metadata_hash": hash(str(all_meta)),
                "timestamp": time.time()
            }
            
            # Save progress embeddings
            if all_embeddings:
                progress_embeddings = np.array(all_embeddings, dtype="float32")
                dim = progress_embeddings.shape[1]
                progress_index = faiss.IndexFlatL2(dim)
                progress_index.add(progress_embeddings)
                faiss.write_index(progress_index, str(progress_file))
                
                # Save checkpoint metadata
                with checkpoint_file.open("w") as f:
                    json.dump(checkpoint_data, f, indent=2)
                
                print(f"[indexer] Checkpoint saved: {len(all_embeddings)} embeddings processed", flush=True)
            
        except Exception as e:
            print(f"[indexer] Error processing batch {batch_idx + 1}: {e}", flush=True)
            # Don't fail completely, continue with next batch
            continue
    
    # Convert to final numpy array
    if all_embeddings:
        embs = np.array(all_embeddings, dtype="float32")
        print(f"[indexer] Final embedding array shape: {embs.shape}", flush=True)
    else:
        print("[indexer] No embeddings generated, creating empty index", flush=True)
        embs = np.zeros((0, 384), dtype="float32")
    
    # Create and save final index
    dim = embs.shape[1] if embs.size > 0 else 384
    index = faiss.IndexFlatL2(dim)
    if embs.size > 0:
        index.add(embs)
    faiss.write_index(index, str(INDEX_FILE))
    
    # Save enhanced metadata
    with META_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)
    
    # Clean up checkpoint files on successful completion
    checkpoint_file.unlink(missing_ok=True)
    progress_file.unlink(missing_ok=True)
    
    print(f"[indexer] index built with {len(all_texts)} chunks using AST-based chunking", flush=True)
    
    # Build BM25 index from the same data (if available and data exists)
    if all_texts and all_meta:
        bm25_success = build_bm25_index_from_chunks(all_texts, all_meta)
        if bm25_success:
            print("[indexer] Successfully built both vector and BM25 indexes", flush=True)
        else:
            print("[indexer] Vector index built successfully, BM25 index failed (continuing)", flush=True)
    else:
        print("[indexer] No data available for BM25 index building", flush=True)
    
    # Build Knowledge Graph from the same data (Phase 2.1 integration)
    if all_texts and all_meta:
        kg_success = build_knowledge_graph_from_chunks(all_meta, build_mode, project)
        if kg_success:
            print("[indexer] Successfully built Knowledge Graph", flush=True)
        else:
            print("[indexer] Knowledge Graph build failed (continuing)", flush=True)
    else:
        print("[indexer] No data available for Knowledge Graph building", flush=True)
    
    # Log chunking statistics
    chunk_types = {}
    file_types = {}
    projects_indexed = set()
    for meta in all_meta:
        chunk_type = meta.get("chunk_type", "unknown")
        file_type = meta.get("file_type", "unknown")
        project_name = meta.get("project", "unknown")
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        file_types[file_type] = file_types.get(file_type, 0) + 1
        projects_indexed.add(project_name)
    
    print(f"[indexer] Chunk types: {dict(chunk_types)}")
    print(f"[indexer] File types: {dict(file_types)}")
    print(f"[indexer] Projects indexed: {sorted(projects_indexed)}")
    
    # Check if BM25 index exists
    bm25_exists = (CACHE_DIR / "bm25_index.json").exists()
    bm25_status = "✅ Built" if bm25_exists else "❌ Missing"
    
    print(f"[indexer] Summary: {processed_files} processed, {failed_files} failed, {total_chunks} chunks created", flush=True)
    print(f"[indexer] Indexes: Vector ✅ Built | BM25 {bm25_status}", flush=True)

    # Record which projects are actually indexed (based on metadata, not directory listing)
    idx_file = CACHE_DIR / "indexed_projects.json"
    try:
        projects_list = sorted(list(projects_indexed))
        idx_file.write_text(json.dumps(projects_list))
        print(f"[indexer] Updated indexed_projects.json with {len(projects_list)} projects: {projects_list}")
    except Exception as e:
        print(f"[indexer] Warning: could not write indexed_projects.json: {e}")
    _build_lock = False
    # If another rebuild request came in while we were working, run it now
    if _pending_rebuild:
        _pending_rebuild = False
        print("[indexer] running queued rebuild", flush=True)
        build_index()

def file_event_listener():
    if not WATCH_ENABLED:
        print("[indexer] file watching disabled (set INDEXER_WATCH=1 to enable) – idling…", flush=True)
        # Keep the process alive so the container stays up
        import time
        while True:
            time.sleep(3600)
    for _ in watch(str(WORKSPACE)):
        print("[indexer] detected file change – rebuilding index", flush=True)
        build_index()

def ensure_index():
    if not INDEX_FILE.exists() or not META_FILE.exists():
        build_index()

if __name__ == "__main__":
    ensure_index()
    # start watch loop only if enabled
    try:
        file_event_listener()
    except KeyboardInterrupt:
        sys.exit(0) 