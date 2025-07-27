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

def embed_batch(texts: List[str]):
    """Generate embeddings locally using MiniLM; returns float32 numpy array."""
    import numpy as np
    try:
        vectors = _local_embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vectors, dtype="float32")
    except Exception as e:
        print(f"[indexer] Local embedding error: {e}", flush=True)
        return np.zeros((len(texts), 384), dtype="float32")

def build_index(project: str | None = None):
    global _build_lock, _pending_rebuild
    if _build_lock:
        # A build is already running – queue another one so that new files are guaranteed to be indexed
        print("[indexer] build already in progress – queueing another rebuild trigger", flush=True)
        _pending_rebuild = True
        return
    _build_lock = True
    print("[indexer] building vector index with AST chunking…", flush=True)
    texts: List[str] = []
    enhanced_meta: List[dict] = []  # Enhanced metadata with chunk information
    
    # Initialize AST chunker
    ast_chunker = ASTChunker()
    
    # If a project scope is provided, adjust the root dir
    root_dir: Path = WORKSPACE if project is None else WORKSPACE / project
    if project and not root_dir.exists():
        print(f"[indexer] Requested project '{project}' not found – falling back to whole workspace", flush=True)
        root_dir = WORKSPACE
    
    # Discover files within the requested scope
    if root_dir == WORKSPACE:
        discovered_files = discover_files()
    else:
        # Temporarily override WORKSPACE for discovery
        original_workspace = WORKSPACE
        try:
            globals()["WORKSPACE"] = root_dir
            discovered_files = discover_files()
        finally:
            globals()["WORKSPACE"] = original_workspace
    
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

                # Derive project name (first path component)
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
    
    if not texts:
        print("[indexer] no text files found")
        return
        
    print(f"[indexer] processed {processed_files} files into {total_chunks} intelligent chunks", flush=True)
    print(f"[indexer] generating embeddings for {len(texts)} chunks", flush=True)
    
    embs = embed_batch(texts)
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    faiss.write_index(index, str(INDEX_FILE))
    
    # Save enhanced metadata
    with META_FILE.open("w", encoding="utf-8") as f:
        json.dump(enhanced_meta, f, indent=2)
    
    print(f"[indexer] index built with {len(texts)} chunks using AST-based chunking", flush=True)
    
    # Log chunking statistics
    chunk_types = {}
    file_types = {}
    for meta in enhanced_meta:
        chunk_type = meta.get("chunk_type", "unknown")
        file_type = meta.get("file_type", "unknown")
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        file_types[file_type] = file_types.get(file_type, 0) + 1
    
    print(f"[indexer] Chunk types: {dict(chunk_types)}")
    print(f"[indexer] File types: {dict(file_types)}")
    print(f"[indexer] Summary: {processed_files} processed, {failed_files} failed, {total_chunks} chunks created", flush=True)

    # Record which top-level projects are indexed
    projects = [p.name for p in Path(WORKSPACE).iterdir() if p.is_dir() and not p.name.startswith('.')]
    idx_file = CACHE_DIR / "indexed_projects.json"
    try:
        idx_file.write_text(json.dumps(projects))
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