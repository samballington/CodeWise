import os, time, json, sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
# type: ignore
import openai
openai.log = "warning"
import faiss
from watchfiles import watch
from file_discovery import FileDiscoveryEngine, FileInfo
from ast_chunker import ASTChunker, CodeChunk
from complexity import choose_chunk_size

WORKSPACE = Path(os.getenv("WORKSPACE_DIR", "/workspace"))
CACHE_DIR = WORKSPACE / ".vector_cache"
CACHE_DIR.mkdir(exist_ok=True)
INDEX_FILE = CACHE_DIR / "index.faiss"
META_FILE = CACHE_DIR / "meta.json"
STATS_FILE = CACHE_DIR / "discovery_stats.json"
CHUNK_SIZE = 400
EMBED_MODEL = "text-embedding-3-small"
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    embs = []
    batch_size = 5  # Reduced from 100 to 5 chunks per batch to avoid token limit
    
    print(f"[indexer] Processing {len(texts)} chunks in batches of {batch_size}")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        try:
            # Validate and clean batch input
            clean_batch = []
            for text in batch:
                if isinstance(text, str) and text.strip():
                    clean_batch.append(text.strip())
                else:
                    clean_batch.append("empty")  # Replace invalid content with placeholder
            
            if not clean_batch:
                print(f"[indexer] Batch {i}-{i+batch_size} contains no valid text, skipping", flush=True)
                for _ in range(len(batch)):
                    embs.append([0.0] * 1536)
                continue
            
            print(f"[indexer] Sending batch of {len(clean_batch)} items to OpenAI", flush=True)
            print(f"[indexer] Sample batch item: {clean_batch[0][:100] if clean_batch else 'empty'}", flush=True)
            resp = openai.embeddings.create(model=EMBED_MODEL, input=clean_batch)
            embs.extend([d.embedding for d in resp.data])
            
            # Progress logging every 50 chunks
            if i % 50 == 0:
                progress = (i / len(texts)) * 100
                print(f"[indexer] Processed {i}/{len(texts)} chunks ({progress:.1f}%)", flush=True)
                
        except openai.BadRequestError as e:
            if "maximum context length" in str(e):
                print(f"[indexer] Batch {i}-{i+batch_size} exceeded token limit, skipping", flush=True)
                # Add empty embeddings to maintain alignment
                for _ in range(len(batch)):
                    embs.append([0.0] * 1536)  # text-embedding-3-small dimension
            else:
                print(f"[indexer] API error for batch {i}-{i+batch_size}: {e}", flush=True)
                # Add empty embeddings to maintain alignment
                for _ in range(len(batch)):
                    embs.append([0.0] * 1536)
                continue
                
        except Exception as e:
            print(f"[indexer] Unexpected error for batch {i}-{i+batch_size}: {e}", flush=True)
            # Add empty embeddings to maintain alignment
            for _ in range(len(batch)):
                embs.append([0.0] * 1536)
            continue
    
    if not embs:
        print("[indexer] No embeddings generated, creating empty index", flush=True)
        return np.array([]).reshape(0, 1536).astype("float32")
    
    print(f"[indexer] Successfully generated {len(embs)} embeddings", flush=True)
    return np.array(embs).astype("float32")

def build_index():
    print("[indexer] building vector index with AST chunking…", flush=True)
    texts: List[str] = []
    enhanced_meta: List[dict] = []  # Enhanced metadata with chunk information
    
    # Initialize AST chunker
    ast_chunker = ASTChunker()
    
    # Use enhanced file discovery
    discovered_files = discover_files()
    
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
        
        # Use AST chunker to create intelligent chunks
        try:
            code_chunks = ast_chunker.chunk_content(content, file_info.path)
            
            for chunk in code_chunks:
                texts.append(chunk.text)
                
                # Create enhanced metadata
                chunk_meta = {
                    "file_path": chunk.file_path,
                    "relative_path": file_info.relative_path,
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
                chunk_meta = {
                    "file_path": str(file_info.path),
                    "relative_path": file_info.relative_path,
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

def file_event_listener():
    for _ in watch(str(WORKSPACE)):
        print("[indexer] detected file change – rebuilding index", flush=True)
        build_index()

def ensure_index():
    if not INDEX_FILE.exists() or not META_FILE.exists():
        build_index()

if __name__ == "__main__":
    ensure_index()
    # start watching in foreground
    try:
        file_event_listener()
    except KeyboardInterrupt:
        sys.exit(0) 