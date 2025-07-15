import os, time, json, sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
# type: ignore
import openai
openai.log = "warning"
import faiss
from watchfiles import watch

WORKSPACE = Path(os.getenv("WORKSPACE_DIR", "/workspace"))
CACHE_DIR = WORKSPACE / ".vector_cache"
CACHE_DIR.mkdir(exist_ok=True)
INDEX_FILE = CACHE_DIR / "index.faiss"
META_FILE = CACHE_DIR / "meta.json"
CHUNK_SIZE = 400
EMBED_MODEL = "text-embedding-3-small"
openai.api_key = os.getenv("OPENAI_API_KEY")

def chunk_text(text: str) -> List[str]:
    return [text[i:i+CHUNK_SIZE] for i in range(0,len(text),CHUNK_SIZE)]

def list_files() -> List[Path]:
    allowed_ext = {".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".txt", ".json", ".html", ".css"}
    paths: List[Path] = []
    for root, _, files in os.walk(WORKSPACE):
        for f in files:
            p = Path(root)/f
            if p.suffix in allowed_ext and not p.name.startswith('.'):
                paths.append(p)
    return paths

def embed_batch(texts: List[str]):
    embs = []
    for i in range(0,len(texts),100):
        batch = texts[i:i+100]
        resp = openai.embeddings.create(model=EMBED_MODEL, input=batch)
        embs.extend([d.embedding for d in resp.data])
    return np.array(embs).astype("float32")

def build_index():
    print("[indexer] building vector index…", flush=True)
    texts: List[str] = []
    meta: List[Tuple[str,str]] = []
    for file in list_files():
        try:
            content = file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for chunk in chunk_text(content):
            texts.append(chunk)
            meta.append((str(file.relative_to(WORKSPACE)), chunk))
    if not texts:
        print("[indexer] no text files found")
        return
    embs = embed_batch(texts)
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    faiss.write_index(index, str(INDEX_FILE))
    with META_FILE.open("w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"[indexer] index built with {len(texts)} chunks", flush=True)

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