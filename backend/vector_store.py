import os
import json
from typing import List, Tuple
import numpy as np
import faiss
# type: ignore
import openai
from pathlib import Path

openai.api_key = os.getenv("OPENAI_API_KEY")

WORKSPACE_DIR = "/workspace"
CACHE_DIR = Path(WORKSPACE_DIR) / ".vector_cache"
INDEX_FILE = CACHE_DIR / "index.faiss"
META_FILE = CACHE_DIR / "meta.json"
CHUNK_SIZE = 400  # characters per chunk
EMBED_MODEL = "text-embedding-3-small"


class VectorStore:
    """Simple FAISS-backed vector store for workspace code retrieval."""

    def __init__(self, workspace_dir: str = "/workspace"):
        self.workspace_dir = workspace_dir
        self.index_path = INDEX_FILE
        self.meta_path = META_FILE
        self.index = None  # type: ignore
        self.meta: List[Tuple[str, str]] = []  # (file_path, snippet)

        if self.index_path.exists() and self.meta_path.exists():
            self._load()
        else:
            # If index not present, fallback to empty index (indexer builds asynchronously)
            self.index = faiss.IndexFlatL2(768)

    # --------------------------- internal helpers ---------------------------
    def _chunk_text(self, text: str) -> List[str]:
        return [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    def _list_files(self) -> List[Path]:
        paths = []
        for root, _, files in os.walk(self.workspace_dir):
            for f in files:
                if f.startswith("."):
                    continue
                if any(f.endswith(ext) for ext in [
                    ".py",
                    ".js",
                    ".ts",
                    ".tsx",
                    ".jsx",
                    ".md",
                    ".txt",
                    ".json",
                    ".html",
                    ".css",
                ]):
                    paths.append(Path(root) / f)
        return paths

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        # split into batches of 100
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            resp = openai.embeddings.create(model=EMBED_MODEL, input=batch)
            embeddings.extend([d.embedding for d in resp.data])
        return embeddings

    def _build(self):
        texts = []
        self.meta = []
        for path in self._list_files():
            try:
                content = Path(path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            chunks = self._chunk_text(content)
            for chunk in chunks:
                texts.append(chunk)
                rel_path = str(Path(path).relative_to(self.workspace_dir))
                self.meta.append((rel_path, chunk))

        if not texts:
            self.index = faiss.IndexFlatL2(768)
            return

        embeddings = np.array(self._embed_batch(texts)).astype("float32")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        self.index = index
        faiss.write_index(index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f)

    def _load(self):
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    # --------------------------- public API ---------------------------
    def query(self, query: str, k: int = 3) -> List[Tuple[str, str]]:
        if self.index is None or not self.meta:
            return []
        emb_vec = np.array(self._embed_batch([query])[0]).astype("float32").reshape(1, -1)
        distances, indices = self.index.search(emb_vec, k)
        results: List[Tuple[str, str]] = []
        for idx in indices[0]:
            if idx < len(self.meta):
                results.append(tuple(self.meta[idx]))
        return results

# Singleton for agent
_vector_store_instance: VectorStore | None = None

def get_vector_store() -> VectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance 