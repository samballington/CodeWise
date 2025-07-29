from fastapi import FastAPI, BackgroundTasks, status
from indexer.main import build_index, ensure_index, INDEX_FILE, META_FILE, WORKSPACE
from pathlib import Path
import json, os

app = FastAPI(title="CodeWise Indexer API")

# Ensure an index is present when container starts
ensure_index()

@app.post("/rebuild", status_code=status.HTTP_202_ACCEPTED)
async def rebuild(payload: dict | None = None, bt: BackgroundTasks = None):
    """Trigger an index rebuild.

    Optionally pass {"project": "my_project"} to limit indexing to a single top-level directory.
    """
    project = None
    if payload and isinstance(payload, dict):
        project = payload.get("project")
    bt.add_task(build_index, project)
    return {"msg": "re-index started", "scope": project or "all"}

@app.get("/status")
async def status():
    indexed_projects_file = WORKSPACE / ".vector_cache" / "indexed_projects.json"
    projects = []
    if indexed_projects_file.exists():
        try:
            projects = json.loads(indexed_projects_file.read_text())
        except Exception:
            pass
    return {
        "index_exists": INDEX_FILE.exists(),
        "meta_exists": META_FILE.exists(),
        "indexed_projects": projects,
    } 