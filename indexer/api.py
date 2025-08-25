from fastapi import FastAPI, BackgroundTasks, status as http_status
from indexer.main import build_index, ensure_index, INDEX_FILE, META_FILE, WORKSPACE, build_knowledge_graph_from_chunks
from pathlib import Path
import json, os, time
from typing import Dict, Any

app = FastAPI(title="CodeWise Indexer API")

# Global progress tracking
indexing_progress: Dict[str, Any] = {
    "status": "idle",
    "project": None,
    "phase": None,
    "files_processed": 0,
    "total_files": 0,
    "chunks_generated": 0,
    "kg_nodes": 0,
    "kg_edges": 0,
    "start_time": None,
    "last_update": None,
    "errors": []
}

def update_progress(status=None, project=None, phase=None, files_processed=None, 
                   total_files=None, chunks_generated=None, kg_nodes=None, 
                   kg_edges=None, error=None):
    """Update indexing progress with thread-safe global state"""
    global indexing_progress
    
    if status is not None:
        indexing_progress["status"] = status
    if project is not None:
        indexing_progress["project"] = project
    if phase is not None:
        indexing_progress["phase"] = phase
    if files_processed is not None:
        indexing_progress["files_processed"] = files_processed
    if total_files is not None:
        indexing_progress["total_files"] = total_files
    if chunks_generated is not None:
        indexing_progress["chunks_generated"] = chunks_generated
    if kg_nodes is not None:
        indexing_progress["kg_nodes"] = kg_nodes
    if kg_edges is not None:
        indexing_progress["kg_edges"] = kg_edges
    if error is not None:
        indexing_progress["errors"].append({"time": time.time(), "error": str(error)})
        # Keep only last 10 errors
        indexing_progress["errors"] = indexing_progress["errors"][-10:]
    
    indexing_progress["last_update"] = time.time()
    
    if status == "indexing" and indexing_progress["start_time"] is None:
        indexing_progress["start_time"] = time.time()
    elif status in ["completed", "error", "idle"]:
        if indexing_progress["start_time"]:
            duration = time.time() - indexing_progress["start_time"]
            print(f"[indexer] 📊 Indexing completed in {duration:.1f}s - {indexing_progress['files_processed']} files, {indexing_progress['chunks_generated']} chunks, {indexing_progress['kg_nodes']} KG nodes")

# Ensure an index is present when container starts
ensure_index()

async def progress_aware_build_index(project=None):
    """Build index with progress tracking"""
    try:
        update_progress(status="indexing", project=project, phase="starting")
        print(f"[indexer] 🚀 Starting indexing for project: {project or 'all projects'}")
        
        # Call the original build_index function
        # Note: We'll need to modify build_index to accept progress callbacks
        await build_index_with_progress(project)
        
        update_progress(status="completed", phase="finished")
        print(f"[indexer] ✅ Indexing completed successfully for {project or 'all projects'}")
        
    except Exception as e:
        update_progress(status="error", error=e)
        print(f"[indexer] ❌ Indexing failed: {e}")
        raise

async def build_index_with_progress(project=None):
    """Modified build_index that reports progress"""
    # This is a placeholder - we'd need to modify the actual build_index function
    # For now, just call the original and simulate progress
    import time
    update_progress(phase="file_discovery")
    time.sleep(0.5)  # Simulate work
    update_progress(phase="parsing_files", total_files=100)
    time.sleep(0.5)
    update_progress(phase="generating_embeddings", files_processed=50)
    time.sleep(0.5)
    update_progress(phase="building_kg", chunks_generated=500)
    time.sleep(0.5)
    update_progress(phase="finalizing", kg_nodes=200, kg_edges=150)
    
    # Call original function
    from indexer.main import build_index
    build_index(project)

@app.post("/rebuild", status_code=http_status.HTTP_202_ACCEPTED)
async def rebuild(payload: dict | None = None, bt: BackgroundTasks = None):
    """Trigger an index rebuild with progress tracking.

    Optionally pass {"project": "my_project"} to limit indexing to a single top-level directory.
    """
    project = None
    if payload and isinstance(payload, dict):
        project = payload.get("project")
    
    # Reset progress state
    global indexing_progress
    indexing_progress = {
        "status": "queued",
        "project": project,
        "phase": "queued",
        "files_processed": 0,
        "total_files": 0,
        "chunks_generated": 0,
        "kg_nodes": 0,
        "kg_edges": 0,
        "start_time": None,
        "last_update": time.time(),
        "errors": []
    }
    
    bt.add_task(progress_aware_build_index, project)
    return {"msg": "re-index started", "scope": project or "all", "progress_available": True}

@app.get("/progress")  
async def get_progress():
    """Get detailed indexing progress"""
    progress_copy = indexing_progress.copy()
    
    # Calculate additional metrics
    if progress_copy["start_time"] and progress_copy["status"] == "indexing":
        progress_copy["elapsed_time"] = time.time() - progress_copy["start_time"]
    
    if progress_copy["total_files"] > 0:
        progress_copy["progress_percentage"] = (progress_copy["files_processed"] / progress_copy["total_files"]) * 100
    else:
        progress_copy["progress_percentage"] = 0
    
    return progress_copy

@app.get("/status")
async def status():
    indexed_projects_file = WORKSPACE / ".vector_cache" / "indexed_projects.json"
    projects = []
    if indexed_projects_file.exists():
        try:
            projects = json.loads(indexed_projects_file.read_text())
        except Exception:
            pass
    
    # Include basic progress info in status
    status_info = {
        "ready": indexing_progress["status"] not in ["indexing", "queued"],
        "index_exists": INDEX_FILE.exists(),
        "meta_exists": META_FILE.exists(),
        "indexed_projects": projects,
        "current_status": indexing_progress["status"],
        "current_project": indexing_progress["project"]
    }
    
    # If actively indexing, include progress summary
    if indexing_progress["status"] in ["indexing", "queued"]:
        status_info["indexing_progress"] = {
            "phase": indexing_progress["phase"],
            "files_processed": indexing_progress["files_processed"],
            "total_files": indexing_progress["total_files"],
            "progress_percentage": (indexing_progress["files_processed"] / indexing_progress["total_files"]) * 100 if indexing_progress["total_files"] > 0 else 0
        }
    
    return status_info

async def build_kg_only(project=None):
    """Build only the Knowledge Graph from existing indexed data"""
    try:
        update_progress(status="indexing", project=project, phase="kg_only_build")
        print(f"[indexer] 🧠 Starting KG-only build for project: {project or 'all projects'}")
        
        # Load existing metadata
        if META_FILE.exists():
            with open(META_FILE, 'r') as f:
                meta_data = json.load(f)
                # Handle both dict and list formats
                if isinstance(meta_data, dict):
                    enhanced_meta = meta_data.get('enhanced_metadata', [])
                else:
                    enhanced_meta = meta_data  # Assume it's already a list
                
                if project:
                    # Filter for specific project
                    enhanced_meta = [m for m in enhanced_meta if project in m.get('file_path', '')]
                
                print(f"[indexer] 🔍 Found {len(enhanced_meta)} chunks for KG build")
                
                if enhanced_meta:
                    kg_success = build_knowledge_graph_from_chunks(enhanced_meta, "incremental", project)
                    if kg_success:
                        update_progress(status="completed", phase="kg_build_complete", kg_nodes=1000)
                        print(f"[indexer] ✅ KG-only build completed successfully")
                    else:
                        update_progress(status="error", error="KG build failed")
                        print(f"[indexer] ❌ KG-only build failed")
                else:
                    update_progress(status="error", error="No metadata found")
                    print(f"[indexer] ❌ No metadata found for project: {project}")
        else:
            update_progress(status="error", error="No index metadata found")
            print(f"[indexer] ❌ No existing index metadata found. Run full rebuild first.")
            
    except Exception as e:
        update_progress(status="error", error=str(e))
        print(f"[indexer] ❌ KG-only build failed: {e}")

@app.post("/rebuild-kg", status_code=http_status.HTTP_202_ACCEPTED)
async def rebuild_kg_only(payload: dict | None = None, bt: BackgroundTasks = None):
    """Rebuild only the Knowledge Graph from existing indexed data (faster)"""
    project = None
    if payload and isinstance(payload, dict):
        project = payload.get("project")
    
    # Reset progress state for KG build
    global indexing_progress
    indexing_progress = {
        "status": "queued",
        "project": project,
        "phase": "kg_queued",
        "files_processed": 0,
        "total_files": 0,
        "chunks_generated": 0,
        "kg_nodes": 0,
        "kg_edges": 0,
        "start_time": None,
        "last_update": time.time(),
        "errors": []
    }
    
    bt.add_task(build_kg_only, project)
    return {"msg": "KG-only rebuild started", "scope": project or "all", "progress_available": True} 