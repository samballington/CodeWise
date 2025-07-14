from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import os
import mimetypes
from datetime import datetime

router = APIRouter(prefix="/projects", tags=["projects"])

# Get workspace directory from environment
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/workspace")

class ProjectService:
    @staticmethod
    def get_projects() -> List[Dict[str, Any]]:
        """Get list of all projects (directories) in workspace, plus workspace root files"""
        workspace_path = Path(WORKSPACE_DIR)
        projects = []
        
        if not workspace_path.exists():
            return projects
        
        # Add workspace root as special project if it contains files
        workspace_files = [f for f in workspace_path.iterdir() if f.is_file() and not f.name.startswith('.')]
        if workspace_files:
            total_size = sum(f.stat().st_size for f in workspace_files)
            modified_time = max((f.stat().st_mtime for f in workspace_files), default=workspace_path.stat().st_mtime)
            
            projects.append({
                "name": "workspace",
                "path": ".",
                "modified": datetime.fromtimestamp(modified_time).isoformat(),
                "size": total_size,
                "is_workspace_root": True
            })
            
        # Add regular project directories
        for item in workspace_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Get basic project info
                project_info = {
                    "name": item.name,
                    "path": str(item.relative_to(workspace_path)),
                    "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                    "size": ProjectService._get_directory_size(item),
                    "is_workspace_root": False
                }
                projects.append(project_info)
        
        return sorted(projects, key=lambda x: x['modified'], reverse=True)
    
    @staticmethod
    def _get_directory_size(directory: Path) -> int:
        """Calculate total size of directory"""
        total_size = 0
        try:
            for item in directory.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
        except (OSError, PermissionError):
            pass
        return total_size
    
    @staticmethod
    def get_file_tree(project_name: str, path: str = "") -> Dict[str, Any]:
        """Get file tree structure for a project"""
        # Handle workspace root specially
        if project_name == "workspace":
            project_path = Path(WORKSPACE_DIR)
            target_path = project_path / path if path else project_path
        else:
            project_path = Path(WORKSPACE_DIR) / project_name
            if not project_path.exists():
                raise HTTPException(status_code=404, detail="Project not found")
            target_path = project_path / path if path else project_path
        
        if not target_path.exists():
            raise HTTPException(status_code=404, detail="Path not found")
        
        return ProjectService._build_tree_node(target_path, project_path, project_name == "workspace")
    
    @staticmethod
    def _build_tree_node(item_path: Path, project_root: Path, is_workspace_root: bool = False) -> Dict[str, Any]:
        """Build a tree node for a file or directory"""
        relative_path = str(item_path.relative_to(project_root))
        if relative_path == ".":
            relative_path = ""
        
        node = {
            "name": item_path.name,
            "path": relative_path,
            "type": "directory" if item_path.is_dir() else "file",
            "modified": datetime.fromtimestamp(item_path.stat().st_mtime).isoformat()
        }
        
        if item_path.is_file():
            node["size"] = item_path.stat().st_size
            node["mime_type"] = mimetypes.guess_type(str(item_path))[0] or "text/plain"
        else:
            # For directories, add children
            children = []
            try:
                for child in sorted(item_path.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
                    # For workspace root, only show files (not subdirectories)
                    if not child.name.startswith('.'):
                        if is_workspace_root and child.is_dir():
                            continue  # Skip directories in workspace root view
                        children.append(ProjectService._build_tree_node(child, project_root, is_workspace_root))
            except (OSError, PermissionError):
                pass
            node["children"] = children
        
        return node
    
    @staticmethod
    def get_file_content(project_name: str, file_path: str) -> Dict[str, Any]:
        """Get content of a specific file"""
        # Handle workspace root specially
        if project_name == "workspace":
            project_path = Path(WORKSPACE_DIR)
            target_file = project_path / file_path
        else:
            project_path = Path(WORKSPACE_DIR) / project_name
            if not project_path.exists():
                raise HTTPException(status_code=404, detail="Project not found")
            target_file = project_path / file_path
        
        if not target_file.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if not target_file.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")
        
        # Security check - ensure file is within project
        try:
            target_file.resolve().relative_to(project_path.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")
        
        try:
            # Try to read as text first
            with open(target_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "content": content,
                "path": file_path,
                "name": target_file.name,
                "size": target_file.stat().st_size,
                "modified": datetime.fromtimestamp(target_file.stat().st_mtime).isoformat(),
                "mime_type": mimetypes.guess_type(str(target_file))[0] or "text/plain",
                "encoding": "utf-8"
            }
        except UnicodeDecodeError:
            # If it's a binary file, return metadata only
            return {
                "content": None,
                "path": file_path,
                "name": target_file.name,
                "size": target_file.stat().st_size,
                "modified": datetime.fromtimestamp(target_file.stat().st_mtime).isoformat(),
                "mime_type": mimetypes.guess_type(str(target_file))[0] or "application/octet-stream",
                "encoding": "binary",
                "is_binary": True
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

@router.get("/")
async def list_projects():
    """List all projects in the workspace"""
    try:
        projects = ProjectService.get_projects()
        return {"projects": projects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing projects: {str(e)}")

@router.get("/{project_name}/tree")
async def get_project_tree(project_name: str, path: str = Query("", description="Path within project")):
    """Get file tree for a specific project"""
    try:
        tree = ProjectService.get_file_tree(project_name, path)
        return tree
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting file tree: {str(e)}")

@router.get("/{project_name}/file")
async def get_file_content(project_name: str, path: str = Query(..., description="File path within project")):
    """Get content of a specific file"""
    try:
        file_data = ProjectService.get_file_content(project_name, path)
        return file_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting file content: {str(e)}") 