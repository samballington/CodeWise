from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import os
import mimetypes
import openai
from datetime import datetime
from vector_store import get_vector_store
import shutil
from pydantic import BaseModel

openai.api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter(prefix="/projects", tags=["projects"])

# Get workspace directory from environment
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/workspace")

class CloneRequest(BaseModel):
    repo_url: str
    target_name: Optional[str] = None

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
    
    @staticmethod
    async def get_project_summary(project_name: str, query: str = "") -> str:
        """Generate AI summary of a project"""
        try:
            # Search for relevant code if query provided
            if query:
                chunks = get_vector_store().query(f"{project_name} {query}")
            else:
                chunks = get_vector_store().query(f"project overview {project_name}")
            
            # Build context
            context_parts = []
            if chunks:
                for file_path, snippet in chunks[:5]:  # Top 5 relevant chunks
                    context_parts.append(f"File: {file_path}\n```\n{snippet}\n```")
            
            context = "\n\n".join(context_parts) if context_parts else "No relevant code found."
            
            # Generate summary
            prompt_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a code analysis assistant. Given code snippets from a project, "
                        "provide a comprehensive summary of the project's purpose, architecture, "
                        "key features, and technologies used. Keep response under 300 tokens."
                    )
                },
                {
                    "role": "user",
                    "content": f"Project: {project_name}\nQuery: {query or 'general overview'}\n\nCode Context:\n{context}\n\nProvide a project summary:"
                }
            ]
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=prompt_messages,
                max_tokens=300,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Project summary (fallback): {project_name} - Unable to generate detailed summary. {str(e)}"
    
    @staticmethod
    async def get_directory_summary(project_name: str, dir_path: str, query: str = "") -> str:
        """Generate AI summary of a directory"""
        try:
            # Get directory path
            if project_name == "workspace":
                project_path = Path(WORKSPACE_DIR)
            else:
                project_path = Path(WORKSPACE_DIR) / project_name
                if not project_path.exists():
                    raise HTTPException(status_code=404, detail="Project not found")
            
            full_dir_path = project_path / dir_path
            if not full_dir_path.exists() or not full_dir_path.is_dir():
                raise HTTPException(status_code=404, detail="Directory not found")
            
            # Get directory structure
            files = []
            dirs = []
            
            for item in full_dir_path.iterdir():
                if item.is_file() and not item.name.startswith('.'):
                    files.append(item.name)
                elif item.is_dir() and not item.name.startswith('.'):
                    dirs.append(item.name)
            
            # Search for relevant code in this directory
            search_query = f"{project_name} {dir_path} {query}" if query else f"{project_name} {dir_path}"
            chunks = get_vector_store().query(search_query)
            
            # Filter chunks to only include files from this directory
            relevant_chunks = []
            for file_path, snippet in chunks:
                if file_path.startswith(dir_path) or dir_path in file_path:
                    relevant_chunks.append((file_path, snippet))
            
            # Build context
            context_parts = []
            if relevant_chunks:
                for file_path, snippet in relevant_chunks[:3]:  # Top 3 relevant chunks
                    context_parts.append(f"File: {file_path}\n```\n{snippet}\n```")
            
            context = "\n\n".join(context_parts) if context_parts else "No relevant code found."
            
            # Create directory info
            structure_info = []
            if dirs:
                structure_info.append(f"Directories: {', '.join(dirs[:10])}")
            if files:
                structure_info.append(f"Files: {', '.join(files[:15])}")
            
            structure = "; ".join(structure_info)
            
            # Generate summary
            prompt_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a code analysis assistant. Given a directory structure and code snippets, "
                        "provide a summary of what this directory contains, its purpose, and key components. "
                        "Keep response under 200 tokens."
                    )
                },
                {
                    "role": "user",
                    "content": f"Directory: {dir_path}\nProject: {project_name}\nQuery: {query or 'general overview'}\nStructure: {structure}\n\nCode Context:\n{context}\n\nProvide a directory summary:"
                }
            ]
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=prompt_messages,
                max_tokens=200,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Directory summary (fallback): {dir_path} in {project_name} - Unable to generate detailed summary. {str(e)}"
    
    @staticmethod
    def delete_project(project_name: str) -> Dict[str, Any]:
        """Delete a project and clean up its vector embeddings"""
        if project_name == "workspace":
            raise HTTPException(status_code=400, detail="Cannot delete workspace root")
        
        workspace_path = Path(WORKSPACE_DIR)
        project_path = workspace_path / project_name
        
        if not project_path.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not project_path.is_dir():
            raise HTTPException(status_code=400, detail="Can only delete directories")
        
        try:
            # Clean up vector embeddings first
            vector_store = get_vector_store()
            vector_cleanup_success = vector_store.remove_project_embeddings(project_name)
            
            # Delete the project directory
            shutil.rmtree(project_path)
            
            return {
                "success": True,
                "message": f"Project '{project_name}' deleted successfully",
                "vector_cleanup": vector_cleanup_success
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")
    
    @staticmethod
    def clone_any_github_repo(repo_url: str, target_name: str = None) -> Dict[str, Any]:
        """Clone any public GitHub repository"""
        import subprocess
        
        # Parse repository URL or user/repo format
        if repo_url.startswith("http"):
            # Full URL provided
            clone_url = repo_url
            if target_name is None:
                # Extract repo name from URL
                target_name = repo_url.rstrip('/').split('/')[-1]
                if target_name.endswith('.git'):
                    target_name = target_name[:-4]
        else:
            # Assume user/repo format
            if '/' not in repo_url:
                raise HTTPException(status_code=400, detail="Invalid repository format. Use 'user/repo' or full URL")
            
            clone_url = f"https://github.com/{repo_url}.git"
            if target_name is None:
                target_name = repo_url.split('/')[-1]
        
        workspace_path = Path(WORKSPACE_DIR)
        target_path = workspace_path / target_name
        
        # Check if project already exists
        if target_path.exists():
            raise HTTPException(status_code=400, detail=f"Project '{target_name}' already exists")
        
        try:
            # Clone the repository
            result = subprocess.run([
                "git", "clone", clone_url, str(target_path)
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode != 0:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Failed to clone repository: {result.stderr}"
                )
            
            # Get project info
            project_size = ProjectService._get_directory_size(target_path)
            
            return {
                "success": True,
                "message": f"Repository cloned successfully as '{target_name}'",
                "project_name": target_name,
                "project_path": str(target_path.relative_to(workspace_path)),
                "size": project_size,
                "clone_url": clone_url
            }
            
        except subprocess.TimeoutExpired:
            # Clean up partial clone on timeout
            if target_path.exists():
                shutil.rmtree(target_path)
            raise HTTPException(status_code=408, detail="Clone operation timed out")
        except Exception as e:
            # Clean up partial clone on error
            if target_path.exists():
                shutil.rmtree(target_path)
            raise HTTPException(status_code=500, detail=f"Clone failed: {str(e)}")

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

@router.get("/{project_name}/summary")
async def get_project_summary(
    project_name: str,
    query: str = Query("", description="Optional query to focus the summary")
):
    """Get an AI-generated summary of the project"""
    try:
        summary = await ProjectService.get_project_summary(project_name, query)
        return JSONResponse(content={"summary": summary, "project": project_name})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_name}/dir/{path:path}")
async def get_directory_summary(
    project_name: str,
    path: str,
    query: str = Query("", description="Optional query to focus the summary")
):
    """Get an AI-generated summary of a directory"""
    try:
        summary = await ProjectService.get_directory_summary(project_name, path, query)
        return JSONResponse(content={"summary": summary, "project": project_name, "path": path})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{project_name}")
async def delete_project(project_name: str):
    """Delete a project and its vector embeddings"""
    try:
        result = ProjectService.delete_project(project_name)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clone")
async def clone_github_repo(request: CloneRequest):
    """Clone any public GitHub repository"""
    try:
        result = ProjectService.clone_any_github_repo(request.repo_url, request.target_name)
        return JSONResponse(content=result)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clone failed: {str(e)}")