from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import subprocess
import aiofiles
from typing import Dict, Any, List
import json
from pathlib import Path

app = FastAPI(title="MCP Server")

# Get workspace directory from environment
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/workspace")

class FileOperation(BaseModel):
    file_path: str
    content: str | None = None

class CommandOperation(BaseModel):
    command: str
    cwd: str | None = None

class DirectoryOperation(BaseModel):
    directory: str

@app.get("/")
async def root():
    return {"message": "MCP Server", "status": "running", "workspace": WORKSPACE_DIR}

@app.post("/tools/read_file")
async def read_file(operation: FileOperation):
    """Read a file from the workspace"""
    try:
        file_path = Path(WORKSPACE_DIR) / operation.file_path
        
        # Security check - ensure path is within workspace
        if not str(file_path.resolve()).startswith(str(Path(WORKSPACE_DIR).resolve())):
            raise HTTPException(status_code=403, detail="Access to file outside workspace is forbidden")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {operation.file_path}")
        
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        return {"result": content, "status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

@app.post("/tools/write_file")
async def write_file(operation: FileOperation):
    """Write content to a file in the workspace"""
    try:
        if operation.content is None:
            raise HTTPException(status_code=400, detail="Content is required for write operation")
        
        file_path = Path(WORKSPACE_DIR) / operation.file_path
        
        # Security check - ensure path is within workspace
        if not str(file_path.resolve()).startswith(str(Path(WORKSPACE_DIR).resolve())):
            raise HTTPException(status_code=403, detail="Access to file outside workspace is forbidden")
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(operation.content)
        
        return {"result": f"File written successfully: {operation.file_path}", "status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing file: {str(e)}")

@app.post("/tools/list_files")
async def list_files(operation: DirectoryOperation):
    """List files in a directory"""
    try:
        dir_path = Path(WORKSPACE_DIR) / operation.directory
        
        # Security check - ensure path is within workspace
        if not str(dir_path.resolve()).startswith(str(Path(WORKSPACE_DIR).resolve())):
            raise HTTPException(status_code=403, detail="Access to directory outside workspace is forbidden")
        
        if not dir_path.exists():
            raise HTTPException(status_code=404, detail=f"Directory not found: {operation.directory}")
        
        files = []
        for item in dir_path.iterdir():
            files.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "path": str(item.relative_to(WORKSPACE_DIR))
            })
        
        return {"result": json.dumps(files, indent=2), "status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

@app.post("/tools/run_command")
async def run_command(operation: CommandOperation):
    """Run a shell command in the workspace"""
    try:
        # Set working directory
        cwd = Path(WORKSPACE_DIR)
        if operation.cwd:
            cwd = cwd / operation.cwd
            
            # Security check
            if not str(cwd.resolve()).startswith(str(Path(WORKSPACE_DIR).resolve())):
                raise HTTPException(status_code=403, detail="Command execution outside workspace is forbidden")
        
        # Run the command
        result = subprocess.run(
            operation.command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\n\nErrors:\n{result.stderr}"
        
        if result.returncode != 0:
            return {
                "result": output,
                "status": "error",
                "return_code": result.returncode
            }
        
        return {"result": output, "status": "success"}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Command execution timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running command: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "workspace": WORKSPACE_DIR,
        "workspace_exists": os.path.exists(WORKSPACE_DIR)
    } 