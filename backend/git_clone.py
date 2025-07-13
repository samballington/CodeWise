from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import httpx

router = APIRouter(prefix="/git", tags=["git"])

class CloneRequest(BaseModel):
    """Request model for cloning a repository"""
    repo_url: str  # Full HTTPS URL to the repository
    destination: str | None = None  # Optional folder name inside /workspace
    depth: int | None = 1  # Shallow clone depth (default=1)


@router.post("/clone")
async def clone_repo(req: CloneRequest):
    """Clone a Git repository into the workspace using MCP run_command.

    * If GITHUB_PAT is set in the environment and the repo URL is on GitHub,
      the token is injected into the clone URL so private repos work.
    * The repo is cloned under /workspace/<destination> (defaults to repo name).
    """
    # Prepare destination directory
    dest_dir = req.destination
    if not dest_dir:
        dest_dir = req.repo_url.rstrip("/").split("/")[-1].removesuffix(".git")

    # Inject personal access token when available
    clone_url = req.repo_url
    pat = os.getenv("GITHUB_PAT")
    if pat and clone_url.startswith("https://github.com"):
        # https://github.com/user/repo.git -> https://<PAT>@github.com/user/repo.git
        clone_url = clone_url.replace("https://", f"https://{pat}@", 1)

    command = f"git clone --depth {req.depth or 1} {clone_url} {dest_dir}"

    # Call MCP run_command
    try:
        mcp_url = os.getenv("MCP_SERVER_URL", "http://mcp_server:8001")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{mcp_url}/tools/run_command",
                json={"command": command, "cwd": ""},
                timeout=60,
            )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            raise HTTPException(status_code=400, detail=data.get("result", "Clone failed"))
        return {"status": "success", "message": f"Cloned into /workspace/{dest_dir}"}
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) 