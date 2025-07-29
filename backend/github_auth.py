import os
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from authlib.integrations.starlette_client import OAuth
from typing import List, Dict, Any
import httpx

router = APIRouter(prefix="/auth", tags=["github"])

oauth = OAuth()
oauth.register(
    name="github",
    client_id=os.getenv("GITHUB_CLIENT_ID"),
    client_secret=os.getenv("GITHUB_CLIENT_SECRET"),
    access_token_url="https://github.com/login/oauth/access_token",
    authorize_url="https://github.com/login/oauth/authorize",
    api_base_url="https://api.github.com/",
    client_kwargs={"scope": "repo user:email"},
)

REDIRECT_URI = "http://localhost:8000/auth/github/callback"

# -------------------------------------------------------
# OAuth login / callback
# -------------------------------------------------------
@router.get("/login/github")
async def login(request: Request):
    """Redirect user to GitHub OAuth consent"""
    return await oauth.github.authorize_redirect(request, REDIRECT_URI)


@router.get("/github/callback")
async def auth_callback(request: Request):
    """Handle GitHub callback and stash token in session"""
    token = await oauth.github.authorize_access_token(request)
    if not token or "access_token" not in token:
        raise HTTPException(status_code=400, detail="Failed to obtain access token")
    request.session["github_token"] = token["access_token"]

    # Close the popup automatically
    html = """
    <html><body><script>
        window.opener && window.opener.postMessage('github-auth-success','*');
        window.close();
    </script><p>You can close this window.</p></body></html>
    """
    return HTMLResponse(content=html)

# Back-compat path so GitHub can hit /oauth/callback
@router.get("/oauth/callback")
async def alias_cb(request: Request):
    return await auth_callback(request)

# -------------------------------------------------------
# API helpers that rely on stored token
# -------------------------------------------------------
async def _get_token(request: Request) -> str:
    token = request.session.get("github_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return token


@router.get("/repos", response_model=List[Dict])
async def list_repos(request: Request):
    """Return user repositories (name, full_name, private, description, updated_at)"""
    token = await _get_token(request)
    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://api.github.com/user/repos",
            headers={"Authorization": f"token {token}"},
            params={"sort": "updated", "per_page": 50},
        )
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail="GitHub API error")
    return r.json()


@router.post("/clone")
async def clone_repo(request: Request, repo: Dict[str, Any]):
    """Clone repo into /workspace using MCP run_command with OAuth token"""
    token = await _get_token(request)
    clone_url = repo.get("clone_url")
    name = repo.get("name")
    if not clone_url or not name:
        raise HTTPException(status_code=400, detail="Missing repo data")

    # Inject token into clone URL
    auth_url = clone_url.replace("https://", f"https://{token}@", 1)
    command = f"git clone --depth 1 {auth_url} {name}"

    mcp_url = os.getenv("MCP_SERVER_URL", "http://mcp_server:8001")
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{mcp_url}/tools/run_command", json={"command": command, "cwd": ""})
    if resp.status_code != 200 or resp.json().get("status") != "success":
        raise HTTPException(status_code=400, detail="Clone failed")

    # Notify indexer
    indexer_url = os.getenv("INDEXER_URL", "http://indexer:8002")
    try:
        await httpx.AsyncClient().post(f"{indexer_url}/rebuild", json={"project": name}, timeout=5)
    except Exception as e:
        print(f"[backend] Warning: could not notify indexer to rebuild: {e}")

    return {"status": "success", "message": f"Cloned {name}"}


@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return {"status": "success"} 