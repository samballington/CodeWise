from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio
from typing import Dict, Any
import os
from dotenv import load_dotenv
from pathlib import Path

from agent import CodeWiseAgent
from chat_memory import ChatMemory
from starlette.middleware.sessions import SessionMiddleware
from github_auth import router as github_oauth_router
from routers.projects import router as projects_router
from api_providers import get_provider_manager
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CodeWise Backend")

# Global agent instance for provider switching
global_agent = None

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session cookies for Auth
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "change_me"))

# Include GitHub OAuth router
app.include_router(github_oauth_router)

# Include Projects router
app.include_router(projects_router)

# Store active connections
active_connections: Dict[str, WebSocket] = {}

@app.get("/")
async def root():
    return {"message": "CodeWise Backend API", "status": "running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connection_id = id(websocket)
    active_connections[connection_id] = websocket
    
    # Initialize or reuse the global agent
    global global_agent
    if global_agent is None:
        global_agent = CodeWiseAgent(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            mcp_server_url=os.getenv("MCP_SERVER_URL", "http://mcp_server:8001")
        )
        logger.info("Global agent initialized")
    
    agent = global_agent
    chat_memory = ChatMemory()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process the message
            if message.get("type") == "user_message":
                user_query = message.get("content", "")
                mentioned_projects = message.get("mentionedProjects", [])
                
                # Log project context if provided
                if mentioned_projects:
                    print(f"[PROJECT SCOPE] User specified projects: {mentioned_projects}")
                
                # Send acknowledgment
                await websocket.send_json({
                    "type": "acknowledgment",
                    "message": "Processing your request..."
                })
                
                # Add user message to memory
                chat_memory.add_message("user", user_query)

                # Execute agent with streaming updates, including project context
                async for update in agent.process_request(
                    user_query, 
                    chat_history=chat_memory.as_langchain_messages(),
                    mentioned_projects=mentioned_projects
                ):
                    # Handle context gathering messages with enhanced logging
                    if update.get("type") in ["context_gathering_start", "context_search", "context_gathering_complete", "context_debug"]:
                        # Log context gathering activities
                        if update.get("type") == "context_gathering_start":
                            print(f"[CONTEXT] Starting context gathering: {update.get('message', '')}")
                        elif update.get("type") == "context_search":
                            print(f"[CONTEXT] Searching {update.get('source', 'unknown source')}: {update.get('query', '')}")
                        elif update.get("type") == "context_gathering_complete":
                            sources = update.get("sources", [])
                            chunks_found = update.get("chunks_found", 0)
                            files_analyzed = update.get("files_analyzed", 0)
                            print(f"[CONTEXT] Complete: {chunks_found} chunks from {files_analyzed} files. Sources: {sources}")
                        elif update.get("type") == "context_debug":
                            print(f"[CONTEXT DEBUG] {update.get('message', '')}")
                            # Print first 500 chars of context for debugging
                            context = update.get("context", "")
                            print(f"[CONTEXT DEBUG] Context preview: {context[:500]}...")
                    
                    # Forward all updates to the frontend
                    await websocket.send_json(update)

                    # If we reach final result, add assistant reply to memory
                    if update.get("type") == "final_result":
                        chat_memory.add_message("assistant", update.get("output", ""))
                
                # Send completion message
                await websocket.send_json({
                    "type": "completion",
                    "message": "Task completed"
                })
                
    except WebSocketDisconnect:
        del active_connections[connection_id]
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Error: {str(e)}"
        })
        del active_connections[connection_id]

# Pydantic models for API provider endpoints
class ProviderSwitchRequest(BaseModel):
    provider: str

# API Provider endpoints
@app.get("/api/provider/info")
async def get_provider_info():
    """Get current API provider information"""
    try:
        provider_manager = get_provider_manager()
        info = provider_manager.get_provider_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get provider info: {str(e)}")

@app.post("/api/provider/switch")
async def switch_provider(request: ProviderSwitchRequest):
    """Switch to a different API provider"""
    try:
        provider_manager = get_provider_manager()
        success = provider_manager.switch_provider(request.provider)
        
        if success:
            # Reinitialize the agent with the new provider
            global global_agent
            if global_agent:
                openai_api_key = os.getenv("OPENAI_API_KEY")
                global_agent.reinitialize_with_provider(openai_api_key)
                logger.info(f"Agent reinitialized after switching to {request.provider}")
            
            return {
                "success": True,
                "message": f"Successfully switched to {request.provider}",
                "current_provider": request.provider
            }
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to switch to provider: {request.provider}"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Provider switch error: {str(e)}")

@app.get("/api/provider/health")
async def get_provider_health():
    """Get health status of all API providers"""
    try:
        provider_manager = get_provider_manager()
        health_status = provider_manager.health_check()
        return health_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "connections": len(active_connections)} 

@app.get("/indexer/status")
async def indexer_status():
    """Return whether the vector index has finished building."""
    ready = Path("/workspace/.vector_cache/index.faiss").exists()
    return {"ready": ready} 