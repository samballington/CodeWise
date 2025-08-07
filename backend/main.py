from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio
from typing import Dict, Any
import os
import logging
from datetime import datetime
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
            
            # Handle keep-alive ping
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            # Process the message
            if message.get("type") == "user_message":
                user_query = message.get("content", "")
                mentioned_projects = message.get("mentionedProjects", [])
                
                # Quick fix: Extract @mentions from query if not provided by frontend
                if not mentioned_projects and user_query:
                    import re
                    mentions = re.findall(r'@([a-zA-Z0-9_-]+)', user_query)
                    mentioned_projects = mentions
                
                # Persistent project context: maintain project context across conversation
                if not hasattr(websocket, 'active_project_context'):
                    websocket.active_project_context = None
                
                # If new @mention found, update active context
                if mentioned_projects:
                    websocket.active_project_context = mentioned_projects[0]  # Use first mentioned project
                    logger.info(f"🎯 PROJECT CONTEXT SET: {websocket.active_project_context}")
                
                # If no @mention but we have active context, use it
                elif websocket.active_project_context:
                    mentioned_projects = [websocket.active_project_context]
                    logger.info(f"🔄 USING ACTIVE PROJECT CONTEXT: {websocket.active_project_context}")
                
                # Log the complete input query
                logger.info("="*80)
                logger.info("🔍 FULL USER QUERY INPUT:")
                logger.info(f"Query: {user_query}")
                logger.info(f"Mentioned Projects: {mentioned_projects}")
                logger.info(f"Timestamp: {datetime.now().isoformat()}")
                logger.info("="*80)
                
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
                        final_output = update.get("output", "")
                        
                        # Task 5: Log enhanced response formatting if available
                        formatted_response = update.get("formatted_response")
                        if formatted_response:
                            logger.info("="*80)
                            logger.info("📝 ENHANCED RESPONSE OUTPUT (Task 5):")
                            logger.info(f"Query Type: {formatted_response.get('query_type', 'unknown')}")
                            logger.info(f"Confidence: {formatted_response.get('confidence_level', 'unknown')} ({formatted_response.get('confidence_score', 0):.2f})")
                            logger.info(f"Code Snippets: {len(formatted_response.get('code_snippets', []))}")
                            logger.info(f"File References: {len(formatted_response.get('file_references', []))}")
                            logger.info(f"Tools Used: {len(formatted_response.get('tools_used', []))}")
                            logger.info(f"Response Time: {formatted_response.get('response_time', 0):.2f}s")
                            logger.info("="*80)
                        else:
                            # Log the complete response output (original format)
                            logger.info("="*80)
                            logger.info("📝 FULL AGENT RESPONSE OUTPUT:")
                            logger.info(f"Response: {final_output}")
                            logger.info(f"Response Length: {len(final_output)} chars")
                            logger.info(f"Timestamp: {datetime.now().isoformat()}")
                            logger.info("="*80)
                        
                        chat_memory.add_message("assistant", final_output)
                
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

# API Provider endpoints removed - OpenAI functionality disabled

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

@app.post("/indexer/refresh")
async def refresh_vector_store():
    """Force refresh the vector store from disk"""
    try:
        from vector_store import get_vector_store
        vs = get_vector_store()
        vs.force_refresh()
        return {"status": "success", "message": "Vector store refreshed from disk"}
    except Exception as e:
        logger.error(f"Failed to refresh vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh vector store: {str(e)}") 