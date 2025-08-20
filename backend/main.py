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

from websocket_adapter_simple import CodeWiseAgent
from chat_memory import ChatMemory
from starlette.middleware.sessions import SessionMiddleware
from github_auth import router as github_oauth_router
from routers.projects import router as projects_router
from api_providers import get_provider_manager
from services import run_startup_kg_population
import time
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

# REQ-3.7.1: KG startup population flag
_kg_startup_completed = False

# DISABLED: This causes 10+ minute startup delays on every restart
# @app.on_event("startup")
# async def startup_event():
#     """
#     Backend startup event - REQ-3.7.1: Automatic KG population
#     """
#     global _kg_startup_completed
#     
#     logger.info("🚀 Backend startup: Beginning automatic Knowledge Graph population")
#     
#     # Run KG population in background (non-blocking)
#     asyncio.create_task(populate_kg_on_startup())

@app.on_event("startup")
async def startup_event():
    """
    Backend startup event - Fast startup with incremental KG indexing
    REQ-3.9.2: Fast startup mode with new project detection
    """
    global _kg_startup_completed
    
    logger.info("🚀 Backend startup: Fast mode with incremental KG indexing")
    logger.info("💡 Existing KG data preserved - checking for new projects to index")
    
    # Run incremental KG population in background (non-blocking)
    asyncio.create_task(populate_incremental_kg_on_startup())
    
    # Mark as completed to avoid blocking queries (incremental runs in background)
    _kg_startup_completed = True

async def populate_kg_on_startup():
    """
    Background task for KG population - REQ-3.7.1
    """
    global _kg_startup_completed
    
    try:
        logger.info("📊 Starting Knowledge Graph population for all workspace projects...")
        result = await run_startup_kg_population()
        
        logger.info(f"✅ KG startup completed: {result.successful_projects}/{result.total_projects} projects "
                   f"indexed in {result.total_processing_time:.2f}s")
        
        _kg_startup_completed = True
        
    except Exception as e:
        logger.error(f"❌ KG startup population failed: {e}")
        _kg_startup_completed = False

async def populate_incremental_kg_on_startup():
    """
    Background task for incremental KG population - REQ-3.9.2
    Only indexes new projects, preserving fast startup
    """
    global _kg_startup_completed
    
    try:
        logger.info("📊 Starting incremental Knowledge Graph indexing for new projects...")
        
        # Import the service
        from services import get_kg_startup_service
        service = get_kg_startup_service()
        
        # Run incremental indexing (only new projects)
        result = await service.populate_all_projects(incremental=True)
        
        if result.project_results:
            logger.info(f"✅ Incremental KG indexing completed: {result.successful_projects} new projects "
                       f"indexed in {result.total_processing_time:.2f}s")
        else:
            logger.info("✅ All projects already indexed - no new projects found")
        
        if result.failed_projects > 0:
            logger.warning(f"⚠️ {result.failed_projects} projects failed indexing")
        
    except Exception as e:
        logger.error(f"❌ Incremental KG indexing failed: {e}")
        # Don't mark as failed - queries can still work with existing data

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
        global_agent = CodeWiseAgent()
        logger.info("Global SDK-compatible agent initialized")
    
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
                # Start timing this request
                request_start_time = time.time()
                logger.info("⏱️  REQUEST START: Processing user message")
                
                user_query = message.get("content", "")
                mentioned_projects = message.get("mentionedProjects", [])
                selected_model = message.get("model", "gpt-oss-120b")  # Default to current model
                
                # Validate model selection
                SUPPORTED_MODELS = ["gpt-oss-120b", "qwen-3-coder-480b", "qwen-3-235b-a22b-thinking-2507"]
                if selected_model not in SUPPORTED_MODELS:
                    logger.warning(f"Unsupported model requested: {selected_model}, defaulting to gpt-oss-120b")
                    selected_model = "gpt-oss-120b"
                
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
                
                # Send acknowledgment (with connection check)
                try:
                    if websocket.client_state.name == 'CONNECTED':
                        await websocket.send_json({
                            "type": "acknowledgment",
                            "message": "Processing your request..."
                        })
                except Exception as ack_error:
                    logger.warning(f"Failed to send acknowledgment to WebSocket {connection_id}: {ack_error}")
                    continue  # Skip processing if we can't acknowledge
                
                # Add user message to memory
                chat_memory.add_message("user", user_query)

                # Execute agent with streaming updates, including project context and model selection
                async for update in agent.process_request(
                    user_query, 
                    chat_history=chat_memory.as_langchain_messages(),
                    mentioned_projects=mentioned_projects,
                    selected_model=selected_model
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
                    
                    # Forward all updates to the frontend (with connection check)
                    try:
                        if websocket.client_state.name == 'CONNECTED':
                            await websocket.send_json(update)
                        else:
                            logger.warning(f"Skipping message to disconnected WebSocket {connection_id}")
                            break
                    except Exception as send_error:
                        logger.warning(f"Failed to send update to WebSocket {connection_id}: {send_error}")
                        break

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
                
                # End timing and log total request time
                total_request_time = time.time() - request_start_time
                logger.info(f"✅ REQUEST END: Total processing time ({total_request_time:.3f}s)")
                
                # Send completion message (with connection check)
                try:
                    if websocket.client_state.name == 'CONNECTED':
                        await websocket.send_json({
                            "type": "completion",
                            "message": "Task completed"
                        })
                except Exception as completion_error:
                    logger.warning(f"Failed to send completion message to WebSocket {connection_id}: {completion_error}")
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket {connection_id} disconnected normally")
        if connection_id in active_connections:
            del active_connections[connection_id]
    except Exception as e:
        logger.error(f"WebSocket {connection_id} error: {str(e)}")
        # Only try to send error message if WebSocket is still open
        try:
            if websocket.client_state.name == 'CONNECTED':
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error: {str(e)}"
                })
        except Exception as send_error:
            logger.warning(f"Could not send error message to closed WebSocket {connection_id}: {send_error}")
        finally:
            if connection_id in active_connections:
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

@app.get("/api/kg/status")
async def kg_status():
    """Get Knowledge Graph population status - REQ-3.7.1"""
    try:
        from services import get_kg_startup_service
        service = get_kg_startup_service()
        status = service.get_kg_status()
        
        return {
            "startup_completed": _kg_startup_completed,
            **status
        }
    except Exception as e:
        return {
            "startup_completed": _kg_startup_completed,
            "status": "error",
            "error": str(e)
        } 

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

# REQ-CACHE-8: Cache Performance Monitoring API Endpoints
@app.get("/api/cache/performance")
async def get_cache_performance():
    """Get comprehensive cache performance dashboard"""
    try:
        from cache.performance_monitor import get_global_performance_monitor
        monitor = get_global_performance_monitor()
        
        # Start monitoring if not already running
        if not monitor._monitoring:
            monitor.start_monitoring()
        
        dashboard = monitor.get_performance_dashboard()
        return dashboard
    except Exception as e:
        logger.error(f"Failed to get cache performance: {e}")
        raise HTTPException(status_code=500, detail=f"Cache performance error: {str(e)}")

@app.post("/api/cache/optimize")
async def force_cache_optimization():
    """Force immediate cache optimization"""
    try:
        from cache.performance_monitor import get_global_performance_monitor
        monitor = get_global_performance_monitor()
        monitor.force_optimization()
        return {"status": "success", "message": "Cache optimization completed"}
    except Exception as e:
        logger.error(f"Failed to optimize cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache optimization error: {str(e)}")

@app.get("/api/cache/metrics")
async def get_cache_metrics():
    """Get detailed cache metrics from all layers"""
    try:
        from cache.cache_metrics import get_global_cache_metrics
        metrics = get_global_cache_metrics()
        
        aggregated = metrics.get_aggregated_metrics()
        recommendations = metrics.get_optimization_recommendations()
        
        return {
            "metrics": aggregated,
            "recommendations": recommendations,
            "performance_report": metrics.generate_performance_report()
        }
    except Exception as e:
        logger.error(f"Failed to get cache metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Cache metrics error: {str(e)}")

@app.post("/api/cache/reset")
async def reset_cache_monitoring():
    """Reset cache monitoring data and statistics"""
    try:
        from cache.performance_monitor import get_global_performance_monitor
        monitor = get_global_performance_monitor()
        monitor.reset_monitoring_data()
        return {"status": "success", "message": "Cache monitoring data reset"}
    except Exception as e:
        logger.error(f"Failed to reset cache monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Cache reset error: {str(e)}") 