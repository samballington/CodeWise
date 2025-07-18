from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from typing import Dict, Any
import os
from dotenv import load_dotenv

from agent import CodeWiseAgent
from chat_memory import ChatMemory
from starlette.middleware.sessions import SessionMiddleware
from github_auth import router as github_oauth_router
from routers.projects import router as projects_router

# Load environment variables
load_dotenv()

app = FastAPI(title="CodeWise Backend")

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
    
    # Initialize the agent and chat memory
    agent = CodeWiseAgent(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        mcp_server_url=os.getenv("MCP_SERVER_URL", "http://mcp_server:8001")
    )
    chat_memory = ChatMemory()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process the message
            if message.get("type") == "user_message":
                user_query = message.get("content", "")
                
                # Send acknowledgment
                await websocket.send_json({
                    "type": "acknowledgment",
                    "message": "Processing your request..."
                })
                
                # Add user message to memory
                chat_memory.add_message("user", user_query)

                # Execute agent with streaming updates
                async for update in agent.process_request(user_query, chat_history=chat_memory.as_langchain_messages()):
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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "connections": len(active_connections)} 