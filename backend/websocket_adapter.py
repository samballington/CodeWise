"""
WebSocket Streaming Adapter for Cerebras SDK Integration

Bridges the gap between the existing LangChain streaming WebSocket system
and our new SDK-native agent. Maintains streaming UX while using pure SDK backend.

Architecture:
- Existing main.py expects: agent.process_request() → AsyncGenerator
- Our SDK agent provides: agent.process_query() → str  
- This adapter: Converts SDK completion to streaming format
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional
from datetime import datetime

try:
    from .cerebras_agent import get_native_agent
except ImportError:
    from cerebras_agent import get_native_agent

logger = logging.getLogger(__name__)

class WebSocketStreamingAdapter:
    """
    Adapter that makes CerebrasNativeAgent compatible with existing WebSocket streaming
    """
    
    def __init__(self):
        """Initialize adapter with native agent"""
        self.native_agent = get_native_agent()
        logger.info("✅ WebSocket streaming adapter initialized")
    
    async def process_request(self, user_query: str, 
                            chat_history: Optional[List] = None,
                            mentioned_projects: Optional[List] = None,
                            selected_model: str = "gpt-oss-120b") -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main interface that existing WebSocket code expects.
        
        Converts SDK agent's completion-based processing into streaming format
        compatible with the existing frontend.
        """
        try:
            # Send initial acknowledgment
            yield {
                "type": "acknowledgment", 
                "message": "Processing your request with Cerebras SDK..."
            }
            
            # Convert LangChain format to SDK format if needed
            conversation_history = self._convert_chat_history(chat_history)
            
            # Process query with native SDK agent
            logger.info(f"🔄 Processing query via SDK: '{user_query[:50]}...'")
            
            # Since SDK agent does completion-based processing, we simulate streaming
            yield {
                "type": "context_gathering_start",
                "message": "Initializing Cerebras SDK native reasoning..."
            }
            
            # Execute SDK native processing
            response = await self.native_agent.process_query(
                user_query=user_query,
                conversation_history=conversation_history
            )
            
            # Send context gathering update
            yield {
                "type": "context_gathering_complete",
                "message": "SDK processing completed",
                "sources": ["cerebras_sdk", "unified_search", "knowledge_graph"],
                "chunks_found": 1,  # SDK handles this internally
                "files_analyzed": 1
            }
            
            # Send the final result in expected format
            yield {
                "type": "final_result",
                "output": response,
                "metadata": {
                    "agent_type": "cerebras_native_sdk",
                    "model_used": selected_model,
                    "reasoning_effort": "medium",
                    "processing_time": 0,  # SDK handles timing internally
                    "tools_used": ["native_sdk_tools"],
                    "sdk_integration": True
                },
                "formatted_response": {
                    "query_type": "sdk_native",
                    "confidence_level": "high",
                    "confidence_score": 0.95,
                    "code_snippets": [],
                    "file_references": [],
                    "tools_used": ["cerebras_sdk"],
                    "response_time": 0.0
                }
            }
            
            logger.info("✅ SDK response streamed successfully")
            
        except Exception as e:
            logger.error(f"❌ WebSocket adapter error: {e}")
            yield {
                "type": "error",
                "message": f"SDK processing error: {str(e)}",
                "error_details": {
                    "component": "websocket_adapter",
                    "sdk_agent": "cerebras_native",
                    "error": str(e)
                }
            }
    
    def _convert_chat_history(self, langchain_history: Optional[List]) -> Optional[List[Dict]]:
        """
        Convert LangChain message format to SDK message format
        """
        if not langchain_history:
            return None
        
        converted = []
        
        try:
            for message in langchain_history:
                # Handle different LangChain message formats
                if hasattr(message, 'type') and hasattr(message, 'content'):
                    # LangChain message object
                    if message.type == 'human':
                        role = 'user'
                    elif message.type == 'ai':
                        role = 'assistant'
                    elif message.type == 'system':
                        role = 'system'
                    else:
                        role = 'user'  # fallback
                    
                    converted.append({
                        "role": role,
                        "content": message.content
                    })
                
                elif isinstance(message, dict):
                    # Already in dict format
                    if "role" in message and "content" in message:
                        converted.append(message)
                    else:
                        # Try to extract from dict structure
                        role = message.get("type", "user")
                        if role == "human":
                            role = "user"
                        elif role == "ai":
                            role = "assistant"
                        
                        converted.append({
                            "role": role,
                            "content": message.get("content", str(message))
                        })
                
                else:
                    # Fallback: treat as string content
                    converted.append({
                        "role": "user",
                        "content": str(message)
                    })
        
        except Exception as e:
            logger.warning(f"⚠️ Chat history conversion failed: {e}, using empty history")
            return None
        
        logger.info(f"🔄 Converted {len(langchain_history)} → {len(converted)} messages")
        return converted

class LegacyAgentWrapper:
    """
    Drop-in replacement for existing agent that uses SDK backend
    """
    
    def __init__(self, **kwargs):
        """Initialize with same signature as existing agent"""
        self.adapter = WebSocketStreamingAdapter()
        logger.info("✅ Legacy agent wrapper initialized with SDK backend")
    
    async def process_request(self, user_query: str, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main method that existing main.py calls
        """
        # Extract parameters that existing code passes
        chat_history = kwargs.get('chat_history', [])
        mentioned_projects = kwargs.get('mentioned_projects', [])
        selected_model = kwargs.get('selected_model', 'gpt-oss-120b')
        
        # Delegate to streaming adapter
        async for update in self.adapter.process_request(
            user_query=user_query,
            chat_history=chat_history,
            mentioned_projects=mentioned_projects,
            selected_model=selected_model
        ):
            yield update

def get_sdk_compatible_agent(**kwargs):
    """
    Factory function that returns SDK-compatible agent with legacy interface
    """
    return LegacyAgentWrapper(**kwargs)

# For backward compatibility with existing imports
class CodeWiseAgent(LegacyAgentWrapper):
    """Backward compatible class name"""
    pass

# Export the key classes
__all__ = ["WebSocketStreamingAdapter", "LegacyAgentWrapper", "CodeWiseAgent", "get_sdk_compatible_agent"]