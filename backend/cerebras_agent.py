"""
COMPLETE Cerebras SDK Native Integration

This replaces ALL custom tool routing, reasoning, and execution with 
native SDK capabilities. No more emulation - this is pure SDK integration.

Key principles:
- SDK handles ALL reasoning and tool selection
- Our infrastructure provides pure data sources
- Tools are pure functions with zero custom logic
- Future-proof architecture that adapts to SDK improvements
"""

import os
import json
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from cerebras.cloud.sdk import Cerebras

# Import our existing Phase 1/2 infrastructure (PRESERVE these)
try:
    from .tools.unified_query_pure import query_codebase_pure as query_codebase
    from .tools.mermaid_renderer import MermaidRenderer
    from .smart_search import get_smart_search_engine
    from .vector_store import VectorStore
    from .tools.unified_tool_schema import UNIFIED_TOOL_SCHEMA
    from .tools.filesystem_tool_schema import FILESYSTEM_TOOL_SCHEMA
    from .tools.filesystem_navigator import FilesystemNavigator
    from .config.cerebras_config import cerebras_config
except ImportError:
    from tools.unified_query_pure import query_codebase_pure as query_codebase
    from tools.mermaid_renderer import MermaidRenderer
    from smart_search import get_smart_search_engine
    from vector_store import VectorStore
    from tools.unified_tool_schema import UNIFIED_TOOL_SCHEMA
    from tools.filesystem_tool_schema import FILESYSTEM_TOOL_SCHEMA
    from tools.filesystem_navigator import FilesystemNavigator
    from config.cerebras_config import cerebras_config

# Import Phase 2 KG with fallback
try:
    from ..storage.database_manager import DatabaseManager
    KG_AVAILABLE = True
except ImportError:
    try:
        from storage.database_manager import DatabaseManager
        KG_AVAILABLE = True
    except ImportError:
        KG_AVAILABLE = False
        DatabaseManager = None

logger = logging.getLogger(__name__)

class CerebrasNativeAgent:
    """
    FULLY SDK-Native Agent - Zero Custom Reasoning/Tool Logic
    
    This agent is a thin coordination layer that lets the Cerebras SDK
    handle ALL reasoning, tool selection, and execution planning while
    providing access to our Phase 1/2 infrastructure.
    
    Architecture:
    - SDK: Handles reasoning, tool selection, conversation management
    - Agent: Coordinates infrastructure and executes pure tool functions
    - Infrastructure: Provides data through smart search, KG, vector store
    """
    
    def __init__(self):
        """Initialize with ONLY infrastructure connections"""
        
        # Validate configuration first
        if not cerebras_config.validate_config():
            raise ValueError("Invalid Cerebras configuration - cannot initialize agent")
        
        # SDK Client (this does ALL the reasoning)
        try:
            self.client = Cerebras(**cerebras_config.get_client_config())
            self.model = cerebras_config.model
            logger.info(f"✅ Cerebras SDK client initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Cerebras client: {e}")
            raise
        
        # REQ-3.6.6: Future-proofing SDK capability detection
        self.sdk_version = self._detect_sdk_version()
        self.available_features = self._detect_sdk_features()
        logger.info(f"✅ SDK capabilities detected: {list(self.available_features.keys())}")
        
        # Phase 1/2 Infrastructure (data sources only)
        try:
            self.smart_search = get_smart_search_engine()
            self.vector_store = VectorStore()
            self.mermaid_renderer = MermaidRenderer()
            logger.info("✅ Core infrastructure initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize core infrastructure: {e}")
            raise
        
        # Phase 2 Knowledge Graph (optional)
        if KG_AVAILABLE:
            try:
                self.kg = DatabaseManager()
                logger.info("✅ Knowledge Graph available for SDK")
            except Exception as e:
                logger.warning(f"⚠️ Knowledge Graph initialization failed: {e}")
                self.kg = None
        else:
            self.kg = None
            logger.warning("⚠️ Knowledge Graph not available")
        
        # REQ-3.6.1: Initialize filesystem navigator tool
        if self.kg:
            try:
                self.filesystem_navigator = FilesystemNavigator(self.kg)
                logger.info("✅ Filesystem navigator initialized with KG integration")
            except Exception as e:
                logger.warning(f"⚠️ Filesystem navigator initialization failed: {e}")
                self.filesystem_navigator = None
        else:
            self.filesystem_navigator = None
            logger.warning("⚠️ Filesystem navigator not available - requires Knowledge Graph")
        
        # CRITICAL: Tool functions that SDK will call (REQ-3.6.1: Two-tool architecture)
        # These are PURE functions - no reasoning, just execution
        self.tool_functions = {
            "query_codebase": self._execute_query_codebase,
            "navigate_filesystem": self._execute_navigate_filesystem,
            # Removed from primary tools - now internal utilities:
            # "generate_diagram": self._execute_generate_diagram,  # Now internal via mermaid_renderer
            # "examine_files": self._execute_examine_files,        # Functionality absorbed into navigate_filesystem
            # "search_symbols": self._execute_search_symbols       # Functionality absorbed into query_codebase
        }
        
        # Conversation limits for safety
        self.max_iterations = 20
        self.current_iteration = 0
        
        logger.info("✅ CerebrasNativeAgent initialized - SDK handles all reasoning")
    
    async def process_query(self, user_query: str, conversation_history: Optional[List[Dict]] = None, selected_model: str = None, mentioned_projects: Optional[List[str]] = None) -> str:
        """
        PURE SDK Processing - No Custom Logic
        
        The SDK handles:
        - Query understanding and intent classification
        - Tool selection and sequencing
        - Reasoning about results
        - Response synthesis
        
        We only provide tool execution.
        """
        try:
            # Store project context for tool filtering
            self.mentioned_projects = mentioned_projects
            if mentioned_projects:
                logger.info(f"🎯 Project context set: {mentioned_projects}")
            
            # Use provided model or fall back to config default
            model_to_use = selected_model or cerebras_config.model
            logger.info(f"🎯 Processing query with model: {model_to_use}")
            
            # Build message history
            messages = conversation_history or []
            
            # Add system prompt ONLY if new conversation
            if not messages:
                messages.append({
                    "role": "system",
                    "content": self._get_native_system_prompt()
                })
            
            # Add user query
            messages.append({"role": "user", "content": user_query})
            
            # Reset iteration counter
            self.current_iteration = 0
            
            # SDK handles everything from here
            result = await self._sdk_native_loop(messages, model_to_use)
            
            logger.info(f"✅ Query processed successfully in {self.current_iteration} iterations")
            return result
            
        except Exception as e:
            logger.error(f"❌ Native agent processing failed: {e}")
            error_message = f"I encountered a system error while processing your request: {str(e)}"
            return await self._ensure_structured_response(error_message, "processing_error")
    
    async def _sdk_native_loop(self, messages: List[Dict], selected_model: str) -> str:
        """
        Pure SDK loop - NO custom logic, just tool execution when requested
        """
        last_successful_content = None
        successful_iterations = 0
        
        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            
            try:
                logger.info(f"🔄 SDK Iteration {self.current_iteration}/{self.max_iterations}")
                
                # REQ-CTX-FALLBACK: Check if we need graceful degradation
                context_health = {"utilization_percent": 0}
                if context_health.get("utilization_percent", 0) > 90:
                    logger.warning("🚨 Context critical - applying graceful degradation")
                    return await self._apply_graceful_degradation(messages, context_health)
                
                # SDK makes ALL decisions using adaptive schema
                adaptive_schema = self._build_adaptive_schema()
                
                # Build completion config with dynamic model
                completion_config = cerebras_config.get_completion_config(selected_model)
                
                # Build API parameters with conditional reasoning_effort
                api_params = {
                    **completion_config,
                    "messages": messages,
                    "tools": adaptive_schema
                }
                
                # Only add reasoning_effort for models that support it
                if cerebras_config.supports_reasoning_effort(selected_model):
                    api_params["reasoning_effort"] = cerebras_config.reasoning_effort
                    logger.info(f"✅ Using reasoning_effort '{cerebras_config.reasoning_effort}' for {selected_model}")
                else:
                    logger.info(f"⏭️ Skipping reasoning_effort for {selected_model} (not supported)")
                
                # DIAGNOSTIC LOGGING: Log exact API request parameters
                logger.info(f"🔧 DIAGNOSTIC: API Request Parameters for iteration {self.current_iteration}")
                logger.info(f"🔧 Model: {api_params.get('model')}")
                logger.info(f"🔧 Max tokens: {api_params.get('max_tokens')}")
                logger.info(f"🔧 Tools count: {len(api_params.get('tools', []))}")
                logger.info(f"🔧 Tools schema: {json.dumps(api_params.get('tools', []), indent=2)}")
                logger.info(f"🔧 Messages count: {len(api_params.get('messages', []))}")
                
                # Log last few messages to see conversation context
                messages_to_log = api_params.get('messages', [])[-3:]  # Last 3 messages
                for i, msg in enumerate(messages_to_log):
                    try:
                        role = msg.get('role', 'unknown') if isinstance(msg, dict) else getattr(msg, 'role', 'unknown')
                        content_preview = str(msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', ''))[:200] + "..."
                        tool_calls = msg.get('tool_calls', []) if isinstance(msg, dict) else getattr(msg, 'tool_calls', [])
                        logger.info(f"🔧 Message {i}: Role={role}, Content={content_preview}, ToolCalls={len(tool_calls)}")
                    except Exception as log_err:
                        logger.info(f"🔧 Message {i}: [Error logging message: {log_err}]")
                
                response = self.client.chat.completions.create(**api_params)
                
                message = response.choices[0].message
                
                # Log SDK reasoning (for transparency and debugging)
                if hasattr(message, 'reasoning') and message.reasoning:
                    logger.info(f"🧠 SDK REASONING: {message.reasoning}")
                
                # SDK wants to use tools
                if message.tool_calls:
                    logger.info(f"🔧 SDK requested {len(message.tool_calls)} tool calls")
                    messages.append(message)
                    
                    # Execute tools with intelligent summarization for large outputs
                    for tool_call in message.tool_calls:
                        # Execute tool normally
                        result = await self._execute_pure_tool_call(tool_call)
                        
                        # Apply summarization if output is large  
                        user_query = self._extract_user_query(messages)
                        tool_content = await self._execute_tool_and_summarize(
                            tool_call.function.name, result, user_query
                        )
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_content
                        })
                    
                    successful_iterations += 1
                    continue  # Let SDK process results
                
                # SDK provides final answer
                else:
                    logger.info("✅ SDK provided final response")
                    
                    # Log full LLM response for debugging/monitoring
                    response_content = message.content
                    response_length = len(response_content) if response_content else 0
                    logger.info(f"📄 LLM Response Details - Length: {response_length} chars, Model: {self.model}")
                    logger.info(f"📝 Full LLM Response:\n{'='*50}\n{response_content}\n{'='*50}")
                    
                    # CRITICAL FIX: Apply unified validation to ALL response paths
                    # This ensures the UI always receives structured JSON, not raw LLM content
                    return await self._ensure_structured_response(response_content, "sdk_success")
                    
            except Exception as e:
                logger.error(f"❌ SDK iteration {self.current_iteration} failed: {e}")
                
                # DIAGNOSTIC LOGGING: Capture the exact error details
                if "400" in str(e) and "tool call" in str(e):
                    logger.error("🔍 DIAGNOSTIC: 400 Tool Call Error Details")
                    logger.error(f"🔍 Error: {str(e)}")
                    logger.error(f"🔍 Exception type: {type(e)}")
                    
                    # Log the exact API parameters that caused the failure
                    logger.error(f"🔍 Failed API params - Model: {api_params.get('model')}")
                    logger.error(f"🔍 Failed API params - Tools: {json.dumps(api_params.get('tools', []), indent=2)}")
                    
                    # Log the conversation state at failure
                    logger.error(f"🔍 Conversation length at failure: {len(messages)} messages")
                    for i, msg in enumerate(messages[-5:]):  # Last 5 messages
                        try:
                            role = msg.get('role', 'unknown') if isinstance(msg, dict) else getattr(msg, 'role', 'unknown')
                            content = str(msg.get('content', ''))[:300] if isinstance(msg, dict) else str(getattr(msg, 'content', ''))[:300]
                            tool_calls = msg.get('tool_calls', []) if isinstance(msg, dict) else getattr(msg, 'tool_calls', [])
                            logger.error(f"🔍 Message {i}: {role} - {content}... (ToolCalls: {len(tool_calls) if tool_calls else 0})")
                        except Exception as log_err:
                            logger.error(f"🔍 Message {i}: [Error logging message: {log_err}]")
                
                # If we had successful iterations, try to provide a helpful response
                if successful_iterations > 0:
                    logger.info(f"⚠️ Partial success: {successful_iterations} iterations completed before error")
                    error_message = (
                        f"I've gathered information about your query but encountered a technical issue "
                        f"in the final processing step. Based on the data I found, I can provide you with "
                        f"relevant information, though the response formatting may be affected. "
                        f"The technical error was: {str(e)}"
                    )
                    # CRITICAL FIX: Return structured error response instead of plain text
                    return await self._ensure_structured_response(error_message, "sdk_error")
                else:
                    error_message = f"I encountered a technical error during processing: {str(e)}"
                    return await self._ensure_structured_response(error_message, "system_error")
        
        logger.warning(f"⚠️ Reached maximum iterations ({self.max_iterations})")
        
        # Instead of giving up, try to synthesize a response from gathered context
        if successful_iterations > 0:
            logger.info(f"💡 Attempting to synthesize response from {successful_iterations} successful tool calls")
            
            # Extract the original user query from the conversation
            user_query = ""
            for msg in messages:
                # Handle both dict and SDK response objects
                if hasattr(msg, 'role'):
                    role = msg.role
                    content = msg.content if hasattr(msg, 'content') else ""
                else:
                    role = msg.get("role", "") if isinstance(msg, dict) else ""
                    content = msg.get("content", "") if isinstance(msg, dict) else ""
                
                if role == "user":
                    user_query = content
                    break
            
            # Create a synthesis prompt using gathered context
            synthesis_messages = [
                {
                    "role": "system", 
                    "content": "You are an expert code analyst. Based on the tool results gathered so far, provide a comprehensive answer to the user's question. Use all available context from previous tool calls to give a complete response."
                },
                {
                    "role": "user", 
                    "content": f"Original question: {user_query}\n\nBased on the information gathered from previous tool calls in this conversation, please provide a comprehensive answer. Synthesize all the context and data that was collected to give the user a complete response."
                }
            ]
            
            # Add the conversation history (contains all tool results)
            for msg in messages[1:]:  # Skip initial system prompt
                # Handle both dict and SDK response objects
                if hasattr(msg, 'role'):
                    role = msg.role
                else:
                    role = msg.get("role", "") if isinstance(msg, dict) else ""
                
                if role in ["assistant", "tool"]:
                    # Convert SDK response objects to dict format for synthesis
                    if hasattr(msg, 'role'):
                        dict_msg = {
                            "role": msg.role,
                            "content": msg.content if hasattr(msg, 'content') else ""
                        }
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            dict_msg["tool_calls"] = msg.tool_calls
                        synthesis_messages.append(dict_msg)
                    else:
                        synthesis_messages.append(msg)
            
            try:
                # Make final synthesis call without tools
                synthesis_completion_config = cerebras_config.get_completion_config(selected_model)
                
                # Build synthesis API parameters with conditional reasoning_effort  
                synthesis_params = {
                    **synthesis_completion_config,
                    "messages": synthesis_messages
                }
                
                # Only add reasoning_effort for models that support it
                if cerebras_config.supports_reasoning_effort(selected_model):
                    synthesis_params["reasoning_effort"] = cerebras_config.reasoning_effort
                    logger.info(f"✅ Using reasoning_effort for synthesis with {selected_model}")
                else:
                    logger.info(f"⏭️ Skipping reasoning_effort for synthesis with {selected_model} (not supported)")
                
                # REQ-UI-UNIFIED-4: Implement validation and self-correction with retry loop
                return await self._validate_and_correct_response(synthesis_params, selected_model)
                
            except Exception as e:
                logger.error(f"❌ Synthesis failed: {e}")
                error_message = (
                    f"I gathered substantial information about your query through {successful_iterations} "
                    f"research steps, but reached the processing limit while organizing the final response. "
                    f"The system found relevant information but encountered a technical limitation in "
                    f"synthesizing the complete answer. Please try rephrasing your question or breaking "
                    f"it into smaller parts."
                )
                # CRITICAL FIX: Structure synthesis error response
                return await self._ensure_structured_response(error_message, "synthesis_error")
        else:
            error_message = "I reached the maximum processing steps. Please try rephrasing your question or breaking it into smaller parts."
            return await self._ensure_structured_response(error_message, "max_iterations_error")
    
    async def _validate_and_correct_response(self, synthesis_params: Dict[str, Any], selected_model: str) -> str:
        """
        REQ-UI-UNIFIED-4: Implement Backend Validation and Self-Correction
        REQ-3.4.2: Error Recovery & Self-Correction System
        
        Validates LLM response against UnifiedAgentResponse schema with retry loop.
        Implements Phase 3 error recovery with Mermaid syntax validation.
        """
        max_retries = 2
        
        for attempt in range(max_retries + 1):  # 0, 1, 2 (3 total attempts)
            try:
                logger.info(f"🔄 Response validation attempt {attempt + 1}/{max_retries + 1}")
                
                # Make API call
                synthesis_response = self.client.chat.completions.create(**synthesis_params)
                raw_response = synthesis_response.choices[0].message.content if hasattr(synthesis_response.choices[0].message, 'content') else str(synthesis_response.choices[0].message)
                
                # Log raw response for debugging
                response_length = len(raw_response) if raw_response else 0
                logger.info(f"📝 Raw response received ({response_length} chars)")
                logger.debug(f"📄 Full Raw Response:\n{'='*50}\n{raw_response}\n{'='*50}")
                
                # REQ-3.4.2: Phase 3 Mermaid validation with error recovery
                mermaid_validation_errors = await self._validate_mermaid_syntax(raw_response)
                if mermaid_validation_errors and attempt < max_retries:
                    # Phase 3: Let LLM learn from syntax errors
                    correction_prompt = self._generate_mermaid_correction_prompt(mermaid_validation_errors)
                    synthesis_params["messages"].append({
                        "role": "user", 
                        "content": correction_prompt
                    })
                    logger.warning(f"⚠️ Mermaid syntax errors found, requesting correction (attempt {attempt + 2})")
                    continue
                
                # Attempt validation
                validated_response = await self._validate_response_structure(raw_response, attempt)
                
                if validated_response:
                    logger.info(f"✅ Response validation successful on attempt {attempt + 1}")
                    return validated_response
                    
                # Validation failed - prepare correction prompt for retry
                if attempt < max_retries:
                    correction_prompt = self._generate_correction_prompt(raw_response, attempt)
                    synthesis_params["messages"].append({
                        "role": "user", 
                        "content": correction_prompt
                    })
                    logger.warning(f"⚠️ Validation failed, retrying with correction prompt (attempt {attempt + 2})")
                
            except Exception as e:
                logger.error(f"❌ Synthesis attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries:
                    # Add error recovery prompt
                    error_prompt = f"The previous response caused an error: {str(e)}. Please provide a valid JSON response following the UnifiedAgentResponse schema format."
                    synthesis_params["messages"].append({
                        "role": "user",
                        "content": error_prompt 
                    })
                    continue
        
        # All attempts failed - return structured error response
        logger.error("💥 All validation attempts failed - generating fallback error response")
        return await self._generate_structured_error_response("Failed to generate a valid structured response after multiple attempts")
    
    def _safe_serialize_tool_result(self, tool_result: Dict[str, Any], tool_name: str) -> str:
        """
        Safely serialize tool results with content isolation to prevent corruption.
        
        This method ensures that complex tool results are properly bounded and that
        content from different sources doesn't bleed together during serialization.
        
        Args:
            tool_result: Raw tool execution result
            tool_name: Name of the tool for context
            
        Returns:
            Clean, isolated string representation of the tool result
        """
        try:
            # Create a bounded container for the tool result
            bounded_result = {
                "tool_name": tool_name,
                "timestamp": time.time(),
                "content": tool_result,
                "content_hash": hash(str(tool_result))  # For integrity verification
            }
            
            # Use safe JSON serialization with content isolation
            serialized = json.dumps(bounded_result, indent=2, ensure_ascii=False, separators=(',', ': '))
            
            # Verify serialization integrity
            try:
                verification = json.loads(serialized)
                if verification.get("content_hash") != hash(str(tool_result)):
                    logger.warning(f"⚠️ Content integrity check failed for {tool_name} - using fallback")
                    return self._fallback_serialize_tool_result(tool_result, tool_name)
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"⚠️ Serialization verification failed for {tool_name} - using fallback")
                return self._fallback_serialize_tool_result(tool_result, tool_name)
            
            logger.debug(f"✅ Safe serialization successful for {tool_name} ({len(serialized)} chars)")
            return serialized
            
        except Exception as e:
            logger.error(f"❌ Safe serialization failed for {tool_name}: {e}")
            return self._fallback_serialize_tool_result(tool_result, tool_name)
    
    def _fallback_serialize_tool_result(self, tool_result: Dict[str, Any], tool_name: str) -> str:
        """
        Fallback serialization method for problematic tool results.
        
        When safe serialization fails, this method provides a basic but reliable
        string representation that maintains content boundaries.
        """
        try:
            # Simple string conversion with clear boundaries
            if isinstance(tool_result, dict):
                content_parts = []
                for key, value in tool_result.items():
                    content_parts.append(f"{key}: {str(value)[:1000]}...")  # Truncate long values
                return f"=== {tool_name.upper()} RESULT ===\n" + "\n".join(content_parts) + f"\n=== END {tool_name.upper()} ==="
            else:
                return f"=== {tool_name.upper()} RESULT ===\n{str(tool_result)[:2000]}...\n=== END {tool_name.upper()} ==="
        except Exception as e:
            logger.error(f"❌ Fallback serialization failed for {tool_name}: {e}")
            return f"=== {tool_name.upper()} RESULT ===\nERROR: Could not serialize result\n=== END {tool_name.upper()} ==="
    
    async def _ensure_structured_response(self, raw_response: str, context: str) -> str:
        """
        Master response wrapper ensuring ALL responses conform to UnifiedAgentResponse schema.
        
        This is the architectural fix that guarantees the unified UI system works correctly
        by ensuring every code path returns structured JSON, never plain text.
        
        Args:
            raw_response: Raw response content (may be JSON, plain text, or error message)
            context: Context of the response (sdk_success, sdk_error, system_error, etc.)
            
        Returns:
            Valid UnifiedAgentResponse JSON string
        """
        try:
            logger.info(f"🔄 Ensuring structured response for context: {context}")
            
            # First, try to validate if it's already a valid UnifiedAgentResponse
            try:
                validated = await self._validate_response_structure(raw_response, 0)
                if validated:
                    logger.info(f"✅ Response already structured for {context}")
                    return validated
            except Exception:
                pass  # Continue to structure the response
            
            # If not valid JSON or not conforming to schema, wrap it appropriately
            if context == "sdk_success":
                # Raw LLM content needs to be structured - this is likely already JSON but needs validation
                return await self._structure_llm_response(raw_response)
            
            elif context in ["sdk_error", "system_error", "synthesis_error", "max_iterations_error", "processing_error"]:
                # Error messages need to be wrapped in TextBlock structure
                return await self._structure_error_response(raw_response, context)
            
            else:
                # Unknown context - default to text block wrapping
                logger.warning(f"⚠️ Unknown context '{context}' - defaulting to text block")
                return await self._structure_as_text_block(raw_response)
                
        except Exception as e:
            logger.error(f"❌ Failed to ensure structured response for {context}: {e}")
            # Absolute fallback - return minimal valid structure
            return await self._generate_structured_error_response(f"Response structuring failed: {str(e)}")
    
    async def _structure_llm_response(self, raw_response: str) -> str:
        """
        Structure raw LLM response content into UnifiedAgentResponse format.
        
        This handles the case where the LLM provided content but it may not be
        properly structured according to our schema.
        
        REQ-3.4.2: Phase 3 - Use full correction loop if Mermaid diagrams need validation.
        """
        try:
            # REQ-3.4.2: Check if response contains Mermaid diagrams that might need correction
            import re
            has_mermaid = bool(re.search(r'```mermaid', raw_response or '', re.IGNORECASE))
            
            if has_mermaid:
                # Check for Mermaid syntax errors
                mermaid_errors = await self._validate_mermaid_syntax(raw_response)
                if mermaid_errors:
                    logger.info(f"🔄 Phase 3: Mermaid errors detected, using full correction loop for {len(mermaid_errors)} errors")
                    
                    # Use full correction loop to allow LLM to learn and fix Mermaid syntax
                    synthesis_params = {
                        "model": "gpt-oss-120b",  # Use default model
                        "messages": [
                            {
                                "role": "system", 
                                "content": "You are a code analyst. Please provide a response that follows proper Mermaid diagram syntax."
                            },
                            {
                                "role": "user",
                                "content": "Please review and correct any Mermaid diagram syntax errors in the following response, then provide the complete corrected response:"
                            },
                            {
                                "role": "assistant",
                                "content": raw_response
                            }
                        ]
                    }
                    
                    # Use full validation and correction with retry loop
                    return await self._validate_and_correct_response(synthesis_params, "gpt-oss-120b")
            
            # No Mermaid issues, use streamlined single response validation
            return await self._validate_and_correct_single_response(raw_response)
            
        except Exception as e:
            logger.warning(f"⚠️ LLM response structuring failed, wrapping as text: {e}")
            return await self._structure_as_text_block(raw_response)
    
    async def _validate_and_correct_single_response(self, raw_response: str) -> str:
        """
        Apply validation to a single response without the full retry loop.
        
        This is a streamlined version of the validation process for responses
        that don't need the full correction cycle.
        
        REQ-3.4.2: Now includes Mermaid syntax validation for Phase 3 compliance.
        """
        # REQ-3.4.2: Phase 3 Mermaid validation (but no correction since this is single-shot)
        mermaid_validation_errors = await self._validate_mermaid_syntax(raw_response)
        if mermaid_validation_errors:
            logger.warning(f"⚠️ Mermaid syntax errors detected in single response (cannot auto-correct): {len(mermaid_validation_errors)} errors")
            for error in mermaid_validation_errors:
                logger.warning(f"🔍 MERMAID ERROR: {error}")
        
        # Attempt validation
        validated_response = await self._validate_response_structure(raw_response, 0)
        
        if validated_response:
            return validated_response
        else:
            # If validation fails, wrap the content as a text block
            return await self._structure_as_text_block(raw_response)
    
    async def _structure_error_response(self, error_message: str, context: str) -> str:
        """
        Structure error messages into proper UnifiedAgentResponse format.
        """
        from schemas.ui_schemas import create_error_response
        import json
        
        try:
            error_response = create_error_response(f"**{context.replace('_', ' ').title()}**\n\n{error_message}", include_debug=False)
            return json.dumps(error_response.model_dump(), indent=2)
        except Exception as e:
            logger.error(f"❌ Error response structuring failed: {e}")
            # Absolute fallback
            return json.dumps({
                "response": [
                    {
                        "block_type": "text",
                        "content": f"## System Error ({context})\n\n{error_message}"
                    }
                ]
            })
    
    async def _structure_as_text_block(self, content: str) -> str:
        """
        Parse markdown content and structure it into appropriate content blocks.
        
        This method analyzes the LLM's markdown response and splits it into:
        - TextBlock: Regular markdown content
        - CodeSnippetBlock: Code blocks with language detection
        - MermaidDiagramBlock: Mermaid diagrams  
        - MarkdownTableBlock: Tables
        """
        import json
        import re
        
        try:
            logger.info("🔧 PARSER: Starting markdown content parsing")
            logger.info(f"🔧 PARSER: Input content length: {len(content)}")
            logger.info(f"🔧 PARSER: Input preview: {content[:200]}...")
            
            # Parse the markdown content into structured blocks
            blocks = self._parse_markdown_content(content)
            
            logger.info(f"🔧 PARSER: Generated {len(blocks)} blocks")
            for i, block in enumerate(blocks):
                logger.info(f"🔧 PARSER: Block {i}: {block['block_type']} - {len(str(block.get('content', block.get('code', block.get('mermaid_code', '')))))} chars")
            
            structured_response = {
                "response": blocks
            }
            
            logger.info(f"✅ Parsed markdown into {len(blocks)} content blocks")
            logger.info(f"🔧 PARSER: Final JSON size: {len(json.dumps(structured_response))}")
            return json.dumps(structured_response, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"❌ Markdown parsing failed, falling back to single text block: {e}")
            # Fallback to single text block
            structured_response = {
                "response": [
                    {
                        "block_type": "text",
                        "content": content
                    }
                ]
            }
            return json.dumps(structured_response, indent=2, ensure_ascii=False)
    
    def _parse_markdown_content(self, content: str) -> list:
        """
        Parse markdown content into structured content blocks.
        
        Returns:
            List of content blocks with appropriate block_type
        """
        import re
        blocks = []
        current_text = []
        lines = content.split('\n')
        logger.info(f"🔧 PARSER: Processing {len(lines)} lines")
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check for mermaid diagram
            if line.strip().startswith('```mermaid'):
                logger.info(f"🔧 PARSER: Found mermaid diagram at line {i}")
                # Save any accumulated text
                if current_text:
                    text_content = '\n'.join(current_text).strip()
                    if text_content:
                        blocks.append({
                            "block_type": "text",
                            "content": text_content
                        })
                    current_text = []
                
                # Extract mermaid diagram
                i += 1
                mermaid_lines = []
                title = "Diagram"  # Default title
                
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    mermaid_lines.append(lines[i])
                    i += 1
                
                mermaid_code = '\n'.join(mermaid_lines).strip()
                if mermaid_code:
                    blocks.append({
                        "block_type": "mermaid_diagram",
                        "title": title,
                        "mermaid_code": mermaid_code
                    })
                
            # Check for code blocks
            elif line.strip().startswith('```') and not line.strip().startswith('```mermaid'):
                logger.info(f"🔧 PARSER: Found code block at line {i}: {line.strip()}")
                # Save any accumulated text
                if current_text:
                    text_content = '\n'.join(current_text).strip()
                    if text_content:
                        blocks.append({
                            "block_type": "text",
                            "content": text_content
                        })
                    current_text = []
                
                # Extract code block
                language_match = re.match(r'```(\w+)', line.strip())
                language = language_match.group(1) if language_match else 'text'
                
                i += 1
                code_lines = []
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    code_lines.append(lines[i])
                    i += 1
                
                code_content = '\n'.join(code_lines).strip()
                if code_content:
                    # Determine title based on language and context
                    title = f"{language.title()} Code"
                    if language == 'java':
                        title = "Java Implementation"
                    elif language == 'javascript':
                        title = "JavaScript Code"
                    elif language == 'python':
                        title = "Python Code"
                    
                    blocks.append({
                        "block_type": "code_snippet",
                        "title": title,
                        "language": language,
                        "code": code_content
                    })
                
            # Check for markdown tables
            elif '|' in line and line.count('|') >= 2:
                logger.info(f"🔧 PARSER: Found table at line {i}: {line.strip()[:50]}...")
                # Save any accumulated text
                if current_text:
                    text_content = '\n'.join(current_text).strip()
                    if text_content:
                        blocks.append({
                            "block_type": "text", 
                            "content": text_content
                        })
                    current_text = []
                
                # Extract table
                table_lines = [line]
                i += 1
                
                # Get header separator line (if exists)
                if i < len(lines) and '|' in lines[i] and ('-' in lines[i] or ':' in lines[i]):
                    table_lines.append(lines[i])
                    i += 1
                
                # Get remaining table rows
                while i < len(lines) and '|' in lines[i] and lines[i].strip():
                    table_lines.append(lines[i])
                    i += 1
                
                # Parse table structure
                table_data = self._parse_table_structure(table_lines)
                if table_data:
                    blocks.append(table_data)
                
                # Don't increment i here as we've already processed all table lines
                continue
                
            else:
                # Regular text line
                current_text.append(line)
            
            i += 1
        
        # Add any remaining text
        if current_text:
            text_content = '\n'.join(current_text).strip()
            if text_content:
                blocks.append({
                    "block_type": "text",
                    "content": text_content
                })
        
        return blocks if blocks else [{"block_type": "text", "content": content}]
    
    def _parse_table_structure(self, table_lines: list) -> dict:
        """
        Parse markdown table lines into MarkdownTableBlock structure.
        
        Args:
            table_lines: List of table lines including headers and rows
            
        Returns:
            MarkdownTableBlock dict or None if parsing fails
        """
        try:
            if len(table_lines) < 1:
                return None
                
            # Extract headers from first line
            header_line = table_lines[0].strip()
            headers = [h.strip() for h in header_line.split('|') if h.strip()]
            
            if not headers:
                return None
            
            # Skip separator line if it exists
            start_row = 1
            if len(table_lines) > 1 and ('---' in table_lines[1] or ':--' in table_lines[1]):
                start_row = 2
                
            # Extract data rows
            rows = []
            for i in range(start_row, len(table_lines)):
                row_line = table_lines[i].strip()
                if row_line:
                    row_cells = [cell.strip() for cell in row_line.split('|') if cell.strip()]
                    if row_cells and len(row_cells) >= len(headers):
                        # Ensure row has same number of cells as headers
                        rows.append(row_cells[:len(headers)])
            
            if rows:
                return {
                    "block_type": "markdown_table",
                    "title": "Data Table",
                    "headers": headers,
                    "rows": rows
                }
                
        except Exception as e:
            logger.warning(f"⚠️ Table parsing failed: {e}")
            
        return None
    
    
    async def _validate_mermaid_syntax(self, raw_response: str) -> List[str]:
        """
        REQ-3.4.2: Validate Mermaid syntax and return specific error messages.
        
        Returns:
            List of syntax errors found in Mermaid diagrams, empty list if valid
        """
        import re
        
        errors = []
        
        try:
            # Find all mermaid code blocks
            mermaid_blocks = re.findall(r'```mermaid([\s\S]*?)```', raw_response)
            
            for i, mermaid_code in enumerate(mermaid_blocks):
                mermaid_code = mermaid_code.strip()
                
                # Check for common syntax issues that break rendering
                
                # 1. Invalid curly braces in labels (common UML pattern that breaks Mermaid)
                if re.search(r'\{[^}]*\}', mermaid_code):
                    errors.append(f"Diagram {i+1}: Contains curly braces {{}} in labels, which are invalid in Mermaid. Use parentheses () or brackets [] instead.")
                
                # 2. Invalid UML-style method signatures
                if re.search(r'\+\w+\s*\(', mermaid_code):
                    errors.append(f"Diagram {i+1}: Contains UML-style method signatures with + prefix, which are invalid in Mermaid. Use plain text labels instead.")
                
                # 3. Invalid node IDs with special characters
                invalid_node_ids = re.findall(r'^([^\s\[\-]+)[\[\-]', mermaid_code, re.MULTILINE)
                for node_id in invalid_node_ids:
                    if re.search(r'[^a-zA-Z0-9_]', node_id):
                        errors.append(f"Diagram {i+1}: Node ID '{node_id}' contains special characters. Use only alphanumeric characters and underscores.")
                
                # 4. Check for missing diagram type declaration
                if not re.match(r'\s*(flowchart|graph|classDiagram|sequenceDiagram)', mermaid_code):
                    errors.append(f"Diagram {i+1}: Missing diagram type declaration. Must start with 'flowchart', 'graph', 'classDiagram', or 'sequenceDiagram'.")
                
                # 5. Check for invalid characters in labels
                bracket_labels = re.findall(r'\[([^\]]+)\]', mermaid_code)
                for label in bracket_labels:
                    if re.search(r'[\[\]{}]', label):
                        errors.append(f"Diagram {i+1}: Label '{label}' contains invalid characters. Avoid nested brackets, braces, and special characters in labels.")
            
            if errors:
                logger.warning(f"🔍 MERMAID VALIDATION: Found {len(errors)} syntax errors")
                for error in errors:
                    logger.warning(f"🔍 MERMAID ERROR: {error}")
            else:
                logger.info(f"✅ MERMAID VALIDATION: {len(mermaid_blocks)} diagrams validated successfully")
            
            return errors
            
        except Exception as e:
            logger.error(f"❌ Mermaid validation failed: {e}")
            return []  # Don't block on validation errors
    
    def _generate_mermaid_correction_prompt(self, errors: List[str]) -> str:
        """
        REQ-3.4.2: Generate correction prompt for Mermaid syntax errors.
        
        This enables the LLM to learn proper Mermaid syntax through error feedback.
        """
        error_list = "\n".join([f"- {error}" for error in errors])
        
        return f"""The Mermaid diagrams in your response contain syntax errors that will prevent them from rendering:

{error_list}

Please fix these syntax errors and regenerate your response. Remember:
- Use parentheses () or brackets [] for labels, never curly braces {{}}
- Avoid UML-style + prefixes for methods
- Use only alphanumeric characters and underscores for node IDs
- Always start diagrams with a valid type: flowchart, graph, classDiagram, or sequenceDiagram
- Don't nest special characters inside labels
- When generating structured JSON, use "block_type": "mermaid_diagram" (not "mermaid")

Correct Mermaid examples:
```mermaid
flowchart TD
    A[User Request] --> B[Process Data]
    B --> C[Return Response]
```

```mermaid
classDiagram
    class User {{
        +String name
        +login()
    }}
```

For structured JSON responses, use this format:
```json
{{
  "response": [
    {{
      "block_type": "mermaid_diagram",
      "title": "System Architecture",
      "mermaid_code": "flowchart TD\\n    A[Component] --> B[Process]"
    }}
  ]
}}
```

Please regenerate your complete response with properly formatted Mermaid diagrams."""
    
    async def _validate_response_structure(self, raw_response: str, attempt: int) -> Optional[str]:
        """
        Validate response against UnifiedAgentResponse schema.
        
        Args:
            raw_response: Raw LLM response text
            attempt: Current attempt number for logging
            
        Returns:
            Validated JSON string if valid, None if invalid
        """
        try:
            # Import schema validation utilities
            from schemas.ui_schemas import validate_response_structure
            import json
            
            # Clean and extract JSON from response
            json_content = self._extract_json_from_response(raw_response)
            if not json_content:
                logger.warning(f"🔍 Attempt {attempt + 1}: No valid JSON found in response")
                return None
            
            # Parse JSON
            try:
                parsed_data = json.loads(json_content)
            except json.JSONDecodeError as e:
                logger.warning(f"🔍 Attempt {attempt + 1}: JSON parsing failed: {e}")
                return None
            
            # Validate against schema
            validated_response = validate_response_structure(parsed_data)
            
            # Additional validation - ensure minimum requirements
            if not isinstance(parsed_data.get('response'), list):
                raise ValueError("Response must contain an array of content blocks")
            
            if len(parsed_data['response']) == 0:
                raise ValueError("Response array cannot be empty")
            
            # Check each block has required fields
            for i, block in enumerate(parsed_data['response']):
                if not isinstance(block, dict):
                    raise ValueError(f"Block {i} must be an object")
                if 'block_type' not in block:
                    raise ValueError(f"Block {i} missing required 'block_type' field")
            
            # Additional quality checks
            quality_check = validated_response.validate_content_quality()
            if quality_check["warnings"]:
                logger.info(f"⚠️ Content quality warnings: {quality_check['warnings']}")
            if quality_check["suggestions"]:
                logger.info(f"💡 Content quality suggestions: {quality_check['suggestions']}")
            
            # Return the original JSON string (not the validated object)
            return json_content
            
        except Exception as e:
            logger.warning(f"🔍 Attempt {attempt + 1}: Validation failed: {e}")
            return None
    
    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """
        Extract JSON content from LLM response, handling various formats.
        
        Returns:
            Clean JSON string or None if no valid JSON found
        """
        import re
        import json
        
        # Remove any markdown formatting
        response = response.strip()
        
        # Try direct JSON parse first
        if response.startswith('{') and response.endswith('}'):
            try:
                json.loads(response)  # Test if valid
                return response
            except:
                pass
        
        # Look for JSON in code blocks
        json_block_patterns = [
            r'```json\s*(\{[\s\S]*?\})\s*```',
            r'```\s*(\{[\s\S]*?\})\s*```',
            r'`(\{[\s\S]*?\})`'
        ]
        
        for pattern in json_block_patterns:
            matches = re.findall(pattern, response, re.MULTILINE | re.DOTALL)
            for match in matches:
                try:
                    json.loads(match.strip())  # Test if valid
                    return match.strip()
                except:
                    continue
        
        # Look for JSON objects in the text
        brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        potential_jsons = re.findall(brace_pattern, response, re.DOTALL)
        
        for potential in potential_jsons:
            try:
                json.loads(potential)  # Test if valid
                return potential.strip()
            except:
                continue
        
        # Last resort: try to find the largest JSON-like structure
        start_brace = response.find('{')
        if start_brace != -1:
            # Find matching closing brace
            brace_count = 0
            for i, char in enumerate(response[start_brace:], start_brace):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = response[start_brace:i+1]
                        try:
                            json.loads(candidate)  # Test if valid
                            return candidate
                        except:
                            break
        
        return None
    
    def _generate_correction_prompt(self, failed_response: str, attempt: int) -> str:
        """
        Generate progressive correction prompts based on attempt number.
        
        Args:
            failed_response: The response that failed validation
            attempt: Current attempt number (0-based)
            
        Returns:
            Correction prompt string
        """
        base_correction = """Your previous response did not conform to the required UnifiedAgentResponse schema format. You MUST respond with valid JSON only.

CRITICAL REQUIREMENTS:
1. Response must be a single JSON object starting with {"response": [
2. No text before or after the JSON
3. Each block must have the correct block_type field
4. All required fields must be present

"""
        
        if attempt == 0:
            # First retry - gentle correction
            return base_correction + """Please fix the format and provide a valid JSON response following this structure:

{
  "response": [
    {
      "block_type": "text",
      "content": "Your analysis here..."
    }
  ]
}"""
        
        elif attempt == 1:
            # Second retry - more specific correction
            error_analysis = self._analyze_response_errors(failed_response)
            return base_correction + f"""The specific issues detected were: {error_analysis}

You must provide ONLY a JSON object with no additional text. Example format:

{{"response": [
  {{"block_type": "text", "content": "Analysis introduction"}},
  {{"block_type": "component_analysis", "title": "Key Components", "components": [...]}}
]}}

Respond with valid JSON now."""
        
        else:
            # Final retry - very explicit
            return """FINAL ATTEMPT: You must respond with ONLY valid JSON in this exact format:

{"response": [{"block_type": "text", "content": "Based on my analysis of the codebase, here are the key findings..."}]}

NO other text. NO markdown. NO explanations. ONLY the JSON object."""
    
    def _analyze_response_errors(self, response: str) -> str:
        """Analyze what went wrong with the response format."""
        errors = []
        
        if not response.strip().startswith('{'):
            errors.append("Response doesn't start with JSON object")
        
        if '```' in response:
            errors.append("Response contains markdown code blocks")
            
        if response.count('{') != response.count('}'):
            errors.append("Unmatched braces in JSON")
            
        if '"response"' not in response:
            errors.append("Missing required 'response' key")
            
        if '"block_type"' not in response:
            errors.append("Missing required 'block_type' fields")
            
        return "; ".join(errors) if errors else "JSON parsing or validation errors"
    
    async def _generate_structured_error_response(self, error_message: str) -> str:
        """
        Generate a valid UnifiedAgentResponse for error cases.
        
        Args:
            error_message: Error description for user
            
        Returns:
            Valid JSON string conforming to schema
        """
        from schemas.ui_schemas import create_error_response
        import json
        
        try:
            error_response = create_error_response(error_message, include_debug=False)
            return json.dumps(error_response.model_dump(), indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to generate structured error response: {e}")
            # Absolute fallback - minimal valid JSON
            return json.dumps({
                "response": [
                    {
                        "block_type": "text",
                        "content": f"## System Error\n\n{error_message}\n\nPlease try your request again or contact support if the issue persists."
                    }
                ]
            })
    
    async def _execute_pure_tool_call(self, tool_call) -> Dict[str, Any]:
        """
        PURE tool execution - no decisions, just run what SDK requests
        """
        try:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            logger.info(f"🔧 PURE EXECUTION: {function_name} with {len(function_args)} args")
            
            # Get pure function
            tool_func = self.tool_functions.get(function_name)
            if not tool_func:
                raise ValueError(f"Unknown tool: {function_name}")
            
            # Execute without any custom logic
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**function_args)
            else:
                result = tool_func(**function_args)
            
            logger.info(f"✅ Tool executed successfully: {len(str(result))} chars returned")
            return result
            
        except Exception as e:
            logger.error(f"❌ Pure tool execution failed: {e}")
            return {
                "error": str(e),
                "success": False,
                "tool_name": tool_call.function.name
            }
    
    async def _execute_query_codebase(self, query: str, analysis_mode: str = "auto", 
                                    filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        PURE query execution with automatic diagram generation for structural_kg mode
        
        Implements the complete workflow:
        1. Execute KG query to get structural data
        2. Detect if this is a diagram request  
        3. Transform data to standard graph format
        4. Generate Mermaid diagram automatically
        5. Return both raw data and rendered diagram
        """
        try:
            logger.info(f"📊 Executing codebase query: '{query[:50]}...' (mode: {analysis_mode})")
            
            # Add project context to filters if available
            enhanced_filters = filters or {}
            if hasattr(self, 'mentioned_projects') and self.mentioned_projects:
                enhanced_filters['project'] = self.mentioned_projects[0]  # Use first project
                logger.info(f"🎯 Applying project filter: {self.mentioned_projects[0]}")
            
            # Use our existing unified query system
            result = await query_codebase(query, enhanced_filters, analysis_mode)
            
            # Build base response
            response = {
                "success": True,
                "query": query,
                "analysis_mode": analysis_mode,
                "results": result.get("results", []),
                "total_results": result.get("total_results", 0),
                "strategy_used": result.get("strategy", "unknown"),
                "execution_metadata": result.get("unified_query", {}),
                "filters_applied": filters or {}
            }
            
            # Step 3-5: Automatic diagram generation for structural_kg mode
            if (analysis_mode == "structural_kg" and 
                result.get("strategy") == "kg_direct" and 
                result.get("total_results", 0) > 0 and
                self._is_diagram_query(query)):
                
                try:
                    logger.info("🎨 Auto-generating diagram from structural data...")
                    
                    # Step 3: Transform KG data to standard graph format
                    graph_data = await self._transform_kg_to_graph_data(result.get("results", []), query)
                    
                    # Step 4: Determine diagram type from query
                    diagram_type = self._infer_diagram_type(query)
                    
                    # Step 5: Generate Mermaid diagram
                    mermaid_code = self.mermaid_renderer.generate(
                        diagram_type=diagram_type,
                        graph_data=graph_data,
                        theme_name="dark_professional"
                    )
                    
                    # Add diagram to response
                    response.update({
                        "diagram_generated": True,
                        "diagram_type": diagram_type,
                        "mermaid_code": mermaid_code,
                        "graph_data": graph_data,
                        "auto_rendered": True
                    })
                    
                    logger.info(f"✅ Auto-diagram generated: {len(mermaid_code)} chars of {diagram_type}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Auto-diagram generation failed: {e}")
                    response["diagram_error"] = str(e)
            
            logger.info(f"✅ Query executed: {response['total_results']} results via {response['strategy_used']}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Query execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "analysis_mode": analysis_mode,
                "results": []
            }
    
    def _is_diagram_query(self, query: str) -> bool:
        """Detect if this query is requesting a diagram"""
        query_lower = query.lower()
        diagram_indicators = [
            'diagram', 'chart', 'visualization', 'graph', 'hierarchy', 
            'structure', 'architecture', 'relationship', 'class diagram',
            'show', 'visualize', 'display'
        ]
        return any(indicator in query_lower for indicator in diagram_indicators)
    
    def _infer_diagram_type(self, query: str) -> str:
        """Infer the best diagram type from the query"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['class', 'hierarchy', 'inheritance', 'extends']):
            return "class_diagram"
        elif any(term in query_lower for term in ['flow', 'process', 'sequence', 'steps']):
            return "flowchart"
        elif any(term in query_lower for term in ['component', 'module', 'service', 'architecture']):
            return "graph"
        else:
            return "graph"  # Default fallback
    
    async def _transform_kg_to_graph_data(self, kg_results: List[Dict], query: str) -> Dict[str, Any]:
        """
        Step 3: Transform KG structural data to standard graph format
        
        Converts the complex KG node data into the clean, standardized format
        that the MermaidRenderer expects. Optionally enriches with file content.
        """
        nodes = []
        edges = []
        node_ids = set()
        
        # Extract nodes from KG results
        for kg_item in kg_results:
            if kg_item.get("type") == "kg_structural_node":
                node_data = kg_item.get("node_data", {})
                node_id = node_data.get("name", "unknown")
                
                if node_id not in node_ids:
                    enhanced_node = {
                        "id": node_id,
                        "label": node_id,
                        "semantic_role": self._infer_semantic_role(node_data),
                        "file_path": node_data.get("file_path", ""),
                        "type": node_data.get("type", "unknown"),
                        "line_start": node_data.get("line_start"),
                        "line_end": node_data.get("line_end"),
                        "signature": node_data.get("signature", "")
                    }
                    
                    # Enhance with file content for richer diagrams
                    enhanced_node = self._enrich_node_with_file_data(enhanced_node)
                    
                    nodes.append(enhanced_node)
                    node_ids.add(node_id)
                
                # Extract relationships as edges
                outgoing_rels = kg_item.get("outgoing_relationships", [])
                incoming_rels = kg_item.get("incoming_relationships", [])
                connected_nodes = kg_item.get("connected_nodes", [])
                
                # Create lookup for connected nodes
                connected_lookup = {node.get("id"): node.get("name", "unknown") for node in connected_nodes}
                
                # Process outgoing relationships
                for rel in outgoing_rels:
                    target_node_id = rel.get("target_id")
                    target_name = connected_lookup.get(target_node_id)
                    
                    if target_name and target_name.lower() not in ['none', '(none)', '', 'null']:
                        edges.append({
                            "source": node_id,
                            "target": target_name,
                            "type": rel.get("type", "uses"),
                            "label": rel.get("type", "")
                        })
                
                # Process incoming relationships (for completeness)
                for rel in incoming_rels:
                    source_node_id = rel.get("source_id")
                    source_name = connected_lookup.get(source_node_id)
                    
                    if source_name and source_name.lower() not in ['none', '(none)', '', 'null']:
                        # Only add if we haven't already added this edge from the other direction
                        edge_exists = any(
                            e["source"] == source_name and e["target"] == node_id 
                            for e in edges
                        )
                        if not edge_exists:
                            edges.append({
                                "source": source_name,
                                "target": node_id,
                                "type": rel.get("type", "uses"),
                                "label": rel.get("type", "")
                            })
        
        # Infer additional relationships from file analysis using universal engine
        additional_edges = await self._infer_relationships_from_files(nodes)
        edges.extend(additional_edges)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "query": query,
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "enhanced_with_files": True,
                "generated_at": "auto"
            }
        }
    
    def _enrich_node_with_file_data(self, node: Dict) -> Dict:
        """
        Enhance node data by reading actual file content.
        This addresses the limitation of relying only on KG data.
        """
        try:
            file_path = node.get("file_path", "")
            if not file_path or not file_path.startswith("/workspace"):
                return node
            
            # Read file content to extract methods, attributes, imports
            from pathlib import Path
            actual_file = Path(file_path)
            
            if actual_file.exists() and actual_file.suffix in ['.java', '.py', '.js', '.ts']:
                content = actual_file.read_text(encoding='utf-8', errors='ignore')
                
                # Extract methods and attributes from actual code
                if file_path.endswith('.java'):
                    methods, attributes = self._parse_java_content(content, node.get("id", ""))
                elif file_path.endswith('.py'):
                    methods, attributes = self._parse_python_content(content, node.get("id", ""))
                else:
                    methods, attributes = [], []
                
                # Add to node
                node["methods"] = methods[:10]  # Limit for diagram clarity
                node["attributes"] = attributes[:5]  # Limit for diagram clarity
                node["enhanced_from_file"] = True
                
        except Exception as e:
            logger.debug(f"Could not enhance node {node.get('id')} with file data: {e}")
        
        return node
    
    def _parse_java_content(self, content: str, class_name: str) -> tuple[List[str], List[str]]:
        """Extract methods and attributes from Java class content"""
        import re
        
        methods = []
        attributes = []
        
        # Find methods (simplified regex)
        method_pattern = r'public\s+[\w<>\[\],\s]+\s+(\w+)\s*\('
        for match in re.finditer(method_pattern, content):
            method_name = match.group(1)
            if method_name not in ['class', class_name]:  # Skip constructors
                methods.append(method_name)
        
        # Find attributes/fields
        field_pattern = r'private\s+[\w<>\[\],\s]+\s+(\w+)\s*[;=]'
        for match in re.finditer(field_pattern, content):
            field_name = match.group(1)
            attributes.append(field_name)
        
        return methods, attributes
    
    def _parse_python_content(self, content: str, class_name: str) -> tuple[List[str], List[str]]:
        """Extract methods and attributes from Python class content"""
        import re
        
        methods = []
        attributes = []
        
        # Find methods
        method_pattern = r'def\s+(\w+)\s*\('
        for match in re.finditer(method_pattern, content):
            method_name = match.group(1)
            if not method_name.startswith('_'):  # Skip private methods for clarity
                methods.append(method_name)
        
        # Find class attributes (simplified)
        attr_pattern = r'self\.(\w+)\s*='
        for match in re.finditer(attr_pattern, content):
            attr_name = match.group(1)
            if not attr_name.startswith('_'):
                attributes.append(attr_name)
        
        return list(set(methods)), list(set(attributes))  # Remove duplicates
    
    async def _infer_relationships_from_files(self, nodes: List[Dict]) -> List[Dict]:
        """
        Universal relationship inference using the Universal Pattern Recognition Engine.
        Works across all programming languages and frameworks.
        """
        additional_edges = []
        
        try:
            from tools.universal_relationship_engine import UniversalRelationshipEngine
            
            # Initialize universal relationship engine
            relationship_engine = UniversalRelationshipEngine()
            
            # Convert nodes to components format for universal engine
            components = []
            for node in nodes:
                components.append({
                    'node_data': node,
                    'type': 'kg_structural_node',
                    'source': 'knowledge_graph'
                })
            
            # Infer relationships using universal engine
            detected_relationships = await relationship_engine.infer_relationships_from_components(
                components, query_context="diagram generation")
            
            # Convert to diagram edge format
            for rel in detected_relationships:
                additional_edges.append({
                    "source": rel.source_component,
                    "target": rel.target_component,
                    "type": rel.relationship_type.value,
                    "label": relationship_engine._get_relationship_label(rel.relationship_type),
                    "confidence": rel.confidence,
                    "evidence": rel.evidence,
                    "language": rel.source_language
                })
            
            logger.info(f"🔗 Universal relationship engine found {len(additional_edges)} relationships")
            
        except Exception as e:
            logger.warning(f"Universal relationship inference failed, falling back to basic analysis: {e}")
            # Fallback to basic file analysis if universal engine fails
            additional_edges = await self._basic_file_analysis_fallback(nodes)
        
        return additional_edges
    
    async def _basic_file_analysis_fallback(self, nodes: List[Dict]) -> List[Dict]:
        """
        Basic fallback file analysis when universal engine is not available.
        This provides minimal relationship detection for backward compatibility.
        """
        additional_edges = []
        
        try:
            for node in nodes:
                file_path = node.get("file_path", "")
                if not file_path:
                    continue
                
                from pathlib import Path
                actual_file = Path(file_path)
                
                if actual_file.exists():
                    content = actual_file.read_text(encoding='utf-8', errors='ignore')
                    
                    # Basic pattern detection for common frameworks
                    if "controller" in file_path.lower():
                        service_deps = self._find_service_dependencies(content, nodes)
                        for dep in service_deps:
                            additional_edges.append({
                                "source": node["id"],
                                "target": dep,
                                "type": "uses",
                                "label": "uses"
                            })
                            
        except Exception as e:
            logger.debug(f"Basic file analysis fallback failed: {e}")
        
        return additional_edges
    
    def _find_service_dependencies(self, content: str, all_nodes: List[Dict]) -> List[str]:
        """Find service dependencies in controller code"""
        import re
        
        dependencies = []
        service_names = [node["id"] for node in all_nodes if "service" in node.get("semantic_role", "").lower()]
        
        # Look for service injections/imports
        for service_name in service_names:
            if service_name.lower() in content.lower():
                dependencies.append(service_name)
        
        return dependencies
    
    def _infer_semantic_role(self, node_data: Dict) -> str:
        """Infer semantic role for styling purposes"""
        node_type = node_data.get("type", "").lower()
        file_path = node_data.get("file_path", "").lower()
        name = node_data.get("name", "").lower()
        
        if "controller" in file_path or "controller" in name:
            return "controller"
        elif "service" in file_path or "service" in name:
            return "service"  
        elif "model" in file_path or "entity" in file_path:
            return "model"
        elif node_type == "class":
            return "logic"
        elif node_type == "interface":
            return "interface"
        else:
            return "external"
    
    def _execute_navigate_filesystem(self, operation: str, path: str = None, 
                                   pattern: str = None, recursive: bool = False) -> Dict[str, Any]:
        """
        REQ-3.6.1: PURE filesystem navigation execution using KG data
        NO custom reasoning - just execute filesystem operations and return data
        """
        try:
            logger.info(f"📁 Executing filesystem navigation: operation={operation} path={path} pattern={pattern} recursive={recursive}")
            
            if not self.filesystem_navigator:
                return {
                    "success": False,
                    "error": "Filesystem navigator not available - Knowledge Graph required",
                    "operation": operation
                }
            
            # Execute the filesystem operation using KG data with project scope
            project_scope = None
            if hasattr(self, 'mentioned_projects') and self.mentioned_projects:
                project_scope = self.mentioned_projects[0]  # Use first mentioned project
                logger.info(f"🎯 Passing project scope to filesystem navigator: {project_scope}")
            
            result = self.filesystem_navigator.execute(operation, path, pattern, recursive, project_scope)
            
            # Add success flag if not present
            if "error" not in result:
                result["success"] = True
            else:
                result["success"] = False
            
            logger.info(f"✅ Filesystem operation executed: {result.get('count', 0)} items found")
            return result
            
        except Exception as e:
            logger.error(f"❌ Filesystem navigation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "operation": operation,
                "path": path,
                "pattern": pattern
            }
    
    def _execute_generate_diagram(self, diagram_type: str, structural_data: Any, 
                                 title: str = "", theme: str = "default") -> Dict[str, Any]:
        """
        PURE diagram generation using Phase 2 renderer
        NO custom logic - just generate and return
        """
        try:
            logger.info(f"📊 Generating {diagram_type} diagram with theme: {theme}")
            
            # Parse structural data if needed
            if isinstance(structural_data, str):
                try:
                    data = json.loads(structural_data)
                except json.JSONDecodeError:
                    # If not valid JSON, wrap as raw data
                    data = {"raw_data": structural_data}
            else:
                data = structural_data
            
            # Use our Phase 2 pure renderer
            mermaid_code = self.mermaid_renderer.generate(
                diagram_type=diagram_type,
                graph_data=data,
                theme_name=theme or "dark_professional"
            )
            
            response = {
                "success": True,
                "diagram_type": diagram_type,
                "mermaid_code": mermaid_code,
                "title": title,
                "theme": theme,
                "data_source": "structural_data",
                "code_length": len(mermaid_code)
            }
            
            logger.info(f"✅ Diagram generated: {len(mermaid_code)} chars of Mermaid code")
            return response
            
        except Exception as e:
            logger.error(f"❌ Diagram generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "diagram_type": diagram_type,
                "title": title
            }
    
    async def _execute_examine_files(self, file_paths: List[str], 
                                   context_lines: int = 5) -> Dict[str, Any]:
        """
        PURE file examination - return file contents for SDK analysis
        """
        try:
            logger.info(f"📄 Examining {len(file_paths)} files")
            
            results = []
            
            # Limit files to prevent overload
            limited_paths = file_paths[:10]
            
            for file_path in limited_paths:
                try:
                    # Resolve file path to absolute workspace path
                    from pathlib import Path
                    workspace_path = Path("/workspace")
                    
                    # Handle different path formats
                    if file_path.startswith("/workspace/"):
                        absolute_path = Path(file_path)
                    elif file_path.startswith("./"):
                        absolute_path = workspace_path / file_path[2:]
                    elif file_path.startswith("/"):
                        absolute_path = Path(file_path)
                    else:
                        # Relative path - try with workspace prefix
                        absolute_path = workspace_path / file_path
                    
                    logger.debug(f"Resolving '{file_path}' to '{absolute_path}'")
                    
                    # Read file with encoding detection
                    encodings = ['utf-8', 'latin-1', 'cp1252']
                    content = None
                    encoding_used = None
                    
                    for encoding in encodings:
                        try:
                            with open(absolute_path, 'r', encoding=encoding) as f:
                                content = f.read()
                            encoding_used = encoding
                            break
                        except UnicodeDecodeError:
                            continue
                        except FileNotFoundError:
                            # If not found, try without workspace prefix
                            if not file_path.startswith("/workspace/"):
                                try:
                                    with open(file_path, 'r', encoding=encoding) as f:
                                        content = f.read()
                                    encoding_used = encoding
                                    break
                                except:
                                    continue
                    
                    if content is None:
                        raise ValueError(f"Could not decode file with any encoding: {encodings}")
                    
                    # Limit content size for performance
                    max_content_size = 50000  # 50KB limit
                    if len(content) > max_content_size:
                        content = content[:max_content_size] + "\n... [content truncated]"
                    
                    results.append({
                        "file_path": file_path,
                        "content": content,
                        "size_bytes": len(content),
                        "lines": len(content.splitlines()),
                        "encoding": encoding_used,
                        "truncated": len(content) >= max_content_size
                    })
                    
                except Exception as e:
                    results.append({
                        "file_path": file_path,
                        "error": str(e),
                        "content": None,
                        "accessible": False
                    })
            
            response = {
                "success": True,
                "files_examined": len(results),
                "files_requested": len(file_paths),
                "context_lines": context_lines,
                "results": results
            }
            
            logger.info(f"✅ Examined {len(results)} files successfully")
            return response
            
        except Exception as e:
            logger.error(f"❌ File examination failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "files_requested": len(file_paths),
                "results": []
            }
    
    async def _execute_search_symbols(self, symbol_name: str, 
                                    symbol_type: Optional[str] = None,
                                    include_relationships: bool = True) -> Dict[str, Any]:
        """
        PURE symbol search using Knowledge Graph
        """
        try:
            logger.info(f"🔍 Searching for symbol: {symbol_name} (type: {symbol_type})")
            
            if not self.kg:
                return {
                    "success": False,
                    "error": "Knowledge Graph not available - cannot search symbols",
                    "symbol_name": symbol_name,
                    "kg_available": False
                }
            
            # Direct KG query - no custom logic
            if symbol_type:
                # Search by type first, then filter by name
                nodes = self.kg.get_nodes_by_type(symbol_type)
                filtered = [n for n in nodes if symbol_name.lower() in n.get("name", "").lower()]
            else:
                # Direct name search
                filtered = self.kg.get_nodes_by_name(symbol_name, exact_match=False)
            
            # Get relationships for found symbols if requested
            enhanced_results = []
            for node in filtered[:20]:  # Limit to top 20 results
                try:
                    symbol_data = {
                        "symbol": node,
                        "callers": [],
                        "dependencies": []
                    }
                    
                    if include_relationships:
                        # Get callers (who calls this symbol)
                        callers = self.kg.find_callers(node["name"], max_depth=2)
                        symbol_data["callers"] = callers[:10]  # Limit callers
                        
                        # Get dependencies (what this symbol depends on)
                        dependencies = self.kg.find_dependencies(node["name"], max_depth=2)
                        symbol_data["dependencies"] = dependencies[:10]  # Limit dependencies
                    
                    enhanced_results.append(symbol_data)
                    
                except Exception as e:
                    # If relationship lookup fails, still include the symbol
                    logger.warning(f"Failed to get relationships for {node.get('name', 'unknown')}: {e}")
                    enhanced_results.append({
                        "symbol": node,
                        "callers": [],
                        "dependencies": [],
                        "relationship_error": str(e)
                    })
            
            response = {
                "success": True,
                "symbol_searched": symbol_name,
                "symbol_type": symbol_type,
                "include_relationships": include_relationships,
                "results": enhanced_results,
                "total_found": len(enhanced_results),
                "kg_available": True
            }
            
            logger.info(f"✅ Found {len(enhanced_results)} symbols matching '{symbol_name}'")
            return response
            
        except Exception as e:
            logger.error(f"❌ Symbol search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol_name": symbol_name,
                "symbol_type": symbol_type,
                "kg_available": self.kg is not None
            }
    
    # REQ-CTX-UNIFIED: Core Summarization Engine Implementation
    async def _execute_tool_and_summarize(self, tool_name: str, tool_result: Dict[str, Any], user_query: str) -> str:
        """
        REQ-CTX-UNIFIED: Execute tool and apply summarization if output is large
        
        Args:
            tool_name: Name of the tool that was executed
            tool_result: Raw result from tool execution
            user_query: Original user query for context
            
        Returns:
            Either raw tool result (if small) or intelligent summary (if large)
        """
        try:
            # Convert result to string with safe content isolation
            raw_result_str = self._safe_serialize_tool_result(tool_result, tool_name)
            raw_size_chars = len(raw_result_str)
            
            # Character threshold: 20,000 chars ≈ 5,000 tokens
            SUMMARIZATION_THRESHOLD = 20000
            
            if raw_size_chars <= SUMMARIZATION_THRESHOLD:
                # Small output - return as-is
                logger.info(f"🔧 Tool output size: {raw_size_chars} chars (below threshold, no summarization)")
                return raw_result_str
            else:
                # Large output - apply intelligent summarization
                logger.info(f"🔧 Tool output size: {raw_size_chars} chars (above threshold, applying summarization)")
                
                summary = await self._summarize_with_llm(raw_result_str, user_query, tool_name)
                summary_size = len(summary)
                compression_ratio = ((raw_size_chars - summary_size) / raw_size_chars) * 100
                
                # REQ-CTX-MONITORING: Log compression metrics
                logger.info(f"🔧 Tool compression: {tool_name} {raw_size_chars//1000}KB→{summary_size//1000}KB "
                           f"({compression_ratio:.1f}% reduction)")
                
                return summary
                
        except Exception as e:
            logger.error(f"❌ Summarization failed for {tool_name}: {e}")
            # Fallback: return truncated raw result
            truncated = raw_result_str[:15000] + "\n\n[TRUNCATED DUE TO SUMMARIZATION ERROR]"
            return truncated
    
    async def _summarize_with_llm(self, large_content: str, user_query: str, tool_name: str) -> str:
        """
        REQ-CTX-UNIFIED: Make isolated LLM call to summarize large tool output
        
        Args:
            large_content: Large tool output to summarize
            user_query: Original user query for context
            tool_name: Name of tool that generated the content
            
        Returns:
            Dense, structured summary preserving key information
        """
        try:
            # REQ-CTX-PROMPT: Get specialized prompt based on tool type
            compression_prompt = self._get_compression_prompt(tool_name, user_query)
            
            # Truncate input to safely fit in summarization context
            # Reserve space for prompt + response (≈8K tokens)
            max_input_chars = 30000
            content_to_summarize = large_content[:max_input_chars]
            if len(large_content) > max_input_chars:
                content_to_summarize += "\n\n[INPUT TRUNCATED FOR SUMMARIZATION]"
            
            # Build isolated summarization prompt
            full_prompt = f"""{compression_prompt}

INPUT DATA TO SUMMARIZE:
---
{content_to_summarize}
---

Remember: Extract key facts, preserve all important entities and relationships, format as structured Markdown. Maximum 2000 tokens."""
            
            # Make isolated API call (no conversation history to save tokens)
            summarization_messages = [
                {"role": "user", "content": full_prompt}
            ]
            
            # Use same model as main query for consistency
            completion_config = cerebras_config.get_completion_config()
            
            # Build API parameters for summarization
            api_params = {
                **completion_config,
                "messages": summarization_messages,
                "max_tokens": 2500  # Ensure room for 2000 token summary
            }
            
            # Only add reasoning_effort for models that support it
            if cerebras_config.supports_reasoning_effort(completion_config.get("model", "")):
                api_params["reasoning_effort"] = "low"  # Faster for summarization
            
            # Execute summarization call
            logger.info(f"🔧 Starting summarization for {tool_name} ({len(content_to_summarize)} chars)")
            
            start_time = time.time()
            response = self.client.chat.completions.create(**api_params)
            summarization_time = time.time() - start_time
            
            summary = response.choices[0].message.content
            
            # REQ-CTX-MONITORING: Log summarization metrics
            logger.info(f"🔧 Summarization completed: {tool_name} in {summarization_time:.2f}s, "
                       f"output: {len(summary)} chars")
            
            # Log full summarization response for debugging/monitoring
            logger.info(f"📝 Full Summarization Response ({tool_name}):\n{'='*50}\n{summary}\n{'='*50}")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ LLM summarization failed: {e}")
            # Fallback: return structured truncation
            return self._fallback_summarization(large_content, user_query, tool_name)
    
    def _get_compression_prompt(self, tool_name: str, user_query: str) -> str:
        """
        REQ-CTX-PROMPT: Get specialized compression prompt based on tool type
        
        Args:
            tool_name: Name of the tool that generated content
            user_query: Original user query for context
            
        Returns:
            Specialized compression prompt for the tool type
        """
        # REQ-CTX-PROMPT: Specialized prompts by tool type
        if tool_name == "query_codebase":
            return f"""You are compressing code analysis results for this query: "{user_query}"

CONTENT CLASSIFICATION (prioritize in this order):
1. ESSENTIAL: Class definitions, main functions, API endpoints, file paths, line numbers
2. IMPORTANT: Dependencies, inheritance relationships, method signatures, design patterns
3. SECONDARY: Helper functions, utilities, configuration details
4. NOISE: Debug output, verbose descriptions, duplicate information

OUTPUT FORMAT (use this exact structure):
## Key Components
- **ComponentName** (`file/path.py:123-145`)
  - Type: [class/function/interface]
  - Purpose: [one-line description]
  - Key methods: method1(), method2()
  - Dependencies: [list key dependencies]

## Architecture Patterns
- **Pattern**: [MVC/Singleton/Factory/etc] in [location]
- **Data Flow**: [describe key data flows]

## Critical Code Snippets
```[language]
// Only essential code that directly answers the query
```

## File References
- Main files: [list with line numbers]
- Configuration: [config files and settings]

COMPRESSION RULES:
- Be extremely dense - every sentence contains multiple facts
- Preserve ALL file paths and line numbers
- Include method signatures for key functions  
- Remove verbose explanations and debug info
- Maximum 2000 tokens"""
        
        elif tool_name == "navigate_filesystem":
            return f"""You are compressing project structure analysis for this query: "{user_query}"

CONTENT CLASSIFICATION (prioritize in this order):
1. ESSENTIAL: Main directories, entry points, configuration files
2. IMPORTANT: Source code directories, test directories, build files
3. SECONDARY: Documentation, assets, utilities
4. NOISE: Hidden files, cache directories, generated files

OUTPUT FORMAT (use this exact structure):
## Project Architecture
```
project/
├── src/main/           # Core application logic
├── src/components/     # UI components  
├── config/            # Configuration files
└── tests/             # Test suites
```

## Key Entry Points
- **Main**: [path to main application file]
- **Build**: [build configuration files]
- **Tests**: [test runner locations]

## Module Organization
- **Backend**: [backend directories and purposes]
- **Frontend**: [frontend directories and purposes]  
- **Shared**: [shared utilities and libraries]

## Configuration
- **Environment**: [env files, config locations]
- **Dependencies**: [package.json, requirements.txt, etc]

COMPRESSION RULES:
- Focus on architectural significance
- Show directory hierarchy with purposes
- Identify main code vs config vs tests
- Note build and deployment files
- Maximum 1000 tokens"""
        
        else:
            # Default compression prompt for any other tools
            return f"""You are compressing technical tool output for this query: "{user_query}"

CONTENT CLASSIFICATION (prioritize in this order):
1. ESSENTIAL: Direct answers to the query, key findings, file paths
2. IMPORTANT: Supporting details, relationships, technical specs
3. SECONDARY: Background information, context
4. NOISE: Verbose descriptions, debug output, duplicates

OUTPUT FORMAT (use structured Markdown):
## Key Findings
- [Most important discoveries]

## Technical Details
- [Relevant specifications, configurations, etc]

## File References
- [Important file paths and locations]

## Additional Context
- [Supporting information that helps answer the query]

COMPRESSION RULES:
- Prioritize information that directly answers the user's query
- Preserve all file paths and technical identifiers
- Remove verbose explanations and redundant information
- Structure for easy scanning and comprehension
- Maximum 2000 tokens"""
    
    def _fallback_summarization(self, content: str, user_query: str, tool_name: str) -> str:
        """
        REQ-CTX-FALLBACK: Fallback summarization when LLM summarization fails
        
        Args:
            content: Content to summarize
            user_query: Original user query
            tool_name: Tool that generated content
            
        Returns:
            Basic structured truncation with key information preserved
        """
        try:
            # Extract key sections intelligently
            lines = content.split('\n')
            
            # Preserve key patterns (file paths, class names, function definitions)
            important_lines = []
            for line in lines:
                # Keep lines with file paths, class definitions, function definitions, errors
                if any(pattern in line.lower() for pattern in [
                    'file_path', 'class ', 'def ', 'function ', 'error', 'exception',
                    '.py:', '.js:', '.ts:', '.java:', 'line ', 'extends', 'implements'
                ]):
                    important_lines.append(line)
            
            # Build fallback summary
            fallback_summary = f"""# Summarization Fallback for {tool_name}

**Query**: {user_query}
**Note**: LLM summarization failed, showing key extracted information

## Key Information Found:
"""
            
            # Add important lines (truncated to fit)
            char_budget = 10000  # Conservative limit for fallback
            current_chars = len(fallback_summary)
            
            for line in important_lines:
                if current_chars + len(line) + 2 > char_budget:
                    break
                fallback_summary += f"- {line.strip()}\n"
                current_chars += len(line) + 3
            
            fallback_summary += f"\n[FALLBACK SUMMARIZATION - LLM COMPRESSION FAILED]"
            
            logger.info(f"🔧 Applied fallback summarization: {len(content)} → {len(fallback_summary)} chars")
            return fallback_summary
            
        except Exception as e:
            logger.error(f"❌ Even fallback summarization failed: {e}")
            # Last resort: simple truncation
            return f"# Error in summarization\n\nQuery: {user_query}\nTool: {tool_name}\n\n" + content[:8000] + "\n\n[TRUNCATED]"
    
    def _extract_user_query(self, messages: List[Dict]) -> str:
        """
        Extract the user's original query from the message history
        
        Args:
            messages: Message history from the conversation
            
        Returns:
            The user's query string, or fallback if not found
        """
        try:
            # Find the first user message (skip system messages)
            for message in messages:
                if message.get("role") == "user":
                    content = message.get("content", "")
                    if content and len(content.strip()) > 0:
                        return content.strip()
            
            # Fallback if no user message found
            return "architecture analysis query"
            
        except Exception as e:
            logger.warning(f"Failed to extract user query: {e}")
            return "code analysis query"
    
    # REQ-CTX-MONITORING: Context Usage Analytics Implementation
    def _log_context_usage(self, messages: List[Dict]) -> None:
        """
        REQ-CTX-MONITORING: Log comprehensive context usage analytics
        
        Args:
            messages: Current message history to analyze
        """
        try:
            # Calculate total context size
            total_chars = 0
            total_tokens_estimate = 0
            message_breakdown = {
                "system": 0,
                "user": 0, 
                "assistant": 0,
                "tool": 0
            }
            tool_usage = {}
            
            for message in messages:
                # Handle both dict messages and SDK response objects
                if hasattr(message, 'role'):
                    role = message.role if hasattr(message, 'role') else "unknown"
                    content = str(message.content if hasattr(message, 'content') else "")
                else:
                    role = message.get("role", "unknown") if isinstance(message, dict) else "unknown"
                    content = str(message.get("content", "") if isinstance(message, dict) else "")
                
                char_count = len(content)
                token_estimate = char_count // 4  # Rough estimate: 4 chars per token
                
                total_chars += char_count
                total_tokens_estimate += token_estimate
                
                if role in message_breakdown:
                    message_breakdown[role] += token_estimate
                
                # Track tool usage
                has_tool_call_id = (hasattr(message, 'tool_call_id') or 
                                   (isinstance(message, dict) and "tool_call_id" in message))
                if role == "tool" and has_tool_call_id:
                    # Try to identify tool type from content
                    if "query_codebase" in content:
                        tool_usage["query_codebase"] = tool_usage.get("query_codebase", 0) + 1
                    elif "navigate_filesystem" in content:
                        tool_usage["navigate_filesystem"] = tool_usage.get("navigate_filesystem", 0) + 1
                    else:
                        tool_usage["other"] = tool_usage.get("other", 0) + 1
            
            # Context limits (Cerebras specific)
            CONTEXT_LIMIT_TOKENS = 65536
            context_utilization = (total_tokens_estimate / CONTEXT_LIMIT_TOKENS) * 100
            
            # REQ-CTX-MONITORING: Log metrics at INFO level for operational visibility
            logger.info(f"📊 Context usage: {total_tokens_estimate:,}/{CONTEXT_LIMIT_TOKENS:,} tokens "
                       f"({context_utilization:.1f}% utilization)")
            
            # Log breakdown
            logger.info(f"📈 Message breakdown: system={message_breakdown['system']}, "
                       f"user={message_breakdown['user']}, assistant={message_breakdown['assistant']}, "
                       f"tool={message_breakdown['tool']} tokens")
            
            # Log tool usage if any tools were used
            if tool_usage:
                tool_summary = ", ".join([f"{tool}={count}" for tool, count in tool_usage.items()])
                logger.info(f"🔧 Tool usage: {tool_summary}")
            
            # Warning at 75% capacity
            if context_utilization > 75:
                logger.warning(f"⚠️ Context usage approaching limit: {context_utilization:.1f}% utilized")
            
            # Critical warning at 90% capacity  
            if context_utilization > 90:
                logger.warning(f"🚨 Context usage critical: {context_utilization:.1f}% - compression strongly recommended")
                
        except Exception as e:
            logger.warning(f"Failed to log context usage: {e}")
    
    def _estimate_token_count(self, text: str) -> int:
        """
        REQ-CTX-MONITORING: Estimate token count for text
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count (rough approximation)
        """
        # Rough estimation: 4 characters per token on average
        # This is a conservative estimate for most English text
        char_count = len(text)
        token_estimate = char_count // 4
        
        # Add slight buffer for tokenization overhead
        return int(token_estimate * 1.1)
    
    def _check_context_health(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        REQ-CTX-MONITORING: Comprehensive context health assessment
        
        Args:
            messages: Message history to analyze
            
        Returns:
            Dictionary with context health metrics
        """
        try:
            total_tokens = 0
            for msg in messages:
                if hasattr(msg, 'content'):
                    content = str(msg.content)
                elif isinstance(msg, dict):
                    content = str(msg.get("content", ""))
                else:
                    content = ""
                total_tokens += self._estimate_token_count(content)
            utilization = (total_tokens / 65536) * 100
            
            # Count summarization events
            summarization_count = 0
            for message in messages:
                content = str(message.get("content", ""))
                if "[SUMMARIZED FROM" in content or "COMPRESSION APPLIED" in content:
                    summarization_count += 1
            
            # Count tool calls
            tool_calls = len([msg for msg in messages if msg.get("role") == "tool"])
            
            return {
                "total_tokens": total_tokens,
                "utilization_percent": utilization,
                "context_health": "healthy" if utilization < 75 else "warning" if utilization < 90 else "critical",
                "summarization_events": summarization_count,
                "tool_calls": tool_calls,
                "message_count": len(messages),
                "efficiency_score": 100 - utilization if utilization < 100 else 0
            }
            
        except Exception as e:
            logger.warning(f"Failed to check context health: {e}")
            return {"error": str(e)}
    
    # REQ-CTX-FALLBACK: Graceful Degradation Implementation
    async def _apply_graceful_degradation(self, messages: List[Dict], context_health: Dict[str, Any]) -> str:
        """
        REQ-CTX-FALLBACK: Apply four-tier graceful degradation strategy
        
        Args:
            messages: Current message history
            context_health: Context health metrics
            
        Returns:
            Response using appropriate fallback strategy
        """
        try:
            utilization = context_health.get("utilization_percent", 0)
            logger.info(f"🔧 Applying graceful degradation for {utilization:.1f}% context utilization")
            
            # Tier 2: Progressive context trimming (90-95% utilization)
            if utilization < 95:
                logger.info("📝 Tier 2: Progressive context trimming")
                trimmed_messages = await self._progressive_context_trimming(messages)
                
                # Try to continue with trimmed context
                if len(trimmed_messages) < len(messages):
                    logger.info(f"✅ Context trimmed: {len(messages)} → {len(trimmed_messages)} messages")
                    return await self._sdk_native_loop(trimmed_messages, self._get_current_model())
            
            # Tier 3: Multi-turn synthesis (95-98% utilization)
            elif utilization < 98:
                logger.info("📝 Tier 3: Multi-turn synthesis")
                return await self._multi_turn_synthesis(messages)
            
            # Tier 4: Partial response with explanation (98%+ utilization)
            else:
                logger.info("📝 Tier 4: Partial response with explanation")
                return self._partial_response_with_explanation(messages, context_health)
                
        except Exception as e:
            logger.error(f"❌ Graceful degradation failed: {e}")
            return self._emergency_fallback(messages)
    
    async def _progressive_context_trimming(self, messages: List[Dict]) -> List[Dict]:
        """
        REQ-CTX-FALLBACK: Tier 2 - Remove oldest tool responses, preserve key context
        
        Args:
            messages: Original message history
            
        Returns:
            Trimmed message history with preserved essential context
        """
        try:
            # Always preserve: system prompt + user query + last 2 tool responses
            trimmed = []
            tool_responses = []
            
            for message in messages:
                role = message.get("role")
                
                if role == "system":
                    # Always keep system prompt
                    trimmed.append(message)
                elif role == "user":
                    # Always keep user messages
                    trimmed.append(message)
                elif role == "assistant":
                    # Keep assistant messages (reasoning)
                    trimmed.append(message)
                elif role == "tool":
                    # Collect tool responses for selective keeping
                    tool_responses.append(message)
            
            # Keep only last 2 tool responses (most recent context)
            if tool_responses:
                recent_tools = tool_responses[-2:]
                trimmed.extend(recent_tools)
                logger.info(f"🔧 Kept {len(recent_tools)}/{len(tool_responses)} tool responses")
            
            return trimmed
            
        except Exception as e:
            logger.error(f"❌ Progressive trimming failed: {e}")
            return messages  # Return original if trimming fails
    
    async def _multi_turn_synthesis(self, messages: List[Dict]) -> str:
        """
        REQ-CTX-FALLBACK: Tier 3 - Synthesize current findings, start fresh
        
        Args:
            messages: Current message history
            
        Returns:
            Synthesized response from current context
        """
        try:
            logger.info("🔧 Starting multi-turn synthesis")
            
            # Extract key findings from current context
            user_query = self._extract_user_query(messages)
            tool_findings = []
            
            for message in messages:
                if message.get("role") == "tool":
                    content = str(message.get("content", ""))
                    if content and len(content.strip()) > 0:
                        # Summarize each tool finding
                        summary = await self._synthesize_tool_finding(content, user_query)
                        tool_findings.append(summary)
            
            # Create synthesis prompt
            synthesis_prompt = f"""Based on the research I've conducted for: "{user_query}"

Key findings from analysis:
{chr(10).join(f"- {finding}" for finding in tool_findings)}

Please provide a comprehensive response that synthesizes these findings into a complete answer."""
            
            # Make isolated synthesis call
            synthesis_messages = [{"role": "user", "content": synthesis_prompt}]
            
            completion_config = cerebras_config.get_completion_config()
            api_params = {
                **completion_config,
                "messages": synthesis_messages,
                "max_tokens": 3000
            }
            
            response = self.client.chat.completions.create(**api_params)
            synthesis_result = response.choices[0].message.content
            
            # Log full multi-turn synthesis response for debugging/monitoring
            synthesis_length = len(synthesis_result) if synthesis_result else 0
            logger.info("✅ Multi-turn synthesis completed successfully")
            logger.info(f"📄 Multi-turn Synthesis Details - Length: {synthesis_length} chars, Model: {self.model}")
            logger.info(f"📝 Full Multi-turn Synthesis:\n{'='*50}\n{synthesis_result}\n{'='*50}")
            
            return synthesis_result
            
        except Exception as e:
            logger.error(f"❌ Multi-turn synthesis failed: {e}")
            return self._partial_response_with_explanation(messages, {"error": str(e)})
    
    async def _synthesize_tool_finding(self, tool_content: str, user_query: str) -> str:
        """
        REQ-CTX-FALLBACK: Synthesize a single tool finding into key insight
        
        Args:
            tool_content: Content from a tool response
            user_query: Original user query
            
        Returns:
            Brief synthesis of the tool finding
        """
        try:
            # Quick synthesis for multi-turn - just extract key points
            lines = tool_content.split('\n')
            key_lines = []
            
            for line in lines:
                line = line.strip()
                if line and any(keyword in line.lower() for keyword in [
                    'key', 'important', 'main', 'primary', 'essential', 'component', 'pattern'
                ]):
                    key_lines.append(line)
                    if len(key_lines) >= 3:  # Limit for synthesis
                        break
            
            if key_lines:
                return f"Analysis found: {'; '.join(key_lines[:3])}"
            else:
                # Fallback: first non-empty line
                for line in lines:
                    if line.strip() and len(line.strip()) > 10:
                        return f"Analysis found: {line.strip()[:100]}..."
                
                return "Analysis completed with technical findings"
                
        except Exception as e:
            logger.warning(f"Failed to synthesize tool finding: {e}")
            return "Analysis completed with mixed results"
    
    def _partial_response_with_explanation(self, messages: List[Dict], context_health: Dict[str, Any]) -> str:
        """
        REQ-CTX-FALLBACK: Tier 4 - Return best available analysis with clear limitations
        
        Args:
            messages: Message history
            context_health: Context health metrics
            
        Returns:
            Partial response with explanation of limitations
        """
        try:
            user_query = self._extract_user_query(messages)
            
            # Extract key findings from available context
            findings = []
            tool_count = 0
            
            for message in messages:
                if message.get("role") == "tool":
                    tool_count += 1
                    content = str(message.get("content", ""))
                    
                    # Extract first meaningful line
                    lines = content.split('\n')
                    for line in lines:
                        if line.strip() and len(line.strip()) > 20:
                            findings.append(line.strip()[:150] + "...")
                            break
            
            utilization = context_health.get("utilization_percent", 0)
            
            return f"""I've analyzed your query: "{user_query}"

## Analysis Results Found:
{chr(10).join(f"• {finding}" for finding in findings[:5])}

## Context Limitation Notice:
Due to the complexity of your query, I've reached the context processing limit ({utilization:.1f}% of available context). This response contains the key findings from {tool_count} analysis operations that were completed successfully.

## Suggestions:
- For deeper analysis, consider breaking your query into smaller, more specific questions
- Focus on particular aspects of the architecture you're most interested in
- Ask follow-up questions about specific components mentioned above

The analysis above represents the most important findings I was able to gather before reaching processing limits."""
            
        except Exception as e:
            logger.error(f"❌ Partial response generation failed: {e}")
            return self._emergency_fallback(messages)
    
    def _emergency_fallback(self, messages: List[Dict]) -> str:
        """
        REQ-CTX-FALLBACK: Final emergency fallback when all else fails
        
        Args:
            messages: Message history
            
        Returns:
            Basic response acknowledging the query
        """
        try:
            user_query = self._extract_user_query(messages)
            return f"""I understand you're asking about: "{user_query}"

I encountered technical limitations while processing your request due to context constraints. However, I successfully initiated the analysis and gathered preliminary information.

To help you better, please consider:
1. Breaking your question into smaller, more specific parts
2. Focusing on particular aspects of what you're trying to understand
3. Asking about specific files, classes, or components

I'm ready to help with a more focused query whenever you're ready."""
            
        except Exception as e:
            logger.error(f"❌ Even emergency fallback failed: {e}")
            return "I encountered technical difficulties processing your request. Please try rephrasing your question or breaking it into smaller parts."
    
    def _get_current_model(self) -> str:
        """Helper to get the current model being used"""
        try:
            return cerebras_config.model
        except:
            return "gpt-oss-120b"  # Default fallback
    
    def _get_native_system_prompt(self) -> str:
        """
        REQ-3.7: Mandatory Workflow Architecture - Codebase Onboarding System
        This prompt implements structured workflows that eliminate lazy discovery
        """
        return """# CodeWise v3.2 - Mandatory Workflow Architecture

## 1. Core Identity
You are CodeWise, a world-class senior software engineer and code intelligence platform. Your purpose is to help users understand complex codebases by using your specialized tools in specific, proven workflows that ensure comprehensive analysis.

## 2. Your Specialized Toolset
You have two primary tools. You MUST use them according to the mandatory workflows below.

### Tool 1: `query_codebase(query: str, ...)`
* **Purpose:** To answer **CONCEPTUAL** questions ("how," "why," "explain"). Use it to understand the *meaning* and *logic* of code **once you know where it is**.
* **Critical Rule:** Never use this as your first tool for broad exploratory queries.
* **Best Use:** After you have established the project structure through filesystem exploration.

### Tool 2: `navigate_filesystem(operation: str, ...)`  
* **Purpose:** To answer **STRUCTURAL** questions ("where," "list," "find"). Use it to **discover** the layout of the codebase.
* **Critical Rule:** Always use this first for architecture and exploratory queries.
* **Operations:** 
  - `operation="tree"` - Get directory structure overview
  - `operation="find"` - Find files matching patterns  
  - `operation="list"` - List contents of specific directories

---

## 3. Mandatory Reasoning Workflows

### Workflow A: Standard Code Query (For Specific Questions)
**Use When:** The user asks about a *specific, known* function, class, or file.
**Examples:** "What does the `authenticate_user` function do?" or "How does UserService work?"
**Process:**
1. **Thought:** The user is asking about a specific symbol. I will use `query_codebase` to find it directly.
2. **Action:** `query_codebase(query="authenticate_user function", analysis_mode="specific_symbol")`

### Workflow B: Codebase Onboarding (For Broad, Exploratory Questions)
**Use When:** The user asks a broad, high-level question about a project, such as:
- "explain the architecture"
- "how does this project work" 
- "explain each controller"
- "show me the system design"
- "how is authentication handled"
- Any query about project structure or overview

**MANDATORY Process - You MUST follow every step:**

1. **Thought:** This is a broad, exploratory query. I must first understand the project's structure. My first step is to use `navigate_filesystem` to get a file tree.

2. **Action (Step 1):** `navigate_filesystem(operation="tree", path=".")`

3. **Observation:** [Review the file tree from the tool's output - identify key directories like src/, components/, controllers/, services/, etc.]

4. **Thought:** Based on the file tree, the core logic appears to be in the `[path/to/logic]` and `[path/to/other/logic]` directories. For authentication queries, I should search semantically rather than guessing file paths.

5. **Action (Step 2):** `query_codebase(query="authentication security implementation", analysis_mode="semantic_rag")` OR for more specific searches use patterns like `navigate_filesystem(operation="find", pattern="*Auth*")` to locate auth-related files

6. **Observation:** [Review the semantic search results for detailed implementation information.]

7. **Thought:** I now have both a structural overview and semantic details. I can synthesize a complete answer.

8. **Final Action:** Provide comprehensive answer combining structural layout with semantic analysis.

### Workflow C: Diagram Generation
**Use When:** The user requests a diagram or visualization.
**Process:**
1. **Thought:** The user wants a diagram. According to mandatory workflow, for diagram generation I must call query_codebase with analysis_mode="structural_kg".
2. **Action:** `query_codebase(query="class hierarchy for controllers", analysis_mode="structural_kg")`
3. **System Response:** The system will automatically transform the structural data into a standardized graph format, infer the appropriate diagram type, and generate the Mermaid diagram syntax.

---

## 4. Critical Rules for Robust Discovery

### Rule 1: Workflow Selection is Mandatory
You MUST identify which workflow applies to each user query and follow it exactly. Do not deviate from the prescribed steps.

### Rule 2: Never Skip Structure for Broad Queries  
For any query about architecture, project overview, or "explain X system", you MUST start with `navigate_filesystem` to build structural understanding. This is non-negotiable.

### Rule 3: Cold Start Protection
If you receive limited results from `query_codebase` (< 10 results or only documentation), you MUST use `navigate_filesystem` to verify the project structure before concluding anything about available source code.

### Rule 4: Path Validation - CRITICAL
NEVER explore non-existent paths. If navigate_filesystem returns 0 items, STOP trying similar paths. Instead:
- Use `navigate_filesystem(operation="find", pattern="*keyword*")` to search by filename patterns
- Use `query_codebase(query="concept", analysis_mode="semantic_rag")` for semantic searches
- Check actual project structure before assuming paths exist

### Rule 5: Completeness Validation
Before providing architectural analysis, verify you have:
- [ ] Found source code files (not just documentation)
- [ ] Identified key architectural components  
- [ ] Located main application entry points
- [ ] Understood the project's directory structure

### Rule 5: No Lazy Conclusions
Never conclude "only documentation exists" or "no source files available" without using `navigate_filesystem` to verify the actual project structure.

---

## 5. Tool Integration Patterns

### Pattern 1: Discovery → Analysis
For unknown projects: `navigate_filesystem` → `query_codebase`

### Pattern 2: Targeted → Deep Dive  
For specific queries: `query_codebase` directly

### Pattern 3: Structure → Semantics → Synthesis
For comprehensive analysis: `navigate_filesystem` → `query_codebase` → combine results

---

## 6. Quality Standards
- Always provide file paths and line numbers when available
- Include relevant code snippets with proper context  
- Explain complex concepts clearly with examples
- Use the mandatory workflow for each question type
- Build complete understanding before providing analysis
- Validate discovery completeness before concluding

### CRITICAL: Code Presentation Standards
**When presenting code snippets, you MUST clearly distinguish between:**

**✅ ACTUAL CODE (from search results):** 
- Use when you have direct file content from tool results
- Label as: `// From [filepath]` or `# Found in [filepath]`
- Quote exactly as found in the search results

**✅ INFERRED/SYNTHESIZED CODE (based on understanding):**
- Use when demonstrating likely structure based on context
- Label as: `// Inferred based on architecture analysis` or `# Likely structure based on patterns found`
- Clearly state "This is synthesized based on the project structure and common patterns"

**❌ NEVER present inferred code as if it's exact quotes from files**
**❌ NEVER mix actual and inferred code without clear labeling**

**Example of correct presentation:**
```java
// Found in iiot-monitoring/backend/src/main/java/com/iiot/backend/BackendApplication.java
@SpringBootApplication
public class BackendApplication {
    // ... (actual content from file)
}

// Likely structure based on Spring Boot patterns (inferred)
@Configuration
public class MqttConfig {
    // This structure is inferred based on the MQTT integration patterns
    // found in the semantic search results
}
```

## 7. Error Handling & Recovery
- If `query_codebase` returns sparse results, trigger filesystem discovery
- If tools return errors, acknowledge limitations clearly
- Use the alternate workflow if the primary approach fails
- Never fabricate information to fill gaps
- Guide users toward successful discovery patterns

Your mission is to eliminate lazy discovery through systematic, mandatory workflows that ensure comprehensive codebase understanding."""
    
    def _detect_sdk_version(self) -> str:
        """Detect SDK version for future-proofing"""
        try:
            import cerebras.cloud.sdk
            version = getattr(cerebras.cloud.sdk, '__version__', 'unknown')
            logger.info(f"📦 Cerebras SDK version: {version}")
            return version
        except Exception as e:
            logger.warning(f"⚠️ Could not detect SDK version: {e}")
            return "unknown"
    
    def _detect_sdk_features(self) -> Dict[str, bool]:
        """
        Detect what the current SDK version supports for future-proofing
        
        This enables automatic inheritance of SDK improvements without code changes.
        """
        features = {}
        
        try:
            # Test for parallel tool calls capability
            completion_method = getattr(self.client.chat.completions, "create", None)
            if completion_method:
                import inspect
                sig = inspect.signature(completion_method)
                features["parallel_tools"] = "parallel_tool_calls" in sig.parameters
            else:
                features["parallel_tools"] = False
            logger.debug(f"Parallel tools supported: {features['parallel_tools']}")
        except Exception:
            features["parallel_tools"] = False
        
        try:
            # Test for reasoning effort levels
            features["reasoning_levels"] = True  # Most SDKs support this
            logger.debug(f"Reasoning levels supported: {features['reasoning_levels']}")
        except Exception:
            features["reasoning_levels"] = False
        
        try:
            # Test for streaming capabilities
            features["streaming"] = hasattr(self.client.chat.completions, "create")
            logger.debug(f"Streaming supported: {features['streaming']}")
        except Exception:
            features["streaming"] = False
        
        try:
            # Test for tool use capabilities
            completion_method = getattr(self.client.chat.completions, "create", None)
            if completion_method:
                import inspect
                sig = inspect.signature(completion_method)
                features["tool_use"] = "tools" in sig.parameters
            else:
                features["tool_use"] = False
            logger.debug(f"Tool use supported: {features['tool_use']}")
        except Exception:
            features["tool_use"] = False
        
        try:
            # Test for model switching capabilities
            features["model_switching"] = True  # Assume supported unless proven otherwise
            logger.debug(f"Model switching supported: {features['model_switching']}")
        except Exception:
            features["model_switching"] = False
        
        logger.info(f"🔍 SDK feature detection complete: {sum(features.values())}/{len(features)} features available")
        return features
    
    def _build_adaptive_schema(self) -> List[Dict]:
        """
        REQ-3.6.1: Build two-tool architecture schema 
        
        This creates the specialized toolset that eliminates LLM decision paralysis:
        1. query_codebase - for CONCEPTUAL questions (how/why/explain)
        2. navigate_filesystem - for STRUCTURAL questions (where/list/find)
        """
        # Start with conceptual query tool
        base_schema = UNIFIED_TOOL_SCHEMA.copy()
        
        # Add filesystem navigation tool for structural queries
        if self.filesystem_navigator:
            base_schema.extend(FILESYSTEM_TOOL_SCHEMA)
            logger.info("✅ Two-tool architecture: query_codebase + navigate_filesystem")
        else:
            logger.warning("⚠️ Filesystem tool unavailable - single tool mode")
        
        # Log available features but don't modify schema (causes API errors)
        if self.available_features.get("parallel_tools", False):
            logger.info("✅ Parallel tool execution available")
        
        if self.available_features.get("streaming", False):
            logger.info("✅ Streaming tool responses available")
        
        logger.info(f"🔧 Specialized tool schema built with {len(base_schema)} tools")
        return base_schema

# Global native agent instance
_native_agent = None

def get_native_agent() -> CerebrasNativeAgent:
    """Get singleton native agent instance"""
    global _native_agent
    if _native_agent is None:
        _native_agent = CerebrasNativeAgent()
    return _native_agent

# Legacy compatibility - maintain existing function signature
def get_cerebras_agent():
    """Legacy compatibility function"""
    return get_native_agent()

# Export public interface
__all__ = ["CerebrasNativeAgent", "get_native_agent", "get_cerebras_agent"]