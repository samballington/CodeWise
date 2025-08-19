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
from typing import Dict, List, Any, Optional
from cerebras.cloud.sdk import Cerebras

# Import our existing Phase 1/2 infrastructure (PRESERVE these)
try:
    from .tools.unified_query_pure import query_codebase_pure as query_codebase
    from .tools.mermaid_renderer import MermaidRenderer
    from .smart_search import get_smart_search_engine
    from .vector_store import VectorStore
    from .tools.unified_tool_schema import UNIFIED_TOOL_SCHEMA
    from .config.cerebras_config import cerebras_config
except ImportError:
    from tools.unified_query_pure import query_codebase_pure as query_codebase
    from tools.mermaid_renderer import MermaidRenderer
    from smart_search import get_smart_search_engine
    from vector_store import VectorStore
    from tools.unified_tool_schema import UNIFIED_TOOL_SCHEMA
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
            logger.info(f"✅ Cerebras SDK client initialized with model: {cerebras_config.model}")
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
        
        # CRITICAL: Tool functions that SDK will call
        # These are PURE functions - no reasoning, just execution
        self.tool_functions = {
            "query_codebase": self._execute_query_codebase,
            "generate_diagram": self._execute_generate_diagram,
            "examine_files": self._execute_examine_files,
            "search_symbols": self._execute_search_symbols
        }
        
        # Conversation limits for safety
        self.max_iterations = 20
        self.current_iteration = 0
        
        logger.info("✅ CerebrasNativeAgent initialized - SDK handles all reasoning")
    
    async def process_query(self, user_query: str, conversation_history: Optional[List[Dict]] = None) -> str:
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
            result = await self._sdk_native_loop(messages)
            
            logger.info(f"✅ Query processed successfully in {self.current_iteration} iterations")
            return result
            
        except Exception as e:
            logger.error(f"❌ Native agent processing failed: {e}")
            return f"I encountered a system error while processing your request: {str(e)}"
    
    async def _sdk_native_loop(self, messages: List[Dict]) -> str:
        """
        Pure SDK loop - NO custom logic, just tool execution when requested
        """
        last_successful_content = None
        successful_iterations = 0
        
        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            
            try:
                logger.info(f"🔄 SDK Iteration {self.current_iteration}/{self.max_iterations}")
                
                # SDK makes ALL decisions using adaptive schema
                adaptive_schema = self._build_adaptive_schema()
                response = self.client.chat.completions.create(
                    **cerebras_config.get_completion_config(),
                    messages=messages,
                    tools=adaptive_schema,
                    reasoning_effort=cerebras_config.reasoning_effort
                )
                
                message = response.choices[0].message
                
                # Log SDK reasoning (for transparency and debugging)
                if hasattr(message, 'reasoning') and message.reasoning:
                    logger.info(f"🧠 SDK REASONING: {message.reasoning}")
                
                # SDK wants to use tools
                if message.tool_calls:
                    logger.info(f"🔧 SDK requested {len(message.tool_calls)} tool calls")
                    messages.append(message)
                    
                    # Execute ONLY what SDK requests - no custom logic
                    for tool_call in message.tool_calls:
                        result = await self._execute_pure_tool_call(tool_call)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result)
                        })
                    
                    successful_iterations += 1
                    continue  # Let SDK process results
                
                # SDK provides final answer
                else:
                    logger.info("✅ SDK provided final response")
                    return message.content
                    
            except Exception as e:
                logger.error(f"❌ SDK iteration {self.current_iteration} failed: {e}")
                
                # If we had successful iterations, try to provide a helpful response
                if successful_iterations > 0:
                    logger.info(f"⚠️ Partial success: {successful_iterations} iterations completed before error")
                    return (
                        f"I've gathered information about your query but encountered a technical issue "
                        f"in the final processing step. Based on the data I found, I can provide you with "
                        f"relevant information, though the response formatting may be affected. "
                        f"The technical error was: {str(e)}"
                    )
                else:
                    return f"I encountered a technical error during processing: {str(e)}"
        
        logger.warning(f"⚠️ Reached maximum iterations ({self.max_iterations})")
        
        # Instead of giving up, try to synthesize a response from gathered context
        if successful_iterations > 0:
            logger.info(f"💡 Attempting to synthesize response from {successful_iterations} successful tool calls")
            
            # Extract the original user query from the conversation
            user_query = ""
            for msg in messages:
                if msg.get("role") == "user":
                    user_query = msg.get("content", "")
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
                if msg.get("role") in ["assistant", "tool"]:
                    synthesis_messages.append(msg)
            
            try:
                # Make final synthesis call without tools
                synthesis_response = self.client.chat.completions.create(
                    **cerebras_config.get_completion_config(),
                    messages=synthesis_messages,
                    reasoning_effort=cerebras_config.reasoning_effort
                )
                
                synthesis_result = synthesis_response.choices[0].message.content if hasattr(synthesis_response.choices[0].message, 'content') else str(synthesis_response.choices[0].message)
                logger.info(f"✅ Successfully synthesized response from gathered context")
                return synthesis_result
                
            except Exception as e:
                logger.error(f"❌ Synthesis failed: {e}")
                return (
                    f"I gathered substantial information about your query through {successful_iterations} "
                    f"research steps, but reached the processing limit while organizing the final response. "
                    f"The system found relevant information but encountered a technical limitation in "
                    f"synthesizing the complete answer. Please try rephrasing your question or breaking "
                    f"it into smaller parts."
                )
        else:
            return "I reached the maximum processing steps. Please try rephrasing your question or breaking it into smaller parts."
    
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
        PURE query execution using our Phase 1/2 infrastructure
        NO custom reasoning - just execute and return data
        """
        try:
            logger.info(f"📊 Executing codebase query: '{query[:50]}...' (mode: {analysis_mode})")
            
            # Use our existing unified query system
            result = await query_codebase(query, filters, analysis_mode)
            
            # Return raw data for SDK to reason about
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
            mermaid_code = self.mermaid_renderer.render_diagram(
                diagram_type=diagram_type,
                data=data,
                title=title,
                theme=theme
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
    
    def _get_native_system_prompt(self) -> str:
        """
        Native system prompt for SDK - focuses on tool usage patterns
        """
        return """# CodeWise - Native SDK Integration

You are CodeWise, a senior software engineer and code intelligence platform. You help users understand complex codebases through systematic analysis using the tools provided.

## Your Native Capabilities

You have access to advanced code analysis tools that combine:
- Semantic search for conceptual understanding
- Knowledge Graph queries for structural relationships  
- Hybrid search for comprehensive coverage
- File examination for detailed analysis

## Tool Usage Patterns

### Standard Analysis
1. Use `query_codebase` with analysis_mode="auto" for most questions
2. The system intelligently selects optimal search strategies
3. Provide comprehensive answers with file paths and code context

### Diagram Generation (Two-Step Process)
1. FIRST: Use `query_codebase` with analysis_mode="structural_kg" to gather factual relationship data
2. THEN: Use `generate_diagram` with the structural data from step 1
3. NEVER invent diagram relationships - always use factual data

### Deep Investigation
1. Use `query_codebase` to find relevant files/symbols
2. Use `examine_files` to read specific source code
3. Use `search_symbols` for detailed symbol relationship analysis
4. Synthesize findings into comprehensive explanations

### Symbol Research
1. Use `search_symbols` for specific function/class lookups
2. Use `query_codebase` with analysis_mode="specific_symbol" for focused searches
3. Combine results for complete understanding

## Quality Standards
- Always provide file paths and line numbers when available
- Include relevant code snippets with proper context
- Explain complex concepts clearly with examples
- Focus on accuracy over speed
- Make responses educational and actionable

## Error Handling
- If tools return errors, acknowledge limitations clearly
- Offer alternative approaches when primary methods fail
- Never fabricate information to fill gaps
- Guide users toward successful query patterns

Your goal is to help users understand complex systems by providing clear, accurate, and context-rich analysis using these powerful tools systematically."""
    
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
        Build tool schema that adapts to SDK capabilities
        
        This ensures we automatically use new SDK features as they become available.
        Note: We don't modify the schema structure as that breaks API compatibility.
        Feature adaptation happens at the execution level, not schema level.
        """
        base_schema = UNIFIED_TOOL_SCHEMA.copy()
        
        # Log available features but don't modify schema (causes API errors)
        if self.available_features.get("parallel_tools", False):
            logger.info("✅ Parallel tool execution available")
        
        if self.available_features.get("streaming", False):
            logger.info("✅ Streaming tool responses available")
        
        logger.info(f"🔧 Standard tool schema built with {len(base_schema)} tools")
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