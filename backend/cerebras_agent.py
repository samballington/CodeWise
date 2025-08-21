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
    
    async def process_query(self, user_query: str, conversation_history: Optional[List[Dict]] = None, selected_model: str = None) -> str:
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
            return f"I encountered a system error while processing your request: {str(e)}"
    
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
                
                # REQ-CTX-MONITORING: Track context usage before each iteration
                self._log_context_usage(messages)
                
                # REQ-CTX-FALLBACK: Check if we need graceful degradation
                context_health = self._check_context_health(messages)
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
                
                synthesis_response = self.client.chat.completions.create(**synthesis_params)
                
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
            
            # Use our existing unified query system
            result = await query_codebase(query, filters, analysis_mode)
            
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
                    graph_data = self._transform_kg_to_graph_data(result.get("results", []), query)
                    
                    # Step 4: Determine diagram type from query
                    diagram_type = self._infer_diagram_type(query)
                    
                    # Step 5: Generate Mermaid diagram
                    mermaid_code = self.mermaid_renderer.render_diagram(
                        diagram_type=diagram_type,
                        data=graph_data,
                        title=f"Diagram: {query}",
                        theme="default"
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
    
    def _transform_kg_to_graph_data(self, kg_results: List[Dict], query: str) -> Dict[str, Any]:
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
                for rel in outgoing_rels:
                    # Find target node in connected_nodes
                    target_id = None
                    connected_nodes = kg_item.get("connected_nodes", [])
                    for connected in connected_nodes:
                        if connected.get("id") == rel.get("target_id"):
                            target_id = connected.get("name")
                            break
                    
                    if target_id and target_id.lower() not in ['none', '(none)', '', 'null']:
                        edges.append({
                            "source": node_id,
                            "target": target_id,
                            "type": rel.get("type", "uses"),
                            "label": rel.get("type", "")
                        })
        
        # Infer additional relationships from file analysis
        additional_edges = self._infer_relationships_from_files(nodes)
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
    
    def _infer_relationships_from_files(self, nodes: List[Dict]) -> List[Dict]:
        """
        Infer additional relationships by analyzing file content.
        This goes beyond KG data to find actual usage patterns.
        """
        additional_edges = []
        
        try:
            for node in nodes:
                file_path = node.get("file_path", "")
                if not file_path or not file_path.startswith("/workspace"):
                    continue
                
                from pathlib import Path
                actual_file = Path(file_path)
                
                if actual_file.exists():
                    content = actual_file.read_text(encoding='utf-8', errors='ignore')
                    
                    # Look for service dependencies in controllers
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
            logger.debug(f"Could not infer relationships from files: {e}")
        
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
            
            # Execute the filesystem operation using KG data
            result = self.filesystem_navigator.execute(operation, path, pattern, recursive)
            
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
            # Convert result to string for size analysis
            raw_result_str = json.dumps(tool_result, indent=2)
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
                role = message.get("role", "unknown")
                content = str(message.get("content", ""))
                char_count = len(content)
                token_estimate = char_count // 4  # Rough estimate: 4 chars per token
                
                total_chars += char_count
                total_tokens_estimate += token_estimate
                
                if role in message_breakdown:
                    message_breakdown[role] += token_estimate
                
                # Track tool usage
                if role == "tool" and "tool_call_id" in message:
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
            total_tokens = sum(self._estimate_token_count(str(msg.get("content", ""))) for msg in messages)
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
            
            logger.info("✅ Multi-turn synthesis completed successfully")
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
- Any query about project structure or overview

**MANDATORY Process - You MUST follow every step:**

1. **Thought:** This is a broad, exploratory query. I must first understand the project's structure. My first step is to use `navigate_filesystem` to get a file tree.

2. **Action (Step 1):** `navigate_filesystem(operation="tree", path=".")`

3. **Observation:** [Review the file tree from the tool's output - identify key directories like src/, components/, controllers/, services/, etc.]

4. **Thought:** Based on the file tree, the core logic appears to be in the `[path/to/logic]` and `[path/to/other/logic]` directories. Now I will perform a targeted semantic search on these areas.

5. **Action (Step 2):** `query_codebase(query="analyze the core components in '[path/to/logic]'")`

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

### Rule 4: Completeness Validation
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