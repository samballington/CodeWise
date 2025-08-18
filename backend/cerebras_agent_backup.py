#!/usr/bin/env python3
"""
Native Cerebras Agent Implementation
Follows Cerebras SDK tool calling patterns exactly as documented
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from typing import Dict, Any, List, AsyncGenerator
from cerebras.cloud.sdk import Cerebras
from pathlib import Path
import re
from .directory_filters import (
    get_find_filter_args, get_grep_filter_args, should_include_file,
    filter_file_list, resolve_workspace_path, get_project_from_path
)
from .project_context import (
    get_context_manager, set_project_context, get_current_context,
    filter_files_by_context
)
from .enhanced_project_structure import EnhancedProjectStructure
from .smart_search import smart_search
from .path_resolver import PathResolver
from .response_formatter import ResponseFormatter, StandardizedResponse
from .tools.mermaid_generator import create_mermaid_diagram
from .tools.mermaid_renderer import MermaidRenderer, KGDiagramGenerator
from .tools.unified_query import query_codebase
from .context.code_annotator import CodeLensAnnotator
from .json_prompt_schema import parse_json_prompt
from .json_prompt_postprocess import improve_json_prompt_readability
from .response_consolidator import ResponseConsolidator, ResponseSource
from .discovery_pipeline import DiscoveryPipeline
from .query_context_manager import QueryContextManager
from .table_generator import TableGenerator, StructuredTable, FileReference
from ..storage.database_manager import DatabaseManager

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class CerebrasNativeAgent:
    """Native Cerebras agent with proper tool calling support"""
    
    def __init__(self, api_key: str, mcp_server_url: str):
        self.client = Cerebras(api_key=api_key, timeout=30.0)  # 30 second timeout to prevent hanging
        self.mcp_server_url = mcp_server_url
        self.tools_schema = self._create_tools_schema()
        self.available_functions = self._create_function_mapping()
        self.current_mentioned_projects = None  # Store current project context
        
        # Rate limiting: Optimized for better performance
        # Reduced from 1.1s to 0.5s for faster processing
        self.min_request_interval = 0.5
        self.last_request_time = 0
        
        # Initialize enhanced project structure analyzer
        self.enhanced_structure = EnhancedProjectStructure(self._call_mcp_tool_wrapper)
        
        # Initialize path resolver for Task 3 fix
        self.path_resolver = PathResolver()
        
        # Initialize response formatter for Task 5
        self.response_formatter = ResponseFormatter()
        
        # Initialize discovery pipeline for Task 5 extension
        self.discovery_pipeline = DiscoveryPipeline(self.path_resolver)
        
        # Initialize query context manager for performance optimization
        self.context_manager = QueryContextManager(
            max_concurrent_contexts=10,
            cleanup_interval_seconds=60,
            enable_health_monitoring=True
        )
        
        # Phase 3.2.3: Initialize new components
        try:
            # Initialize Knowledge Graph database manager
            db_path = os.getenv('KG_DATABASE_PATH', '/workspace/.vector_cache/codewise.db')
            self.kg_database = DatabaseManager(db_path)
            
            # Initialize Phase 3 components
            self.mermaid_renderer = MermaidRenderer()
            self.kg_diagram_generator = KGDiagramGenerator(self.kg_database, self.mermaid_renderer.theme_manager)
            self.code_annotator = CodeLensAnnotator(self.kg_database)
            
            logger.info("Phase 3 components initialized: KG diagrams, code annotations")
        except Exception as e:
            logger.warning(f"Phase 3 components initialization failed: {e}")
            self.kg_database = None
            self.mermaid_renderer = MermaidRenderer()  # Fallback renderer
            self.kg_diagram_generator = None
            self.code_annotator = None
        
        # Tool context for result chaining
        self.tool_context = {
            'last_search_results': None,
            'discovered_files': [],
            'query_intent': None,
            'main_files': [],
            'entities_found': []
        }
    
    def _repair_truncated_json(self, truncated_json: str) -> str:
        """Attempt to repair truncated JSON for mermaid diagrams"""
        try:
            # Remove any trailing incomplete text
            json_str = truncated_json.strip()
            
            # Find the last complete object/array structure
            bracket_count = 0
            brace_count = 0
            last_valid_pos = 0
            in_string = False
            escape_next = False
            
            for i, char in enumerate(json_str):
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\' and in_string:
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                    elif char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                    
                    # Check if we have a complete structure
                    if brace_count == 0 and bracket_count == 0 and i > 0:
                        last_valid_pos = i + 1
            
            # If we found a complete structure, use it
            if last_valid_pos > 0:
                repaired = json_str[:last_valid_pos]
                logger.info(f"🔧 JSON repair: truncated from {len(json_str)} to {len(repaired)} chars")
                return repaired
            
            # Otherwise, attempt to close unclosed structures
            repaired = json_str
            
            # Close unclosed strings
            if in_string:
                repaired += '"'
            
            # Close unclosed arrays
            while bracket_count > 0:
                repaired += ']'
                bracket_count -= 1
            
            # Close unclosed objects
            while brace_count > 0:
                repaired += '}'
                brace_count -= 1
            
            logger.info(f"🔧 JSON repair: added {len(repaired) - len(json_str)} closing characters")
            return repaired
            
        except Exception as e:
            logger.error(f"❌ JSON repair failed: {e}")
            return None

    async def _call_mcp_tool_wrapper(self, tool_name: str, params):
        """Wrapper method for MCP tool calls to match the expected signature"""
        if tool_name == "run_command":
            if isinstance(params, str):
                command = params
            else:
                command = params.get("command", params)
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
        elif tool_name == "read_file":
            try:
                with open(params, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {e}"
        else:
            return f"Tool {tool_name} not supported in wrapper"
        
    def _filter_results_by_projects(self, results: List, mentioned_projects: List[str] = None) -> List:
        """Filter search results by mentioned projects using centralized context management"""
        if not mentioned_projects:
            return results
        
        # Use centralized filtering logic
        filtered_results = []
        for result in results:
            # Get file path from result (handle both search results and file paths)
            file_path = result.file_path if hasattr(result, 'file_path') else str(result)
            
            # Use centralized context manager for consistent filtering
            if get_context_manager().is_file_in_current_context(file_path):
                filtered_results.append(result)
        
        return filtered_results
    
    def _detect_negative_result(self, result_text: str, function_name: str) -> bool:
        """Detect if a tool result indicates no findings (Phase 1 fix)"""
        negative_patterns = {
            "smart_search": ["no results found", "not found", "no matches"],
            "examine_files": ["file not found", "does not exist", "error reading"],
            "analyze_relationships": ["not found", "error analyzing", "no references found"]
        }
        
        patterns = negative_patterns.get(function_name, ["not found", "no results"])
        return any(pattern in result_text.lower() for pattern in patterns)
    
    def _get_fallback_guidance(self, function_name: str) -> str:
        """Get fallback strategy guidance for negative results (Phase 1 fix)"""
        fallback_strategies = {
            "smart_search": "Try different search terms or more general queries - smart_search adapts automatically",
            "examine_files": "Verify file paths are correct or try smart_search to find the files first",
            "analyze_relationships": "Ensure target exists or try smart_search to find the correct symbol/file names"
        }
        
        return fallback_strategies.get(function_name, "Try alternative approaches or different search terms")
    
    async def _get_smart_fallback_files(self, query_intent: str = None, query: str = "") -> List[str]:
        """Get intelligent fallback files when search fails"""
        import subprocess
        from pathlib import Path
        
        workspace = Path('/workspace')
        fallback_files = []
        
        try:
            # Intent-based fallbacks
            if query_intent == "ARCHITECTURE" or any(word in query.lower() for word in ['architecture', 'system', 'overview', 'structure']):
                # Look for architectural files
                arch_patterns = ['README*', 'package.json', 'pom.xml', 'setup.py', 'requirements.txt', 'Dockerfile', 'docker-compose*']
                for pattern in arch_patterns:
                    matches = list(workspace.rglob(pattern))
                    fallback_files.extend([str(f) for f in matches[:2]])  # Limit per pattern
                
                # Look for main entry points
                entry_patterns = ['main.py', 'app.py', 'index.js', 'server.js', 'Application.java', 'manage.py']
                for pattern in entry_patterns:
                    matches = list(workspace.rglob(pattern))
                    fallback_files.extend([str(f) for f in matches[:1]])
            
            elif query_intent == "ENTITY" or any(word in query.lower() for word in ['entity', 'model', 'database', 'schema']):
                # Look for entity-like files
                try:
                    cmd = ["find", "/workspace", "-name", "*model*", "-o", "-name", "*entity*", "-o", "-name", "*Entity*", "-type", "f"]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
                        fallback_files.extend(files[:5])
                except Exception:
                    pass
            
            else:
                # General fallbacks - look for common important files
                general_patterns = ['README*', 'main.*', 'app.*', 'index.*', 'package.json']
                for pattern in general_patterns:
                    matches = list(workspace.rglob(pattern))
                    fallback_files.extend([str(f) for f in matches[:2]])
            
            # Remove duplicates and apply project filtering
            unique_files = []
            for f in fallback_files:
                if f not in unique_files:
                    if self.current_mentioned_projects:
                        for project in self.current_mentioned_projects:
                            if f"/{project}/" in f or f.endswith(f"/{project}"):
                                unique_files.append(f)
                                break
                    else:
                        unique_files.append(f)
            
            # Limit to reasonable number
            return unique_files[:8]
            
        except Exception as e:
            logger.debug(f"Error in smart fallback discovery: {e}")
            return []
    
    def _extract_files_from_search_results(self, search_result: str) -> List[str]:
        """Extract file paths from smart_search results"""
        import re
        
        files = []
        
        # Look for FILE: patterns
        file_matches = re.findall(r'FILE:\s*([^\n]+)', search_result)
        files.extend(file_matches)
        
        # Look for file paths in results
        path_matches = re.findall(r'([a-zA-Z0-9_./\\-]+\.[a-zA-Z]{2,4})', search_result)
        files.extend(path_matches)
        
        # Clean and deduplicate
        cleaned_files = []
        for f in files:
            f = f.strip()
            if f and f not in cleaned_files and not f.startswith('Error'):
                cleaned_files.append(f)
        
        return cleaned_files[:10]  # Limit to 10 files
    
    def _extract_query_intent_from_search(self, search_result: str) -> str:
        """Extract query intent from smart_search results"""
        import re
        
        intent_match = re.search(r'Query Intent:\s*(\w+)', search_result)
        if intent_match:
            return intent_match.group(1)
        return "GENERAL"
    
    def _extract_main_target_from_examine_results(self, examine_result: str) -> str:
        """Extract main target for relationship analysis from examine_files results"""
        import re
        
        # Look for the first successfully examined file
        file_match = re.search(r'FILE:\s*([^\n]+)', examine_result)
        if file_match:
            return file_match.group(1).strip()
        
        # Look for class or function names that could be analyzed
        class_matches = re.findall(r'class\s+(\w+)', examine_result)
        if class_matches:
            return class_matches[0]
        
        function_matches = re.findall(r'def\s+(\w+)', examine_result)
        if function_matches:
            return function_matches[0]
        
        return "main"  # Default fallback
    
    def _detect_incomplete_response(self, response_text: str) -> bool:
        """Detect if response indicates intention to do more work (Phase 2 fix)"""
        continuation_signals = [
            "let me try", "i will", "let me search", "i'll examine", 
            "let me check", "i'll look", "let me find", "i will search",
            "let me explore", "i'll investigate", "let me analyze"
        ]
        return any(signal in response_text.lower() for signal in continuation_signals)
    
    def _detect_inadequate_response(self, response_text: str, tool_call_count: int) -> bool:
        """Detect if GPT-OSS-120B provided inadequate response after calling tools (Two-Stage Fix)"""
        if tool_call_count == 0:
            return False  # No tools called, response is expected to be brief
        
        # More conservative inadequate response detection
        response_lower = response_text.lower().strip()
        
        # Only trigger for very short responses that are clearly inadequate
        if len(response_text.strip()) < 30 and tool_call_count > 2:
            # Check if it's just a generic completion message
            inadequate_patterns = [
                "task completed",
                "done",
                "completed", 
                "finished"
            ]
            if any(pattern in response_lower for pattern in inadequate_patterns):
                return True
            
        # Don't trigger synthesis for Mermaid generation or intermediate responses
        if "mermaid" in response_lower or "diagram" in response_lower:
            return False
            
        return False
    
    def _compile_tool_results_summary(self, tool_results: List[Dict]) -> str:
        """Compile tool results into a comprehensive summary for synthesis stage"""
        if not tool_results:
            return "No tool results available."
        
        summary_parts = []
        
        for i, result in enumerate(tool_results, 1):
            tool_name = result.get('tool_name', 'unknown_tool')
            result_content = result.get('result', '')
            
            # Truncate very long results but preserve key information
            if len(result_content) > 2000:
                result_content = result_content[:1800] + "...\n[Content truncated for synthesis]"
            
            summary_parts.append(f"**Tool {i}: {tool_name.upper()}**\n{result_content}")
        
        return "\n\n".join(summary_parts)
    
    async def _enforce_rate_limit(self):
        """Enforce rate limiting to prevent 429 errors - 1 request per second max"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last_request
            logger.info(f"🕒 RATE LIMITING: Waiting {wait_time:.1f}s before next API request")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def _calculate_investigation_completeness(self, tool_calls_made: List[str], messages: List[Dict]) -> float:
        """Calculate investigation quality score to prevent premature termination (Updated for 3-tool system)"""
        score = 0.0
        
        # Extract function names from tool call signatures
        tool_types = set()
        for call in tool_calls_made:
            if ':' in call:
                func_name = call.split(':')[0]
                tool_types.add(func_name)
        
        # Points for tool diversity (encourages using different approaches)
        score += len(tool_types) * 2.0  # Increased weight for tool diversity
        
        # Check recent results for negative outcomes
        recent_results = []
        for msg in messages[-10:]:  # Check last 10 messages
            if msg.get('role') == 'tool':
                recent_results.append(msg.get('content', ''))
        
        # Deduct for negative results without alternatives
        negative_results = sum(1 for result in recent_results if "not found" in result.lower() or "no matches" in result.lower())
        if negative_results > 0 and len(tool_types) < 2:
            score -= 1  # Reduced penalty
        
        # Bonus for comprehensive investigation patterns (updated for 3-tool system)
        if "smart_search" in tool_types and "examine_files" in tool_types:
            score += 2
        
        # Bonus for complete workflow
        if "smart_search" in tool_types and "examine_files" in tool_types and "analyze_relationships" in tool_types:
            score += 3
        
        return score
    
    def _create_tools_schema(self) -> List[Dict[str, Any]]:
        """Create enhanced 7-tool architecture schema with Phase 2.3 KG tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "smart_search",
                    "description": "Intelligent unified search combining vector search, keyword search, and entity discovery. Automatically detects query intent (entity, file, general) and routes to optimal search strategies. Use this as your primary search tool.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query - can be about entities, files, code functionality, or general questions"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 10)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "examine_files",
                    "description": "Flexible file inspection with multiple detail levels. Automatically uses files from previous smart_search if no paths provided. Can show file summaries, full content, or structural analysis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_paths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of file paths to examine. Optional - will use smart fallbacks from previous search if not provided."
                            },
                            "detail_level": {
                                "type": "string",
                                "enum": ["summary", "full", "structure"],
                                "description": "Level of detail: 'summary' (key sections), 'full' (complete content), 'structure' (outline only)"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_relationships",
                    "description": "Analyze code relationships and dependencies using AST-based analysis. Automatically uses targets from previous examine_files if no target provided. Find imports, references, related files, and predict impact of changes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {
                                "type": "string",
                                "description": "Target file, class, or function to analyze relationships for. Optional - will use smart fallbacks from previous tools if not provided."
                            },
                            "analysis_type": {
                                "type": "string",
                                "enum": ["imports", "usage", "impact", "all"],
                                "description": "Type of analysis: 'imports' (what it uses), 'usage' (what uses it), 'impact' (change prediction), 'all' (comprehensive)"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mermaid_generator",
                    "description": "Generate professional Mermaid.js diagrams from structured data. Converts semantic diagram descriptions into syntactically correct, styled Mermaid code using predefined templates.",
                    "parameters": {
                        "type": "object", 
                        "properties": {
                            "diagram_type": {
                                "type": "string",
                                "enum": ["Full-Stack Application", "API / Microservice Interaction", "Database Schema", "Internal Component Flow", "CI/CD Pipeline", "General System Architecture"],
                                "description": "The type of diagram template to use for styling and structure"
                            },
                            "nodes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string", "description": "Unique alphanumeric identifier"},
                                        "label": {"type": "string", "description": "Display text (will be auto-escaped)"},
                                        "shape": {"type": "string", "enum": ["square", "round-edge", "database", "rhombus", "circle", "hexagon"], "description": "Mermaid node shape"},
                                        "group": {"type": "string", "description": "Optional subgraph name"}
                                    },
                                    "required": ["id", "label"]
                                },
                                "description": "List of all diagram nodes/components"
                            },
                            "edges": {
                                "type": "array", 
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "from": {"type": "string", "description": "Source node ID"},
                                        "to": {"type": "string", "description": "Target node ID"}, 
                                        "label": {"type": "string", "description": "Optional edge label"}
                                    },
                                    "required": ["from", "to"]
                                },
                                "description": "List of connections between nodes"
                            }
                        },
                        "required": ["diagram_type", "nodes", "edges"]
                    }
                }
            },
            # ==================== PHASE 2.3: KNOWLEDGE GRAPH TOOLS ====================
            {
                "type": "function",
                "function": {
                    "name": "kg_find_symbol",
                    "description": "Find symbols in the Knowledge Graph by name. Returns detailed symbol information with location, type, and metadata. Use this to locate specific functions, classes, or variables across the codebase.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol_name": {
                                "type": "string",
                                "description": "Name of the symbol to find (function, class, variable)"
                            },
                            "exact_match": {
                                "type": "boolean",
                                "description": "Whether to use exact name matching (default: true)"
                            }
                        },
                        "required": ["symbol_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "kg_explore_neighborhood",
                    "description": "Explore the complete 'neighborhood' of relationships around a symbol. Returns comprehensive relationship analysis including callers, dependencies, inheritance, and file context.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol_name": {
                                "type": "string",
                                "description": "Symbol to explore relationships for"
                            },
                            "max_depth": {
                                "type": "integer",
                                "description": "Maximum relationship depth to explore (default: 3)"
                            }
                        },
                        "required": ["symbol_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "kg_find_callers",
                    "description": "Find all symbols that call the target symbol through the Knowledge Graph. Returns accurate caller information with relationship context.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol_name": {
                                "type": "string",
                                "description": "Symbol to find callers for"
                            },
                            "max_depth": {
                                "type": "integer",
                                "description": "Maximum caller depth to explore (default: 3)"
                            }
                        },
                        "required": ["symbol_name"]
                    }
                }
            }
        ]
    
    def _create_function_mapping(self) -> Dict[str, callable]:
        """Map function names to enhanced 7-tool implementations"""
        return {
            "smart_search": self._smart_search_router,  # Context-aware router
            "examine_files": self._examine_files,
            "analyze_relationships": self._analyze_relationships,
            "mermaid_generator": self._mermaid_generator_wrapper,
            # Phase 3.3.2: Unified query tool
            "query_codebase": self._query_codebase_wrapper,
            # Phase 2.3 KG tools (legacy - replaced by query_codebase)
            "kg_find_symbol": self._kg_find_symbol,
            "kg_explore_neighborhood": self._kg_explore_neighborhood,
            "kg_find_callers": self._kg_find_callers
        }
    
    async def _mermaid_generator_wrapper(self, diagram_type: str, nodes: List[Dict], edges: List[Dict]) -> str:
        """
        Phase 3.2.3: Enhanced mermaid generator with KG-powered diagram capabilities.
        
        Intelligently routes between KG-powered factual diagrams and traditional 
        LLM-based diagrams based on available data and diagram type.
        """
        try:
            # Phase 3.2.3: Try KG-powered diagram generation first for better accuracy
            if hasattr(self, 'kg_diagram_generator') and self.kg_diagram_generator:
                try:
                    # KG-powered diagram generation for supported types
                    if diagram_type == 'classDiagram' and 'class_name' in str(nodes):
                        # Extract class name for KG lookup
                        class_name = self._extract_class_name_from_nodes(nodes)
                        if class_name:
                            logger.info(f"🎯 Using KG-powered class diagram generation for: {class_name}")
                            kg_result = self.kg_diagram_generator.generate_class_hierarchy(class_name)
                            if kg_result and not kg_result.startswith('graph TD\n    error'):
                                return self._format_mermaid_response(kg_result, diagram_type, "KG-powered", len(nodes), len(edges))
                    
                    elif diagram_type.startswith('graph') and 'function' in str(nodes):
                        # Try call flow diagram
                        function_name = self._extract_function_name_from_nodes(nodes)
                        if function_name:
                            logger.info(f"🎯 Using KG-powered call flow generation for: {function_name}")
                            kg_result = self.kg_diagram_generator.generate_call_flow(function_name, max_depth=3)
                            if kg_result and not kg_result.startswith('graph TD\n    error'):
                                return self._format_mermaid_response(kg_result, diagram_type, "KG-powered", len(nodes), len(edges))
                    
                    elif 'module' in diagram_type.lower() or 'dependency' in diagram_type.lower():
                        # Try module dependency diagram
                        module_path = self._extract_module_path_from_nodes(nodes)
                        if module_path:
                            logger.info(f"🎯 Using KG-powered module dependency generation for: {module_path}")
                            kg_result = self.kg_diagram_generator.generate_module_dependencies(module_path)
                            if kg_result and not kg_result.startswith('graph TD\n    error'):
                                return self._format_mermaid_response(kg_result, diagram_type, "KG-powered", len(nodes), len(edges))
                    
                except Exception as e:
                    logger.warning(f"KG-powered diagram generation failed, falling back to traditional: {e}")
            
            # Fallback to traditional LLM-based diagram generation
            logger.info(f"🎨 Using traditional diagram generation for {diagram_type} with {len(nodes)} nodes and {len(edges)} edges")
            
            # Validate input for traditional generation
            if not nodes:
                return "❌ **Error**: No nodes provided for diagram generation"
               
            if not edges:
                return "❌ **Error**: No edges provided for diagram generation"
            
            # Use Phase 3.2.1 Pure Rendering approach
            if hasattr(self, 'mermaid_renderer') and self.mermaid_renderer:
                # Use new pure renderer
                graph_data = {'nodes': nodes, 'edges': edges}
                result = self.mermaid_renderer.generate(diagram_type, graph_data)
                return self._format_mermaid_response(result, diagram_type, "Pure Renderer", len(nodes), len(edges))
            else:
                # Legacy fallback
                diagram_data = {
                    "diagram_type": diagram_type,
                    "nodes": nodes,
                    "edges": edges
                }
                result = create_mermaid_diagram(diagram_data)
                return self._format_mermaid_response(result, diagram_type, "Legacy", len(nodes), len(edges))
            
        except ValueError as ve:
            logger.error(f"Mermaid validation error: {ve}")
            return f"❌ **Mermaid Validation Error:** {str(ve)}\n\nPlease check your input data format and try again."
        except Exception as e:
            logger.error(f"Mermaid generator error: {e}")
            return f"❌ **Mermaid Generation Failed:** {str(e)}\n\nThe diagram could not be generated. Please verify your input data."

    def _format_mermaid_response(self, mermaid_content: str, diagram_type: str, generation_method: str, node_count: int, edge_count: int) -> str:
        """
        Phase 3.2.3: Standardized mermaid response formatting with generation method tracking.
        """
        try:
            # DEBUG: Log raw mermaid output (Step 6 diagnosis)
            logger.info(f"🔍 MERMAID DEBUG STEP 6 - {generation_method} output: {mermaid_content[:200]}...")
            if '--&gt;' in mermaid_content or '--&amp;gt;' in mermaid_content or '--@gt' in mermaid_content:
                logger.error(f"🚨 CORRUPTION DETECTED IN {generation_method} OUTPUT: {mermaid_content}")
            
            # Format for structured response
            response = f"✅ **Mermaid Diagram Generated Successfully** ({generation_method})\n\n"
            response += f"**📊 Diagram Details:**\n"
            response += f"- **Type:** {diagram_type}\n"
            response += f"- **Method:** {generation_method}\n"
            response += f"- **Nodes:** {node_count} components\n"
            response += f"- **Connections:** {edge_count} relationships\n"
            response += f"- **Diagram Length:** {len(mermaid_content)} characters\n\n"
            response += "```mermaid\n"
            response += mermaid_content
            response += "\n```"
            
            # DEBUG: Log final tool response before sending to LLM (Step 7 diagnosis)
            logger.info(f"🔍 MERMAID DEBUG STEP 7 - Tool response to LLM: {response[:300]}...")
            if '--&gt;' in response or '--&amp;gt;' in response or '--@gt' in response:
                logger.error(f"🚨 CORRUPTION DETECTED IN TOOL RESPONSE: {response}")
            
            logger.info(f"🎨 Successfully generated {generation_method} Mermaid diagram ({len(mermaid_content)} chars)")
            return response
            
        except Exception as e:
            logger.error(f"Mermaid response formatting error: {e}")
            return f"❌ **Diagram Formatting Error:** {str(e)}"

    def _extract_class_name_from_nodes(self, nodes: List[Dict]) -> str:
        """Extract class name from node data for KG lookup."""
        try:
            for node in nodes:
                if isinstance(node, dict):
                    # Look for class name in various possible fields
                    for field in ['id', 'name', 'label', 'class_name']:
                        if field in node and node[field]:
                            value = str(node[field])
                            # Check if it looks like a class name (starts with capital)
                            if value and value[0].isupper():
                                return value
            return None
        except Exception as e:
            logger.debug(f"Class name extraction failed: {e}")
            return None

    def _extract_function_name_from_nodes(self, nodes: List[Dict]) -> str:
        """Extract function name from node data for KG lookup."""
        try:
            for node in nodes:
                if isinstance(node, dict):
                    # Look for function indicators
                    for field in ['id', 'name', 'label', 'function_name']:
                        if field in node and node[field]:
                            value = str(node[field])
                            # Check if it looks like a function name
                            if value and ('(' in value or field == 'function_name'):
                                return value.split('(')[0]  # Remove parentheses if present
                            elif value and not value[0].isupper():  # lowercase suggests function
                                return value
            return None
        except Exception as e:
            logger.debug(f"Function name extraction failed: {e}")
            return None

    def _extract_module_path_from_nodes(self, nodes: List[Dict]) -> str:
        """Extract module path from node data for KG lookup."""
        try:
            for node in nodes:
                if isinstance(node, dict):
                    # Look for file path or module indicators
                    for field in ['id', 'name', 'label', 'file_path', 'module', 'path']:
                        if field in node and node[field]:
                            value = str(node[field])
                            # Check if it looks like a file path
                            if '/' in value or '\\' in value or '.' in value:
                                return value
            return None
        except Exception as e:
            logger.debug(f"Module path extraction failed: {e}")
            return None
    
    async def _query_codebase_wrapper(self, query: str, filters: dict = None, analysis_mode: str = 'auto') -> str:
        """
        Phase 3.3.2: Unified query tool wrapper that eliminates tool choice complexity.
        
        This single tool replaces the need to choose between smart_search, kg_find_symbol,
        kg_explore_neighborhood, and kg_find_callers by intelligently routing queries
        to the optimal strategy.
        
        Args:
            query: Natural language query about the codebase
            filters: Optional filters (file_type, directory, symbol_type, etc.)
            analysis_mode: 'auto', 'structural_kg', 'semantic_rag', 'specific_symbol'
            
        Returns:
            Formatted results with strategy information
        """
        try:
            logger.info(f"🔍 UNIFIED QUERY: '{query}' (mode: {analysis_mode})")
            
            # Use the unified query system
            result = await query_codebase(query, filters, analysis_mode)
            
            # Format response for the agent
            response_parts = []
            response_parts.append(f"✅ **Query Results** ({result['strategy']})")
            response_parts.append(f"**Query Analysis:** {result.get('query_analysis', {}).get('intent', 'unknown')} (confidence: {result.get('query_analysis', {}).get('confidence', 0.0):.2f})")
            response_parts.append(f"**Results Found:** {result.get('total_results', 0)}")
            
            if result.get('error'):
                response_parts.append(f"**Error:** {result['error']}")
                return '\n'.join(response_parts)
            
            # Add results
            results = result.get('results', [])
            if results:
                response_parts.append("\n**📋 Results:**")
                for i, item in enumerate(results[:10], 1):  # Limit to top 10
                    if isinstance(item, dict):
                        # Handle different result formats
                        name = item.get('name', item.get('file_path', 'Unknown'))
                        file_path = item.get('file_path', '')
                        item_type = item.get('type', 'unknown')
                        
                        if 'snippet' in item:
                            # Smart search result format
                            snippet = item['snippet'][:200] + '...' if len(item['snippet']) > 200 else item['snippet']
                            response_parts.append(f"   {i}. **{name}** ({item_type})")
                            response_parts.append(f"      📁 {file_path}")
                            response_parts.append(f"      📝 {snippet}")
                        else:
                            # KG result format
                            line_info = f":{item.get('line_start', '')}" if item.get('line_start') else ""
                            response_parts.append(f"   {i}. **{name}** ({item_type})")
                            response_parts.append(f"      📁 {file_path}{line_info}")
                            
                            if 'relationship' in item:
                                response_parts.append(f"      🔗 {item['relationship']}")
                    else:
                        # Handle other formats
                        response_parts.append(f"   {i}. {str(item)}")
                    
                    response_parts.append("")  # Add spacing
            else:
                response_parts.append("\n❌ No results found for this query.")
            
            # Add metadata
            if result.get('execution_time'):
                response_parts.append(f"⏱️ **Execution Time:** {result['execution_time']:.2f}s")
            
            # Add strategy explanation
            strategy_explanations = {
                'kg_direct_find_callers': '🔍 Used Knowledge Graph to find function callers',
                'kg_direct_explore_neighborhood': '🕸️ Used Knowledge Graph to explore symbol relationships',
                'kg_direct_find_symbol': '🎯 Used Knowledge Graph for direct symbol lookup',
                'symbol_expand': '🔄 Found symbol and expanded with neighborhood context',
                'hybrid_intelligent': '🧠 Used intelligent hybrid search with Phase 3 classification',
                'hybrid_legacy': '🔧 Used legacy hybrid search as fallback',
                'error': '❌ Query processing encountered an error'
            }
            
            strategy_explanation = strategy_explanations.get(result['strategy'], f"Used {result['strategy']} strategy")
            response_parts.append(f"\n💡 **Strategy:** {strategy_explanation}")
            
            return '\n'.join(response_parts)
            
        except Exception as e:
            logger.error(f"Query codebase wrapper failed: {e}")
            return f"❌ **Query Failed:** {str(e)}\n\nPlease check your query and try again."
    
    async def _smart_search_router(self, query: str, max_results: int = 10) -> str:
        """Route to context-aware or legacy smart search based on availability"""
        if hasattr(self, 'current_query_context') and self.current_query_context is not None:
            # Use context-aware version (THE PERFORMANCE FIX)
            return await self._smart_search_context_aware(query, self.current_query_context, max_results)
        else:
            # Fallback to legacy version
            logger.warning("⚠️ Using legacy smart search - context not available")
            return await self._smart_search_legacy(query, max_results)
    
    async def _smart_search_context_aware(self, query: str, query_context, max_results: int = 10) -> str:
        """Context-aware intelligent unified search (PERFORMANCE OPTIMIZED)"""
        try:
            logger.info(f"🧠 CONTEXT-AWARE SMART SEARCH: '{query}' (context: {query_context.query_id[:8]}, max_results={max_results})")
            
            # Use the context-aware smart search engine (THE KEY FIX)
            from backend.smart_search import SmartSearchEngine
            search_engine = SmartSearchEngine()  # Get the search engine instance
            search_result = await search_engine.search_with_context(
                query=query, 
                context=query_context,
                k=max_results, 
                mentioned_projects=self.current_mentioned_projects
            )
            
            results = search_result['results']
            query_analysis = search_result['query_analysis']
            strategies_used = search_result['search_strategies_used']
            
            # Store context for tool chaining
            self.tool_context['last_search_results'] = search_result
            self.tool_context['query_intent'] = query_analysis['intent'].value
            
            # Apply project filtering if mentioned projects are specified (same as legacy)
            if self.current_mentioned_projects:
                filtered_results = []
                for result in results:
                    file_path = result.file_path
                    if not file_path.startswith('/workspace/'):
                        file_path = f"/workspace/{file_path.lstrip('/')}"
                    
                    if get_context_manager().is_file_in_current_context(file_path):
                        filtered_results.append(result)
                results = filtered_results
            
            # Extract discovered files for tool chaining
            discovered_files = [result.file_path for result in results]
            self.tool_context['discovered_files'] = discovered_files
            query_context.discovered_files.extend(discovered_files)  # Update context
            
            # Handle auto-examination recommendations from discovery pipeline
            auto_examine_files = search_result.get('auto_examine_files', [])
            if auto_examine_files:
                logger.info(f"🔍 AUTO-EXAMINATION: Discovery pipeline recommends examining {len(auto_examine_files)} files")
                logger.info(f"🔍 AUTO-EXAMINATION: Files: {auto_examine_files}")
                self.tool_context['auto_examine_files'] = auto_examine_files
            else:
                auto_examine_files = []
            
            # Rest of the method logic (same as legacy)...
            if not results:
                filter_msg = f" (filtered for projects: {self.current_mentioned_projects})" if self.current_mentioned_projects else ""
                intent_for_fallback = query_analysis['intent'].value
                
                fallback_files = await self._get_smart_fallback_files(intent_for_fallback, query)
                if fallback_files:
                    fallback_msg = f"\n\n🎯 **Smart Fallback Guidance:**\n"
                    for i, file_path in enumerate(fallback_files[:3], 1):
                        fallback_msg += f"{i}. `{file_path}`\n"
                    fallback_msg += "\nConsider examining these files that might contain relevant information."
                else:
                    fallback_msg = ""
                
                return f"❌ **No results found**{filter_msg} for query: '{query}'\n\n📊 **Search Analysis:**\n- Intent: {intent_for_fallback.title()}\n- Strategies: {', '.join(strategies_used)}\n- Results: 0 matches{fallback_msg}"
            
            # Create structured response with both markdown and JSON table
            markdown_summary = []
            markdown_summary.append(f"🧠 **SMART SEARCH RESULTS** ({len(results)} matches)")
            markdown_summary.append(f"📊 **Query Analysis:** Intent={query_analysis['intent'].value}, Confidence={query_analysis['confidence']:.2f}")
            markdown_summary.append(f"🎯 **Strategies Used:** {', '.join(strategies_used)}")
            
            if auto_examine_files:
                markdown_summary.append(f"🔍 **Discovery Pipeline:** Found {len(auto_examine_files)} files for auto-examination")
            
            execution_time = search_result.get('execution_time', 0)
            markdown_summary.append(f"⚡ **Execution time:** {execution_time:.2f}s")
            
            # Add auto-examine suggestion if files were found
            if auto_examine_files and len(auto_examine_files) > 0:
                markdown_summary.append(f"\n💡 **Auto-Examine Suggestion:** The discovery pipeline found {len(auto_examine_files)} related files. Consider examining them for deeper insights.")
            
            # Create structured table for search results
            search_table = TableGenerator.create_search_results_table(
                [
                    {
                        'file_path': result.file_path.replace('/workspace/', ''),
                        'relevance_score': result.relevance_score,
                        'snippet': result.snippet,
                        'matched_terms': getattr(result, 'matched_terms', []),
                        'search_strategy': getattr(result, 'search_strategy', ['hybrid'])
                    } 
                    for result in results[:max_results]
                ],
                title="Search Results"
            )
            
            # Create file references for JSON
            file_refs = [
                FileReference(
                    path=result.file_path.replace('/workspace/', ''),
                    line_start=getattr(result, 'line_start', None),
                    line_end=getattr(result, 'line_end', None)
                )
                for result in results[:max_results]
            ]
            
            # Combine markdown and structured data
            markdown_content = "\n".join(markdown_summary)
            result_text = TableGenerator.wrap_response_with_structured_data(
                markdown_content,
                tables=[search_table],
                references=file_refs
            )
            logger.info(f"✅ CONTEXT-AWARE SMART SEARCH COMPLETE: {len(results)} results, {execution_time:.2f}s")
            return result_text
            
        except Exception as e:
            logger.error(f"❌ CONTEXT-AWARE SMART SEARCH ERROR: {e}")
            return f"❌ **Search Error:** {str(e)}\n\nPlease try a different query or check the system logs for more details."
    
    async def _smart_search_legacy(self, query: str, max_results: int = 10) -> str:
        """Intelligent unified search combining vector, BM25, and entity discovery"""
        try:
            logger.error(f"🚀 SMART_SEARCH START CHECKPOINT: '{query}' (max_results={max_results})")
            logger.info(f"🧠 SMART SEARCH: '{query}' (max_results={max_results})")
            
            # Use the smart search engine (with project filtering)
            search_result = await smart_search(query, k=max_results, mentioned_projects=self.current_mentioned_projects)
            
            results = search_result['results']
            query_analysis = search_result['query_analysis']
            strategies_used = search_result['search_strategies_used']
            
            # Store context for tool chaining
            self.tool_context['last_search_results'] = search_result
            self.tool_context['query_intent'] = query_analysis['intent'].value
            
            # Apply project filtering if mentioned projects are specified
            if self.current_mentioned_projects:
                filtered_results = []
                for result in results:
                    # Normalize file path to full workspace path for filtering
                    file_path = result.file_path
                    if not file_path.startswith('/workspace/'):
                        file_path = f"/workspace/{file_path.lstrip('/')}"
                    
                    if get_context_manager().is_file_in_current_context(file_path):
                        filtered_results.append(result)
                results = filtered_results
            
            # Extract discovered files for tool chaining
            discovered_files = [result.file_path for result in results]
            self.tool_context['discovered_files'] = discovered_files
            
            # NEW: Handle auto-examination recommendations from discovery pipeline
            auto_examine_files = search_result.get('auto_examine_files', [])
            if auto_examine_files:
                logger.info(f"🔍 AUTO-EXAMINATION: Discovery pipeline recommends examining {len(auto_examine_files)} files")
                logger.info(f"🔍 AUTO-EXAMINATION: Files: {auto_examine_files}")
                # Store for potential auto-examination
                self.tool_context['auto_examine_files'] = auto_examine_files
            else:
                # Ensure we have an empty list if no files found
                auto_examine_files = []
            
            if not results:
                filter_msg = f" (filtered for projects: {self.current_mentioned_projects})" if self.current_mentioned_projects else ""
                
                # Store intent even when no results found for fallback
                intent_for_fallback = query_analysis['intent'].value
                self.tool_context['query_intent'] = intent_for_fallback
                
                return f"No results found for query '{query}'{filter_msg}.\n\nQuery Analysis:\n- Intent: {intent_for_fallback}\n- Confidence: {query_analysis['confidence']:.2f}\n- Strategies tried: {', '.join(strategies_used)}\n\n💡 TIP: Other tools can use smart fallbacks based on the detected intent."
            
            # Create structured table for search results
            search_table = TableGenerator.create_search_results_table([
                {
                    'file_path': result.file_path.replace('/workspace/', ''),
                    'relevance_score': result.confidence,
                    'snippet': result.snippet,
                    'matched_terms': getattr(result, 'matched_terms', []),
                    'search_strategy': result.search_strategy
                } 
                for result in results[:max_results]
            ], title="Smart Search Results")
            
            # Build markdown response with metadata
            markdown_parts = []
            markdown_parts.append(f"🧠 **SMART SEARCH RESULTS** ({len(results)} found)")
            markdown_parts.append(f"📊 **Query Analysis:**")
            markdown_parts.append(f"   • Intent: {query_analysis['intent'].value} (confidence: {query_analysis['confidence']:.2f})")
            markdown_parts.append(f"   • Strategies: {', '.join(strategies_used)}")
            markdown_parts.append(f"   • Execution Time: {search_result.get('execution_time', 0):.2f}s")
            markdown_parts.append("")
            
            # Store main files for potential relationship analysis
            main_files = []
            for result in results:
                # Collect high-confidence files as potential main files
                if result.confidence > 0.7:
                    main_files.append(result.file_path)
            
            self.tool_context['main_files'] = main_files[:3]  # Keep top 3
            
            # Add project filter info if applicable
            if self.current_mentioned_projects:
                markdown_parts.append(f"[Filtered for projects: {', '.join(self.current_mentioned_projects)}]")
            
            # Add tool chaining hint
            markdown_parts.append(f"💡 **DISCOVERED:** {len(discovered_files)} files ready for examination")
            
            # NEW: Auto-examination integration for Task 5 Extension
            logger.info(f"🔍 AUTO-EXAMINATION DEBUG: auto_examine_files = {auto_examine_files}")
            logger.info(f"🔍 AUTO-EXAMINATION DEBUG: len(auto_examine_files) = {len(auto_examine_files) if auto_examine_files else 0}")
            
            if auto_examine_files and len(auto_examine_files) > 0:
                logger.info(f"🔍 AUTO-EXAMINATION: CONDITION MET - Starting examination of {len(auto_examine_files)} files")
                markdown_parts.append("")
                markdown_parts.append(f"🔍 **DISCOVERY PIPELINE:** Found {len(auto_examine_files)} files for auto-examination")
                markdown_parts.append(f"   📁 Auto-examine files: {', '.join(auto_examine_files)}")
                
                # Automatically examine the discovered files
                try:
                    logger.info(f"🔍 AUTO-EXAMINATION: Starting examination of {len(auto_examine_files)} files")
                    auto_exam_result = await self._examine_files(auto_examine_files, "summary")
                    
                    # Add the auto-examination results to the response
                    markdown_parts.append("")
                    markdown_parts.append("---")
                    markdown_parts.append("🔍 **AUTO-EXAMINATION RESULTS:**")
                    markdown_parts.append("---")
                    markdown_parts.append(auto_exam_result)
                    
                    logger.info("✅ AUTO-EXAMINATION: Completed successfully")
                    
                except Exception as auto_exam_error:
                    logger.error(f"❌ AUTO-EXAMINATION: Failed: {auto_exam_error}")
                    markdown_parts.append(f"\n⚠️ Auto-examination failed: {auto_exam_error}")
            else:
                logger.info(f"🔍 AUTO-EXAMINATION: CONDITION NOT MET - No files to examine")
            
            # Use TableGenerator to wrap response with structured data
            markdown_response = "\n".join(markdown_parts)
            return TableGenerator.wrap_response_with_structured_data(markdown_response, [search_table])
            
        except Exception as e:
            logger.error(f"Smart search error: {e}")
            return f"Error during smart search: {str(e)}"
    

    
    def _apply_syntax_highlighting(self, content: str, file_extension: str) -> str:
        """Apply basic syntax highlighting using markdown code blocks"""
        
        # Map file extensions to language identifiers for markdown code blocks
        language_map = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.jsx': 'jsx',
            '.tsx': 'tsx',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.kt': 'kotlin',
            '.swift': 'swift',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.sass': 'sass',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.ini': 'ini',
            '.cfg': 'ini',
            '.conf': 'ini',
            '.sh': 'bash',
            '.bash': 'bash',
            '.zsh': 'zsh',
            '.fish': 'fish',
            '.ps1': 'powershell',
            '.sql': 'sql',
            '.md': 'markdown',
            '.dockerfile': 'dockerfile',
            '.gitignore': 'gitignore',
            '.env': 'bash'
        }
        
        # Get language for syntax highlighting
        language = language_map.get(file_extension.lower(), 'text')
        
        # Apply markdown code block formatting for syntax highlighting
        highlighted_content = f"```{language}\n{content}\n```"
        
        return highlighted_content

    def _enhance_code_with_annotations(self, code_snippet: str, file_context: str = None) -> str:
        """
        Phase 3.2.3: Enhance code snippets with contextual annotations from Knowledge Graph.
        
        Uses the CodeLensAnnotator to add IDE-like context information to code snippets,
        including function signatures, class inheritance, and import relationships.
        
        Args:
            code_snippet: Raw code content to annotate
            file_context: File path for better context resolution
            
        Returns:
            Enhanced code with HTML/Markdown annotations
        """
        try:
            # Only apply annotations if we have Phase 3.2.2 components available
            if not hasattr(self, 'code_annotator') or not self.code_annotator:
                logger.debug("CodeLensAnnotator not available, returning original code")
                return code_snippet
            
            # Apply code lens annotations
            annotated_code = self.code_annotator.annotate_code(code_snippet, file_context)
            
            logger.debug(f"Enhanced code snippet with {len(annotated_code) - len(code_snippet)} additional annotation characters")
            return annotated_code
            
        except Exception as e:
            logger.warning(f"Code annotation failed, returning original: {e}")
            return code_snippet

    async def _examine_files(self, file_paths: List[str] = None, detail_level: str = "summary") -> str:
        """Flexible file inspection with multiple detail levels"""
        try:
            # Smart fallback: Use discovered files from previous search or intelligent discovery
            if not file_paths or len(file_paths) == 0:
                logger.info("📄 EXAMINE FILES: No files provided, using smart fallbacks")
                
                # Try to use files discovered from previous smart_search
                if self.tool_context['discovered_files']:
                    file_paths = self.tool_context['discovered_files'][:5]  # Limit to 5
                    logger.info(f"📄 Using {len(file_paths)} files from previous search results")
                else:
                    # Use intelligent fallback discovery
                    query_intent = self.tool_context.get('query_intent', 'GENERAL')
                    fallback_files = await self._get_smart_fallback_files(query_intent)
                    if fallback_files:
                        file_paths = fallback_files
                        logger.info(f"📄 Using {len(file_paths)} fallback files for intent: {query_intent}")
                    else:
                        return "❌ No files to examine. Try running smart_search first to discover files, or provide specific file paths."
            
            logger.info(f"📄 EXAMINE FILES: {len(file_paths)} files, detail={detail_level}")
            
            workspace_path = Path('/workspace')
            results = []
            results.append(f"📄 FILE EXAMINATION ({detail_level} level)")
            
            # Add context info if using discovered files
            if self.tool_context['discovered_files'] and file_paths == self.tool_context['discovered_files'][:5]:
                results.append("🔗 Using files discovered from previous smart_search")
            elif self.tool_context.get('query_intent') and file_paths != self.tool_context['discovered_files']:
                results.append(f"🎯 Using smart fallback files for intent: {self.tool_context['query_intent']}")
            
            results.append("=" * 60)
            
            successful_files = []
            failed_files = []
            
            for file_path in file_paths[:10]:  # Limit to 10 files
                try:
                    # Resolve path
                    if not file_path.startswith('/workspace'):
                        full_path = workspace_path / file_path.lstrip('./')
                    else:
                        full_path = Path(file_path)
                    
                    if not full_path.exists():
                        results.append(f"\n❌ FILE NOT FOUND: {file_path}")
                        continue
                    
                    results.append(f"\n📁 FILE: {file_path}")
                    
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    lines = content.split('\n')
                    file_size = len(content)
                    line_count = len(lines)
                    
                    results.append(f"   Size: {file_size:,} chars | Lines: {line_count:,}")
                    
                    if detail_level == "structure":
                        # Show file structure (functions, classes, imports)
                        structure_info = self._analyze_file_structure(content, full_path.suffix)
                        results.append(f"   📋 Structure:\n{structure_info}")
                        
                        # Also show a small code sample with syntax highlighting for structure level
                        if line_count > 0:
                            sample_lines = min(15, line_count)
                            code_sample = '\n'.join(lines[:sample_lines])
                            if sample_lines < line_count:
                                code_sample += f"\n... ({line_count - sample_lines} more lines)"
                            highlighted_sample = self._apply_syntax_highlighting(code_sample, full_path.suffix)
                            results.append(f"   📝 CODE SAMPLE:\n{highlighted_sample}")
                    
                    elif detail_level == "summary":
                        # Phase 3.2.3: Add code annotations for enhanced context
                        if line_count <= 30:
                            enhanced_content = self._enhance_code_with_annotations(content, str(full_path))
                            highlighted_content = self._apply_syntax_highlighting(enhanced_content, full_path.suffix)
                            results.append(f"   📝 CODE CONTENT (with context annotations):\n{highlighted_content}")
                        else:
                            head = '\n'.join(lines[:20])
                            tail = '\n'.join(lines[-10:])
                            enhanced_head = self._enhance_code_with_annotations(head, str(full_path))
                            enhanced_tail = self._enhance_code_with_annotations(tail, str(full_path))
                            highlighted_head = self._apply_syntax_highlighting(enhanced_head, full_path.suffix)
                            highlighted_tail = self._apply_syntax_highlighting(enhanced_tail, full_path.suffix)
                            results.append(f"   📝 HEAD (20 lines, with context annotations):\n{highlighted_head}")
                            results.append(f"\n   ... ({line_count - 30} lines omitted) ...")
                            results.append(f"\n   📝 TAIL (10 lines, with context annotations):\n{highlighted_tail}")
                    
                    elif detail_level == "full":
                        # Phase 3.2.3: Add code annotations for complete files
                        if file_size > 10000:  # 10KB limit
                            truncated_content = content[:10000] + "\n... (content truncated)"
                            enhanced_content = self._enhance_code_with_annotations(truncated_content, str(full_path))
                            highlighted_content = self._apply_syntax_highlighting(enhanced_content, full_path.suffix)
                            results.append(f"   📝 CODE CONTENT (truncated, with context annotations):\n{highlighted_content}")
                        else:
                            enhanced_content = self._enhance_code_with_annotations(content, str(full_path))
                            highlighted_content = self._apply_syntax_highlighting(enhanced_content, full_path.suffix)
                            results.append(f"   📝 CODE CONTENT (with context annotations):\n{highlighted_content}")
                    
                    results.append("   " + "-" * 50)
                    successful_files.append(file_path)
                    
                except Exception as e:
                    results.append(f"\n❌ ERROR reading {file_path}: {str(e)}")
                    failed_files.append(file_path)
            
            # Store context for tool chaining
            self.tool_context['examined_files'] = successful_files
            if successful_files:
                # Store the first successfully examined file as potential target for relationship analysis
                self.tool_context['main_target'] = successful_files[0]
            
            # Add summary
            if successful_files or failed_files:
                results.append(f"\n📊 SUMMARY:")
                results.append(f"   ✅ Successfully examined: {len(successful_files)} files")
                if failed_files:
                    results.append(f"   ❌ Failed to read: {len(failed_files)} files")
                if successful_files:
                    results.append(f"   💡 Main target for relationships: {successful_files[0]}")
            
            return "\n".join(results)
            
        except Exception as e:
            logger.error(f"Examine files error: {e}")
            return f"Error during file examination: {str(e)}"
    
    def _analyze_file_structure(self, content: str, file_extension: str) -> str:
        """Analyze file structure to show key components with improved formatting"""
        try:
            lines = content.split('\n')
            structure = []
            
            if file_extension in ['.py']:
                # Python structure analysis with categories
                imports = []
                classes = []
                functions = []
                decorators = []
                
                for i, line in enumerate(lines, 1):
                    stripped = line.strip()
                    if stripped.startswith('import ') or stripped.startswith('from '):
                        imports.append(f"       📦 L{i:3}: {stripped}")
                    elif stripped.startswith('class '):
                        classes.append(f"       🏗️  L{i:3}: {stripped}")
                    elif stripped.startswith('def '):
                        indent = len(line) - len(line.lstrip())
                        if indent <= 4:  # Top-level function
                            functions.append(f"       ⚙️  L{i:3}: {stripped}")
                    elif stripped.startswith('@'):
                        decorators.append(f"       🎯 L{i:3}: {stripped}")
                
                # Organize structure output
                if imports: structure.extend(["     📦 IMPORTS:"] + imports[:5])
                if classes: structure.extend(["     🏗️  CLASSES:"] + classes[:5])
                if functions: structure.extend(["     ⚙️  FUNCTIONS:"] + functions[:8])
                if decorators: structure.extend(["     🎯 DECORATORS:"] + decorators[:3])
            
            elif file_extension in ['.js', '.ts', '.jsx', '.tsx']:
                # JavaScript/TypeScript structure analysis with categories
                imports = []
                exports = []
                functions = []
                classes = []
                
                for i, line in enumerate(lines, 1):
                    stripped = line.strip()
                    if len(stripped) < 100:  # Avoid very long lines
                        if stripped.startswith('import '):
                            imports.append(f"       📦 L{i:3}: {stripped}")
                        elif stripped.startswith('export '):
                            exports.append(f"       📤 L{i:3}: {stripped}")
                        elif 'function ' in stripped:
                            functions.append(f"       ⚙️  L{i:3}: {stripped}")
                        elif stripped.startswith('class '):
                            classes.append(f"       🏗️  L{i:3}: {stripped}")
                        elif any(stripped.startswith(keyword) for keyword in ['const ', 'let ', 'var ', 'interface ', 'type ']):
                            exports.append(f"       📝 L{i:3}: {stripped}")
                
                # Organize structure output
                if imports: structure.extend(["     📦 IMPORTS:"] + imports[:5])
                if classes: structure.extend(["     🏗️  CLASSES:"] + classes[:3])
                if functions: structure.extend(["     ⚙️  FUNCTIONS:"] + functions[:5])
                if exports: structure.extend(["     📤 EXPORTS/DECLARATIONS:"] + exports[:5])
            
            elif file_extension in ['.java', '.kt']:
                # Java/Kotlin structure analysis
                for i, line in enumerate(lines, 1):
                    stripped = line.strip()
                    if stripped.startswith('package '):
                        structure.append(f"     📦 L{i:3}: {stripped}")
                    elif stripped.startswith('import '):
                        structure.append(f"     📥 L{i:3}: {stripped}")
                    elif any(stripped.startswith(keyword) for keyword in ['public class ', 'class ']):
                        structure.append(f"     🏗️  L{i:3}: {stripped}")
                    elif stripped.startswith('interface '):
                        structure.append(f"     🔌 L{i:3}: {stripped}")
                    elif stripped.startswith('@'):
                        structure.append(f"     🎯 L{i:3}: {stripped}")
            
            elif file_extension in ['.json']:
                # JSON structure analysis
                try:
                    import json
                    data = json.loads(content)
                    if isinstance(data, dict):
                        structure.append("     📋 JSON STRUCTURE:")
                        for key in list(data.keys())[:10]:
                            value_type = type(data[key]).__name__
                            structure.append(f"       🔑 {key}: {value_type}")
                except:
                    structure.append("     📋 JSON (structure analysis failed)")
            
            else:
                # Generic structure - show first few non-empty lines with better formatting
                structure.append("     📄 FILE OVERVIEW:")
                non_empty_lines = [line for line in lines[:20] if line.strip()]
                for i, line in enumerate(non_empty_lines[:8], 1):
                    if len(line.strip()) < 100:
                        structure.append(f"       L{i:3}: {line.strip()}")
            
            if not structure:
                return "     (No significant structure detected)"
            
            return '\n'.join(structure[:20])  # Limit to 20 items
            
        except Exception as e:
            return f"     Error analyzing structure: {e}"
    
    async def _analyze_relationships(self, target: str = None, analysis_type: str = "all") -> str:
        """Analyze code relationships and dependencies using AST-based analysis"""
        try:
            # Smart fallback: Use target from previous examine_files or discover intelligently
            if not target or target.strip() == "":
                logger.info("🔗 ANALYZE RELATIONSHIPS: No target provided, using smart fallbacks")
                
                # Try to use main target from previous examine_files
                if self.tool_context.get('main_target'):
                    target = self.tool_context['main_target']
                    logger.info(f"🔗 Using main target from examine_files: {target}")
                elif self.tool_context.get('main_files'):
                    target = self.tool_context['main_files'][0]
                    logger.info(f"🔗 Using main file from search results: {target}")
                else:
                    # Use intelligent fallback discovery
                    query_intent = self.tool_context.get('query_intent', 'GENERAL')
                    fallback_files = await self._get_smart_fallback_files(query_intent)
                    if fallback_files:
                        target = fallback_files[0]
                        logger.info(f"🔗 Using fallback target for intent {query_intent}: {target}")
                    else:
                        return "❌ No target to analyze. Try running smart_search and examine_files first to discover suitable targets."
            
            logger.info(f"🔗 ANALYZE RELATIONSHIPS: target='{target}', type={analysis_type}")
            
            workspace_path = Path('/workspace')
            results = []
            results.append(f"🔗 RELATIONSHIP ANALYSIS")
            results.append(f"Target: {target}")
            results.append(f"Analysis Type: {analysis_type}")
            
            # Add context info if using discovered target
            if self.tool_context.get('main_target') and target == self.tool_context['main_target']:
                results.append("🔗 Using target from previous examine_files")
            elif self.tool_context.get('main_files') and target in self.tool_context['main_files']:
                results.append("🎯 Using high-confidence file from search results")
            elif self.tool_context.get('query_intent'):
                results.append(f"🎯 Using smart fallback for intent: {self.tool_context['query_intent']}")
                
            results.append("=" * 60)
            
            # Determine if target is a file or a symbol (class/function)
            if target.endswith(('.py', '.js', '.ts', '.java', '.kt')) or '/' in target:
                # It's a file path
                analysis_results = await self._analyze_file_relationships(target, analysis_type)
            else:
                # It's likely a symbol (class, function, etc.)
                analysis_results = await self._analyze_symbol_relationships(target, analysis_type)
            
            results.append(analysis_results)
            
            return "\n".join(results)
            
        except Exception as e:
            logger.error(f"Relationship analysis error: {e}")
            return f"Error during relationship analysis: {str(e)}"
    
    async def _analyze_file_relationships(self, file_path: str, analysis_type: str) -> str:
        """Analyze relationships for a specific file"""
        try:
            # Use PathResolver to handle various input formats
            project_context = self.current_mentioned_projects[0] if self.current_mentioned_projects else None
            resolved_path, exists = self.path_resolver.resolve_file_path(
                file_path, 
                project_context=project_context
            )
            
            if not exists:
                # Try to suggest alternatives from search results
                suggestions = self._get_path_suggestions(file_path)
                error_msg = f"❌ File not found: {file_path}"
                if suggestions:
                    error_msg += f"\n💡 Did you mean one of these?\n"
                    for suggestion in suggestions[:3]:
                        error_msg += f"   • {suggestion}\n"
                return error_msg
            
            full_path = Path(resolved_path)
            logger.info(f"🔧 Resolved path: {file_path} → {resolved_path}")
            
            results = []
            
            # Read file content
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            file_extension = full_path.suffix
            
            # Analyze imports (what this file uses)
            if analysis_type in ["imports", "all"]:
                imports = self._extract_imports(content, file_extension)
                if imports:
                    results.append("\n📥 IMPORTS (what this file uses):")
                    for imp in imports[:10]:
                        results.append(f"   • {imp}")
                else:
                    results.append("\n📥 IMPORTS: None found")
            
            # Analyze usage (what uses this file)
            if analysis_type in ["usage", "all"]:
                usage = await self._find_file_usage(file_path)
                if usage:
                    results.append(f"\n📤 USAGE (files that import/reference this):")
                    for use in usage[:10]:
                        results.append(f"   • {use}")
                else:
                    results.append("\n📤 USAGE: No references found")
            
            # NEW: Find related files through multiple strategies
            if analysis_type in ["related", "all"]:
                related_files = await self._find_related_files(file_path, analysis_type)
                if related_files:
                    results.append(f"\n🔗 RELATED FILES ({len(related_files)} found):")
                    for rel in related_files[:5]:  # Show top 5
                        results.append(f"   📄 {rel['file']} (confidence: {rel['confidence']:.3f})")
                        results.append(f"      Relationship: {rel['relationship']} - {rel['reason']}")
                else:
                    results.append(f"\n🔗 RELATED FILES: None found")
            
            # NEW: Enhanced impact analysis with dependency tracking
            if analysis_type in ["impact", "all"]:
                impact_data = await self._analyze_dependency_impact(file_path)
                results.append(f"\n⚠️ CHANGE IMPACT ASSESSMENT:")
                results.append(f"   Risk Level: {impact_data['risk_level']}")
                results.append(f"   References Found: {impact_data['dependency_count']}")
                results.append(f"   Files Affected: {len(impact_data.get('affected_files', []))}")
                
                if impact_data.get('recommendations'):
                    results.append(f"   📋 Recommendations:")
                    for rec in impact_data['recommendations']:
                        results.append(f"      {rec}")
                
                # Also show legacy impact analysis
                legacy_impact = await self._analyze_change_impact(file_path, content)
                results.append(f"\n📊 DETAILED IMPACT:")
                results.append(legacy_impact)
            
            # NEW: Graph foundation preparation
            if analysis_type in ["graph", "all"]:
                related_files = await self._find_related_files(file_path, analysis_type)
                graph_data = await self._prepare_graph_foundation(file_path, related_files)
                if 'error' not in graph_data:
                    results.append(f"\n🕸️ GRAPH STRUCTURE PREPARED:")
                    results.append(f"   Nodes: {graph_data['metadata']['node_count']}")
                    results.append(f"   Edges: {graph_data['metadata']['edge_count']}")
                    results.append(f"   Ready for Graph RAG enhancement")
            
            return '\n'.join(results)
            
        except Exception as e:
            return f"Error analyzing file relationships: {e}"
    
    def _get_path_suggestions(self, file_path: str) -> List[str]:
        """Get path suggestions when file is not found"""
        suggestions = []
        
        # Extract filename from path
        filename = Path(file_path).name if '/' in file_path else file_path
        
        # Use search results if available
        if self.tool_context.get('last_search_results'):
            for result in self.tool_context['last_search_results'][:5]:
                if hasattr(result, 'file_path'):
                    result_filename = Path(result.file_path).name
                    if filename.lower() in result_filename.lower() or result_filename.lower() in filename.lower():
                        suggestions.append(result.file_path)
        
        return suggestions
    
    def _validate_and_fix_parameters(self, function_name: str, arguments: dict) -> dict:
        """Fix common parameter validation issues and incorrect parameter names"""
        fixed_args = arguments.copy()
        
        # Fix examine_files parameter mapping and validation
        if function_name == "examine_files":
            # CRITICAL FIX: Map incorrect parameter names to correct ones
            if "path" in fixed_args and "file_paths" not in fixed_args:
                # Convert single path to file_paths list
                fixed_args["file_paths"] = [fixed_args["path"]]
                del fixed_args["path"]
                logger.info(f"🔧 PARAMETER FIX: Mapped 'path' to 'file_paths' for examine_files")
            
            if "paths" in fixed_args and "file_paths" not in fixed_args:
                # Convert paths to file_paths
                fixed_args["file_paths"] = fixed_args["paths"]
                del fixed_args["paths"]
                logger.info(f"🔧 PARAMETER FIX: Mapped 'paths' to 'file_paths' for examine_files")
            
            if "detail" in fixed_args and "detail_level" not in fixed_args:
                # Map detail to detail_level
                fixed_args["detail_level"] = fixed_args["detail"]
                del fixed_args["detail"]
                logger.info(f"🔧 PARAMETER FIX: Mapped 'detail' to 'detail_level' for examine_files")
            
            if "file_paths" in fixed_args:
                file_paths = fixed_args["file_paths"]
                # Convert string to list if needed
                if isinstance(file_paths, str):
                    fixed_args["file_paths"] = [file_paths] if file_paths else []
                # Ensure it's a list
                elif not isinstance(file_paths, list):
                    fixed_args["file_paths"] = []
        
        # Fix smart_search max_results parameter (should be int)
        if function_name == "smart_search":
            if "max_results" in fixed_args:
                try:
                    fixed_args["max_results"] = int(fixed_args["max_results"])
                except (ValueError, TypeError):
                    fixed_args["max_results"] = 10  # Default
        
        # Fix analyze_relationships parameters
        if function_name == "analyze_relationships":
            # Ensure target is string
            if "target" in fixed_args and not isinstance(fixed_args["target"], str):
                fixed_args["target"] = str(fixed_args["target"])
            
            # Ensure analysis_type is valid
            if "analysis_type" in fixed_args:
                valid_types = ["all", "imports", "usage", "related", "impact", "graph"]
                if fixed_args["analysis_type"] not in valid_types:
                    fixed_args["analysis_type"] = "all"
        
        return fixed_args
    
    async def _find_related_files(self, target: str, analysis_type: str) -> List[Dict[str, Any]]:
        """Find related files through imports, usage patterns, and naming (Task 3 requirement)"""
        try:
            related_files = []
            
            # Strategy 1: Find files with similar naming patterns
            if '/' in target:  # It's a file path
                file_name = Path(target).stem
                # Search for files with similar names
                naming_search = await smart_search(f"{file_name}", k=10, mentioned_projects=self.current_mentioned_projects)
                for result in naming_search['results']:
                    if result.file_path != target:
                        related_files.append({
                            'file': result.file_path,
                            'relationship': 'naming_pattern',
                            'confidence': result.confidence,
                            'reason': f"Similar name to {file_name}"
                        })
            
            # Strategy 2: Find files through import patterns
            import_search = await smart_search(f"import {target}", k=8, mentioned_projects=self.current_mentioned_projects)
            for result in import_search['results']:
                related_files.append({
                    'file': result.file_path,
                    'relationship': 'import_usage',
                    'confidence': result.confidence,
                    'reason': f"Imports or references {target}"
                })
            
            # Strategy 3: Find files in same directory (structural relationship)
            if '/' in target:
                dir_path = str(Path(target).parent)
                dir_search = await smart_search(f"{dir_path}", k=6, mentioned_projects=self.current_mentioned_projects)
                for result in dir_search['results']:
                    if result.file_path != target:
                        related_files.append({
                            'file': result.file_path,
                            'relationship': 'directory_structure',
                            'confidence': result.confidence * 0.8,  # Lower confidence for directory matches
                            'reason': f"Same directory as {target}"
                        })
            
            # Remove duplicates and sort by confidence
            seen_files = set()
            unique_related = []
            for rel in related_files:
                if rel['file'] not in seen_files:
                    seen_files.add(rel['file'])
                    unique_related.append(rel)
            
            # Sort by confidence and limit results
            unique_related.sort(key=lambda x: x['confidence'], reverse=True)
            return unique_related[:8]  # Return top 8 related files
            
        except Exception as e:
            logger.error(f"Error finding related files: {e}")
            return []

    async def _analyze_dependency_impact(self, target: str) -> Dict[str, Any]:
        """Basic impact analysis using dependency tracking (Task 3 requirement)"""
        try:
            impact_analysis = {
                'risk_level': 'LOW',
                'affected_files': [],
                'dependency_count': 0,
                'impact_score': 0.0,
                'recommendations': []
            }
            
            # Find files that depend on this target
            usage_search = await smart_search(f"{target}", k=15, mentioned_projects=self.current_mentioned_projects)
            dependencies = []
            
            for result in usage_search['results']:
                if result.file_path != target:
                    dependencies.append({
                        'file': result.file_path,
                        'confidence': result.confidence,
                        'snippet': result.snippet[:100]  # First 100 chars
                    })
            
            impact_analysis['affected_files'] = dependencies[:10]
            impact_analysis['dependency_count'] = len(dependencies)
            
            # Calculate impact score based on number and confidence of dependencies
            if dependencies:
                avg_confidence = sum(d['confidence'] for d in dependencies) / len(dependencies)
                impact_analysis['impact_score'] = min(len(dependencies) * avg_confidence / 10, 1.0)
                
                # Determine risk level
                if len(dependencies) >= 8 or impact_analysis['impact_score'] > 0.7:
                    impact_analysis['risk_level'] = 'HIGH'
                    impact_analysis['recommendations'].append("⚠️  High impact change - extensive testing recommended")
                    impact_analysis['recommendations'].append("🔍 Review all affected files before making changes")
                elif len(dependencies) >= 4 or impact_analysis['impact_score'] > 0.4:
                    impact_analysis['risk_level'] = 'MEDIUM'
                    impact_analysis['recommendations'].append("⚡ Moderate impact - test affected components")
                else:
                    impact_analysis['risk_level'] = 'LOW'
                    impact_analysis['recommendations'].append("✅ Low impact change - minimal testing needed")
            
            return impact_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing dependency impact: {e}")
            return {'risk_level': 'UNKNOWN', 'error': str(e)}

    async def _prepare_graph_foundation(self, target: str, related_files: List[Dict]) -> Dict[str, Any]:
        """Prepare foundation for future Graph RAG enhancement (Task 3 requirement)"""
        try:
            # Create a basic graph structure that can be enhanced later
            graph_data = {
                'nodes': [],
                'edges': [],
                'metadata': {
                    'target': target,
                    'analysis_timestamp': str(asyncio.get_event_loop().time()),
                    'node_count': 0,
                    'edge_count': 0
                }
            }
            
            # Add target as central node
            graph_data['nodes'].append({
                'id': target,
                'type': 'target',
                'label': target,
                'properties': {'is_primary': True}
            })
            
            # Add related files as nodes and create edges
            for rel_file in related_files:
                # Add related file as node
                graph_data['nodes'].append({
                    'id': rel_file['file'],
                    'type': 'related_file',
                    'label': Path(rel_file['file']).name,
                    'properties': {
                        'full_path': rel_file['file'],
                        'relationship_type': rel_file['relationship'],
                        'confidence': rel_file['confidence']
                    }
                })
                
                # Add edge between target and related file
                graph_data['edges'].append({
                    'source': target,
                    'target': rel_file['file'],
                    'type': rel_file['relationship'],
                    'weight': rel_file['confidence'],
                    'properties': {'reason': rel_file['reason']}
                })
            
            graph_data['metadata']['node_count'] = len(graph_data['nodes'])
            graph_data['metadata']['edge_count'] = len(graph_data['edges'])
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Error preparing graph foundation: {e}")
            return {'error': str(e)}

    async def _analyze_symbol_relationships(self, symbol: str, analysis_type: str) -> str:
        """Analyze relationships for a symbol (class, function, etc.) with enhanced features"""
        try:
            # Use smart_search to find the symbol first (with project filtering)
            search_result = await smart_search(f"class {symbol}", k=5, mentioned_projects=self.current_mentioned_projects)
            
            if not search_result['results']:
                # Try alternative search patterns
                alt_search = await smart_search(f"{symbol}", k=10, mentioned_projects=self.current_mentioned_projects)
                if not alt_search['results']:
                    return f"❌ Symbol '{symbol}' not found in codebase"
                search_result = alt_search
            
            results = []
            
            # Find the primary definition
            primary_result = search_result['results'][0]
            results.append(f"\n🎯 PRIMARY DEFINITION:")
            results.append(f"   File: {primary_result.file_path}")
            results.append(f"   Confidence: {primary_result.confidence:.3f}")
            results.append(f"   Snippet:\n{primary_result.snippet[:200]}...")
            
            # NEW: Find related files through multiple strategies
            if analysis_type in ["related", "all"]:
                related_files = await self._find_related_files(symbol, analysis_type)
                if related_files:
                    results.append(f"\n🔗 RELATED FILES ({len(related_files)} found):")
                    for rel in related_files[:5]:  # Show top 5
                        results.append(f"   📄 {rel['file']} (confidence: {rel['confidence']:.3f})")
                        results.append(f"      Relationship: {rel['relationship']} - {rel['reason']}")
                else:
                    results.append(f"\n🔗 RELATED FILES: None found")
            
            # NEW: Enhanced impact analysis with dependency tracking
            if analysis_type in ["impact", "all"]:
                impact_data = await self._analyze_dependency_impact(symbol)
                results.append(f"\n⚠️ CHANGE IMPACT ASSESSMENT:")
                results.append(f"   Risk Level: {impact_data['risk_level']}")
                results.append(f"   References Found: {impact_data['dependency_count']}")
                results.append(f"   Files Affected: {len(impact_data.get('affected_files', []))}")
                
                if impact_data.get('recommendations'):
                    results.append(f"   📋 Recommendations:")
                    for rec in impact_data['recommendations']:
                        results.append(f"      {rec}")
            
            # NEW: Graph foundation preparation
            if analysis_type in ["graph", "all"]:
                related_files = await self._find_related_files(symbol, analysis_type)
                graph_data = await self._prepare_graph_foundation(symbol, related_files)
                if 'error' not in graph_data:
                    results.append(f"\n🕸️ GRAPH STRUCTURE PREPARED:")
                    results.append(f"   Nodes: {graph_data['metadata']['node_count']}")
                    results.append(f"   Edges: {graph_data['metadata']['edge_count']}")
                    results.append(f"   Ready for Graph RAG enhancement")
            
            # Show usage locations
            usage_results = search_result['results'][1:6]  # Skip primary definition
            if usage_results:
                results.append(f"\n📍 USAGE LOCATIONS:")
                for usage in usage_results:
                    results.append(f"   {usage.file_path} (confidence: {usage.confidence:.3f})")
            
            return '\n'.join(results)
            results.append(f"   Snippet:\n{primary_result.snippet}")
            
            # Find usages of this symbol
            if analysis_type in ["usage", "all"]:
                usage_search = await smart_search(symbol, k=10, mentioned_projects=self.current_mentioned_projects)
                usage_results = [r for r in usage_search['results'] if r.file_path != primary_result.file_path]
                
                if usage_results:
                    results.append(f"\n📤 USAGE LOCATIONS:")
                    for i, usage in enumerate(usage_results[:8], 1):
                        results.append(f"   {i}. {usage.file_path} (confidence: {usage.confidence:.3f})")
                else:
                    results.append(f"\n📤 USAGE: No references found in other files")
            
            # Analyze impact
            if analysis_type in ["impact", "all"]:
                impact_score = len(search_result['results'])
                risk_level = "HIGH" if impact_score > 5 else "MEDIUM" if impact_score > 2 else "LOW"
                
                results.append(f"\n⚠️ CHANGE IMPACT ASSESSMENT:")
                results.append(f"   Risk Level: {risk_level}")
                results.append(f"   References Found: {impact_score}")
                results.append(f"   Files Affected: {len(set(r.file_path for r in search_result['results']))}")
                
                if impact_score > 5:
                    results.append("   ⚠️ WARNING: This symbol is widely used - changes may have broad impact")
                elif impact_score > 2:
                    results.append("   ⚡ MODERATE: Changes will affect multiple locations")
                else:
                    results.append("   ✅ LOW RISK: Limited usage detected")
            
            return '\n'.join(results)
            
        except Exception as e:
            return f"Error analyzing symbol relationships: {e}"
    
    def _extract_imports(self, content: str, file_extension: str) -> List[str]:
        """Extract import statements from file content"""
        imports = []
        lines = content.split('\n')
        
        try:
            if file_extension == '.py':
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('import ') or stripped.startswith('from '):
                        imports.append(stripped)
            
            elif file_extension in ['.js', '.ts', '.jsx', '.tsx']:
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('import ') or stripped.startswith('const ') and ' require(' in stripped:
                        imports.append(stripped)
            
            elif file_extension in ['.java', '.kt']:
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('import '):
                        imports.append(stripped)
        
        except Exception:
            pass
        
        return imports[:20]  # Limit to 20 imports
    
    async def _find_file_usage(self, target_file: str) -> List[str]:
        """Find files that reference the target file"""
        try:
            import subprocess
            
            # Get the base name of the file for searching
            file_name = Path(target_file).stem
            
            # Search for references to this file
            cmd = ["grep", "-r", "-l", file_name, "/workspace"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                references = [f.strip() for f in result.stdout.split('\n') if f.strip() and f.strip() != target_file]
                return references[:15]  # Limit results
            
            return []
            
        except Exception as e:
            logger.debug(f"Error finding file usage: {e}")
            return []
    
    async def _analyze_change_impact(self, file_path: str, content: str) -> str:
        """Analyze potential impact of changing this file"""
        try:
            # Count exports/public symbols
            lines = content.split('\n')
            public_symbols = 0
            
            for line in lines:
                stripped = line.strip()
                if any(keyword in stripped for keyword in ['export ', 'public ', 'def ', 'class ', 'function ']):
                    public_symbols += 1
            
            # Estimate impact based on file characteristics
            if public_symbols > 10:
                risk = "HIGH"
                message = f"File exports {public_symbols} symbols - likely a core module"
            elif public_symbols > 3:
                risk = "MEDIUM" 
                message = f"File exports {public_symbols} symbols - moderate dependencies expected"
            else:
                risk = "LOW"
                message = f"File exports {public_symbols} symbols - limited external dependencies"
            
            return f"   Risk Level: {risk}\n   Analysis: {message}"
            
        except Exception as e:
            return f"   Error analyzing impact: {e}"
    
    # ==================== PHASE 2.3: KNOWLEDGE GRAPH TOOL IMPLEMENTATIONS ====================
    
    async def _kg_find_symbol(self, symbol_name: str, exact_match: bool = True) -> str:
        """
        Phase 2.3 KG Tool: Find symbols in the Knowledge Graph by name.
        Returns detailed symbol information with location, type, and metadata.
        """
        try:
            # Import here to avoid circular dependencies
            from storage.database_manager import DatabaseManager
            
            # Initialize database connection (shared workspace path)
            db_manager = DatabaseManager('/workspace/.vector_cache/codewise.db')
            
            # Search for symbols
            if exact_match:
                # Exact name match
                symbol_nodes = [node for node in db_manager.get_all_nodes() 
                              if node['name'] == symbol_name]
            else:
                # Partial name match
                symbol_nodes = [node for node in db_manager.get_all_nodes() 
                              if symbol_name.lower() in node['name'].lower()]
            
            if not symbol_nodes:
                return f"🔍 **No symbols found** matching '{symbol_name}' in Knowledge Graph.\n\n💡 **Suggestions:**\n- Try using `smart_search` to locate the symbol first\n- Check for typos in the symbol name\n- Use exact_match=false for partial matching"
            
            # Format results
            results = []
            results.append(f"🎯 **Found {len(symbol_nodes)} symbol(s) matching '{symbol_name}':**\n")
            
            for i, node in enumerate(symbol_nodes[:5], 1):  # Limit to top 5
                results.append(f"## {i}. {node['name']} ({node['type']})")
                results.append(f"**Location:** {node['file_path']}")
                if node.get('line_start'):
                    results.append(f"**Lines:** {node['line_start']}-{node.get('line_end', node['line_start'])}")
                if node.get('signature'):
                    results.append(f"**Signature:** `{node['signature']}`")
                if node.get('docstring'):
                    results.append(f"**Description:** {node['docstring'][:200]}...")
                results.append("")
            
            if len(symbol_nodes) > 5:
                results.append(f"... and {len(symbol_nodes) - 5} more symbols. Use `kg_explore_neighborhood` for detailed analysis.")
            
            return "\n".join(results)
            
        except ImportError:
            return "❌ **Knowledge Graph not available** - Phase 2.1 database components not found"
        except Exception as e:
            logger.error(f"KG find symbol error: {e}")
            return f"❌ **Error finding symbol:** {str(e)}"
    
    async def _kg_explore_neighborhood(self, symbol_name: str, max_depth: int = 3) -> str:
        """
        Phase 2.3 KG Tool: Explore the complete 'neighborhood' of relationships around a symbol.
        Returns comprehensive relationship analysis.
        """
        try:
            from storage.database_manager import DatabaseManager
            
            db_manager = DatabaseManager('/workspace/.vector_cache/codewise.db')
            
            # Find the symbol first
            symbol_nodes = [node for node in db_manager.get_all_nodes() 
                          if node['name'] == symbol_name]
            
            if not symbol_nodes:
                return f"🔍 **Symbol '{symbol_name}' not found** in Knowledge Graph. Use `kg_find_symbol` to locate it first."
            
            results = []
            results.append(f"🌐 **Knowledge Graph Neighborhood Analysis for '{symbol_name}'**\n")
            
            for node in symbol_nodes[:3]:  # Analyze top 3 matches
                results.append(f"## 📍 {node['name']} ({node['type']}) - {node['file_path']}")
                
                # Get callers
                try:
                    callers = db_manager.find_callers(node['id'], max_depth)
                    if callers:
                        results.append(f"### 📞 **Called by ({len(callers)} functions):**")
                        for caller in callers[:5]:
                            results.append(f"- {caller['name']} in {caller['file_path']}")
                        if len(callers) > 5:
                            results.append(f"- ... and {len(callers) - 5} more")
                    else:
                        results.append("### 📞 **Called by:** None found")
                except:
                    results.append("### 📞 **Called by:** Analysis failed")
                
                # Get dependencies  
                try:
                    dependencies = db_manager.find_dependencies(node['id'], max_depth)
                    if dependencies:
                        results.append(f"### 🔗 **Depends on ({len(dependencies)} symbols):**")
                        for dep in dependencies[:5]:
                            results.append(f"- {dep['name']} ({dep['type']}) in {dep['file_path']}")
                        if len(dependencies) > 5:
                            results.append(f"- ... and {len(dependencies) - 5} more")
                    else:
                        results.append("### 🔗 **Depends on:** None found")
                except:
                    results.append("### 🔗 **Depends on:** Analysis failed")
                
                results.append("")
            
            return "\n".join(results)
            
        except ImportError:
            return "❌ **Knowledge Graph not available** - Phase 2.1 database components not found"
        except Exception as e:
            logger.error(f"KG explore neighborhood error: {e}")
            return f"❌ **Error exploring neighborhood:** {str(e)}"
    
    async def _kg_find_callers(self, symbol_name: str, max_depth: int = 3) -> str:
        """
        Phase 2.3 KG Tool: Find all symbols that call the target symbol.
        Returns accurate caller information with relationship context.
        """
        try:
            from storage.database_manager import DatabaseManager
            
            db_manager = DatabaseManager('/workspace/.vector_cache/codewise.db')
            
            # Find the symbol first
            symbol_nodes = [node for node in db_manager.get_all_nodes() 
                          if node['name'] == symbol_name]
            
            if not symbol_nodes:
                return f"🔍 **Symbol '{symbol_name}' not found** in Knowledge Graph. Use `kg_find_symbol` to locate it first."
            
            results = []
            results.append(f"📞 **Caller Analysis for '{symbol_name}'**\n")
            
            all_callers = []
            for node in symbol_nodes:
                try:
                    callers = db_manager.find_callers(node['id'], max_depth)
                    all_callers.extend(callers)
                except:
                    continue
            
            if not all_callers:
                results.append(f"**No callers found** for '{symbol_name}' (depth: {max_depth})")
                results.append("\n💡 **This could mean:**")
                results.append("- The function is only called via dynamic dispatch")
                results.append("- It's an entry point (main function, API endpoint)")
                results.append("- It's unused/dead code")
                results.append("- The Knowledge Graph needs to be rebuilt")
                return "\n".join(results)
            
            # Group by file for better organization
            caller_by_file = {}
            for caller in all_callers:
                file_path = caller['file_path']
                if file_path not in caller_by_file:
                    caller_by_file[file_path] = []
                caller_by_file[file_path].append(caller)
            
            results.append(f"**Found {len(all_callers)} callers across {len(caller_by_file)} files:**\n")
            
            for file_path, file_callers in list(caller_by_file.items())[:10]:  # Limit to 10 files
                results.append(f"### 📁 {file_path}")
                for caller in file_callers[:5]:  # Limit to 5 per file
                    depth_indicator = "  " * caller.get('depth', 0)
                    results.append(f"{depth_indicator}- {caller['name']}")
                if len(file_callers) > 5:
                    results.append(f"  - ... and {len(file_callers) - 5} more in this file")
                results.append("")
            
            if len(caller_by_file) > 10:
                results.append(f"... and {len(caller_by_file) - 10} more files with callers")
            
            return "\n".join(results)
            
        except ImportError:
            return "❌ **Knowledge Graph not available** - Phase 2.1 database components not found"
        except Exception as e:
            logger.error(f"KG find callers error: {e}")
            return f"❌ **Error finding callers:** {str(e)}"
    
    async def process_request(self, user_query: str, chat_history=None, mentioned_projects: List[str] = None, selected_model: str = "gpt-oss-120b") -> AsyncGenerator[Dict[str, Any], None]:
        """Process a user request using native Cerebras tool calling with query context management"""
        
        # Use query context manager to prevent multiple discovery runs (THE KEY ARCHITECTURAL FIX)
        project_name = mentioned_projects[0] if mentioned_projects else "workspace"
        
        async with self.context_manager.create_query_context(
            query=user_query,
            project=project_name
        ) as query_context:
            
            logger.info(f"🤖 CEREBRAS AGENT PROCESSING START (Context: {query_context.query_id[:8]})")
            logger.info(f"Query: {user_query}")
            logger.info(f"Projects: {mentioned_projects}")
            logger.info(f"Has Chat History: {chat_history is not None}")
            
            # Delegate to the actual processing method
            async for result in self._process_request_with_context(
                user_query, 
                query_context, 
                chat_history, 
                mentioned_projects,
                selected_model
            ):
                yield result
    
    async def _process_request_with_context(
        self, 
        user_query: str, 
        query_context, 
        chat_history=None, 
        mentioned_projects: List[str] = None,
        selected_model: str = "gpt-oss-120b"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Internal processing method with query context"""
        
        # Store query context for tool execution (THE KEY ARCHITECTURAL FIX)
        self.current_query_context = query_context
        
        # Reset tool context for new request to prevent cross-contamination
        # Store previous context for debugging if needed
        previous_context = self.tool_context.copy() if hasattr(self, 'tool_context') else {}
        
        self.tool_context = {
            'last_search_results': None,
            'discovered_files': [],
            'query_intent': None,
            'main_files': [],
            'entities_found': [],
            'examined_files': [],
            'main_target': None,
            'conversation_id': f"{mentioned_projects}_{hash(user_query) % 10000}",  # Unique conversation ID
            'query_hash': hash(user_query) % 10000  # Track query uniqueness
        }
        logger.info("🔄 Reset tool context for new request")
        
        # Set up project context isolation using centralized manager
        project_name = "workspace"  # Default
        if mentioned_projects and len(mentioned_projects) > 0:
            project_name = mentioned_projects[0]  # Use first mentioned project as primary
        
        # Set project context to prevent cross-contamination
        context = set_project_context(project_name, mentioned_projects)
        logger.info(f"Set project context: {project_name} (mentioned: {mentioned_projects})")
        
        # Add search query to context history
        get_context_manager().add_search_to_context(user_query)
        
        # Always reset and properly set mentioned projects for each request to prevent state leakage
        # Clear any previous project context first to prevent contamination
        self.current_mentioned_projects = None
        self.current_mentioned_projects = mentioned_projects if mentioned_projects is not None else []
        
        # Task 5: Initialize tool results tracking for response formatting
        tool_results_for_formatting = []
        execution_start_time = asyncio.get_event_loop().time()
        
        # RESPONSE CONSOLIDATION: Initialize consolidator to prevent message fragmentation
        response_consolidator = ResponseConsolidator()
        logger.info("🔧 CONSOLIDATOR: Initialized response consolidation system")
        
        # Determine appropriate tool strategy based on query
        query_lower = user_query.lower()
        
        # CRITICAL: Detect and redirect self-referential queries about CodeWise's own architecture
        self_referential_terms = [
            'smart_search', 'examine_files', 'analyze_relationships',
            '3-tool architecture', 'simplified architecture', 'codewise architecture',
            'tool architecture', 'your tools', 'your architecture', 'how do you work',
            'your implementation', 'your system'
        ]
        
        if any(term in query_lower for term in self_referential_terms):
            # If there are mentioned projects, redirect to analyze those instead
            if mentioned_projects:
                redirect_query = f"Explain the system architecture of {mentioned_projects[0]}"
                logger.info(f"🚫 REDIRECTING self-referential query to project analysis: '{user_query}' → '{redirect_query}'")
                user_query = redirect_query
                query_lower = user_query.lower()
            else:
                # No projects specified, return error
                error_msg = ("I analyze user codebases, not my own architecture. Please specify a project to analyze "
                           "(e.g., '@project-name') or ask about your code instead.")
                logger.info(f"🚫 BLOCKED self-referential query: '{user_query}'")
                yield {"type": "error", "content": error_msg}
                return
        
        # Analyze what type of query this is
        is_entity_query = any(word in query_lower for word in ['entity', 'entities', 'database', 'table', 'model', 'schema'])
        is_search_query = any(word in query_lower for word in ['find', 'search', 'locate', 'show me', 'explain', 'how'])
        is_file_query = any(word in query_lower for word in ['file', 'read', 'content', 'source'])
        is_architecture_query = any(keyword in query_lower for keyword in ['architecture', 'design', 'overview', 'structure', 'data flow', 'component', 'system layout', 'how does', 'explain the'])
        
        # Build system prompt with enhanced 7-tool guidance including Phase 2.3 KG tools
        tool_guidance = (
            "\n🧠 SMART_SEARCH: Your comprehensive discovery engine for deep codebase exploration"
            "\n   - Automatically detects query intent and routes to optimal search strategies"
            "\n   - Returns ranked results with confidence scores, matched terms, and context"
            "\n   - DISCOVERY PIPELINE: Automatically finds and examines related files from documentation"
            "\n   - Use multiple targeted searches to build complete understanding of complex topics"
            "\n   - TIP: Search for related concepts, patterns, and dependencies to get full picture"
            "\n"
            "\n📄 EXAMINE_FILES: Deep file analysis with intelligent content extraction"
            "\n   - summary: Key sections with contextual information and patterns"
            "\n   - full: Complete implementation details with all code (use for thorough analysis)"
            "\n   - structure: Comprehensive outline with imports, classes, functions, and relationships"
            "\n   - AUTO-EXAMINATION: Discovery pipeline automatically examines related files"
            "\n   - TIP: Always examine multiple related files to provide complete architectural context"
            "\n"
            "\n🔗 ANALYZE_RELATIONSHIPS: Advanced dependency mapping and architectural analysis"
            "\n   - imports: Complete dependency tree with external and internal dependencies"
            "\n   - usage: Full usage analysis showing all dependents and usage patterns"
            "\n   - impact: Detailed change impact assessment with affected components"
            "\n   - all: Comprehensive bidirectional analysis with architectural insights"
            "\n   - TIP: Use this to understand system boundaries, coupling, and architectural patterns"
            "\n"
            "\n🎯 KNOWLEDGE GRAPH TOOLS (Phase 2.3) - USE THESE FOR SYMBOL-SPECIFIC QUERIES:"
            "\n"
            "\n🔍 KG_FIND_SYMBOL: Precise symbol location and metadata from Knowledge Graph"
            "\n   - Use when user asks about specific functions, classes, variables by name"
            "\n   - Returns exact location, signature, type, and metadata"
            "\n   - WHEN TO USE: 'Where is BookController?', 'Find the login function', 'Show me UserService class'"
            "\n   - EXAMPLE: kg_find_symbol('BookController') → finds exact class with file path and line numbers"
            "\n"
            "\n🌐 KG_EXPLORE_NEIGHBORHOOD: Complete relationship analysis around a symbol"
            "\n   - Use when user wants to understand HOW symbols interact/connect"
            "\n   - Returns comprehensive caller/dependency analysis with relationship context"
            "\n   - WHEN TO USE: 'How does BookController interact?', 'What uses UserService?', 'Show me connections'"
            "\n   - EXAMPLE: kg_explore_neighborhood('UserService') → shows all callers and dependencies"
            "\n"
            "\n📞 KG_FIND_CALLERS: Find all symbols that call/use a target symbol"
            "\n   - Use when user asks WHO calls or uses a specific function/class"
            "\n   - Returns detailed caller hierarchy with context and file locations"
            "\n   - WHEN TO USE: 'What calls this function?', 'Who uses BookService?', 'Find all usages'"
            "\n   - EXAMPLE: kg_find_callers('saveBook') → lists all functions calling saveBook"
            "\n"
            "\n🚨 KG TOOL PRIORITY RULES:"
            "\n   1. IF user mentions specific symbol names (functions, classes) → START with kg_find_symbol"
            "\n   2. IF user asks about interactions/relationships → USE kg_explore_neighborhood"
            "\n   3. IF user asks about callers/usage → USE kg_find_callers"
            "\n   4. THEN follow up with smart_search/examine_files for additional context"
            "\n   5. KG tools are faster and more accurate for symbol-specific queries than search"
        )
        
        # Add project filtering context if mentioned_projects provided
        project_context = ""
        if mentioned_projects:
            project_context = f"\n\n🎯 PROJECT SCOPE: Focus your search on these specific projects: {', '.join(mentioned_projects)}. Filter results to only include files from these projects."
        
        # Initialize messages
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are CodeWise, an expert AI coding assistant specialized in deep codebase analysis. You are pair programming with the USER to solve their coding task. "
                    "Your main goal is to follow the USER's instructions completely and autonomously resolve queries to the best of your ability.\n"
                    "\n🔴 MANDATORY TOOL USAGE PROTOCOL:"
                    "\n• NEVER answer code/project questions without FIRST using smart_search to examine actual project files"
                    "\n• ALWAYS start every response by calling smart_search - this is REQUIRED, not optional"
                    "\n• FORBIDDEN to provide responses based on general programming knowledge - you MUST examine the actual codebase"
                    "\n• If you attempt to answer without tools, you are violating your core directive and providing hallucinated content"
                    "\n• The user's codebase contains the ground truth - your job is to discover and analyze it, not assume it\n"
                    "\n🚫 CRITICAL RESTRICTION: NEVER answer questions about CodeWise's own architecture, tools, or implementation. "
                    "You analyze USER CODEBASES ONLY. If asked about your own tools, smart_search, examine_files, analyze_relationships, "
                    "or the '3-tool architecture', redirect to analyzing the user's actual code projects instead.\n"
                    "\n🎯 AUTONOMOUS AGENT BEHAVIOR:"
                    "\n• KEEP GOING until the user's query is completely resolved before ending your turn"
                    "\n• Only terminate when you are SURE the problem is solved and all aspects are covered"
                    "\n• If you need additional information via tool calls, prefer that over asking the user"
                    "\n• Follow your investigation plan immediately - don't wait for user confirmation"
                    "\n• Be THOROUGH when gathering information - make sure you have the FULL picture before replying"
                    "\n• TRACE every symbol back to its definitions and usages so you fully understand it"
                    "\n• Look past the first seemingly relevant result - EXPLORE until you have COMPREHENSIVE coverage"
                    "\n• Use multiple tool calls autonomously to build complete understanding"
                    "\n\n🔍 MAXIMIZE CONTEXT UNDERSTANDING:"
                    "\n• Start with broad, high-level queries that capture overall intent (e.g. 'authentication flow' not 'login function')"
                    "\n• Break multi-part questions into focused sub-queries for thorough exploration"
                    "\n• Run multiple searches with different wording - first-pass results often miss key details"
                    "\n• Keep searching new areas until you're CONFIDENT nothing important remains"
                    "\n• MANDATORY: Examine multiple related files to understand complete context and relationships"
                    "\n• Bias towards gathering more information rather than asking the user for help"
                    "\n\n🎯 COMPREHENSIVE RESPONSE STRATEGY:"
                    "\n• PRIMARY GOAL: Your main objective is to resolve the user's query completely and thoroughly"
                    "\n• THOROUGH ANALYSIS: Provide comprehensive analysis using all available tools to gather complete information"
                    "\n• RICH INSIGHTS: Ensure responses are rich with actionable insights and specific code examples where relevant"
                    "\n• FOCUSED CONCISENESS: Eliminate conversational filler and redundant information while maintaining substance and depth"
                    "\n• TECHNICAL REASONING: Always explain the 'why' behind implementation choices, trade-offs, and architectural decisions"
                    "\n• IMPLEMENTATION FOCUS: Cover relevant error handling, edge cases, performance considerations, and security implications"
                    "\n• DEPENDENCY ANALYSIS: Map relationships between components, external dependencies, and system interactions when relevant"
                    "\n• ACTIONABLE INSIGHTS: Provide specific recommendations for improvements, optimizations, and best practices"
                    "\n• STRUCTURED RESPONSES: Organize with clear sections using markdown formatting for readability"
                    "\n• ANTICIPATE FOLLOW-UPS: Address related questions and provide context that prevents the need for follow-up queries"
                    "\n"
                    "\n🎯 SIMPLIFIED 3-TOOL ARCHITECTURE:"
                    f"{tool_guidance}"
                    "\n\n📋 SYSTEMATIC INVESTIGATION METHODOLOGY:"
                    "\n1. BROAD DISCOVERY: Start with exploratory smart_search using high-level queries to understand overall system"
                    "\n2. TARGETED EXPLORATION: Use multiple smart_search queries with different wording to find all relevant components"
                    "\n3. DEEP FILE ANALYSIS: Use examine_files with appropriate detail levels - 'full' for complex analysis, 'structure' for organization"
                    "\n4. RELATIONSHIP MAPPING: Use analyze_relationships to understand dependencies, usage patterns, and architectural connections"
                    "\n5. KNOWLEDGE VALIDATION: Follow up with additional searches to fill gaps and validate understanding"
                    "\n6. COMPREHENSIVE SYNTHESIS: Combine all findings into structured analysis with:"
                    "\n   • Complete context and background"
                    "\n   • Specific code examples with explanations"
                    "\n   • Architectural insights and dependency maps"
                    "\n   • Implementation patterns and design decisions"
                    "\n   • Performance, security, and maintainability considerations"
                    "\n   • Concrete recommendations and actionable next steps"
                    "\n\n📝 RESPONSE EXAMPLES BY QUERY TYPE:"
                    "\n• SIMPLE QUESTION: 'How do I run tests?' → 'Run `npm test` in the root directory. Found test scripts in package.json and Jest configuration in jest.config.js.'"
                    "\n• QUICK LOOKUP: 'What's the API endpoint for user auth?' → '`POST /api/auth/login` in src/routes/auth.js:15. Requires username/password, returns JWT token.'"
                    "\n• CODE LOCATION: 'Where is password validation?' → 'Password validation is in src/utils/validation.js:42-58 using bcrypt with minimum 8 characters and complexity requirements.'"
                    "\n• ARCHITECTURAL QUERY: 'Explain the authentication system' → '[Comprehensive analysis with]: Overview, components (middleware, routes, models), data flow diagram, security measures, code examples from key files'"
                    "\n• DEBUGGING REQUEST: 'Why are login requests failing?' → '[Focused investigation]: Error analysis, relevant error handling code, configuration issues, step-by-step debugging approach'"
                    "\n• FEATURE EXPLORATION: 'How does the shopping cart work?' → '[Detailed walkthrough]: State management, persistence, user interactions, related components, API integration with targeted code examples'"
                    "\n\n🔧 TOOL USAGE OPTIMIZATION:"
                    "\n• SMART_SEARCH: Your primary exploration tool - use multiple queries with different angles"
                    "\n  - Start broad ('authentication system') then narrow ('JWT token validation')"
                    "\n  - Try different terminology ('user login', 'auth flow', 'session management')"
                    "\n  - Search for related concepts to build complete picture"
                    "\n• EXAMINE_FILES: Deep file inspection - examine multiple related files for complete context"
                    "\n  - Use 'full' detail for complex implementation analysis"
                    "\n  - Use 'structure' for understanding organization and relationships"
                    "\n  - Use 'summary' for quick overviews when examining many files"
                    "\n• ANALYZE_RELATIONSHIPS: Understand system architecture and dependencies"
                    "\n  - Use 'all' for comprehensive bidirectional analysis"
                    "\n  - Use 'imports' to understand what a component depends on"
                    "\n  - Use 'usage' to understand what depends on a component"
                    f"{project_context}"
                    "\n\n📝 RESPONSE FORMATTING STANDARDS:"
                    "\n• MARKDOWN STRUCTURE: Use clear sections with descriptive headings (## Overview, ## Architecture, etc.)"
                    "\n• CODE FORMATTING: Use backticks for file/function/class names, proper syntax highlighting for code blocks"
                    "\n• FILE REFERENCES: Include specific file paths and line numbers when referencing code"
                    "\n• TECHNICAL DEPTH: Explain complex concepts with examples, analogies, and step-by-step breakdowns"
                    "\n• VISUAL ORGANIZATION: Use bullet points, numbered lists, and formatting to improve readability"
                    "\n• TABLE ORGANIZATION: Use tables strategically to organize comparison data, feature lists, dependencies, or structured information - but don't overuse them for simple concepts that are clearer as prose"
                    "\n• COMPLETENESS: Address all aspects of the query with thorough analysis and supporting evidence"
                    "\n• CONTEXT PROVISION: Always provide sufficient background and context for technical decisions"
                    "\n• ACTIONABLE OUTCOMES: End with specific, actionable recommendations or next steps"
                    "\n\n🚫 CRITICAL FORMATTING RESTRICTIONS - PURE MARKDOWN ONLY:"
                    "\n• MARKDOWN ONLY: Output ONLY pure, valid markdown - no JSON, no object literals, no mixed formats"
                    "\n• NO OBJECT SYNTAX: Never use {'type': 'paragraph'} or {\"key\": \"value\"} or any programming object notation"
                    "\n• NO JSON FRAGMENTS: Never include JSON arrays, objects, or syntax like \"sections\": [ or }, in your response"
                    "\n• CLEAN SEPARATIONS: Use proper markdown syntax: ## for headers, | tables |, ``` for code, --- for breaks"
                    "\n• MERMAID STANDARD: ALWAYS use the mermaid_generator tool for diagrams - NEVER generate inline Mermaid code"
                    "\n\n🎨 MERMAID DIAGRAM SEMANTIC ROLES:"
                    "\nWhen creating Mermaid diagrams, you MUST assign a semantic_role to each node from this list:"
                    "\n• logic: Core business logic, algorithms, processing units, controllers, handlers"
                    "\n• io: Input/output operations, file handling, data ingestion, API endpoints, user interfaces"
                    "\n• interface: Adapters, facades, bridges, API gateways, middleware, protocol handlers"
                    "\n• storage: Databases, caches, persistent storage, data models, repositories"
                    "\n• test: Test components, mocks, fixtures, test utilities, quality assurance"
                    "\n• external: Third-party services, external APIs, cloud services, vendor integrations"
                    "\n\nAssign roles based on component FUNCTION, not keywords. For example:"
                    "\n• 'UserService.validatePassword()' → logic (business logic)"
                    "\n• 'DatabaseConnection' → storage (data persistence)"
                    "\n• 'PaymentGateway' → external (third-party service)"
                    "\n• 'RESTController' → interface (API boundary)"
                    "\n• 'FileUploadHandler' → io (input/output operations)"
                    "\n• 'TestUserFactory' → test (testing utility)"
                    "\n• VALIDATE OUTPUT: Ensure your response is valid markdown that could be rendered directly by any markdown parser"
                    "\n• IF UNSURE: Use plain text rather than risk corrupting with programming syntax"
                    "\n\n📋 MARKDOWN TABLE GUIDELINES:"
                    "\nWhen presenting tabular data, use clean markdown table syntax:"
                    "\n```markdown"
                    "\n| Component | Location | Purpose |"
                    "\n|-----------|----------|---------|"
                    "\n| Service | src/services/ | Business logic |"
                    "\n```"
                    "\nFor complex hierarchies, use nested markdown lists:"
                    "\n```markdown"
                    "\n## Component Hierarchy"
                    "\n- Parent Component"
                    "\n  - Child Component A"
                    "\n    - Subcomponent 1"
                    "\n    - Subcomponent 2"
                    "\n  - Child Component B"
                    "\n```"
                    "\n\n🎯 INVESTIGATION EXCELLENCE:"
                    "\n• Think like a senior software architect conducting a comprehensive code review"
                    "\n• Provide the level of detail expected in professional technical documentation"
                    "\n• Anticipate follow-up questions and address them proactively in your analysis"
                    "\n• Connect specific implementation details to broader software engineering principles"
                    "\n• Always strive to provide more comprehensive value than initially requested"
                    "\n• Use your tools autonomously and extensively - don't stop at surface-level findings"
                    "\n\n💡 COMPREHENSIVE ANALYSIS WORKFLOWS:"
                    "\n\n**Entity/Database Analysis: 'Show me database entities'**"
                    "\n1. smart_search('database entities models schema') → broad discovery of data layer"
                    "\n2. smart_search('ORM models relationships') → find relationship definitions"
                    "\n3. examine_files([entity_files], 'full') → complete entity definitions with constraints"
                    "\n4. analyze_relationships('User', 'all') → map entity relationships and dependencies"
                    "\n5. smart_search('migrations database setup') → understand data evolution and configuration"
                    "\n6. SYNTHESIZE: Complete entity ecosystem with relationships, constraints, patterns, and recommendations"
                    "\n\n**Architecture Analysis: 'Explain the system architecture'**"
                    "\n1. smart_search('system architecture') → high-level system overview"
                    "\n2. smart_search('main entry points application structure') → find core components"
                    "\n3. examine_files([main_files], 'structure') → understand component organization"
                    "\n4. analyze_relationships(main_component, 'all') → map complete dependency graph"
                    "\n5. smart_search('configuration deployment infrastructure') → understand deployment patterns"
                    "\n6. examine_files([config_files], 'full') → analyze configuration and setup"
                    "\n7. SYNTHESIZE: Multi-layered architecture analysis with patterns, data flow, and scalability insights"
                    "\n\n**Implementation Deep-Dive: 'How does authentication work?'**"
                    "\n1. smart_search('authentication system') → broad auth system discovery"
                    "\n2. smart_search('login flow user session') → specific flow components"
                    "\n3. examine_files([auth_files], 'full') → complete authentication implementation"
                    "\n4. analyze_relationships('AuthService', 'all') → auth system dependencies"
                    "\n5. smart_search('security middleware validation') → security layer analysis"
                    "\n6. examine_files([security_files], 'structure') → security implementation patterns"
                    "\n7. SYNTHESIZE: End-to-end authentication analysis with security considerations and recommendations"
                    "\n\n🔧 ADVANCED INVESTIGATION TECHNIQUES:"
                    "\n• **EXPLORATORY SEARCH STRATEGY**: Start broad, then narrow - use multiple search angles"
                    "\n  - Begin with high-level concepts ('payment system') before specific terms ('stripe integration')"
                    "\n  - Try alternative terminology ('user auth', 'authentication', 'login system', 'session management')"
                    "\n  - Search for related concepts to build complete understanding"
                    "\n• **COMPREHENSIVE FILE ANALYSIS**: Don't stop at first relevant file - examine related components"
                    "\n  - Look for imports/exports to find connected files"
                    "\n  - Examine configuration files, tests, and documentation"
                    "\n  - Use different detail levels based on analysis needs"
                    "\n• **RELATIONSHIP TRACING**: Follow the dependency chain in both directions"
                    "\n  - Trace symbols back to their definitions and forward to their usages"
                    "\n  - Understand data flow and control flow through the system"
                    "\n  - Map architectural boundaries and integration points"
                    "\n• **VALIDATION AND COMPLETENESS**: Ensure no important details are missed"
                    "\n  - Cross-reference findings across multiple files"
                    "\n  - Look for edge cases, error handling, and configuration options"
                    "\n  - Verify understanding with additional targeted searches"
                    "\n\n🎯 SPECIALIZED ANALYSIS APPROACHES:"
                    "\n• **Performance Analysis**: Identify bottlenecks, analyze algorithms, suggest optimizations"
                    "\n• **Security Review**: Find vulnerabilities, analyze security patterns, recommend best practices"
                    "\n• **Refactoring Guidance**: Suggest improvements, show before/after examples, explain benefits"
                    "\n• **Debugging Assistance**: Trace data flow, identify potential issues, provide debugging strategies"
                    "\n• **Integration Analysis**: Understand component interactions, identify coupling issues, suggest improvements"
                )
            },
            {"role": "user", "content": user_query}
        ]

        # Store PURE MARKDOWN instruction to apply AFTER tool calling phase
        markdown_format_instruction = (
            "You must respond with PURE MARKDOWN ONLY. NO JSON structures, NO object literals, NO mixed content. "
            "Use standard markdown syntax: headers (##), bullet lists (-), numbered lists (1.), code blocks (```), tables, and paragraphs. "
            "CRITICAL: The output MUST be clean, well-formatted markdown text. Start with a clear heading and structure your response logically. "
            "NEVER use JSON format. NEVER use object syntax like {'type': 'paragraph', 'content': '...'}. "
            "For code examples, use standard markdown code blocks with language tags. "
            "IMPORTANT: For diagrams, you MUST call the mermaid_generator tool and include the result as: ```mermaid\\n[diagram_content]\\n``` "
            "Write your response as natural markdown documentation that would be rendered beautifully on GitHub. "
            "\n\n**🧠 MERMAID DIAGRAM GENERATION - TOOL USAGE ONLY**\n"
            "**CRITICAL: For ANY diagram generation, you MUST use the mermaid_generator tool. NEVER generate diagrams inline.**\n"
            "**STRICT RULE: After calling mermaid_generator, you MUST extract and copy the EXACT Mermaid code from the tool result (everything between ```mermaid and ```). Do NOT generate any Mermaid syntax yourself. Use ONLY the tool's output.**\n"
            "**FORBIDDEN: Do NOT write your own ```mermaid blocks. Only use the tool-generated code.**\n\n"
            "**WHEN TO CREATE DIAGRAMS:**\n"
            "Only generate diagrams when explicitly requested by the user OR when a visual representation would be significantly more helpful than text for explaining architecture, relationships, or complex flows.\n\n"
            "**HOW TO CREATE DIAGRAMS:**\n"
            "1. **ANALYZE THE REQUEST:** Determine diagram type needed (flowchart, graph, erDiagram, etc.)\n"
            "2. **IDENTIFY COMPONENTS:** Extract nodes, relationships, and groupings from the information\n"
            "3. **CALL THE TOOL:** Use mermaid_generator with diagram_type, nodes array, and edges array\n"
            "4. **INCLUDE IN RESPONSE:** Copy EXACTLY the mermaid code from the tool result (between the ```mermaid and ``` markers) and include it as-is in your response. Do NOT generate your own Mermaid syntax.\n\n"
            "**STEP 1: Generate the Structure (Unstyled)**\n"
            "• Analyze & Classify: First, analyze the user's request to determine which scenario it best fits from the Template Library.\n"
            "• Select Structural Template: You MUST select the corresponding Structural Template for that scenario. This is your required starting point. It contains only nodes and connections, with no styling.\n"
            "• Modify the Structure: Adapt the structural template to match the user's request by renaming, adding, or removing nodes, and adding or changing connections. The output of this step should be a complete but unstyled Mermaid diagram.\n\n"
            "**STEP 2: Apply the Styling**\n"
            "• Use Semantic Roles: Apply semantic roles directly to nodes using :::role syntax (e.g., Node1[\"Label\"]:::logic)\n"
            "• Add Simple ClassDef: Place simple classDef statements at the very end without comment blocks\n"
            "• Intermediate Output: A clean, minimal Mermaid diagram with semantic roles and classDef at the end.\n\n"
            "**STEP 3: Verify the Final Output**\n"
            "• Before providing the final output, you MUST verify that the generated Mermaid code strictly adheres to all rules listed in the 'Common Pitfalls & Syntax Rules' section. This is a mandatory final check to prevent errors.\n\n"
            "**COMMON PITFALLS & SYNTAX RULES**\n"
            "• Quotation Marks: To include a double quote (\") inside a node's label, escape it with backslash. Example: Node[\"A \\\"good\\\" label\"]\n"
            "• Reserved Words: Words like 'graph', 'end', 'subgraph' are reserved. If you must use them in a label, enclose the label in quotes. Example: A --> B[\"end\"]\n"
            "• Node IDs: A node's ID must be a single, unbroken string. Example: MyNode[\"This is the label\"]\n"
            "• Special Characters: Use standard arrow syntax '-->' for connections. NEVER use --&gt; (HTML entities). For labels with special characters, quote them. Example: A --> B[\"API < 2.0\"]\n"
            "• Closing Subgraphs: Every 'subgraph' MUST be closed with a corresponding 'end' keyword.\n\n"
            "**TEMPLATE LIBRARY**\n\n"
            "**Full-Stack Application**\n"
            "Structural Template:\n"
            "graph TD\n"
            "    User[\"User\"]:::io\n"
            "    WebApp[\"Frontend\"]:::interface\n"
            "    Server[\"Backend API\"]:::logic\n"
            "    Database[\"Database\"]:::storage\n"
            "    PaymentGateway[\"Payment Gateway\"]:::external\n\n"
            "    User --> WebApp\n"
            "    WebApp --> Server\n"
            "    Server --> Database\n"
            "    Server -->|Processes Payment| PaymentGateway\n\n"
            "    classDef logic fill:#2563eb,stroke:#1d4ed8,color:#ffffff,stroke-width:2px\n"
            "    classDef io fill:#059669,stroke:#047857,color:#ffffff,stroke-width:2px\n"
            "    classDef interface fill:#7c3aed,stroke:#6d28d9,color:#ffffff,stroke-width:2px\n"
            "    classDef storage fill:#dc2626,stroke:#b91c1c,color:#ffffff,stroke-width:3px\n"
            "    classDef external fill:#6b7280,stroke:#4b5563,color:#ffffff,stroke-width:2px\n\n"
            "**API / Microservice Interaction**\n"
            "Structural Template:\n"
            "graph TD\n"
            "    ApiGateway[\"API Gateway\"]:::interface\n"
            "    ServiceA[\"Authentication Service\"]:::logic\n"
            "    ServiceB[\"Order Processing Service\"]:::logic\n"
            "    ThirdParty[\"3rd Party API\"]:::external\n\n"
            "    ApiGateway -->|Routes to| ServiceA\n"
            "    ApiGateway -->|Routes to| ServiceB\n"
            "    ServiceB -->|Fetches Data| ThirdParty\n\n"
            "    classDef logic fill:#2563eb,stroke:#1d4ed8,color:#ffffff,stroke-width:2px\n"
            "    classDef interface fill:#7c3aed,stroke:#6d28d9,color:#ffffff,stroke-width:2px\n"
            "    classDef external fill:#6b7280,stroke:#4b5563,color:#ffffff,stroke-width:2px\n\n"
            "**Database Schema**\n"
            "Structural Template:\n"
            "erDiagram\n"
            "    USERS {\n"
            "        int id PK\n"
            "        string username\n"
            "    }\n"
            "    POSTS {\n"
            "        int id PK\n"
            "        int user_id FK\n"
            "    }\n"
            "    USERS ||--o{ POSTS : \"writes\"\n\n"
            "**Internal Component Flow**\n"
            "Structural Template:\n"
            "graph TD\n"
            "    A_Request[\"Request\"]:::io\n"
            "    B_Controller[\"Controller\"]:::interface\n"
            "    C_Service[\"Service Logic\"]:::logic\n"
            "    D_Model[\"Data Model\"]:::storage\n\n"
            "    A_Request --> B_Controller\n"
            "    B_Controller --> C_Service\n"
            "    C_Service --> D_Model\n\n"
            "    classDef logic fill:#2563eb,stroke:#1d4ed8,color:#ffffff,stroke-width:2px\n"
            "    classDef io fill:#059669,stroke:#047857,color:#ffffff,stroke-width:2px\n"
            "    classDef interface fill:#7c3aed,stroke:#6d28d9,color:#ffffff,stroke-width:2px\n"
            "    classDef storage fill:#dc2626,stroke:#b91c1c,color:#ffffff,stroke-width:3px\n\n"
            "**CI/CD Pipeline**\n"
            "Structural Template:\n"
            "graph LR\n"
            "    A[\"Code Commit\"]:::io\n"
            "    B[\"Build Server\"]:::logic\n"
            "    C[\"Run Tests\"]:::test\n"
            "    D[\"Deploy to Staging\"]:::logic\n"
            "    E[\"Production\"]:::logic\n\n"
            "    A --> B\n"
            "    B --> C\n"
            "    C --> D\n"
            "    D --> E\n\n"
            "    classDef logic fill:#2563eb,stroke:#1d4ed8,color:#ffffff,stroke-width:2px\n"
            "    classDef io fill:#059669,stroke:#047857,color:#ffffff,stroke-width:2px\n"
            "    classDef test fill:#ea580c,stroke:#c2410c,color:#ffffff,stroke-width:2px\n\n"
            "**General System Architecture**\n"
            "Structural Template:\n"
            "graph TD\n"
            "    Input_Source[\"Input Source\"]:::io\n"
            "    Processing_Unit[\"Processing Unit\"]:::logic\n"
            "    Data_Store[\"Data Store\"]:::storage\n"
            "    System_Consumer[\"System Consumer\"]:::io\n\n"
            "    Input_Source --> Processing_Unit\n"
            "    Processing_Unit <--> Data_Store\n"
            "    Processing_Unit --> System_Consumer\n\n"
            "    classDef logic fill:#2563eb,stroke:#1d4ed8,color:#ffffff,stroke-width:2px\n"
            "    classDef io fill:#059669,stroke:#047857,color:#ffffff,stroke-width:2px\n"
            "    classDef storage fill:#dc2626,stroke:#b91c1c,color:#ffffff,stroke-width:3px"
        )
        
        # Add chat history if provided
        if chat_history:
            # Insert chat history before the current user message
            history_messages = []
            for msg in chat_history[-5:]:  # Last 5 messages
                if hasattr(msg, 'content'):
                    role = "user" if msg.type == "human" else "assistant"
                    history_messages.append({"role": role, "content": msg.content})
            
            # Insert history before current user message
            messages = [messages[0]] + history_messages + [messages[1]]
        
        # CONTEXTUAL PROMPT INJECTION: Add specific requirements for architecture queries
        if is_architecture_query:
            messages.append({
                "role": "system",
                "content": """**MANDATORY: COMPREHENSIVE ARCHITECTURE ANALYSIS PROTOCOL**

🚨 SYSTEM REQUIREMENT: You are conducting a mandatory architecture analysis. This is NOT optional.

**STRICT REQUIREMENTS - FAILURE TO COMPLY IS UNACCEPTABLE:**

1. **MINIMUM TOOL USAGE:** You MUST use AT LEAST 3 different tools before providing any final response
2. **MANDATORY SECTIONS:** Your response MUST include ALL of these sections:
   - **System Overview:** High-level summary of codebase purpose and structure
   - **Core Components:** Main components/modules/services with file names and code snippets
   - **Relationships & Data Flow:** Component interactions and data flow through system
   - **Key Patterns:** Design patterns and technologies used
3. **INVESTIGATION DEPTH:** You MUST examine multiple files, not just search results
4. **NO SHORTCUTS:** "Task completed" or brief responses are STRICTLY FORBIDDEN

**ENFORCEMENT:** Any attempt to provide a minimal response will be rejected. Continue investigating until you have comprehensive coverage of all required sections.

**TOOL USAGE MANDATE:** You must call smart_search, examine_files, AND analyze_relationships before attempting a final response."""
            })
        
        yield {"type": "context_gathering_start", "message": "Starting analysis with native Cerebras tools..."}
        
        # Multi-turn tool calling loop with duplicate detection
        # Dynamic iteration limit based on query complexity
        query_lower = user_query.lower()
        needs_diagrams = any(keyword in query_lower for keyword in ['mermaid', 'diagram', 'architecture', 'chart', 'visualization', 'flowchart'])
        max_iterations = 8 if needs_diagrams else 6  # Optimized to reduce rate limiting while maintaining quality
        iteration = 0
        recent_tool_calls = []  # Track recent tool calls to avoid repetition
        tool_call_count = 0  # Track total tool calls made
        synthesis_mode = False  # Track if we're in synthesis stage (no more tools allowed)
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🔄 ITERATION {iteration}/{max_iterations} (Tool calls so far: {tool_call_count})")
            
            try:
                # PHASE 3: Investigation Quality Gating - Fixed to allow initial tool usage
                investigation_score = self._calculate_investigation_completeness(recent_tool_calls, messages)
                
                # Smart decision making: check if we have enough information
                has_search_results = bool(self.tool_context.get('last_search_results'))
                has_examined_files = bool(self.tool_context.get('examined_files'))
                has_relationships = any('analyze_relationships' in call for call in recent_tool_calls)
                
                # Check if user explicitly requested diagrams
                user_wants_diagram = any(
                    keyword in user_query.lower() 
                    for keyword in ['diagram', 'chart', 'visual', 'mermaid', 'graph', 'flow', 'architecture diagram']
                )
                
                # CRITICAL FIX: Prevent tools during synthesis stage to avoid message pairing issues
                if synthesis_mode:
                    use_tools = None  # NO TOOLS during synthesis - prevents message structure corruption
                    logger.info(f"🚫 SYNTHESIS MODE: Tools disabled (synthesis_mode={synthesis_mode})")
                elif tool_call_count == 0:
                    use_tools = self.tools_schema  # Always allow tools initially
                    logger.info(f"🔧 TOOLS ENABLED: Initial iteration (tool_count={tool_call_count})")
                elif tool_call_count < 6:
                    # DYNAMIC THRESHOLD: Set higher threshold for architecture queries
                    investigation_threshold = 12 if is_architecture_query else 8
                    
                    if investigation_score < investigation_threshold:
                        use_tools = self.tools_schema
                        logger.info(f"🔧 TOOLS ENABLED: Continuing investigation (tool_count={tool_call_count}, score={investigation_score:.1f}, threshold={investigation_threshold})")
                    else:
                        use_tools = None
                        logger.info(f"🚫 TOOLS DISABLED: Investigation threshold reached (tool_count={tool_call_count}, score={investigation_score:.1f}, threshold={investigation_threshold})")
                elif user_wants_diagram and tool_call_count < 10:
                    # SPECIAL CASE: Allow all tools when user wants diagrams (don't be too restrictive)
                    use_tools = self.tools_schema
                    logger.info(f"🎨 DIAGRAM TOOLS ENABLED: User requested diagram, allowing all tools (tool_count={tool_call_count})")
                else:
                    use_tools = None
                    logger.info(f"🚫 TOOLS DISABLED: Quality threshold reached (tool_count={tool_call_count}, score={investigation_score:.1f})")
                    # Add markdown format instruction when ready for final answer
                    messages.append({
                        "role": "system", 
                        "content": f"You have completed your investigation. Now provide a COMPREHENSIVE, DETAILED final analysis. This should be thorough and extensive - include multiple sections, detailed explanations, code examples, architectural insights, and actionable recommendations. DO NOT provide brief responses. Use this format: {markdown_format_instruction}"
                    })
                
                # RATE LIMITING: Enforce 1.1s delay to prevent 429 errors  
                await self._enforce_rate_limit()
                
                # DEBUG: Log message structure before API call
                logger.info(f"🔍 DEBUG: About to make API call with {len(messages)} messages, tools={use_tools is not None}")
                for i, msg in enumerate(messages[-3:]):  # Log last 3 messages
                    msg_type = msg.get('role', 'unknown')
                    has_tools = 'tool_calls' in msg
                    has_tool_id = 'tool_call_id' in msg
                    logger.info(f"🔍 MSG[{len(messages)-3+i}]: {msg_type}, tools={has_tools}, tool_id={has_tool_id}")
                
                # Make API call to Cerebras
                api_params = {
                    "model": selected_model,
                    "messages": messages,
                    "max_tokens": 8000,  # Allow longer responses for comprehensive analysis
                    "temperature": 0.3   # Slightly more creative but still focused
                }
                if use_tools is not None:
                    api_params["tools"] = use_tools
                logger.info(f"🔍 API PARAMS: model={selected_model}, tools_provided={use_tools is not None}, synthesis_mode={synthesis_mode}")
                
                response = self.client.chat.completions.create(**api_params)
                # parallel_tool_calls parameter removed - not supported by GPT-OSS-120B
                
                choice = response.choices[0].message
                
                # Check if there are tool calls
                if choice.tool_calls:
                    tool_call_count += len(choice.tool_calls)
                    
                    # Add the assistant message with tool calls
                    messages.append({
                        "role": "assistant",
                        "content": choice.content or "",
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                            for tool_call in choice.tool_calls
                        ]
                    })
                    
                    # Collect system messages to add AFTER all tool responses (prevent API pairing issues)
                    pending_system_messages = []
                    
                    # Process each tool call
                    for tool_call in choice.tool_calls:
                        function_name = tool_call.function.name
                        function_args = tool_call.function.arguments
                        
                        # Track tool call for duplicate detection
                        call_signature = f"{function_name}:{function_args}"
                        recent_tool_calls.append(call_signature)
                        
                        # Keep only last 5 calls for duplicate detection
                        if len(recent_tool_calls) > 5:
                            recent_tool_calls.pop(0)
                        
                        # Log detailed tool call information
                        logger.info(f"🔧 TOOL CALL: {function_name}")
                        logger.info(f"Input: {function_args}")
                        
                        yield {
                            "type": "tool_start",
                            "tool": function_name,
                            "input": function_args
                        }
                        
                        # Check for repetitive tool calling - improved logic
                        call_count = recent_tool_calls.count(call_signature)
                        
                        # Allow legitimate 3-tool pattern: smart_search → examine_files → analyze_relationships
                        # Only block if same tool called 3+ times with same args, or excessive total calls
                        should_block = (
                            call_count >= 3 or  # Same call 3+ times
                            (tool_call_count >= 8 and call_count >= 2)  # Many calls + repetition
                        )
                        
                        if should_block:
                            warning_msg = f"Info: {function_name} has been called {call_count} times with same parameters. Skipping to avoid excessive repetition."
                            
                            # Special handling for mermaid_generator repetition
                            if function_name == "mermaid_generator":
                                # Try to repair and process the diagram one more time
                                logger.info("🔧 Final attempt to repair mermaid diagram after repetition...")
                                repaired_args = self._repair_truncated_json(function_args)
                                if repaired_args:
                                    try:
                                        arguments = json.loads(repaired_args)
                                        arguments = self._validate_and_fix_parameters(function_name, arguments)
                                        result = await self.available_functions[function_name](**arguments)
                                        logger.info("✅ Final repair attempt successful!")
                                    except Exception as e:
                                        result = f"❌ **Diagram Generation Issue**: Multiple attempts to generate the diagram encountered technical limitations. Architecture analysis completed successfully. Technical details: {str(e)}"
                                else:
                                    result = "❌ **Diagram Data Issue**: Could not process diagram parameters. Architecture analysis completed successfully."
                            else:
                                result = warning_msg
                        else:
                            # Normalize known aliases (e.g., 'functions.smart_search' -> 'smart_search')
                            normalized_name = function_name
                            if function_name.startswith("functions."):
                                normalized_name = function_name.split(".", 1)[1]

                            if normalized_name in self.available_functions:
                                function_name = normalized_name
                                try:
                                    # Parse arguments and call function
                                    arguments = json.loads(function_args)
                                    
                                    # Fix common parameter validation issues
                                    arguments = self._validate_and_fix_parameters(function_name, arguments)
                                    
                                    logger.info(f"🔧 PARSED ARGS: {arguments}")
                                    result = await self.available_functions[function_name](**arguments)
                                    
                                except json.JSONDecodeError as e:
                                    result = f"Error parsing {function_name} arguments: {str(e)}"
                                    logger.error(f"JSON decode error for {function_name}: {function_args}")
                                    
                                    # Special handling for truncated mermaid diagrams - repair the JSON
                                    if function_name == "mermaid_generator" and ("Expecting ',' delimiter" in str(e) or "Unterminated string" in str(e)):
                                        logger.info("🔧 Attempting to repair truncated mermaid JSON...")
                                        repaired_args = self._repair_truncated_json(function_args)
                                        if repaired_args:
                                            try:
                                                arguments = json.loads(repaired_args)
                                                arguments = self._validate_and_fix_parameters(function_name, arguments)
                                                result = await self.available_functions[function_name](**arguments)
                                                logger.info("✅ Successfully repaired and processed mermaid diagram")
                                            except Exception as repair_error:
                                                logger.error(f"❌ Repair attempt failed: {repair_error}")
                                                result = f"Error: Could not repair diagram data - {str(repair_error)}"
                                        else:
                                            result = "Error: Could not repair truncated diagram data"
                                except TypeError as e:
                                    result = f"Error executing {function_name}: {str(e)}"
                                    logger.error(f"Parameter error for {function_name}: {arguments} -> {str(e)}")
                                except Exception as e:
                                    result = f"Error executing {function_name}: {str(e)}"
                                    logger.error(f"General error for {function_name}: {str(e)}")
                            else:
                                # Suppress noisy unknown-tool messages in synthesis; provide silent no-op
                                result = ""
                        
                        # PHASE 1: Negative Result Fallback System - COLLECT instead of adding immediately
                        if self._detect_negative_result(result, function_name) and tool_call_count < 5:
                            fallback_guidance = self._get_fallback_guidance(function_name)
                            pending_system_messages.append({
                                "role": "system",
                                "content": f"The previous search returned no results. {fallback_guidance}. Continue investigating before providing final answer."
                            })
                        
                        # Log tool output (truncated for readability)
                        # Avoid logging empty no-op outputs
                        result_preview = (result[:300] + "...") if (isinstance(result, str) and len(result) > 300) else result
                        if result_preview == "":
                            result_preview = "(no output)"
                        logger.info(f"🔧 TOOL RESULT ({len(result)} chars): {result_preview}")
                        
                        # Task 5: Collect tool results for response formatting
                        tool_results_for_formatting.append({
                            'tool_name': function_name,
                            'arguments': arguments if 'arguments' in locals() else {},
                            'result': result,
                            'execution_time': 0.0,  # Could be measured if needed
                            'success': not result.startswith('Error'),
                            'results': []  # Could extract structured results if available
                        })
                        
                        # Avoid emitting empty no-op tool outputs to the UI
                        yield {
                            "type": "tool_end",
                            "output": result if result != "" else ""
                        }
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })
                    
                    # CRITICAL FIX: Add pending system messages AFTER all tool responses are processed
                    for system_msg in pending_system_messages:
                        messages.append(system_msg)
                        logger.info(f"📝 ADDED DELAYED SYSTEM MESSAGE: {system_msg['content'][:100]}...")
                    
                    # After 4+ tool calls, add guidance but allow continued investigation if needed
                    if tool_call_count >= 4:
                        messages.append({
                            "role": "system",
                            "content": f"You have made several tool calls and gathered substantial information. If you have found sufficient information, provide a COMPREHENSIVE, DETAILED final analysis with multiple sections, extensive explanations, code examples, and actionable insights. DO NOT provide brief responses - be thorough and extensive. If you need to investigate further with different approaches, continue, but focus on completing the investigation efficiently.\n\nWhen providing your final answer, use this format: {markdown_format_instruction}"
                        })
                
                else:
                    # No tool calls - we have the final response
                    final_content = choice.content or "Task completed"
                    logger.info(f"✅ FINAL RESPONSE ({len(final_content)} chars): {final_content[:500]}...")
                    
                    # FORCED TOOL CONTINUATION: Override model stopping for architecture queries
                    if is_architecture_query and tool_call_count < 3:
                        logger.warning(f"🚫 ARCHITECTURE OVERRIDE: AI attempted to stop after {tool_call_count} tools. Forcing continuation.")
                        messages.append({
                            "role": "assistant", 
                            "content": final_content
                        })
                        messages.append({
                            "role": "system",
                            "content": f"""**RESPONSE REJECTED - INSUFFICIENT INVESTIGATION**

You attempted to provide a final response after only {tool_call_count} tool calls. This violates the architecture analysis protocol.

**REQUIRED ACTIONS:**
- You MUST use at least 3 different tools (smart_search, examine_files, analyze_relationships)
- Current tools used: {tool_call_count}/3 minimum required
- Continue investigating immediately - do not provide another final response until requirements are met

**NEXT STEP:** Call examine_files to analyze the discovered files in detail."""
                        })
                        continue  # Force another iteration
                    
                    # CRITICAL FIX: Check max iterations BEFORE processing response
                    if iteration >= max_iterations:
                        logger.warning(f"🔄 MAX ITERATIONS REACHED: {iteration}/{max_iterations} - Forcing final response")
                        
                        execution_time = asyncio.get_event_loop().time() - execution_start_time
                        execution_metadata = {
                            'total_time': execution_time,
                            'iterations': iteration,
                            'tool_calls': tool_call_count,
                            'max_iterations_reached': True
                        }
                        
                        try:
                            formatted_response = self.response_formatter.format_response(
                                raw_response=final_content,
                                tool_results=tool_results_for_formatting,
                                query=user_query,
                                execution_metadata=execution_metadata
                            )
                            response_consolidator.add_response_data(
                                raw_output=final_content,
                                source=ResponseSource.MAX_ITERATIONS,
                                formatted_response=formatted_response.to_dict(),
                                execution_metadata=execution_metadata,
                                max_iterations_reached=True
                            )
                        except Exception as e:
                            logger.error(f"Response formatting failed in max iterations: {e}")
                            response_consolidator.add_response_data(
                                raw_output=final_content,
                                source=ResponseSource.MAX_ITERATIONS,
                                execution_metadata=execution_metadata,
                                max_iterations_reached=True,
                                error=f"Max iterations formatting failed: {e}"
                            )
                        break
                    
                    # PHASE 2: Incomplete Response Detection
                    if self._detect_incomplete_response(final_content) and tool_call_count < 5:
                        # Don't terminate - add follow-through prompt
                        messages.append({
                            "role": "system", 
                            "content": "You mentioned you would perform additional actions. Please follow through on what you just said you would do instead of providing a final answer."
                        })
                        continue  # Skip termination, continue loop
                    
                    # TWO-STAGE PROCESSING: Detect inadequate GPT-OSS-120B responses and trigger synthesis
                    if self._detect_inadequate_response(final_content, tool_call_count) and tool_call_count > 0:
                        logger.warning(f"⚠️ INADEQUATE RESPONSE DETECTED: '{final_content[:100]}...' after {tool_call_count} tool calls")
                        logger.info("🔄 TRIGGERING SYNTHESIS STAGE: Re-prompting with tool results")
                        
                        # CRITICAL FIX: Add the assistant's inadequate response to messages first
                        messages.append({
                            "role": "assistant",
                            "content": final_content
                        })
                        
                        # Compile all tool results for synthesis
                        tool_summary = self._compile_tool_results_summary(tool_results_for_formatting)
                        
                        synthesis_prompt = (
                            f"You called tools and gathered valuable information, but your response was inadequate. "
                            f"Please provide a comprehensive analysis of the original query based on the information you gathered:\n\n"
                            f"**ORIGINAL QUERY:** {user_query}\n\n"
                            f"**INFORMATION YOU GATHERED:**\n{tool_summary}\n\n"
                            f"Now provide a detailed, comprehensive response that actually uses this information to answer the user's query.\n\n"
                            f"RESPONSE FORMAT: {markdown_format_instruction}"
                        )
                        
                        messages.append({
                            "role": "system",
                            "content": synthesis_prompt
                        })
                        
                        # CRITICAL FIX: Enter synthesis mode to prevent additional tool calls
                        synthesis_mode = True
                        # Give synthesis mode extra iterations to complete
                        max_iterations = min(max_iterations + 2, 12)  # Add 2 more iterations for synthesis
                        logger.info(f"🚫 SYNTHESIS MODE ACTIVATED: No more tools allowed, extended max_iterations to {max_iterations}")
                        continue  # Skip termination, continue to synthesis stage
                    
                    # RESPONSE CONSOLIDATION: Collect all response data instead of yielding multiple final_result messages
                    execution_time = asyncio.get_event_loop().time() - execution_start_time
                    execution_metadata = {
                        'total_time': execution_time,
                        'iterations': iteration,
                        'tool_calls': tool_call_count,
                        'synthesis_triggered': synthesis_mode
                    }
                    
                    # UNIFIED PIPELINE: Always use unified response processing
                    try:
                        from unified_response_pipeline import get_unified_pipeline
                        
                        unified_pipeline = get_unified_pipeline()
                        unified_response = unified_pipeline.process_response(
                            raw_response=final_content,
                            tool_results=tool_results_for_formatting,
                            query=user_query,
                            execution_metadata=execution_metadata
                        )
                        
                        logger.info(f"🔄 AGENT: Unified pipeline success, pipeline={unified_response.processing_info.pipeline_used.value}")
                        
                        # Always use unified format - consistent output regardless of AI response format
                        response_consolidator.add_response_data(
                            raw_output=unified_response.content,
                            source=ResponseSource.UNIFIED,
                            formatted_response=unified_response.to_dict(),
                            execution_metadata=execution_metadata,
                            synthesis_triggered=synthesis_mode
                        )
                        
                        logger.info("✅ AGENT: Successfully processed unified response")
                        
                    except Exception as e:
                        logger.error(f"💥 UNIFIED PIPELINE FAILED: {e}")
                        # Emergency fallback to prevent complete failure
                        response_consolidator.add_response_data(
                            raw_output=final_content,
                            source=ResponseSource.RAW,
                            execution_metadata=execution_metadata,
                            synthesis_triggered=synthesis_mode,
                            error=f"Structured parsing failed: {e}"
                        )
                    break
                    
            except Exception as e:
                # RESPONSE CONSOLIDATION: Add API error to consolidator instead of yielding directly
                response_consolidator.add_error(f"Cerebras API error: {str(e)}", ResponseSource.RAW)
                break
        
        # POST-LOOP MAX ITERATIONS HANDLER: Handle case where loop exits due to max iterations without adequate response
        # Check for both no data AND inadequate responses like "Task completed"
        has_inadequate_response = False
        if response_consolidator.has_data():
            primary_source = response_consolidator.get_primary_source()
            if primary_source == ResponseSource.MAX_ITERATIONS:
                # Check if the response is inadequate
                for data in response_consolidator.response_data:
                    if data.source == ResponseSource.MAX_ITERATIONS:
                        content = data.raw_output.strip().lower()
                        if content in ['task completed', 'task completed.', 'task completed...'] or len(content) < 50:
                            has_inadequate_response = True
                            logger.warning(f"🔄 INADEQUATE MAX ITERATIONS RESPONSE: '{data.raw_output[:50]}...' - Will trigger synthesis")
                            break
        
        if iteration >= max_iterations and (not response_consolidator.has_data() or has_inadequate_response):
            logger.warning(f"🔄 MAX ITERATIONS POST-PROCESSING: Making final synthesis call (iteration {iteration}/{max_iterations})")
            
            # Only clear data if this is the first inadequate response (avoid overriding better attempts)
            synthesis_already_attempted = any(data.synthesis_triggered for data in response_consolidator.response_data)
            
            if has_inadequate_response and not synthesis_already_attempted:
                logger.info("🧹 CLEARING INADEQUATE RESPONSE DATA: First synthesis attempt")
                response_consolidator.response_data.clear()
            elif synthesis_already_attempted:
                logger.info("🔄 SYNTHESIS ALREADY ATTEMPTED: Keeping existing synthesis attempts, will only retry if empty")
                if response_consolidator.has_data():
                    logger.info("📦 KEEPING EXISTING SYNTHESIS DATA: Not performing redundant post-loop synthesis")
                    return  # Don't do post-loop synthesis if we already have synthesis data
            
            try:
                # Compile tool results for better synthesis prompt
                tool_summary = self._compile_tool_results_summary(tool_results_for_formatting)
                logger.info(f"🔍 SYNTHESIS DEBUG: tool_results count = {len(tool_results_for_formatting)}")
                logger.info(f"🔍 SYNTHESIS DEBUG: tool_summary length = {len(tool_summary)} chars")
                logger.info(f"🔍 SYNTHESIS DEBUG: tool_summary preview = {tool_summary[:200]}...")
                
                # Use the same detailed synthesis prompt as main loop
                synthesis_prompt = (
                    f"CRITICAL: You have completed a comprehensive investigation using multiple tools and reached the maximum iteration limit. "
                    f"You examined {len(tool_results_for_formatting)} tools and gathered extensive information. "
                    f"You MUST now provide a detailed, comprehensive final analysis - NOT just 'Analysis complete'.\n\n"
                    f"**ORIGINAL QUERY:** {user_query}\n\n"
                    f"**INFORMATION YOU GATHERED:**\n{tool_summary}\n\n"
                    f"REQUIREMENTS FOR YOUR RESPONSE:\n"
                    f"- Provide a comprehensive analysis of what you discovered\n"
                    f"- Include specific details from the files you examined\n"
                    f"- Answer the user's query thoroughly using the gathered information\n"
                    f"- Minimum 500 words - this is NOT optional\n"
                    f"- Structure your response with clear sections and findings\n\n"
                    f"DO NOT respond with just 'Analysis complete' or other brief responses.\n\n"
                    f"RESPONSE FORMAT: {markdown_format_instruction}"
                )
                
                # Prepare messages for synthesis - add comprehensive system prompt
                synthesis_messages = messages + [{
                    "role": "system",
                    "content": synthesis_prompt
                }]
                
                # Make final API call without tools to synthesize results
                logger.info(f"🔄 SYNTHESIS: Making final API call without tools")
                response = self.client.chat.completions.create(
                    model=selected_model,
                    messages=synthesis_messages,
                    tools=None,  # No tools for final synthesis
                    temperature=0.3,  # Slightly higher for more detailed responses
                    max_tokens=12000,  # Increased to allow for comprehensive analysis
                    timeout=180.0  # Longer timeout for detailed synthesis
                )
                
                if response.choices and response.choices[0].message:
                    final_content = response.choices[0].message.content or "Analysis complete based on investigation results"
                    logger.info(f"✅ SYNTHESIS: Received final response ({len(final_content)} chars)")
                    
                    # Calculate execution metadata
                    execution_time = asyncio.get_event_loop().time() - execution_start_time
                    execution_metadata = {
                        'total_time': execution_time,
                        'iterations': iteration,
                        'tool_calls': tool_call_count,
                        'max_iterations_reached': True,
                        'synthesis_triggered': True
                    }
                    
                    # Process through unified pipeline
                    from unified_response_pipeline import get_unified_pipeline
                    unified_response = get_unified_pipeline().process_response(
                        raw_response=final_content,
                        tool_results=tool_results_for_formatting,
                        query=user_query,
                        execution_metadata=execution_metadata
                    )
                    
                    # Add to consolidator with max iterations source
                    response_consolidator.add_response_data(
                        raw_output=unified_response.content,
                        source=ResponseSource.UNIFIED,
                        formatted_response=unified_response.to_dict(),
                        execution_metadata=execution_metadata,
                        max_iterations_reached=True,
                        synthesis_triggered=True
                    )
                    
                    logger.info("✅ SYNTHESIS: Successfully processed and added final response to consolidator")
                    
                else:
                    logger.error("💥 SYNTHESIS: No response content received from API")
                    response_consolidator.add_response_data(
                        raw_output="Error: No final response received from synthesis call",
                        source=ResponseSource.MAX_ITERATIONS,
                        execution_metadata={'max_iterations_reached': True},
                        max_iterations_reached=True,
                        error="No response content from synthesis API call"
                    )
                    
            except Exception as e:
                logger.error(f"💥 SYNTHESIS FAILED: {e}")
                # Emergency fallback to prevent complete failure
                response_consolidator.add_response_data(
                    raw_output=f"Error: Failed to synthesize final response - {str(e)}",
                    source=ResponseSource.MAX_ITERATIONS,
                    execution_metadata={'max_iterations_reached': True},
                    max_iterations_reached=True,
                    error=f"Synthesis API call failed: {str(e)}"
                )
        
        # RESPONSE CONSOLIDATION: Single yield point - consolidate all response data and yield once
        if response_consolidator.has_data():
            primary_source = response_consolidator.get_primary_source()
            logger.info(f"🔧 CONSOLIDATOR: Consolidating {len(response_consolidator.response_data)} data sources into single final_result")
            logger.info(f"🎯 CONSOLIDATOR: Primary source will be: {primary_source.value if primary_source else 'unknown'}")
            
            try:
                consolidated_response = await response_consolidator.consolidate(
                    original_query=user_query,
                    llm_provider=self  # Pass self as LLM provider for regeneration
                )
                
                # Enhanced logging for debugging
                metadata = consolidated_response.get('consolidation_metadata', {})
                logger.info(f"✅ CONSOLIDATOR: Successfully consolidated response")
                logger.info(f"   - Primary source: {metadata.get('primary_source', 'unknown')}")
                logger.info(f"   - Total sources: {metadata.get('total_sources', 0)}")
                logger.info(f"   - Sources used: {metadata.get('sources_used', [])}")
                logger.info(f"   - Synthesis triggered: {metadata.get('synthesis_triggered', False)}")
                logger.info(f"   - Max iterations: {metadata.get('max_iterations_reached', False)}")
                logger.info(f"   - Output length: {len(consolidated_response.get('output', ''))}")
                logger.info(f"   - Has structured: {'structured_response' in consolidated_response}")
                logger.info(f"   - Has formatted: {'formatted_response' in consolidated_response}")
                
                # CRITICAL DEBUG: Log the full actual response being yielded
                logger.info("🔍 FULL CONSOLIDATED RESPONSE BEING YIELDED:")
                logger.info("=" * 80)
                logger.info(f"RESPONSE TYPE: {consolidated_response.get('type', 'MISSING')}")
                logger.info(f"OUTPUT CONTENT ({len(consolidated_response.get('output', ''))} chars):")
                logger.info(f"'{consolidated_response.get('output', 'MISSING_OUTPUT')}'")
                
                # DEBUG: Check for corruption in final JSON payload to frontend (Step 9 diagnosis)
                final_output = consolidated_response.get('output', '')
                if '--&gt;' in final_output or '--&amp;gt;' in final_output or '--@gt' in final_output:
                    logger.error(f"🚨 CORRUPTION DETECTED IN FINAL JSON TO FRONTEND (STEP 9): {final_output}")
                else:
                    logger.info(f"✅ MERMAID DEBUG STEP 9 - No corruption detected in final output")
                
                if 'structured_response' in consolidated_response:
                    logger.info("STRUCTURED_RESPONSE PRESENT:")
                    logger.info(str(consolidated_response['structured_response'])[:500] + "..." if len(str(consolidated_response['structured_response'])) > 500 else str(consolidated_response['structured_response']))
                
                if 'formatted_response' in consolidated_response:
                    logger.info("FORMATTED_RESPONSE PRESENT:")
                    logger.info(str(consolidated_response['formatted_response'])[:500] + "..." if len(str(consolidated_response['formatted_response'])) > 500 else str(consolidated_response['formatted_response']))
                
                logger.info("CONSOLIDATION_METADATA:")
                logger.info(str(metadata))
                logger.info("=" * 80)
                
                if metadata.get('errors_encountered', 0) > 0:
                    logger.warning(f"⚠️ CONSOLIDATOR: Response contains {metadata['errors_encountered']} errors")
                
                yield consolidated_response
                
            except Exception as e:
                logger.error(f"💥 CONSOLIDATOR: Consolidation failed: {e}")
                yield {
                    "type": "final_result", 
                    "output": f"Error during response consolidation: {e}",
                    "consolidation_metadata": {
                        "primary_source": "consolidation_error",
                        "total_sources": len(response_consolidator.response_data),
                        "error": f"Consolidation failed: {e}"
                    }
                }
        else:
            logger.error("💥 CONSOLIDATOR: No response data collected during entire request processing")
            yield {
                "type": "final_result",
                "output": "Error: No response data was collected during processing. This may indicate a serious issue with the agent logic.",
                "consolidation_metadata": {
                    "primary_source": "no_data_error",
                    "total_sources": 0,
                    "error": "No response data collected"
                }
            } 