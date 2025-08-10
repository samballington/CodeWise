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
from directory_filters import (
    get_find_filter_args, get_grep_filter_args, should_include_file,
    filter_file_list, resolve_workspace_path, get_project_from_path
)
from project_context import (
    get_context_manager, set_project_context, get_current_context,
    filter_files_by_context
)
from enhanced_project_structure import EnhancedProjectStructure
from backend.smart_search import smart_search
from backend.path_resolver import PathResolver
from backend.response_formatter import ResponseFormatter, StandardizedResponse
from backend.json_prompt_schema import parse_json_prompt
from backend.json_prompt_postprocess import improve_json_prompt_readability
from backend.response_consolidator import ResponseConsolidator, ResponseSource
from backend.discovery_pipeline import DiscoveryPipeline
from backend.query_context_manager import QueryContextManager
from backend.table_generator import TableGenerator, StructuredTable, FileReference

logger = logging.getLogger(__name__)


class CerebrasNativeAgent:
    """Native Cerebras agent with proper tool calling support"""
    
    def __init__(self, api_key: str, mcp_server_url: str):
        self.client = Cerebras(api_key=api_key, timeout=30.0)  # 30 second timeout to prevent hanging
        self.mcp_server_url = mcp_server_url
        self.tools_schema = self._create_tools_schema()
        self.available_functions = self._create_function_mapping()
        self.current_mentioned_projects = None  # Store current project context
        
        # Rate limiting: Enforced as 1 request per second (not just 30/minute window)  
        # Use 1.1s for safety buffer
        self.min_request_interval = 1.1
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
        
        # Tool context for result chaining
        self.tool_context = {
            'last_search_results': None,
            'discovered_files': [],
            'query_intent': None,
            'main_files': [],
            'entities_found': []
        }
    
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
        
        # Check for clearly inadequate responses
        inadequate_patterns = [
            "task completed",
            "done",
            "completed", 
            "finished"
        ]
        
        response_lower = response_text.lower().strip()
        
        # Short response with tools called is likely inadequate
        if len(response_text.strip()) < 50 and tool_call_count > 0:
            return True
            
        # Check for generic completion messages
        if any(pattern in response_lower for pattern in inadequate_patterns):
            return True
            
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
        """Create simplified 3-tool architecture schema with smart tools"""
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
            }
        ]
    
    def _create_function_mapping(self) -> Dict[str, callable]:
        """Map function names to simplified 3-tool implementations"""
        return {
            "smart_search": self._smart_search_router,  # Context-aware router
            "examine_files": self._examine_files,
            "analyze_relationships": self._analyze_relationships
        }
    
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
                        # Show first 20 and last 10 lines with syntax highlighting
                        if line_count <= 30:
                            highlighted_content = self._apply_syntax_highlighting(content, full_path.suffix)
                            results.append(f"   📝 CODE CONTENT:\n{highlighted_content}")
                        else:
                            head = '\n'.join(lines[:20])
                            tail = '\n'.join(lines[-10:])
                            highlighted_head = self._apply_syntax_highlighting(head, full_path.suffix)
                            highlighted_tail = self._apply_syntax_highlighting(tail, full_path.suffix)
                            results.append(f"   📝 HEAD (20 lines):\n{highlighted_head}")
                            results.append(f"\n   ... ({line_count - 30} lines omitted) ...")
                            results.append(f"\n   📝 TAIL (10 lines):\n{highlighted_tail}")
                    
                    elif detail_level == "full":
                        # Show complete content with syntax highlighting (truncated if very large)
                        if file_size > 10000:  # 10KB limit
                            truncated_content = content[:10000] + "\n... (content truncated)"
                            highlighted_content = self._apply_syntax_highlighting(truncated_content, full_path.suffix)
                            results.append(f"   📝 CODE CONTENT (truncated):\n{highlighted_content}")
                        else:
                            highlighted_content = self._apply_syntax_highlighting(content, full_path.suffix)
                            results.append(f"   📝 CODE CONTENT:\n{highlighted_content}")
                    
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
    
    async def process_request(self, user_query: str, chat_history=None, mentioned_projects: List[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
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
                mentioned_projects
            ):
                yield result
    
    async def _process_request_with_context(
        self, 
        user_query: str, 
        query_context, 
        chat_history=None, 
        mentioned_projects: List[str] = None
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
        
        # Build system prompt with simplified 3-tool guidance
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
                    "\n\n🎯 EXCEPTIONAL RESPONSE STANDARDS - COMPREHENSIVE BY DEFAULT:"
                    "\n• ALWAYS PROVIDE COMPREHENSIVE ANALYSIS: Every response should be thorough, detailed, and complete - never give brief or surface-level answers"
                    "\n• EXPLAIN EVERYTHING: Assume the user wants to understand the full context, background, and implications of every aspect"
                    "\n• MULTI-LAYERED TECHNICAL ANALYSIS: Include specific code examples, file paths, implementation details, and architectural context"
                    "\n• ARCHITECTURAL DEPTH: Explain design patterns, architectural decisions, data flow, and system boundaries with concrete evidence"
                    "\n• CODE EVIDENCE: Always include relevant code snippets with proper syntax highlighting and detailed explanations"
                    "\n• TECHNICAL REASONING: Explain the 'why' behind implementation choices, trade-offs, and architectural decisions"
                    "\n• IMPLEMENTATION DETAILS: Cover error handling, edge cases, performance considerations, and security implications"
                    "\n• DEPENDENCY ANALYSIS: Map relationships between components, external dependencies, and system interactions"
                    "\n• ACTIONABLE INSIGHTS: Provide specific recommendations for improvements, optimizations, and best practices"
                    "\n• MULTIPLE PERSPECTIVES: Analyze from developer, architect, security, performance, and maintainability viewpoints"
                    "\n• STRUCTURED RESPONSES: Organize with clear sections using markdown formatting for readability"
                    "\n• COMPLETE COVERAGE: Address all aspects of the query with thorough analysis and supporting evidence"
                    "\n• DEFAULT TO VERBOSE: When in doubt, provide more detail rather than less - users can always ask for summaries if needed"
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
                    "\n\n📦 STRUCTURED OUTPUT (REQUIRED WHEN PRESENTING TABLES OR HIERARCHIES):"
                    "\nWhen you present tabular data (e.g., dependency tables) or hierarchical trees (e.g., parent/child POMs), ALSO include a fenced JSON block conforming to this schema so the UI can render it natively:"
                    "\n```json"
                    "\n{"
                    "\n  \"version\": \"codewise_structured_v1\","
                    "\n  \"tables\": ["
                    "\n    { \"title\": string, \"columns\": [string, ...], \"rows\": [[string|number|null, ...]], \"note\": string? }"
                    "\n  ],"
                    "\n  \"trees\": ["
                    "\n    { \"title\": string, \"root\": { \"label\": string, \"children\": [ {\"label\": string, \"children\": [...] } ] } }"
                    "\n  ],"
                    "\n  \"references\": [ { \"path\": string, \"line_start\": number?, \"line_end\": number? } ]"
                    "\n}"
                    "\n```"
                    "\nRules: The JSON must be valid and appear exactly once. Keep Markdown prose separate; do not render ASCII tables if JSON is provided."
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

        # Store JSON prompt instruction to apply AFTER tool calling phase
        json_prompt_instruction = (
            "You must respond with VALID JSON only (no markdown). Use this exact envelope: "
            "{ \"response\": { \"metadata\": { \"query_type\": \"architecture|dependencies|database|general\", \"confidence\": 0.0 }, "
            "\"sections\": [], \"follow_up_suggestions\": [] } }. "
            "The output MUST begin with '{' as the first character and end with '}'. Do not prepend headings, prose, or code fences. "
            "Sections must be typed using: paragraph, heading(level 1-6), table(columns, rows, note), list(style bullet|numbered), "
            "code_block(language, content), callout(style info|warning|error|success), tree(root.label, root.children), diagram(format mermaid, content). "
            "If you include a diagram, you MUST use a dedicated diagram section: {\"type\":\"diagram\", \"format\":\"mermaid\", \"content\":\"graph TD;subgraph FE[\\\"Frontend\\\"];UI[\\\"Component\\\"];end\"}. Do NOT emit markdown or code fences for diagrams. "
            "CRITICAL: The diagram content is a JSON string. ESCAPE ALL internal double quotes as \\\" (e.g., UI[\\\"React Canvas\\\"]). Use semicolons (;) as line delimiters for proper formatting. "
            "Do not include markdown in contents; use the section types. "
            "\n\n**🧠 MERMAID DIAGRAM GENERATION GUIDELINES**\n"
            "**IMPORTANT: Only generate diagrams when explicitly requested by the user OR when a visual representation would be significantly more helpful than text for explaining architecture, relationships, or complex flows. Do not create diagrams for simple explanations that can be communicated clearly with text and tables.**\n\n"
            "You are an expert AI assistant specializing in generating styled Mermaid.js diagrams to visualize software architecture. Your primary function is to analyze a user's request, select the most appropriate architectural template from the library below, and then populate it with the specific details provided.\n\n"
            "**CRITICAL INSTRUCTIONS: YOU MUST FOLLOW THIS PROCESS EXACTLY.**\n\n"
            "**ANALYZE AND CLASSIFY:** First, analyze the user's request to determine which of the following scenarios it best fits:\n"
            "• Full-Stack Application: A high-level overview of a complete application.\n"
            "• API / Microservice Interaction: The communication flow between different services or APIs.\n"
            "• Database Schema: The structure and relationships of database tables.\n"
            "• Internal Component Flow: A detailed look at how components within a single service interact.\n"
            "• CI/CD Pipeline: The flow of code from commit to deployment.\n"
            "• General System Architecture: A flexible, high-level diagram for requests that don't fit other categories.\n\n"
            "**SELECT MANDATORY TEMPLATE:** Once you have classified the request, you MUST select the corresponding template from the Template Library below. This is your required starting point. DO NOT START FROM SCRATCH.\n\n"
            "**MODIFY THE TEMPLATE:** Your task is to adapt the chosen template to perfectly match the user's request.\n"
            "• Rename the generic nodes in the template to match the entities in the user's request.\n"
            "• Add new nodes and define their relationships using arrows (-->).\n"
            "• Remove any nodes from the template that are not relevant to the user's specific scenario.\n"
            "• Create multiple subgraph blocks to logically group different parts of the architecture and draw connections between them.\n"
            "• Optionally, add text to arrows to describe the relationship (e.g., -->|Uses|, -->|Sends Data To|). This helps clarify the interaction between components.\n"
            "• You MUST keep the existing classDef styling and apply the appropriate classes to new nodes you create.\n\n"
            "**SYNTAX REQUIREMENTS:** You MUST generate only valid Mermaid.js syntax inside a single code block. Do not include any explanatory text or conversational filler.\n\n"
            "**CRITICAL SYNTAX RULES (MUST FOLLOW):**\n"
            "• **Subgraph Syntax**: Use `subgraph \"Title\"` NOT `subgraph Title[\"Label\"]`. Example: `subgraph \"Frontend\"` is CORRECT.\n"
            "• **Nested Quotes**: Use `&quot;` for inner quotes inside node labels. Example: `Node[\"Label &quot;with quotes&quot;\"]` is CORRECT.\n"
            "• **Node Labels**: If a label contains double quotes, escape them with `&quot;`. Example: `FalAI[\"Fal.AI &quot;Service&quot;\"]`\n"
            "• **Consistent Escaping**: Always escape ALL inner double quotes within node and edge labels using `&quot;`\n\n"
            "**TEMPLATE LIBRARY (CHOOSE ONE AND BUILD UPON IT)**\n\n"
            "**Template 1: Full-Stack Application** (Use for high-level, end-to-end application views)\n"
            "graph TD\n    %% --- Style Definitions ---\n    classDef userStyle fill:#99d98c,stroke:#333,stroke-width:2px\n    classDef frontendStyle fill:#76c893,stroke:#333,stroke-width:2px\n    classDef backendStyle fill:#52b69a,stroke:#333,stroke-width:2px\n    classDef dbStyle fill:#34a0a4,stroke:#333,stroke-width:2px\n    classDef externalStyle fill:#d9ed92,stroke:#333,stroke-width:2px\n\n    %% --- Core Structure ---\n    User([User]):::userStyle\n\n    subgraph \\\"Primary Application\\\"\n        direction LR\n        WebApp[Frontend]:::frontendStyle\n        Server[Backend API]:::backendStyle\n        Database[(Database)]:::dbStyle\n    end\n    \n    subgraph \\\"External Services\\\"\n        PaymentGateway[(Payment Gateway)]:::externalStyle\n    end\n\n    %% --- Relationships ---\n    User --> WebApp\n    WebApp --> Server\n    Server --> Database\n    Server -->|Processes Payment| PaymentGateway\n\n"
            "**Template 2: API / Microservice Interaction** (Use for showing how different services communicate)\n"
            "graph TD\n    %% --- Style Definitions ---\n    classDef apiStyle fill:#1a759f,stroke:#333,stroke-width:2px,color:#fff\n    classDef serviceStyle fill:#184e77,stroke:#333,stroke-width:2px,color:#fff\n    classDef externalStyle fill:#d9ed92,stroke:#333,stroke-width:2px\n\n    %% --- Core Structure ---\n    ApiGateway[API Gateway]:::apiStyle\n\n    subgraph \\\"User Service\\\"\n        ServiceA{{Authentication}}:::serviceStyle\n    end\n    \n    subgraph \\\"Order Service\\\"\n        ServiceB{{Order Processing}}:::serviceStyle\n    end\n\n    ThirdParty[(3rd Party API)]:::externalStyle\n\n    %% --- Relationships ---\n    ApiGateway -->|Routes to| ServiceA\n    ApiGateway -->|Routes to| ServiceB\n    ServiceB -->|Fetches Data| ThirdParty\n\n"
            "**Template 3: Database Schema** (Use for visualizing database tables and their relationships)\n"
            "erDiagram\n    %% --- Table Definitions ---\n    USERS {\n        int id PK\n        string username\n        string email\n    }\n    POSTS {\n        int id PK\n        string title\n        string content\n        int user_id FK\n    }\n    COMMENTS {\n        int id PK\n        string text\n        int post_id FK\n        int user_id FK\n    }\n\n    %% --- Relationships ---\n    USERS ||--o{ POSTS : \\\"writes\\\"\n    POSTS ||--o{ COMMENTS : \\\"has\\\"\n    USERS ||--o{ COMMENTS : \\\"writes\\\"\n\n"
            "**Template 4: Internal Component Flow** (Use for detailed views inside a single application or service)\n"
            "graph TD\n    %% --- Style Definitions ---\n    classDef entrypointStyle fill:#ef476f,stroke:#333,stroke-width:2px,color:white\n    classDef controllerStyle fill:#f78c6b,stroke:#333,stroke-width:2px\n    classDef serviceStyle fill:#ffd166,stroke:#333,stroke-width:2px\n    classDef modelStyle fill:#06d6a0,stroke:#333,stroke-width:2px\n\n    %% --- Core Structure ---\n    subgraph \\\"Backend Logic Flow\\\"\n        direction LR\n        A_Request[Request]\n        B_Controller{Controller}\n        C_Service[Service Logic]\n        D_Model[(Data Model)]\n    end\n\n    %% --- Relationships ---\n    A_Request --> B_Controller\n    B_Controller --> C_Service\n    C_Service --> D_Model\n\n    %% --- Apply Styles ---\n    class A_Request entrypointStyle\n    class B_Controller controllerStyle\n    class C_Service serviceStyle\n    class D_Model modelStyle\n\n"
            "**Template 5: CI/CD Pipeline** (Use for visualizing code deployment flows)\n"
            "graph LR\n    %% --- Style Definitions ---\n    classDef vcsStyle fill:#fca311,stroke:#333,stroke-width:2px\n    classDef buildStyle fill:#14213d,stroke:#333,stroke-width:2px,color:white\n    classDef testStyle fill:#5a189a,stroke:#333,stroke-width:2px,color:white\n    classDef deployStyle fill:#008000,stroke:#333,stroke-width:2px,color:white\n\n    %% --- Core Structure ---\n    subgraph \\\"CI/CD Pipeline\\\"\n        A(Code Commit) --> B{Build Server}\n        B --> C(Run Tests)\n        C -- On Success --> D[Deploy to Staging]\n        D --> E((Production))\n    end\n\n    %% --- Apply Styles ---\n    class A vcsStyle\n    class B buildStyle\n    class C testStyle\n    class D,E deployStyle\n\n"
            "**Template 6: General System Architecture** (Use as a flexible, high-level template for various architectures)\n"
            "graph TD\n    %% --- Style Definitions ---\n    classDef sourceStyle fill:#8ecae6,stroke:#333,stroke-width:2px\n    classDef processStyle fill:#219ebc,stroke:#333,stroke-width:2px,color:white\n    classDef dataStyle fill:#023047,stroke:#333,stroke-width:2px,color:white\n    classDef consumerStyle fill:#ffb703,stroke:#333,stroke-width:2px\n\n    %% --- Core Structure ---\n    subgraph \\\"Data Ingestion\\\"\n        Input_Source([Input Source]):::sourceStyle\n    end\n\n    subgraph \\\"Core Processing\\\"\n        Processing_Unit[/Processing Unit/]:::processStyle\n        Data_Store[(Data Store)]:::dataStyle\n    end\n    \n    subgraph \\\"Data Consumption\\\"\n        System_Consumer([System Consumer]):::consumerStyle\n    end\n\n    %% --- Relationships ---\n    Input_Source --> Processing_Unit\n    Processing_Unit <--> Data_Store\n    Processing_Unit --> System_Consumer"
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
        
        yield {"type": "context_gathering_start", "message": "Starting analysis with native Cerebras tools..."}
        
        # Multi-turn tool calling loop with duplicate detection
        max_iterations = 10  # Increased for more comprehensive responses
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
                
                # CRITICAL FIX: Prevent tools during synthesis stage to avoid message pairing issues
                if synthesis_mode:
                    use_tools = None  # NO TOOLS during synthesis - prevents message structure corruption
                    logger.info(f"🚫 SYNTHESIS MODE: Tools disabled (synthesis_mode={synthesis_mode})")
                elif tool_call_count == 0:
                    use_tools = self.tools_schema  # Always allow tools initially
                    logger.info(f"🔧 TOOLS ENABLED: Initial iteration (tool_count={tool_call_count})")
                elif tool_call_count < 6 and investigation_score < 8:  # More permissive threshold
                    use_tools = self.tools_schema
                    logger.info(f"🔧 TOOLS ENABLED: Continuing investigation (tool_count={tool_call_count}, score={investigation_score:.1f})")
                else:
                    use_tools = None
                    logger.info(f"🚫 TOOLS DISABLED: Quality threshold reached (tool_count={tool_call_count}, score={investigation_score:.1f})")
                    # Add JSON format instruction when ready for final answer
                    messages.append({
                        "role": "system", 
                        "content": f"You have completed your investigation. Now provide a COMPREHENSIVE, DETAILED final analysis. This should be thorough and extensive - include multiple sections, detailed explanations, code examples, architectural insights, and actionable recommendations. DO NOT provide brief responses. Use this format: {json_prompt_instruction}"
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
                    "model": "gpt-oss-120b",
                    "messages": messages,
                    "max_tokens": 8000,  # Allow longer responses for comprehensive analysis
                    "temperature": 0.3   # Slightly more creative but still focused
                }
                if use_tools is not None:
                    api_params["tools"] = use_tools
                logger.info(f"🔍 API PARAMS: tools_provided={use_tools is not None}, synthesis_mode={synthesis_mode}")
                
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
                            "content": f"You have made several tool calls and gathered substantial information. If you have found sufficient information, provide a COMPREHENSIVE, DETAILED final analysis with multiple sections, extensive explanations, code examples, and actionable insights. DO NOT provide brief responses - be thorough and extensive. If you need to investigate further with different approaches, continue, but focus on completing the investigation efficiently.\n\nWhen providing your final answer, use this format: {json_prompt_instruction}"
                        })
                
                else:
                    # No tool calls - we have the final response
                    final_content = choice.content or "Task completed"
                    logger.info(f"✅ FINAL RESPONSE ({len(final_content)} chars): {final_content[:500]}...")
                    
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
                            f"RESPONSE FORMAT: {json_prompt_instruction}"
                        )
                        
                        messages.append({
                            "role": "system",
                            "content": synthesis_prompt
                        })
                        
                        # CRITICAL FIX: Enter synthesis mode to prevent additional tool calls
                        synthesis_mode = True
                        logger.info("🚫 SYNTHESIS MODE ACTIVATED: No more tools allowed to prevent message pairing issues")
                        continue  # Skip termination, continue to synthesis stage
                    
                    # RESPONSE CONSOLIDATION: Collect all response data instead of yielding multiple final_result messages
                    execution_time = asyncio.get_event_loop().time() - execution_start_time
                    execution_metadata = {
                        'total_time': execution_time,
                        'iterations': iteration,
                        'tool_calls': tool_call_count,
                        'synthesis_triggered': synthesis_mode
                    }
                    
                    # CRITICAL FIX: Single pipeline selection based on response type
                    try:
                        structured = parse_json_prompt(final_content)
                        if structured is not None:
                            # JSON Pipeline ONLY - for structured responses
                            logger.info("📊 AGENT: Detected structured JSON response - using JSON pipeline")
                            
                            try:
                                improved = improve_json_prompt_readability(structured.model_dump())
                                response_consolidator.add_response_data(
                                    raw_output=final_content,
                                    source=ResponseSource.STRUCTURED,
                                    structured_response=improved,
                                    execution_metadata=execution_metadata,
                                    synthesis_triggered=synthesis_mode
                                )
                                logger.info("✅ AGENT: Successfully processed structured response")
                            except Exception as e:
                                logger.error(f"💥 AGENT: Structured response processing failed: {e}")
                                # Fallback to raw response
                                response_consolidator.add_response_data(
                                    raw_output=final_content,
                                    source=ResponseSource.RAW,
                                    execution_metadata=execution_metadata,
                                    synthesis_triggered=synthesis_mode,
                                    error=f"Structured processing failed: {e}"
                                )
                        else:
                            # Markdown Pipeline ONLY - for non-structured responses
                            logger.info("📊 AGENT: Detected markdown response - using markdown pipeline")
                            
                            try:
                                formatted_response = self.response_formatter.format_response(
                                    raw_response=final_content,
                                    tool_results=tool_results_for_formatting,
                                    query=user_query,
                                    execution_metadata=execution_metadata
                                )
                                response_consolidator.add_response_data(
                                    raw_output=final_content,
                                    source=ResponseSource.FORMATTED,
                                    formatted_response=formatted_response.to_dict(),
                                    execution_metadata=execution_metadata,
                                    synthesis_triggered=synthesis_mode
                                )
                                logger.info("✅ AGENT: Successfully processed formatted response")
                            except Exception as e:
                                logger.error(f"💥 AGENT: Response formatting failed: {e}")
                                # Fallback to raw response
                                response_consolidator.add_response_data(
                                    raw_output=final_content,
                                    source=ResponseSource.RAW,
                                    execution_metadata=execution_metadata,
                                    synthesis_triggered=synthesis_mode,
                                    error=f"Formatting failed: {e}"
                                )
                    except Exception as e:
                        logger.error(f"Structured response parsing failed: {e}")
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