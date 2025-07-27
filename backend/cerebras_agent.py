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

logger = logging.getLogger(__name__)


class CerebrasNativeAgent:
    """Native Cerebras agent with proper tool calling support"""
    
    def __init__(self, api_key: str, mcp_server_url: str):
        self.client = Cerebras(api_key=api_key)
        self.mcp_server_url = mcp_server_url
        self.tools_schema = self._create_tools_schema()
        self.available_functions = self._create_function_mapping()
        self.current_mentioned_projects = None  # Store current project context
        
        # Initialize enhanced project structure analyzer
        self.enhanced_structure = EnhancedProjectStructure(self._call_mcp_tool_wrapper)
    
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
            "list_entities": ["no entity-like files found", "no entities found", "entity scan complete: no"],
            "code_search": ["no matches", "no results found", "not found"],
            "file_glimpse": ["file not found", "does not exist"],
            "read_file": ["file not found", "permission denied"],
            "search_by_extension": ["no .* files found"],
            "search_file_content": ["no matches found"]
        }
        
        patterns = negative_patterns.get(function_name, ["not found", "no results"])
        return any(pattern in result_text.lower() for pattern in patterns)
    
    def _get_fallback_guidance(self, function_name: str) -> str:
        """Get fallback strategy guidance for negative results (Phase 1 fix)"""
        fallback_strategies = {
            "list_entities": "Try code_search for 'entity', 'model', 'database', '@Entity' keywords",
            "code_search": "Try different search terms or examine src/main directories with file_glimpse",
            "file_glimpse": "Try reading complete files with read_file or search in parent directories",
            "search_by_extension": "Try alternative file extensions or use search_file_content for patterns",
            "search_file_content": "Try broader search terms or different file types"
        }
        
        return fallback_strategies.get(function_name, "Try alternative approaches or different search terms")
    
    def _detect_incomplete_response(self, response_text: str) -> bool:
        """Detect if response indicates intention to do more work (Phase 2 fix)"""
        continuation_signals = [
            "let me try", "i will", "let me search", "i'll examine", 
            "let me check", "i'll look", "let me find", "i will search",
            "let me explore", "i'll investigate", "let me analyze"
        ]
        return any(signal in response_text.lower() for signal in continuation_signals)
    
    def _calculate_investigation_completeness(self, tool_calls_made: List[str], messages: List[Dict]) -> float:
        """Calculate investigation quality score to prevent premature termination (Phase 3 fix)"""
        score = 0.0
        
        # Extract function names from tool call signatures
        tool_types = set()
        for call in tool_calls_made:
            if ':' in call:
                func_name = call.split(':')[0]
                tool_types.add(func_name)
        
        # Points for tool diversity (encourages using different approaches)
        score += len(tool_types) * 1.5
        
        # Check recent results for negative outcomes
        recent_results = []
        for msg in messages[-10:]:  # Check last 10 messages
            if msg.get('role') == 'tool':
                recent_results.append(msg.get('content', ''))
        
        # Deduct for negative results without alternatives
        negative_results = sum(1 for result in recent_results if "not found" in result.lower() or "no matches" in result.lower())
        if negative_results > 0 and len(tool_types) < 3:
            score -= 2
        
        # Bonus for comprehensive investigation patterns
        if "code_search" in tool_types and ("read_file" in tool_types or "file_glimpse" in tool_types):
            score += 2
        
        # Bonus for following up on entity searches with alternative approaches
        if "list_entities" in tool_types and ("search_file_content" in tool_types or "search_by_extension" in tool_types):
            score += 1.5
        
        return score
    
    def _create_tools_schema(self) -> List[Dict[str, Any]]:
        """Create Cerebras-compatible tool schemas with strict=True"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "code_search",
                    "strict": True,
                    "description": "Search the codebase for relevant code snippets using hybrid search. Use this to find specific functions, classes, or implementation details.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant code"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "file_glimpse",
                    "strict": True,
                    "description": "Get a quick view of a file showing the first and last 20 lines. Use this to inspect specific files found through code search.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to glimpse"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_entities",
                    "strict": True,
                    "description": "List files that likely contain database entities or schema definitions. Use this specifically when looking for database entities.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Optional pattern to filter entities (default: searches for common entity patterns)"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "strict": True,
                    "description": "Read the complete contents of a specific file. Use this when you need full file content after finding it through other tools.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to read"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_by_extension",
                    "strict": True,
                    "description": "Search for files with a specific extension (e.g., .java, .py, .sql). Use this to find all files of a particular type.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "extension": {
                                "type": "string",
                                "description": "The file extension to search for (e.g., .java, .py, .sql)"
                            }
                        },
                        "required": ["extension"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_file_content",
                    "strict": True,
                    "description": "Search for specific text patterns within files across the codebase. Returns matches with context lines.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "The text pattern to search for"
                            },
                            "file_extension": {
                                "type": "string",
                                "description": "Optional: file extension to limit search (e.g., 'py', 'js')"
                            }
                        },
                        "required": ["pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_project_structure",
                    "strict": True,
                    "description": "Generate enhanced project structure analysis with framework detection, @ annotations for codebase highlighting, entry point identification, and context awareness.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Optional: specific project name (e.g., 'Gymmy', 'finance_RAG'). If not specified, analyzes workspace root."
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_related_files",
                    "strict": True,
                    "description": "Find files related to a given file through imports, tests, configs, and naming patterns.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The file path to find relationships for"
                            },
                            "relationship_type": {
                                "type": "string",
                                "description": "Type of relationship to find: 'import', 'test', 'config', or 'all'",
                                "enum": ["import", "test", "config", "all"]
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            }
        ]
    
    def _create_function_mapping(self) -> Dict[str, callable]:
        """Map function names to actual implementations"""
        return {
            "code_search": self._code_search,
            "file_glimpse": self._file_glimpse,
            "list_entities": self._list_entities,
            "read_file": self._read_file,
            "search_by_extension": self._search_by_extension,
            "search_file_content": self._search_file_content,
            "get_project_structure": self._get_project_structure,
            "find_related_files": self._find_related_files
        }
    
    async def _code_search(self, query: str = None, **kwargs) -> str:
        """Search codebase for relevant snippets"""
        # Defensive parameter handling for malformed tool calls
        if query is None:
            if kwargs:
                # Try to extract query from other possible parameter names
                query = kwargs.get('q', kwargs.get('search', kwargs.get('term', '')))
            if not query:
                return "Error: No search query provided. Please provide a query parameter."
        
        try:
            # Import here to avoid circular imports
            from backend.hybrid_search import HybridSearchEngine
            
            search_engine = HybridSearchEngine()
            results = search_engine.search(query, k=10)
            
            # Apply project filtering if mentioned projects are specified
            if self.current_mentioned_projects:
                results = self._filter_results_by_projects(results, self.current_mentioned_projects)
            
            if not results:
                filter_msg = f" (filtered for projects: {self.current_mentioned_projects})" if self.current_mentioned_projects else ""
                return f"No results found for the search query{filter_msg}."
            
            formatted_results = []
            for result in results:
                formatted_results.append(f"FILE: {result.file_path}\n{result.snippet}\n---")
            
            filter_info = f"\n\n[Filtered for projects: {', '.join(self.current_mentioned_projects)}]" if self.current_mentioned_projects else ""
            return "\n".join(formatted_results) + filter_info
            
        except Exception as e:
            return f"Error during code search: {str(e)}"
    
    async def _file_glimpse(self, file_path: str) -> str:
        """Get first and last 20 lines of a file"""
        try:
            workspace_path = Path('/workspace')
            full_path = workspace_path / file_path
            
            if not full_path.exists():
                return f"File not found: {file_path}"
            
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            if len(lines) <= 40:
                return f"--- COMPLETE FILE: {file_path} ---\n{''.join(lines)}"
            
            head = ''.join(lines[:20])
            tail = ''.join(lines[-20:])
            
            return f"--- START OF {file_path} ---\n{head}\n... ({len(lines)-40} lines omitted) ...\n{tail}--- END OF {file_path} ---"
            
        except Exception as e:
            return f"Error reading file {file_path}: {str(e)}"
    
    async def _search_by_extension(self, extension: str) -> str:
        """Search for files with a specific extension"""
        try:
            import subprocess
            
            # Clean the extension (remove leading dots and wildcards)
            clean_ext = extension.lstrip("*.")
            
            # Build find command to search workspace
            command = [
                "find", "/workspace", 
                "-name", f"*.{clean_ext}", 
                "-type", "f"
            ]
            
            # Execute command
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return f"Error searching for .{clean_ext} files: {result.stderr}"
            
            files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
            
            # Filter out common ignore patterns using standardized filtering
            filtered_files = filter_file_list(files)
            
            # Apply project filtering if mentioned projects are specified
            if self.current_mentioned_projects:
                project_filtered = []
                for file_path in filtered_files:
                    for project in self.current_mentioned_projects:
                        if f"/{project}/" in file_path or file_path.endswith(f"/{project}"):
                            project_filtered.append(file_path)
                            break
                filtered_files = project_filtered
            
            if not filtered_files:
                filter_msg = f" in projects: {self.current_mentioned_projects}" if self.current_mentioned_projects else ""
                return f"No .{clean_ext} files found{filter_msg}"
            
            # Count and format results
            total_files = len(filtered_files)
            file_list = "\n".join(filtered_files[:50])  # Show first 50 files
            truncated_msg = f"\n... ({total_files - 50} more files)" if total_files > 50 else ""
            
            filter_info = f"\n[Filtered for projects: {', '.join(self.current_mentioned_projects)}]" if self.current_mentioned_projects else ""
            
            return f"""Found {total_files} .{clean_ext} files:

{file_list}{truncated_msg}{filter_info}"""
            
        except subprocess.TimeoutExpired:
            return f"Search for .{clean_ext} files timed out"
        except Exception as e:
            return f"Error searching for .{clean_ext} files: {str(e)}"
    
    async def _search_file_content(self, pattern: str, file_extension: str = None) -> str:
        """Search for text patterns within files with context"""
        try:
            import subprocess
            import shlex
            
            # Build grep command with context and file filtering
            command = ["grep", "-r", "-n", "-C", "2"]  # recursive, line numbers, 2 lines context
            
            # Add file type filtering if specified
            if file_extension:
                clean_ext = file_extension.lstrip("*.")
                command.extend(["--include", f"*.{clean_ext}"])
            
            # Add the search pattern (safely quoted)
            command.append(pattern)
            
            # Search in workspace
            command.append("/workspace")
            
            # Execute command
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            
            if result.returncode != 0 and result.returncode != 1:  # 1 means no matches, which is OK
                return f"Error searching for pattern '{pattern}': {result.stderr}"
            
            if not result.stdout.strip():
                ext_msg = f" in .{file_extension} files" if file_extension else ""
                return f"No matches found for pattern '{pattern}'{ext_msg}"
            
            # Parse and structure the grep results
            lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            
            # Filter out common ignore patterns using standardized filtering
            filtered_lines = []
            for line in lines:
                # Extract file path from grep output (format: filepath:linenumber:content)
                file_path = line.split(':')[0] if ':' in line else line
                if should_include_file(file_path):
                    filtered_lines.append(line)
            
            if not filtered_lines:
                return f"No matches found for pattern '{pattern}' (after filtering)"
            
            # Apply project filtering if mentioned projects are specified
            if self.current_mentioned_projects:
                project_filtered = []
                for line in filtered_lines:
                    if ':' in line:
                        file_path = line.split(':', 1)[0]
                        for project in self.current_mentioned_projects:
                            if f"/{project}/" in file_path:
                                project_filtered.append(line)
                                break
                filtered_lines = project_filtered
            
            if not filtered_lines:
                filter_msg = f" in projects: {self.current_mentioned_projects}" if self.current_mentioned_projects else ""
                return f"No matches found for pattern '{pattern}'{filter_msg}"
            
            # Group results by file and format nicely
            current_file = None
            formatted_results = []
            match_count = 0
            file_count = 0
            
            for line in filtered_lines[:100]:  # Limit to 100 lines
                if ':' in line:
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        file_path, line_num, content = parts[0], parts[1], parts[2]
                        
                        # Check if this is a new file
                        if file_path != current_file:
                            if current_file is not None:
                                formatted_results.append("")  # Add separator
                            formatted_results.append(f"📁 {file_path}")
                            current_file = file_path
                            file_count += 1
                        
                        # Format the line with context indicator
                        if line_num.isdigit():
                            formatted_results.append(f"   {line_num}: {content}")
                            match_count += 1
                        else:
                            # Context line (has -- instead of line number)
                            formatted_results.append(f"   {line_num}: {content}")
            
            # Create summary
            ext_msg = f" in .{file_extension} files" if file_extension else ""
            filter_info = f"\n[Filtered for projects: {', '.join(self.current_mentioned_projects)}]" if self.current_mentioned_projects else ""
            
            summary = f"Found {match_count} matches in {file_count} files for pattern '{pattern}'{ext_msg}{filter_info}\n\n"
            
            # Join results and truncate if needed
            content = "\n".join(formatted_results)
            if len(content) > 4000:  # Truncate very long results
                content = content[:4000] + "\n... (results truncated)"
            
            return summary + content
            
        except subprocess.TimeoutExpired:
            return f"Search for pattern '{pattern}' timed out"
        except Exception as e:
            return f"Error searching for pattern '{pattern}': {str(e)}"
    
    async def _list_entities(self, pattern: str = None) -> str:
        """List files that likely contain database entities"""
        try:
            workspace = Path('/workspace')
            entity_patterns = re.compile(r"(@Entity\b|class .*Entity\b|CREATE TABLE)", re.IGNORECASE)
            matches = []
            
            for file_path in workspace.rglob('*'):
                if file_path.is_file() and file_path.suffix in ['.java', '.kt', '.py', '.sql']:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            sample = f.read(2048)  # Read first 2KB
                            if entity_patterns.search(sample):
                                rel_path = file_path.relative_to(workspace)
                                matches.append(str(rel_path))
                    except Exception:
                        continue
            
            # Apply project filtering if mentioned projects are specified
            if self.current_mentioned_projects:
                matches = self._filter_results_by_projects(matches, self.current_mentioned_projects)
            
            if not matches:
                filter_msg = f" (filtered for projects: {self.current_mentioned_projects})" if self.current_mentioned_projects else ""
                return f"ENTITY SCAN COMPLETE: No entity-like files found in the workspace{filter_msg}. This suggests the codebase may not contain traditional ORM entities or the entities use different naming conventions."
            
            filter_info = f" (filtered for projects: {', '.join(self.current_mentioned_projects)})" if self.current_mentioned_projects else ""
            result = f"ENTITY SCAN COMPLETE: Found {len(matches)} files with database entity patterns{filter_info}:\n"
            result += "\n".join(f"- {match}" for match in matches)
            result += f"\n\nTo examine specific entity details, use file_glimpse or read_file with any of these paths."
            
            return result
            
        except Exception as e:
            return f"Error scanning for entities: {str(e)}"
    
    async def _read_file(self, file_path: str) -> str:
        """Read complete file contents"""
        try:
            workspace_path = Path('/workspace')
            full_path = workspace_path / file_path
            
            if not full_path.exists():
                return f"File not found: {file_path}"
            
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            return f"--- CONTENT OF {file_path} ---\n{content}\n--- END OF {file_path} ---"
            
        except Exception as e:
            return f"Error reading file {file_path}: {str(e)}"
    
    async def _get_project_structure(self, project_name: str = None) -> str:
        """Generate enhanced project structure analysis with context awareness"""
        try:
            # Use enhanced project structure analyzer
            directory = f"/workspace/{project_name}" if project_name else "/workspace"
            return await self.enhanced_structure.analyze_project(directory, max_depth=4, include_files=True, project_name=project_name)
        except Exception as e:
            return f"Error analyzing project structure: {str(e)}"
    
    def _build_tree_structure_cerebras(self, directories: List[str], files: List[str], base_dir: str) -> str:
        """Build a tree-like display of directories and files"""
        import os
        
        # Create a mapping of path -> items
        tree_items = {}
        
        # Add directories
        for dir_path in directories:
            if dir_path.startswith(base_dir):
                rel_path = os.path.relpath(dir_path, base_dir)
                if rel_path != '.':
                    tree_items[rel_path] = {'type': 'dir', 'items': []}
        
        # Add files
        for file_path in files:
            if file_path.startswith(base_dir):
                rel_path = os.path.relpath(file_path, base_dir)
                parent_dir = os.path.dirname(rel_path) if '/' in rel_path else '.'
                
                if parent_dir == '.':
                    parent_dir = ''
                    
                if parent_dir not in tree_items:
                    tree_items[parent_dir] = {'type': 'dir', 'items': []}
                tree_items[parent_dir]['items'].append(os.path.basename(rel_path))
        
        # Sort and format
        lines = []
        sorted_paths = sorted(tree_items.keys())
        
        for path in sorted_paths[:20]:  # Limit to first 20 items
            # Calculate depth for indentation
            depth = path.count('/') if path else 0
            indent = "  " * depth
            
            # Directory name
            dir_name = os.path.basename(path) if path else base_dir.split('/')[-1]
            
            if path:
                lines.append(f"{indent}📁 {dir_name}/")
            else:
                lines.append(f"{dir_name}/")
            
            # Files in this directory
            for file_name in sorted(tree_items[path]['items'])[:10]:  # Max 10 files per dir
                file_indent = "  " * (depth + 1)
                file_icon = self._get_file_icon_cerebras(file_name)
                lines.append(f"{file_indent}{file_icon} {file_name}")
        
        if len(sorted_paths) > 20:
            lines.append(f"... ({len(sorted_paths) - 20} more directories)")
        
        return "\n".join(lines)
    
    def _get_file_icon_cerebras(self, filename: str) -> str:
        """Get appropriate icon for file type"""
        if filename.lower().startswith('readme'):
            return "📖"
        elif filename in ['package.json', 'requirements.txt', 'Pipfile', 'setup.py', 'pyproject.toml']:
            return "📦"
        elif filename in ['manage.py', 'app.py', 'main.py', 'index.js']:
            return "🚀"
        elif filename.startswith('.env') or 'config' in filename.lower():
            return "⚙️"
        elif filename in ['Dockerfile', 'docker-compose.yml']:
            return "🐳"
        elif filename.endswith('.py'):
            return "🐍"
        elif filename.endswith(('.js', '.ts', '.jsx', '.tsx')):
            return "📄"
        else:
            return "📄"
    
    def _detect_project_type_cerebras(self, files: List[str], directories: List[str]) -> str:
        """Detect project framework/type based on files and structure"""
        import os
        
        frameworks = []
        
        file_basenames = [os.path.basename(f) for f in files]
        
        # Check for specific frameworks
        if 'manage.py' in file_basenames:
            frameworks.append("Django")
        if 'package.json' in file_basenames:
            frameworks.append("Node.js/JavaScript")
        if any('requirements.txt' in f for f in file_basenames):
            frameworks.append("Python")
        if any('Dockerfile' in f for f in file_basenames):
            frameworks.append("Containerized")
        if any('tsconfig.json' in f for f in file_basenames):
            frameworks.append("TypeScript")
        
        # Check directory patterns
        dir_names = [os.path.basename(d) for d in directories]
        if 'src' in dir_names and 'public' in dir_names:
            frameworks.append("React/Frontend")
        if any('templates' in d for d in directories):
            frameworks.append("Web Framework")
        
        if frameworks:
            return f"Project Type: {', '.join(frameworks)}"
        else:
            return "Project Type: Unknown/General"
    
    async def _find_related_files(self, file_path: str, relationship_type: str = "all") -> str:
        """Find files related to a given file through various relationships"""
        try:
            import subprocess
            import os
            
            # Resolve file path to workspace
            if not file_path.startswith('/workspace'):
                if file_path.startswith('./'):
                    file_path = file_path[2:]
                file_path = f"/workspace/{file_path.lstrip('/')}"
            
            # Check if the source file exists
            if not os.path.exists(file_path):
                return f"File not found: {file_path}"
            
            # Get file info
            file_name = os.path.basename(file_path)
            module_name = os.path.splitext(file_name)[0]
            file_dir = os.path.dirname(file_path)
            
            related_files = []
            
            # 1. Find files with similar names
            cmd = ["find", "/workspace", "-name", f"*{module_name}*", "-type", "f"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                files = [f.strip() for f in result.stdout.split('\n') if f.strip() and f != file_path]
                related_files.extend(files[:10])
            
            # 2. Find test files
            test_cmd = ["find", "/workspace", "-name", f"*test*{module_name}*", "-o", "-name", f"test_{module_name}*", "-type", "f"]
            test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=5)
            if test_result.returncode == 0:
                test_files = [f.strip() for f in test_result.stdout.split('\n') if f.strip()]
                related_files.extend(test_files[:5])
            
            # 3. Find files in same directory
            dir_cmd = ["find", file_dir, "-maxdepth", "1", "-type", "f"]
            dir_result = subprocess.run(dir_cmd, capture_output=True, text=True, timeout=5)
            if dir_result.returncode == 0:
                dir_files = [f.strip() for f in dir_result.stdout.split('\n') if f.strip() and f != file_path]
                related_files.extend(dir_files[:5])
            
            # Remove duplicates and apply project filtering
            unique_files = []
            for f in related_files:
                if f not in unique_files:
                    # Apply project filtering if specified
                    if self.current_mentioned_projects:
                        for project in self.current_mentioned_projects:
                            if f"/{project}/" in f:
                                unique_files.append(f)
                                break
                    else:
                        unique_files.append(f)
            
            # Format results
            if not unique_files:
                return f"No related files found for: {file_name}"
            
            result_lines = [f"Related files for: {file_name}"]
            result_lines.append("=" * 50)
            
            for i, rel_file in enumerate(unique_files[:15], 1):
                rel_name = os.path.basename(rel_file)
                rel_dir = os.path.dirname(rel_file).replace('/workspace/', '')
                result_lines.append(f"{i}. {rel_name}")
                result_lines.append(f"   └─ {rel_dir}")
            
            if len(unique_files) > 15:
                result_lines.append(f"... and {len(unique_files) - 15} more files")
            
            return "\n".join(result_lines)
            
        except Exception as e:
            return f"Error finding related files for {file_path}: {str(e)}"
    
    async def process_request(self, user_query: str, chat_history=None, mentioned_projects: List[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a user request using native Cerebras tool calling"""
        
        # Log the agent processing start
        logger.info("🤖 CEREBRAS AGENT PROCESSING START")
        logger.info(f"Query: {user_query}")
        logger.info(f"Projects: {mentioned_projects}")
        logger.info(f"Has Chat History: {chat_history is not None}")
        
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
        self.current_mentioned_projects = mentioned_projects if mentioned_projects is not None else []
        
        # Determine appropriate tool strategy based on query
        query_lower = user_query.lower()
        
        # Analyze what type of query this is
        is_entity_query = any(word in query_lower for word in ['entity', 'entities', 'database', 'table', 'model', 'schema'])
        is_search_query = any(word in query_lower for word in ['find', 'search', 'locate', 'show me', 'explain', 'how'])
        is_file_query = any(word in query_lower for word in ['file', 'read', 'content', 'source'])
        
        # Build system prompt with better tool guidance
        if is_entity_query:
            tool_guidance = (
                "\n- Start with list_entities to find database entities"
                "\n- Use file_glimpse to examine specific entity files" 
                "\n- Use read_file to get complete entity implementations"
                "\n- Use code_search to understand relationships and usage patterns"
            )
        elif is_file_query:
            tool_guidance = (
                "\n- Use code_search to find relevant files"
                "\n- Use file_glimpse to examine file structure and key sections"
                "\n- Use read_file for complete file contents when needed"
                "\n- Follow up with related files found in imports/references"
            )
        else:
            tool_guidance = (
                "\n- Start with code_search to find relevant code"
                "\n- Use file_glimpse to examine key files found"
                "\n- Use read_file to get complete implementation details"
                "\n- Search for related functionality, dependencies, and usage patterns"
                "\n- Only use list_entities if the query specifically involves database entities"
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
                    "You are CodeWise, an AI coding assistant with access to powerful tools for analyzing codebases. "
                    "Use tools strategically to gather relevant information, then provide a comprehensive answer.\n"
                    "\nTool usage strategy:"
                    f"{tool_guidance}"
                    "\n- Start with searches to find relevant files"
                    "\n- Examine key files with file_glimpse or read_file when needed"
                    "\n- Use 2-4 tool calls to gather sufficient context"
                    "\n- Provide final answer once you have enough information to address the query"
                    f"{project_context}"
                    "\n\nEXAMPLE WORKFLOW:"
                    "\n1. code_search('project main functionality') → find key files"
                    "\n2. file_glimpse('src/main/app.js') → understand structure"
                    "\n3. read_file('package.json') → understand dependencies"
                    "\n4. code_search('specific feature implementation') → dive deeper"
                    "\n5. file_glimpse('feature-specific-file.js') → examine implementation"
                    "\n6. Provide comprehensive analysis based on all findings"
                    "\n\nProvide detailed analysis based on the actual code you discover through thorough tool usage."
                )
            },
            {"role": "user", "content": user_query}
        ]
        
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
        max_iterations = 6  # Reduced to prevent infinite loops
        iteration = 0
        recent_tool_calls = []  # Track recent tool calls to avoid repetition
        tool_call_count = 0  # Track total tool calls made
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🔄 ITERATION {iteration}/{max_iterations} (Tool calls so far: {tool_call_count})")
            
            try:
                # PHASE 3: Investigation Quality Gating - Enhanced tool call limiting logic
                investigation_score = self._calculate_investigation_completeness(recent_tool_calls, messages)
                use_tools = self.tools_schema if investigation_score < 5 and tool_call_count < 6 else None
                
                # Make API call to Cerebras
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b",  # Use the recommended model
                    messages=messages,
                    tools=use_tools,
                    parallel_tool_calls=False  # Required for some models
                )
                
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
                        
                        # Check for repetitive tool calling - be more aggressive
                        if recent_tool_calls.count(call_signature) >= 2:
                            warning_msg = f"Info: {function_name} has been called multiple times. Skipping to avoid repetition."
                            result = warning_msg
                        elif function_name in self.available_functions:
                            try:
                                # Parse arguments and call function
                                arguments = json.loads(function_args)
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
                            result = f"Unknown function: {function_name}"
                        
                        # PHASE 1: Negative Result Fallback System
                        if self._detect_negative_result(result, function_name) and tool_call_count < 5:
                            fallback_guidance = self._get_fallback_guidance(function_name)
                            messages.append({
                                "role": "system",
                                "content": f"The previous search returned no results. {fallback_guidance}. Continue investigating before providing final answer."
                            })
                        
                        # Log tool output (truncated for readability)
                        result_preview = result[:300] + "..." if len(result) > 300 else result
                        logger.info(f"🔧 TOOL RESULT ({len(result)} chars): {result_preview}")
                        
                        yield {
                            "type": "tool_end",
                            "output": result
                        }
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })
                    
                    # After 4+ tool calls, add guidance but allow continued investigation if needed
                    if tool_call_count >= 4:
                        messages.append({
                            "role": "system",
                            "content": "You have made several tool calls. If you have found sufficient information, provide a comprehensive final answer. If you need to investigate further with different approaches, continue, but focus on completing the investigation efficiently."
                        })
                
                else:
                    # No tool calls - we have the final response
                    final_content = choice.content or "Task completed"
                    logger.info(f"✅ FINAL RESPONSE ({len(final_content)} chars): {final_content[:500]}...")
                    
                    # PHASE 2: Incomplete Response Detection
                    if self._detect_incomplete_response(final_content) and tool_call_count < 5:
                        # Don't terminate - add follow-through prompt
                        messages.append({
                            "role": "system", 
                            "content": "You mentioned you would perform additional actions. Please follow through on what you just said you would do instead of providing a final answer."
                        })
                        continue  # Skip termination, continue loop
                    
                    yield {
                        "type": "final_result",
                        "output": final_content
                    }
                    break
                    
            except Exception as e:
                yield {
                    "type": "error", 
                    "message": f"Cerebras API error: {str(e)}"
                }
                break
        
        if iteration >= max_iterations:
            # Force a final call without tools to get an answer
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b",
                    messages=messages + [{
                        "role": "system", 
                        "content": "Please provide a final answer based on the information you've gathered. No more tool calls are available."
                    }],
                    tools=None
                )
                final_content = response.choices[0].message.content
                yield {
                    "type": "final_result",
                    "output": final_content or "Unable to provide a complete response due to tool call limits."
                }
            except Exception:
                yield {
                    "type": "final_result",
                    "output": "Maximum iterations reached. The response may be incomplete."
                } 