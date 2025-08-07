import asyncio
import os
import openai
from typing import AsyncGenerator, Dict, Any, List, Tuple
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.llms.base import LLM
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage, ChatResult, ChatGeneration
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.callbacks.base import AsyncCallbackHandler
import httpx
import json
import re
import logging
import shlex
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache, SQLiteCache
from vector_store import get_vector_store
from api_providers import get_provider_manager
from backend.hybrid_search import HybridSearchEngine
from context_delivery import ContextDeliverySystem
from directory_filters import (
    get_find_filter_args, get_grep_filter_args, should_include_file,
    filter_file_list, resolve_workspace_path, get_project_from_path
)
from project_context import (
    get_context_manager, set_project_context, get_current_context,
    filter_files_by_context
)
from enhanced_project_structure import EnhancedProjectStructure

# Set up enhanced logging for context retrieval
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional Redis cache if redis client is installed and LC_CACHE starts with redis://
_cache_uri = os.getenv("LC_CACHE", "sqlite")

try:
    if _cache_uri.startswith("redis://"):
        try:
            from langchain.cache import RedisCache  # type: ignore
            set_llm_cache(RedisCache.from_url(_cache_uri))
        except Exception:
            # Redis not available; fall back to SQLite
            set_llm_cache(SQLiteCache(database_path=".langchain.db"))
    elif _cache_uri in {"sqlite", "sqlite://", "sqlite:///"}:
        set_llm_cache(SQLiteCache(database_path=".langchain.db"))
    else:
        set_llm_cache(InMemoryCache())
except Exception:
    # Ensure cache setup never breaks the agent
    set_llm_cache(InMemoryCache())

class ContextExtractor:
    """Enhanced context extractor for better key term extraction and query analysis"""
    
    def __init__(self):
        # Common project/framework names to boost
        self.known_projects = {
            'fastapi', 'django', 'flask', 'react', 'vue', 'angular', 'node', 'express',
            'spring', 'springboot', 'hibernate', 'maven', 'gradle', 'docker', 'kubernetes',
            'postgres', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'kafka', 'rabbitmq',
            'nextjs', 'nuxt', 'svelte', 'laravel', 'symfony', 'rails', 'phoenix', 'gin',
            'echo', 'fiber', 'nestjs', 'koa', 'hapi', 'tensorflow', 'pytorch', 'sklearn'
        }
        
        # Technical indicators that suggest code-related queries
        self.technical_indicators = {
            'function', 'class', 'method', 'variable', 'import', 'export', 'module',
            'component', 'service', 'controller', 'model', 'repository', 'entity',
            'interface', 'type', 'enum', 'struct', 'async', 'await', 'promise',
            'database', 'api', 'endpoint', 'route', 'middleware', 'authentication',
            'authorization', 'validation', 'configuration', 'deployment', 'schema',
            'migration', 'seed', 'fixture', 'test', 'spec', 'mock', 'stub', 'factory'
        }
        
        # File extension patterns for better context matching
        self.file_extensions = {
            'python': ['.py', '.pyx', '.pyi'],
            'javascript': ['.js', '.jsx', '.mjs', '.cjs'],
            'typescript': ['.ts', '.tsx', '.d.ts'],
            'java': ['.java', '.class', '.jar'],
            'csharp': ['.cs', '.csx'],
            'cpp': ['.cpp', '.cc', '.cxx', '.c++', '.hpp', '.h'],
            'go': ['.go'],
            'rust': ['.rs'],
            'php': ['.php', '.phtml'],
            'ruby': ['.rb', '.rake'],
            'config': ['.json', '.yaml', '.yml', '.toml', '.ini', '.env'],
            'web': ['.html', '.css', '.scss', '.sass', '.less'],
            'sql': ['.sql', '.ddl', '.dml'],
            'docker': ['Dockerfile', '.dockerignore'],
            'build': ['Makefile', '.gradle', 'pom.xml', 'package.json', 'requirements.txt']
        }
        
        # Directory patterns that indicate project structure
        self.project_directories = {
            'src', 'lib', 'app', 'components', 'services', 'controllers', 'models',
            'views', 'templates', 'static', 'public', 'assets', 'config', 'tests',
            'test', 'spec', 'docs', 'scripts', 'utils', 'helpers', 'middleware',
            'api', 'routes', 'handlers', 'repositories', 'entities', 'migrations'
        }
        
        # Context history for learning from previous successful queries
        self.successful_terms_history = []
        self.max_history_size = 50
    
    def extract_key_terms(self, query: str, context_history: List[str] = None) -> List[str]:
        """Enhanced key term extraction with context awareness"""
        logger.info(f"Extracting key terms from query: {query}")
        
        terms = []
        query_lower = query.lower()
        
        # Always include the original query
        terms.append(query)
        
        # Extract project names from directory-like patterns
        project_patterns = [
            r'\b([A-Z][a-zA-Z0-9_]*(?:[A-Z][a-zA-Z0-9_]*)*)\b',  # PascalCase projects
            r'\b([a-z][a-z0-9_]*(?:_[a-z0-9]+)*)\b',  # snake_case projects
            r'\b([a-z][a-z0-9-]*(?:-[a-z0-9]+)*)\b',  # kebab-case projects
        ]
        
        for pattern in project_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if len(match) > 2 and match.lower() not in {'the', 'and', 'for', 'with', 'from'}:
                    terms.append(match)
        
        # Extract file extensions and technical terms
        technical_patterns = [
            r'\b\w+\.[a-zA-Z]{1,4}\b',  # Files with extensions
            r'\b[a-zA-Z]+[A-Z][a-zA-Z]*\b',  # CamelCase
            r'\b[a-z]+_[a-z_]+\b',  # snake_case (multi-word)
            r'\b[a-zA-Z]+-[a-zA-Z-]+\b',  # kebab-case (multi-word)
            r'\b[A-Z]{2,}\b',  # Acronyms
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, query)
            terms.extend(matches)
        
        # Extract quoted terms (high priority)
        quoted_terms = re.findall(r'"([^"]*)"', query) + re.findall(r"'([^']*)'", query)
        terms.extend(quoted_terms)
        
        # Extract known project/framework names
        words = re.findall(r'\b\w+\b', query_lower)
        for word in words:
            if word in self.known_projects:
                terms.append(word)
                logger.debug(f"Found known project/framework: {word}")
        
        # Extract technical indicators
        for word in words:
            if word in self.technical_indicators:
                terms.append(word)
        
        # Context-aware term extraction from history
        if context_history:
            for prev_query in context_history[-3:]:  # Last 3 queries
                prev_words = re.findall(r'\b[A-Z][a-zA-Z]+\b', prev_query)
                for word in prev_words:
                    if word.lower() in query_lower:
                        terms.append(word)
                        logger.debug(f"Added context term from history: {word}")
        
        # Clean and deduplicate terms
        unique_terms = []
        seen = set()
        
        for term in terms:
            clean_term = term.strip()
            if (clean_term and 
                len(clean_term) > 1 and 
                clean_term.lower() not in seen and
                not clean_term.isdigit()):
                unique_terms.append(clean_term)
                seen.add(clean_term.lower())
        
        # Prioritize terms (quoted terms and known projects first)
        prioritized_terms = []
        regular_terms = []
        
        for term in unique_terms:
            if (term in quoted_terms or 
                term.lower() in self.known_projects or
                any(char in term for char in ['.', '_', '-']) or
                term[0].isupper()):
                prioritized_terms.append(term)
            else:
                regular_terms.append(term)
        
        final_terms = prioritized_terms + regular_terms
        
        logger.info(f"Extracted {len(final_terms)} key terms: {final_terms[:5]}{'...' if len(final_terms) > 5 else ''}")
        return final_terms
    
    def analyze_query_intent(self, query: str) -> Dict[str, any]:
        """Analyze query to understand user intent and context needs"""
        query_lower = query.lower()
        
        # Determine query type
        query_type = "general"
        if any(word in query_lower for word in ["explain", "how", "what", "describe", "show"]):
            query_type = "explanation"
        elif any(word in query_lower for word in ["find", "search", "locate", "where"]):
            query_type = "search"
        elif any(word in query_lower for word in ["implement", "create", "build", "add"]):
            query_type = "implementation"
        elif any(word in query_lower for word in ["fix", "debug", "error", "issue", "problem"]):
            query_type = "debugging"
        
        # Detect project references
        project_hints = []
        words = query.split()
        for word in words:
            # Look for capitalized words that might be project names
            if word[0].isupper() and len(word) > 2:
                project_hints.append(word)
        
        # Detect file type preferences
        file_type_hints = []
        if any(ext in query_lower for ext in ['.py', 'python']):
            file_type_hints.append('python')
        if any(ext in query_lower for ext in ['.js', '.ts', 'javascript', 'typescript']):
            file_type_hints.append('javascript')
        if any(ext in query_lower for ext in ['.java', 'spring']):
            file_type_hints.append('java')
        
        # Calculate complexity score
        complexity_indicators = len(re.findall(r'\b(?:and|or|with|using|for|in|on|at)\b', query_lower))
        complexity_score = min(1.0, complexity_indicators / 5.0)
        
        return {
            'query_type': query_type,
            'project_hints': project_hints,
            'file_type_hints': file_type_hints,
            'complexity_score': complexity_score,
            'has_technical_terms': any(word in query_lower for word in self.technical_indicators),
            'is_specific_search': '"' in query or "'" in query
        }

class StreamingCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming agent updates"""
    
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.completion_tokens = 0
    
    async def on_agent_action(self, action, **kwargs):
        await self.queue.put({
            "type": "agent_action",
            "action": action.tool,
            "input": action.tool_input,
            "log": action.log
        })
    
    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        logger.info(f"Tool started: {tool_name} with input: {input_str}")
        
        await self.queue.put({
            "type": "tool_start",
            "tool": tool_name,
            "input": input_str
        })
    
    async def on_tool_end(self, output: str, **kwargs):
        logger.info(f"Tool completed with output length: {len(output)} characters")
        
        await self.queue.put({
            "type": "tool_end",
            "output": output
        })

    async def on_llm_new_token(self, token: str, **kwargs):
        # Stream partial tokens to frontend
        self.completion_tokens += 1
        await self.queue.put({
            "type": "stream_token",
            "token": token
        })
    
    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        await self.queue.put({
            "type": "chain_start",
            "name": serialized.get("name", "unknown")
        })
    
    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        await self.queue.put({
            "type": "chain_end",
            "outputs": outputs
        })

class MCPToolWrapper:
    """Wrapper for MCP server tools"""
    
    def __init__(self, mcp_server_url: str):
        self.mcp_server_url = mcp_server_url
        self.client = httpx.AsyncClient()
    
    async def call_mcp_tool(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Call an MCP server tool"""
        try:
            response = await self.client.post(
                f"{self.mcp_server_url}/tools/{tool_name}",
                json=params
            )
            response.raise_for_status()
            result = response.json()
            return result.get("result", "Tool executed successfully")
        except Exception as e:
            return f"Error calling MCP tool {tool_name}: {str(e)}"
    
    async def read_file(self, file_path: str) -> str:
        """Read a file via MCP server"""
        return await self.call_mcp_tool("read_file", {"file_path": file_path})
    
    async def write_file(self, file_path: str, content: str) -> str:
        """Write to a file via MCP server"""
        return await self.call_mcp_tool("write_file", {"file_path": file_path, "content": content})
    
    async def list_files(self, directory: str) -> str:
        """List files in a directory via MCP server"""
        return await self.call_mcp_tool("list_files", {"directory": directory})
    
    async def run_command(self, command: str) -> str:
        """Run a shell command via MCP server"""
        return await self.call_mcp_tool("run_command", {"command": command})
    
    async def find_files(self, pattern: str, directory: str = ".") -> str:
        """Find files matching a pattern using shell commands"""
        command = f"find {directory} -name '{pattern}' -type f | head -20"
        return await self.call_mcp_tool("run_command", {"command": command})
    
    async def grep_search(self, pattern: str, directory: str = ".", file_type: str = "*") -> str:
        """Search for text patterns across files"""
        command = f"grep -r --include='*.{file_type}' '{pattern}' {directory} | head -20"
        return await self.call_mcp_tool("run_command", {"command": command})
    
    async def get_file_info(self, file_path: str) -> str:
        """Get file information (size, type, etc.)"""
        command = f"file '{file_path}' && ls -la '{file_path}'"
        return await self.call_mcp_tool("run_command", {"command": command})
    
    async def explore_directory_tree(self, directory: str, max_depth: int = 3) -> str:
        """Get a tree view of directory structure"""
        command = f"find {directory} -maxdepth {max_depth} -type d | head -30"
        return await self.call_mcp_tool("run_command", {"command": command})
    
    async def search_by_extension(self, extensions: List[str], directory: str = ".") -> str:
        """Find files by extension patterns with composition analysis"""
        # Ensure we're searching in the workspace if directory is relative
        if directory == "." or not directory.startswith("/"):
            if directory == ".":
                search_dir = "/workspace"
            else:
                search_dir = f"/workspace/{directory.lstrip('./')}"
        else:
            search_dir = directory
        
        # Build find command with multiple -name patterns joined by -o
        patterns = []
        for ext in extensions:
            clean_ext = ext.lstrip("*.")
            patterns.append(f'-name "*.{clean_ext}"')
        
        pattern_str = " -o ".join(patterns) if patterns else '-name "*"'
        
        # Find files using standardized filtering
        filter_args = get_find_filter_args()
        command = f"""find {search_dir} \\( {pattern_str} \\) -type f {filter_args} | head -100"""
        
        result = await self.call_mcp_tool("run_command", {"command": command})
        
        # Parse results for composition analysis
        if not result or "Error" in result:
            return result or "No files found"
        
        files = [f.strip() for f in result.split('\n') if f.strip()]
        
        # Count by extension and calculate composition
        ext_counts = {}
        total_files = len(files)
        
        for file_path in files:
            if '.' in file_path:
                ext = file_path.rsplit('.', 1)[-1].lower()
                ext_counts[ext] = ext_counts.get(ext, 0) + 1
        
        # Format results with composition analysis
        composition_lines = []
        for ext, count in sorted(ext_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_files) * 100 if total_files > 0 else 0
            composition_lines.append(f"  .{ext}: {count} files ({percentage:.1f}%)")
        
        composition_summary = "\n".join(composition_lines) if composition_lines else "  No files found"
        
        file_list = "\n".join(files[:20])  # Show first 20 files
        truncated_msg = f"\n... ({total_files - 20} more files)" if total_files > 20 else ""
        
        return f"""Found {total_files} files matching extensions: {extensions}

Project Composition:
{composition_summary}

Files:
{file_list}{truncated_msg}"""
    
    async def search_file_content(self, pattern: str, directory: str = ".", file_types: List[str] = None, context_lines: int = 2) -> str:
        """Search for text patterns across files with context"""
        # Ensure we're searching in the workspace if directory is relative
        if directory == "." or not directory.startswith("/"):
            if directory == ".":
                search_dir = "/workspace"
            else:
                search_dir = f"/workspace/{directory.lstrip('./')}"
        else:
            search_dir = directory
        
        # Build include patterns for file types
        if file_types:
            includes = " ".join([f'--include="*.{ft.lstrip("*.")}"' for ft in file_types])
        else:
            includes = '--include="*"'
        
        # Escape pattern for shell safety
        escaped_pattern = shlex.quote(pattern)
        
        # Build grep command with context and limits using standardized filtering
        grep_filter = get_grep_filter_args()
        command = f"""grep -r {includes} -n -C {context_lines} {escaped_pattern} {search_dir} | grep -v -E '{grep_filter}' | head -50"""
        
        result = await self.call_mcp_tool("run_command", {"command": command})
        
        if not result or "Error" in result:
            return result or f"No matches found for pattern: {pattern}"
        
        # Parse and structure the grep results
        lines = [line.strip() for line in result.split('\n') if line.strip()]
        
        if not lines:
            return f"No matches found for pattern: {pattern}"
        
        # Group results by file and format nicely
        current_file = None
        formatted_results = []
        match_count = 0
        file_count = 0
        
        for line in lines:
            if ':' in line:
                parts = line.split(':', 2)  # Split into file, line_num, content (max 2 splits)
                if len(parts) >= 3:
                    file_path, line_num, content = parts[0], parts[1], parts[2]
                    
                    # Check if this is a new file
                    if file_path != current_file:
                        if current_file is not None:
                            formatted_results.append("")  # Add separator between files
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
        summary = f"Found {match_count} matches in {file_count} files for pattern: {pattern}\n\n"
        
        # Join results and truncate if too long
        content = "\n".join(formatted_results)
        if len(content) > 3000:  # Truncate very long results
            content = content[:3000] + "\n... (results truncated)"
        
        return summary + content
    
    async def get_project_structure(self, directory: str = ".", max_depth: int = 4, include_files: bool = True) -> str:
        """Generate comprehensive project tree with intelligent filtering"""
        # Ensure we're searching in the workspace if directory is relative
        if directory == "." or not directory.startswith("/"):
            if directory == ".":
                search_dir = "/workspace"
            else:
                search_dir = f"/workspace/{directory.lstrip('./')}"
        else:
            search_dir = directory
        
        # Get directory structure with standardized filtering
        filter_args = get_find_filter_args()
        dirs_cmd = f"""find {search_dir} -maxdepth {max_depth} -type d {filter_args} | sort"""
        dirs_result = await self.call_mcp_tool("run_command", {"command": dirs_cmd})
        
        if not dirs_result or "Error" in dirs_result:
            return "Error retrieving directory structure"
        
        directories = [d.strip() for d in dirs_result.split('\n') if d.strip()]
        
        # Get important files with multiple strategies
        important_files_cmd = f"""find {search_dir} -maxdepth {max_depth} -type f \\( \\
            -name 'README*' -o -name 'readme*' -o \\
            -name 'package.json' -o -name 'requirements.txt' -o -name 'Pipfile' -o \\
            -name 'manage.py' -o -name 'app.py' -o -name 'main.py' -o -name 'index.js' -o \\
            -name 'Dockerfile' -o -name 'docker-compose.yml' -o -name '.env*' -o \\
            -name '*.config.js' -o -name 'tsconfig.json' -o -name 'setup.py' -o \\
            -name 'Makefile' -o -name 'pyproject.toml' \\
        \\) | head -30"""
        
        files_result = await self.call_mcp_tool("run_command", {"command": important_files_cmd})
        important_files = [f.strip() for f in (files_result or "").split('\n') if f.strip() and "Error" not in f]
        
        # Get some source files for context (limited) with standardized filtering
        if include_files:
            filter_args = get_find_filter_args()
            source_files_cmd = f"""find {search_dir} -maxdepth {max_depth} -type f \\( -name '*.py' -o -name '*.js' -o -name '*.ts' -o -name '*.jsx' -o -name '*.tsx' \\) {filter_args} | head -20"""
            source_result = await self.call_mcp_tool("run_command", {"command": source_files_cmd})
            source_files = [f.strip() for f in (source_result or "").split('\n') if f.strip() and "Error" not in f]
        else:
            source_files = []
        
        # Combine and organize all files
        all_files = important_files + source_files
        
        # Build tree structure
        tree_structure = self._build_tree_structure(directories, all_files, search_dir)
        
        # Detect framework/project type
        framework_info = self._detect_project_type(important_files, directories)
        
        # Use enhanced project structure analyzer
        project_name = os.path.basename(search_dir) if search_dir != "/workspace" else "workspace"
        return await self.enhanced_structure.analyze_project(directory, max_depth, include_files, project_name)
    
    def _build_tree_structure(self, directories: List[str], files: List[str], base_dir: str) -> str:
        """Build a tree-like display of directories and files"""
        # Create a mapping of path -> items
        tree_items = {}
        
        # Add directories
        for dir_path in directories:
            if dir_path.startswith('./'):
                dir_path = dir_path[2:]  # Remove './' prefix
            tree_items[dir_path] = {'type': 'dir', 'items': []}
        
        # Add files
        for file_path in files:
            if file_path.startswith('./'):
                file_path = file_path[2:]  # Remove './' prefix
            parent_dir = os.path.dirname(file_path) if '/' in file_path else '.'
            if parent_dir not in tree_items:
                tree_items[parent_dir] = {'type': 'dir', 'items': []}
            tree_items[parent_dir]['items'].append(os.path.basename(file_path))
        
        # Sort and format
        lines = []
        sorted_paths = sorted(tree_items.keys())
        
        for path in sorted_paths[:20]:  # Limit to first 20 items
            # Calculate depth for indentation
            depth = path.count('/') if path != '.' else 0
            indent = "  " * depth
            
            # Directory name
            dir_name = os.path.basename(path) if path != '.' else base_dir
            if path == '.':
                lines.append(f"{dir_name}/")
            else:
                lines.append(f"{indent}📁 {dir_name}/")
            
            # Files in this directory
            for file_name in sorted(tree_items[path]['items'])[:10]:  # Max 10 files per dir
                file_indent = "  " * (depth + 1)
                file_icon = self._get_file_icon(file_name)
                lines.append(f"{file_indent}{file_icon} {file_name}")
        
        if len(sorted_paths) > 20:
            lines.append(f"... ({len(sorted_paths) - 20} more directories)")
        
        return "\n".join(lines)
    
    def _get_file_icon(self, filename: str) -> str:
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
    
    def _detect_project_type(self, files: List[str], directories: List[str]) -> str:
        """Detect project framework/type based on files and structure"""
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
    
    async def find_related_files(self, file_path: str, relationship_types: List[str] = ["import", "test", "config"]) -> str:
        """Find files related to a given file through various relationships"""
        # Resolve file path to workspace
        if not file_path.startswith("/workspace"):
            if file_path.startswith('./'):
                file_path = file_path[2:]
            file_path = f"/workspace/{file_path.lstrip('/')}"
        
        # Extract file information
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        file_dir = os.path.dirname(file_path)
        file_ext = os.path.splitext(file_path)[1]
        
        results = {}
        
        # 1. Import/Dependency Analysis
        if "import" in relationship_types:
            import_results = await self._find_import_relationships(file_path, base_name, file_ext)
            if import_results:
                results["imports"] = import_results
        
        # 2. Test File Discovery
        if "test" in relationship_types:
            test_results = await self._find_test_relationships(base_name, file_dir, file_ext)
            if test_results:
                results["tests"] = test_results
        
        # 3. Configuration Relationships
        if "config" in relationship_types:
            config_results = await self._find_config_relationships(file_path, file_dir)
            if config_results:
                results["configs"] = config_results
        
        # 4. Naming Pattern Relationships
        if "naming" in relationship_types:
            naming_results = await self._find_naming_relationships(base_name, file_dir, file_ext)
            if naming_results:
                results["naming_patterns"] = naming_results
        
        # Format results
        return self._format_relationship_results(file_path, results)
    
    async def _find_import_relationships(self, file_path: str, base_name: str, file_ext: str) -> List[str]:
        """Find files that import or are imported by the target file"""
        relationships = []
        
        # Look for files that import this module
        if file_ext in ['.py']:
            # Python imports: "from module import" or "import module"
            grep_filter = get_grep_filter_args()
            import_cmd = f"""grep -r 'from.*{base_name}\\|import.*{base_name}' --include='*.py' . | grep -v -E '{grep_filter}' | head -10"""
        elif file_ext in ['.js', '.ts', '.jsx', '.tsx']:
            # JavaScript/TypeScript imports: "import ... from" or "require(...)"
            grep_filter = get_grep_filter_args()
            import_cmd = f"""grep -r "import.*{base_name}\\|require.*{base_name}" --include='*.js' --include='*.ts' --include='*.jsx' --include='*.tsx' . | grep -v -E '{grep_filter}' | head -10"""
        else:
            return relationships
        
        result = await self.call_mcp_tool("run_command", {"command": import_cmd})
        
        if result and "Error" not in result:
            lines = [line.strip() for line in result.split('\n') if line.strip()]
            for line in lines:
                if ':' in line:
                    file_ref = line.split(':', 1)[0]
                    if file_ref != file_path:  # Don't include self-references
                        relationships.append(f"Imported by: {file_ref}")
        
        return relationships
    
    async def _find_test_relationships(self, base_name: str, file_dir: str, file_ext: str) -> List[str]:
        """Find test files related to the target file"""
        test_files = []
        
        # Multiple test naming conventions
        test_patterns = [
            f"test_{base_name}*",
            f"*_{base_name}_test*",
            f"*{base_name}*test*",
            f"test*{base_name}*",
            f"{base_name}*spec*"
        ]
        
        for pattern in test_patterns:
            if file_ext in ['.py']:
                cmd = f"find /workspace -name '{pattern}.py' -o -name '{pattern}_test.py' | head -5"
            elif file_ext in ['.js', '.ts', '.jsx', '.tsx']:
                cmd = f"find /workspace -name '{pattern}.js' -o -name '{pattern}.test.js' -o -name '{pattern}.spec.js' -o -name '{pattern}.ts' -o -name '{pattern}.test.ts' | head -5"
            else:
                cmd = f"find /workspace -name '{pattern}*' | head -3"
            
            result = await self.call_mcp_tool("run_command", {"command": cmd})
            
            if result and "Error" not in result:
                files = [f.strip() for f in result.split('\n') if f.strip()]
                for file_ref in files:
                    if file_ref not in test_files:
                        test_files.append(f"Test file: {file_ref}")
        
        return test_files
    
    async def _find_config_relationships(self, file_path: str, file_dir: str) -> List[str]:
        """Find configuration files that might affect the target file"""
        config_files = []
        
        # Look for config files in current and parent directories
        config_patterns = [
            "settings*", "config*", ".env*", "*.conf", "*.ini", 
            "*.config.js", "tsconfig.json", "package.json", "requirements.txt"
        ]
        
        # Search in current directory and parent directories 
        # Ensure we're using workspace paths
        search_dirs = [file_dir]
        parent_dir = os.path.dirname(file_dir)
        if parent_dir != file_dir and parent_dir.startswith('/workspace'):
            search_dirs.append(parent_dir)
        
        for search_dir in search_dirs[:2]:  # Limit to 2 levels up
            for pattern in config_patterns:
                cmd = f"find {search_dir} -maxdepth 1 -name '{pattern}' | head -3"
                result = await self.call_mcp_tool("run_command", {"command": cmd})
                
                if result and "Error" not in result:
                    files = [f.strip() for f in result.split('\n') if f.strip()]
                    for file_ref in files:
                        if file_ref not in config_files:
                            config_files.append(f"Config file: {file_ref}")
        
        return config_files
    
    async def _find_naming_relationships(self, base_name: str, file_dir: str, file_ext: str) -> List[str]:
        """Find files with similar naming patterns"""
        similar_files = []
        
        # Look for files with similar names
        patterns = [
            f"{base_name}_*",
            f"*_{base_name}",
            f"{base_name[:-1]}*" if len(base_name) > 3 else f"{base_name}*"  # Partial match
        ]
        
        for pattern in patterns:
            filter_args = get_find_filter_args()
            cmd = f"find {file_dir} -name '{pattern}*' {filter_args} | head -5"
            result = await self.call_mcp_tool("run_command", {"command": cmd})
            
            if result and "Error" not in result:
                files = [f.strip() for f in result.split('\n') if f.strip()]
                for file_ref in files:
                    if os.path.basename(file_ref) != f"{base_name}{file_ext}":  # Don't include self
                        similar_files.append(f"Similar name: {file_ref}")
        
        return similar_files[:5]  # Limit results
    
    def _format_relationship_results(self, file_path: str, results: Dict[str, List[str]]) -> str:
        """Format the relationship analysis results"""
        if not results:
            return f"No related files found for: {file_path}"
        
        output_lines = [f"Related files for: {file_path}\n"]
        
        for relationship_type, files in results.items():
            if files:
                output_lines.append(f"🔗 {relationship_type.replace('_', ' ').title()}:")
                for file_ref in files[:5]:  # Limit to 5 per category
                    output_lines.append(f"  • {file_ref}")
                output_lines.append("")
        
        # Summary
        total_relationships = sum(len(files) for files in results.values())
        output_lines.append(f"Total: {total_relationships} relationships found across {len(results)} categories")
        
        return "\n".join(output_lines)

class ProviderManagedChatModel(BaseChatModel):
    """Custom LangChain chat model that routes through the provider manager"""
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, provider_manager, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'provider_manager', provider_manager)
    
    def _generate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
        """Generate chat completion using the provider manager"""
        # Convert LangChain messages to provider format
        provider_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                provider_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                provider_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                provider_messages.append({"role": "assistant", "content": msg.content})
        
        # Get response from provider manager
        response = self.provider_manager.chat_completion(provider_messages, **kwargs)
        
        # Convert back to LangChain format
        ai_message = AIMessage(content=response.content)
        generation = ChatGeneration(message=ai_message)
        
        return ChatResult(generations=[generation])
    
    async def _agenerate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
        """Async version - for now just call sync version"""
        return self._generate(messages, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        return "provider_managed_chat"

class CodeWiseAgent:
    """Main agent class for CodeWise"""
    
    def __init__(self, openai_api_key: str, mcp_server_url: str):
        # Initialize API provider manager
        self.provider_manager = get_provider_manager()
        
        # Get current provider info for LLM initialization
        provider_info = self.provider_manager.get_provider_info()
        current_provider = self.provider_manager.get_current_provider()
        
        # Initialize LLM based on current provider
        if current_provider and provider_info.get('provider') == 'cerebras':
            # Use native Cerebras agent for proper tool calling
            from cerebras_agent import CerebrasNativeAgent
            self.use_native_cerebras = True
            self.cerebras_agent = CerebrasNativeAgent(
                api_key=os.getenv("CEREBRAS_API_KEY", ""),
                mcp_server_url=mcp_server_url
            )
            logger.info("✅ Using native Cerebras agent with proper tool calling")
            # Skip LangChain initialization entirely for native Cerebras
            return
            
        else:
            # Use LangChain implementation for OpenAI and other providers
            self.use_native_cerebras = False
            
            if current_provider and provider_info.get('provider') == 'openai':
                self.llm = ChatOpenAI(
                    api_key=openai_api_key,
                    model=provider_info.get('chat_model', 'gpt-4-turbo-preview'),
                    temperature=0,
                    streaming=True
                )
                logger.info(f"Using OpenAI provider with model: {provider_info.get('chat_model')}")
            elif current_provider and provider_info.get('provider') == 'kimi':
                # Use custom provider-managed chat model for Kimi
                self.llm = ProviderManagedChatModel(
                    provider_manager=self.provider_manager,
                    temperature=0
                )
                logger.info("✅ Using Kimi K2 provider with custom LangChain wrapper")
            else:
                # Fallback to OpenAI if no provider available
                self.llm = ChatOpenAI(
                    api_key=openai_api_key,
                    model="gpt-4-turbo-preview",
                    temperature=0,
                    streaming=True
                )
                logger.info("Using fallback OpenAI provider")
        
        # Initialize shared components for LangChain agents only
        if not self.use_native_cerebras:
            self.mcp_wrapper = MCPToolWrapper(mcp_server_url)
        
            # Initialize hybrid search and context delivery systems
            try:
                self.hybrid_search = HybridSearchEngine()
                self.context_delivery = ContextDeliverySystem(self.hybrid_search)
                logger.info("Hybrid search and context delivery systems initialized")
            except Exception as e:
                logger.error(f"Failed to initialize hybrid search: {e}, falling back to vector search")
                self.hybrid_search = None
                self.context_delivery = None
            
            # Initialize enhanced project structure analyzer
            self.enhanced_structure = EnhancedProjectStructure(lambda tool, params: self.mcp_wrapper.call_mcp_tool(tool, params))
        
            self.tools = self._create_tools()
            self.agent = self._create_agent()
            self.context_extractor = ContextExtractor()
    
    def reinitialize_with_provider(self, openai_api_key: str):
        """Reinitialize the agent with the current provider settings"""
        logger.info("Reinitializing agent with current provider settings")
        
        # Get updated provider info
        provider_info = self.provider_manager.get_provider_info()
        current_provider = self.provider_manager.get_current_provider()
        
        # Reinitialize LLM based on current provider
        if current_provider and provider_info.get('provider') == 'openai':
            self.llm = ChatOpenAI(
                api_key=openai_api_key,
                model=provider_info.get('chat_model', 'gpt-4-turbo-preview'),
                temperature=0,
                streaming=True
            )
            logger.info(f"Reinitialized with OpenAI provider using model: {provider_info.get('chat_model')}")
        elif current_provider and provider_info.get('provider') == 'kimi':
            # For Kimi, use OpenAI wrapper but actual calls go through provider manager
            self.llm = ChatOpenAI(
                api_key=openai_api_key or "dummy-key",
                model="gpt-3.5-turbo",
                temperature=0,
                streaming=True
            )
            logger.info("Reinitialized with Kimi provider using LangChain OpenAI wrapper")
        else:
            # Fallback to OpenAI
            self.llm = ChatOpenAI(
                api_key=openai_api_key,
                model="gpt-4-turbo-preview",
                temperature=0,
                streaming=True
            )
            logger.info("Reinitialized with fallback OpenAI provider")
        
        # Recreate the agent with the new LLM
        self.agent = self._create_agent()
        logger.info("Agent successfully reinitialized with new provider")
    
    async def auto_search_context(self, query: str, callback_queue: asyncio.Queue, chat_history=None, mentioned_projects: List[str] = None) -> str:
        """Enhanced automatic context search with improved fallback strategies"""
        logger.info(f"Starting enhanced auto-context search for query: {query}")
        
        # Send context gathering notification
        await callback_queue.put({
            "type": "context_gathering_start",
            "message": "Analyzing query and gathering relevant context..."
        })
        
        # Log project scope if mentioned projects are provided
        if mentioned_projects:
            logger.info(f"Project scope specified: {mentioned_projects}")
            await callback_queue.put({
                "type": "context_search",
                "source": "project scope",
                "query": f"Filtering by projects: {', '.join(mentioned_projects)}"
            })
        
        # Extract previous queries from chat history for context
        previous_queries = []
        if chat_history:
            for message in chat_history[-5:]:  # Last 5 messages for context
                if hasattr(message, 'content'):
                    previous_queries.append(message.content)
        
        # Analyze query intent and extract key terms with context history
        query_intent = self.context_extractor.analyze_query_intent(query)
        key_terms = self.context_extractor.extract_key_terms(query, context_history=previous_queries)
        
        logger.info(f"Query analysis: type={query_intent['query_type']}, "
                   f"projects={query_intent['project_hints']}, "
                   f"complexity={query_intent['complexity_score']:.2f}")
        
        # Helper function to filter results by mentioned projects using centralized context management
        def filter_by_projects(chunks: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
            if not mentioned_projects:
                return chunks
            
            # Use centralized context manager for consistent filtering
            filtered_chunks = []
            for file_path, snippet in chunks:
                if get_context_manager().is_file_in_current_context(file_path):
                    filtered_chunks.append((file_path, snippet))
            
            logger.info(f"Project filtering: {len(chunks)} -> {len(filtered_chunks)} chunks for projects {mentioned_projects}")
            
            # Log sample filtered paths for debugging
            if filtered_chunks:
                sample_paths = [path for path, _ in filtered_chunks[:5]]
                logger.info(f"Sample filtered paths: {sample_paths}")
            
            return filtered_chunks
        
        all_context = []
        context_sources = []
        search_attempts = []
        
        # Clean query by removing @project mentions to improve semantic matching
        clean_query = re.sub(r'@\w+', '', query).strip()
        if clean_query != query:
            logger.info(f"Cleaned query for search: '{clean_query}' (from '{query}')")
        search_query = clean_query or query  # Fallback to original if empty
        
        # Primary search with original query
        await callback_queue.put({
            "type": "context_search",
            "source": "semantic search",
            "query": search_query[:50] + "..." if len(search_query) > 50 else search_query
        })
        
        # Determine initial relevance threshold based on query complexity
        base_threshold = 0.25 if query_intent['complexity_score'] > 0.5 else 0.3
        
        # Use hybrid search if available, otherwise fallback to vector search
        if self.hybrid_search:
            try:
                logger.info(f"Attempting hybrid search with threshold {base_threshold}")
                search_results = self.hybrid_search.search(search_query, k=4, min_relevance=base_threshold)
                chunks = [(result.file_path, result.snippet) for result in search_results]
                search_attempts.append(f"Hybrid search: {len(chunks)} results")
                logger.info(f"Hybrid search successful: {len(chunks)} results found")
            except Exception as e:
                logger.error(f"Hybrid search failed: {e}")
                logger.info(f"Falling back to vector search with threshold {base_threshold}")
                chunks = get_vector_store().query(search_query, k=4, min_relevance=base_threshold)
                search_attempts.append(f"Vector search (fallback): {len(chunks)} results")
                logger.info(f"Vector search fallback: {len(chunks)} results found")
        else:
            logger.info(f"Using vector search with threshold {base_threshold}")
            chunks = get_vector_store().query(search_query, k=4, min_relevance=base_threshold)
            search_attempts.append(f"Vector search: {len(chunks)} results")
            logger.info(f"Vector search: {len(chunks)} results found")
        
        # Apply project filtering to main search results
        chunks = filter_by_projects(chunks)
        
        if chunks:
            logger.info(f"Found {len(chunks)} chunks for main query (after project filtering)")
            all_context.extend(chunks)
            context_sources.append(f"Main query: {len(chunks)} relevant chunks")
        
        # Enhanced key term search with prioritization
        prioritized_terms = key_terms[:4]  # Increased from 3 to 4
        for i, term in enumerate(prioritized_terms):
            if term.lower() != query.lower() and len(term) > 2:  # Avoid duplicate and short terms
                await callback_queue.put({
                    "type": "context_search", 
                    "source": "key term analysis",
                    "query": term
                })
                
                # Use lower threshold for key terms to find more context
                term_threshold = base_threshold * 0.7
                
                try:
                    if self.hybrid_search:
                        term_results = self.hybrid_search.search(term, k=2, min_relevance=term_threshold)
                        term_chunks = [(result.file_path, result.snippet) for result in term_results]
                    else:
                        term_chunks = get_vector_store().query(term, k=2, min_relevance=term_threshold)
                    
                    # Apply project filtering to term search results
                    term_chunks = filter_by_projects(term_chunks)
                    
                    if term_chunks:
                        logger.info(f"Found {len(term_chunks)} chunks for term: {term} (after project filtering)")
                        all_context.extend(term_chunks)
                        context_sources.append(f"Term '{term}': {len(term_chunks)} chunks")
                        search_attempts.append(f"Term '{term}': {len(term_chunks)} results")
                    else:
                        search_attempts.append(f"Term '{term}': 0 results")
                        
                except Exception as e:
                    logger.error(f"Search failed for term '{term}': {e}")
                    search_attempts.append(f"Term '{term}': failed")
        
        # Remove duplicates while preserving order and relevance
        seen_contexts = set()
        unique_context = []
        for path, snippet in all_context:
            # Create a more robust deduplication key
            context_key = f"{path}:{hash(snippet[:200])}"  # Use hash for better deduplication
            if context_key not in seen_contexts:
                seen_contexts.add(context_key)
                unique_context.append((path, snippet))
        
        # Adaptive context limit based on query complexity
        max_context = 6 if query_intent['complexity_score'] > 0.5 else 5
        unique_context = unique_context[:max_context]
        
        if unique_context:
            # Build comprehensive context summary
            context_summary = self._build_context_summary(unique_context, query)
            
            # Send context gathering complete notification
            await callback_queue.put({
                "type": "context_gathering_complete",
                "sources": context_sources,
                "chunks_found": len(unique_context),
                "files_analyzed": len(set(path for path, _ in unique_context))
            })
            
            logger.info(f"Auto-context complete: {len(unique_context)} chunks from "
                       f"{len(set(path for path, _ in unique_context))} files")
            logger.debug(f"Search attempts: {'; '.join(search_attempts)}")
            
            return context_summary
        else:
            # Enhanced fallback strategies
            logger.warning("No context found with primary searches, trying fallback strategies")
            
            # Fallback 1: Try with very low threshold
            await callback_queue.put({
                "type": "context_search",
                "source": "low threshold fallback",
                "query": "broader search"
            })
            
            try:
                fallback_chunks = get_vector_store().query(search_query, k=3, min_relevance=0.1)
                
                # Apply project filtering to fallback search results
                fallback_chunks = filter_by_projects(fallback_chunks)
                
                if fallback_chunks:
                    logger.info(f"Fallback search found {len(fallback_chunks)} chunks (after project filtering)")
                    context_summary = self._build_context_summary(fallback_chunks, query)
                    
                    await callback_queue.put({
                        "type": "context_gathering_complete",
                        "sources": ["Low threshold fallback"],
                        "chunks_found": len(fallback_chunks),
                        "files_analyzed": len(set(path for path, _ in fallback_chunks)),
                        "fallback_used": True
                    })
                    
                    return context_summary
            except Exception as e:
                logger.error(f"Fallback search failed: {e}")
            
            # Fallback 2: Directory-based search
            await callback_queue.put({
                "type": "context_search",
                "source": "directory fallback",
                "query": "project structure"
            })
            
            directory_summary = self._get_directory_fallback_summary(query)
            if directory_summary:
                await callback_queue.put({
                    "type": "context_gathering_complete",
                    "sources": ["Directory structure analysis"],
                    "chunks_found": 0,
                    "files_analyzed": 0,
                    "fallback_used": True
                })
                return f"DIRECTORY CONTEXT:\n{directory_summary}"
            
            # Final fallback: No context found
            await callback_queue.put({
                "type": "context_gathering_complete",
                "sources": [],
                "chunks_found": 0,
                "files_analyzed": 0,
                "no_context": True
            })
            
            logger.error(f"All context search strategies failed. Search attempts: {'; '.join(search_attempts)}")
            return "No relevant context found in the codebase after trying multiple search strategies."
    
    def _build_context_summary(self, chunks: List[tuple], query: str) -> str:
        """Build a comprehensive context summary from retrieved chunks with citations"""
        logger.info(f"Building context summary from {len(chunks)} chunks")
        
        # Group chunks by file
        files_context = {}
        citation_map = {}  # Map for citation references
        citation_counter = 1
        
        for file_path, snippet in chunks:
            if file_path not in files_context:
                files_context[file_path] = []
            files_context[file_path].append(snippet)
            
            # Create citation reference
            citation_key = f"[{citation_counter}]"
            citation_map[citation_key] = file_path
            citation_counter += 1
        
        # Build formatted context with citation markers
        context_parts = []
        context_parts.append(f"RELEVANT CONTEXT (from {len(files_context)} files):")
        context_parts.append("\nIMPORTANT: When answering, reference sources using [1], [2], etc. format")
        
        citation_counter = 1
        for file_path, snippets in files_context.items():
            context_parts.append(f"\n=== [{citation_counter}] {file_path} ===")
            for i, snippet in enumerate(snippets, 1):
                if len(snippets) > 1:
                    context_parts.append(f"Chunk {i}:")
                context_parts.append(f"```\n{snippet}\n```")
            citation_counter += 1
        
        # Add citation reference guide
        context_parts.append(f"\nCITATION REFERENCES:")
        for citation_key, file_path in citation_map.items():
            context_parts.append(f"{citation_key} = {file_path}")
        
        context_parts.append(f"\nUSER QUERY: {query}")
        context_parts.append(f"\nINSTRUCTION: Always include citations [1], [2], etc. when referencing specific code or information from the files above.")
        
        return "\n".join(context_parts)

    def _search_code_with_summary(self, query: str) -> str:
        """Search code and return both raw results and summary"""
        if self.hybrid_search is None:
            return f"Code search not available. Using fallback directory summary: {self._get_directory_fallback_summary(query)}"
        
        try:
            results = self.hybrid_search.search(query, k=10)
            if not results:
                return f"No code snippets found for query: {query}. Using fallback directory summary: {self._get_directory_fallback_summary(query)}"
            
            # Generate context summary using context delivery system
            if self.context_delivery:
                summary = self.context_delivery.get_enhanced_context(query)
            else:
                # Fallback to simple concatenation
                context_parts = []
                for i, result in enumerate(results[:5], 1):
                    rel_path = result.metadata.get('relative_path', result.metadata.get('file_path', 'unknown'))
                    snippet = result.snippet[:500] + "..." if len(result.snippet) > 500 else result.snippet
                    context_parts.append(f"[{i}] FILE: {rel_path}\n{snippet}")
                summary = "\n\n".join(context_parts)
            
            return summary
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Search failed: {str(e)}. Using fallback: {self._get_directory_fallback_summary(query)}"
    
    async def _handle_find_files(self, input_str: str) -> str:
        """Handle find_files tool with JSON parameter parsing"""
        try:
            params = json.loads(input_str)
            pattern = params.get("pattern", "*")
            directory = params.get("directory", ".")
            return await self.mcp_wrapper.find_files(pattern, directory)
        except json.JSONDecodeError:
            # Fallback: treat as simple pattern
            return await self.mcp_wrapper.find_files(input_str, ".")
    
    async def _handle_grep_search(self, input_str: str) -> str:
        """Handle grep_search tool with JSON parameter parsing"""
        try:
            params = json.loads(input_str)
            pattern = params.get("pattern", "")
            directory = params.get("directory", ".")
            file_type = params.get("file_type", "*")
            return await self.mcp_wrapper.grep_search(pattern, directory, file_type)
        except json.JSONDecodeError:
            # Fallback: treat as simple pattern
            return await self.mcp_wrapper.grep_search(input_str, ".", "*")
    
    async def _handle_explore_tree(self, input_str: str) -> str:
        """Handle explore_directory_tree tool with JSON parameter parsing"""
        try:
            params = json.loads(input_str)
            directory = params.get("directory", ".")
            max_depth = params.get("max_depth", 3)
            return await self.mcp_wrapper.explore_directory_tree(directory, max_depth)
        except json.JSONDecodeError:
            # Fallback: treat as simple directory
            return await self.mcp_wrapper.explore_directory_tree(input_str, 3)
    
    async def _handle_search_by_extension(self, input_str: str) -> str:
        """Handle search_by_extension tool with JSON parameter parsing"""
        try:
            params = json.loads(input_str)
            extensions = params.get("extensions", [])
            directory = params.get("directory", ".")
            return await self.mcp_wrapper.search_by_extension(extensions, directory)
        except json.JSONDecodeError:
            # Fallback: treat as simple extension (e.g., "py" or "*.py")
            clean_ext = input_str.lstrip("*.")
            return await self.mcp_wrapper.search_by_extension([clean_ext], ".")
    
    async def _handle_search_file_content(self, input_str: str) -> str:
        """Handle search_file_content tool with JSON parameter parsing"""
        try:
            params = json.loads(input_str)
            pattern = params.get("pattern", "")
            directory = params.get("directory", ".")
            file_types = params.get("file_types", None)
            context_lines = params.get("context_lines", 2)
            return await self.mcp_wrapper.search_file_content(pattern, directory, file_types, context_lines)
        except json.JSONDecodeError:
            # Fallback: treat as simple search pattern
            return await self.mcp_wrapper.search_file_content(input_str, ".", None, 2)
    
    async def _handle_get_project_structure(self, input_str: str) -> str:
        """Handle get_project_structure tool with JSON parameter parsing"""
        try:
            params = json.loads(input_str)
            directory = params.get("directory", ".")
            max_depth = params.get("max_depth", 4)
            include_files = params.get("include_files", True)
            project_name = params.get("project_name")
            return await self.enhanced_structure.analyze_project(directory, max_depth, include_files, project_name)
        except json.JSONDecodeError:
            # Fallback: treat as simple directory path
            return await self.enhanced_structure.analyze_project(input_str, 4, True)
    
    async def _handle_find_related_files(self, input_str: str) -> str:
        """Handle find_related_files tool with JSON parameter parsing"""
        try:
            params = json.loads(input_str)
            file_path = params.get("file_path", "")
            relationship_types = params.get("relationship_types", ["import", "test", "config"])
            return await self.mcp_wrapper.find_related_files(file_path, relationship_types)
        except json.JSONDecodeError:
            # Fallback: treat as simple file path
            return await self.mcp_wrapper.find_related_files(input_str, ["import", "test", "config"])

    def _get_directory_fallback_summary(self, query: str) -> str:
        """Get directory-based summary when vector search is insufficient."""
        try:
            import os
            from pathlib import Path
            
            workspace_dir = "/workspace"
            workspace_path = Path(workspace_dir)
            
            if not workspace_path.exists():
                return ""
            
            # Look for directories that might match the query
            relevant_dirs = []
            query_lower = query.lower()
            
            for item in workspace_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # Check if directory name is relevant to query
                    if (query_lower in item.name.lower() or 
                        item.name.lower() in query_lower or
                        any(word in item.name.lower() for word in query_lower.split())):
                        relevant_dirs.append(item)
            
            if not relevant_dirs:
                # Fallback to common directories
                common_dirs = ['src', 'app', 'lib', 'components', 'utils', 'api', 'services']
                for dir_name in common_dirs:
                    dir_path = workspace_path / dir_name
                    if dir_path.exists() and dir_path.is_dir():
                        relevant_dirs.append(dir_path)
                        break
            
            if not relevant_dirs:
                return ""
            
            # Generate summary for the most relevant directory
            target_dir = relevant_dirs[0]
            relative_path = str(target_dir.relative_to(workspace_path))
            
            # Get directory structure
            files = []
            dirs = []
            
            for item in target_dir.iterdir():
                if item.is_file() and not item.name.startswith('.'):
                    files.append(item.name)
                elif item.is_dir() and not item.name.startswith('.'):
                    dirs.append(item.name)
            
            # Create structure info
            structure_info = []
            if dirs:
                structure_info.append(f"Directories: {', '.join(dirs[:5])}")
            if files:
                structure_info.append(f"Files: {', '.join(files[:10])}")
            
            structure = "; ".join(structure_info)
            
            prompt_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a code analysis assistant. Given a directory structure and user query, "
                        "provide a helpful summary of what this directory likely contains and how it "
                        "relates to the user's query. Keep response under 200 tokens."
                    )
                },
                {
                    "role": "user",
                    "content": f"User Query: {query}\n\nDirectory: {relative_path}\nStructure: {structure}\n\nProvide a directory summary:"
                }
            ]
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=prompt_messages,
                max_tokens=200,
                temperature=0.1,
            )
            
            return f"Directory '{relative_path}' summary: {response.choices[0].message.content.strip()}"
            
        except Exception as e:
            return f"Directory fallback failed: {str(e)}"

    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools from MCP wrapper methods"""
        return [
            Tool(
                name="read_file",
                description="Read the contents of a file. Provide the file path relative to the workspace. If the file doesn't exist, use list_files to explore the directory structure first.",
                func=lambda x: asyncio.run(self.mcp_wrapper.read_file(x)),
                coroutine=self.mcp_wrapper.read_file
            ),
            Tool(
                name="code_search",
                description="Retrieve up to 10 relevant code snippets from the project for a natural language query. Use this for initial exploration and to find files related to your query. Always try multiple search terms if first search doesn't yield enough results.",
                func=lambda x: self._search_code_with_summary(x),
                coroutine=lambda x: asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._search_code_with_summary(x),
                ),
            ),
            Tool(
                name="file_glimpse",
                description="Same as read_file but optimized for quickly scanning file contents. Use this when you need to check multiple files rapidly or when read_file fails.",
                func=lambda x: asyncio.run(self.mcp_wrapper.read_file(x)),
                coroutine=self.mcp_wrapper.read_file
            ),
            Tool(
                name="list_files",
                description="List files in a directory. Always use this first when exploring an unknown project structure. Provide the directory path relative to the workspace. Use '.' for root workspace.",
                func=lambda x: asyncio.run(self.mcp_wrapper.list_files(x)),
                coroutine=self.mcp_wrapper.list_files
            ),
            Tool(
                name="list_entities",
                description="List all directories and files in a path recursively. Use this to get a comprehensive view of project structure when list_files isn't enough.",
                func=lambda x: asyncio.run(self.mcp_wrapper.list_files(x)),
                coroutine=self.mcp_wrapper.list_files
            ),
            Tool(
                name="write_file",
                description="Write content to a file. Provide a JSON string with 'file_path' and 'content' keys.",
                func=lambda x: asyncio.run(self.mcp_wrapper.write_file(**json.loads(x))),
                coroutine=lambda x: self.mcp_wrapper.write_file(**json.loads(x))
            ),
            Tool(
                name="run_command",
                description="Run a shell command. Use for running tests, linters, or installing packages. Also useful for finding files with commands like 'find . -name \"*.py\" | head -20'.",
                func=lambda x: asyncio.run(self.mcp_wrapper.run_command(x)),
                coroutine=self.mcp_wrapper.run_command
            ),
            Tool(
                name="find_files",
                description="Find files matching a pattern using shell commands. Use JSON format: {\"pattern\": \"*.py\", \"directory\": \".\"}. Returns up to 20 results.",
                func=lambda x: asyncio.run(self._handle_find_files(x)),
                coroutine=lambda x: self._handle_find_files(x)
            ),
            Tool(
                name="grep_search",
                description="Search for text patterns across files. Use JSON format: {\"pattern\": \"search_term\", \"directory\": \".\", \"file_type\": \"py\"}. Returns up to 20 results.",
                func=lambda x: asyncio.run(self._handle_grep_search(x)),
                coroutine=lambda x: self._handle_grep_search(x)
            ),
            Tool(
                name="get_file_info",
                description="Get detailed file information (size, type, permissions). Provide a file path as a simple string.",
                func=lambda x: asyncio.run(self.mcp_wrapper.get_file_info(x)),
                coroutine=self.mcp_wrapper.get_file_info
            ),
            Tool(
                name="explore_directory_tree",
                description="Get a tree view of directory structure. Use JSON format: {\"directory\": \".\", \"max_depth\": 3}. Returns up to 30 directory entries.",
                func=lambda x: asyncio.run(self._handle_explore_tree(x)),
                coroutine=lambda x: self._handle_explore_tree(x)
            ),
            Tool(
                name="search_by_extension",
                description="Search for files by specific extension patterns with project composition analysis. Use JSON format: {\"extensions\": [\"py\", \"js\"], \"directory\": \"Gymmy\"} for specific projects, or simple string like 'py' for all Python files. Automatically searches /workspace. Returns up to 100 results with composition stats.",
                func=lambda x: asyncio.run(self._handle_search_by_extension(x)),
                coroutine=lambda x: self._handle_search_by_extension(x)
            ),
            Tool(
                name="search_file_content",
                description="Search for specific text patterns within files, including context lines. Use JSON format: {\"pattern\": \"search_term\", \"directory\": \".\", \"file_types\": [\"py\", \"js\"], \"context_lines\": 2}. Returns up to 50 lines of context per match.",
                func=lambda x: asyncio.run(self._handle_search_file_content(x)),
                coroutine=lambda x: self._handle_search_file_content(x)
            ),
            Tool(
                name="get_project_structure",
                description="Generate enhanced project structure analysis with framework detection, @ annotations for codebase highlighting, and context awareness. Use JSON format: {\"directory\": \".\", \"max_depth\": 4, \"include_files\": true, \"project_name\": \"ProjectName\"}. Returns comprehensive analysis with framework detection, entry points, and structured tree view.",
                func=lambda x: asyncio.run(self._handle_get_project_structure(x)),
                coroutine=lambda x: self._handle_get_project_structure(x)
            ),
            Tool(
                name="find_related_files",
                description="Find files related to a given file through various relationships (imports, tests, configs, naming patterns). Use JSON format: {\"file_path\": \"path/to/file.py\", \"relationship_types\": [\"import\", \"test\", \"config\"]}. Returns a formatted summary of related files.",
                func=lambda x: asyncio.run(self._handle_find_related_files(x)),
                coroutine=lambda x: self._handle_find_related_files(x)
            )
        ]
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """<role>
Expert software engineer and code architect specializing in full-stack development, system design, and code optimization.
</role>

<expertise>
- Full-stack development (frontend, backend, databases)
- System architecture and design patterns
- Code optimization and performance tuning
- DevOps, containerization, and deployment
- Security best practices and vulnerability assessment
- API design and integration
- Testing strategies and debugging
</expertise>

<capabilities>
Available tools for direct code manipulation and exploration:

**Core File Operations:**
1. read_file - Analyze existing code, configurations, and documentation
2. file_glimpse - Quick file content scanning
3. write_file - Create new files or modify existing code
4. list_files - Explore project structure and codebase organization
5. run_command - Execute commands for testing, building, package management

**Phase 1 Enhanced Discovery Tools:**
6. search_by_extension - Find files by extension patterns with project composition analysis
7. search_file_content - Search for text patterns within files with context lines
8. get_project_structure - Generate intelligent project tree with framework detection
9. find_related_files - Discover file relationships (imports, tests, configs, naming patterns)

**Advanced Search & Analysis:**
10. code_search - Search for relevant code snippets using natural language queries
11. list_entities - Get comprehensive directory listings
12. find_files - Find files matching patterns (e.g., "*.py", "urls*")
13. grep_search - Search for text patterns across multiple files
14. get_file_info - Get detailed file information (size, type, permissions)
15. explore_directory_tree - Get tree view of directory structure with depth control
</capabilities>

<response_guidelines>
- Be direct and technically precise
- Provide working code solutions, not just explanations
- Challenge inefficient or insecure approaches when spotted
- Focus on practical implementation over theory
- Don't over-explain basic programming concepts
- Give specific, actionable recommendations
- When creating code, make it production-ready with proper error handling
- **CRITICAL EXPLORATION RULE**: If initial context doesn't contain sufficient information, you MUST use tools aggressively to explore:
  1. Use list_files to understand project structure
  2. Use code_search with multiple different search terms
  3. Use read_file to examine relevant files found through exploration
  4. Use run_command for file discovery (find, grep, etc.)
- Never give up after one failed file lookup - explore alternative paths and file locations
- Always assume project names mentioned by the user correspond to repositories or directories in the accessible workspace
- When referencing code or information from files, ALWAYS include citations using [1], [2], etc. format
- For every new user question, extract key nouns/noun phrases and explore them thoroughly before answering
</response_guidelines>

<communication_style>
- Concise and solution-oriented
- Assume user has technical competence
- Point out potential issues or better alternatives
- Not overly agreeable - provide honest technical assessment
- Use technical terminology appropriately
- Prioritize working code examples over lengthy descriptions
</communication_style>

<execution_approach>
1. Analyze the request and identify the core technical challenge
2. Use tools to gather comprehensive context before responding
3. Plan the most efficient implementation approach
4. Provide working, tested code solutions
</execution_approach>"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=25,
            early_stopping_method="generate"
        )
    
    async def process_request(self, user_query: str, chat_history=None, mentioned_projects: List[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a user request and yield updates"""
        
        # Set up project context isolation
        project_name = "workspace"  # Default
        if mentioned_projects and len(mentioned_projects) > 0:
            project_name = mentioned_projects[0]  # Use first mentioned project as primary
        
        # Set project context to prevent cross-contamination
        context = set_project_context(project_name, mentioned_projects)
        logger.info(f"Set project context: {project_name} (mentioned: {mentioned_projects})")
        
        # Add search query to context history
        get_context_manager().add_search_to_context(user_query)
        
        # Route to appropriate agent implementation
        if self.use_native_cerebras:
            # Use native Cerebras agent
            async for update in self.cerebras_agent.process_request(user_query, chat_history, mentioned_projects):
                yield update
        else:
            # Use LangChain agent (existing implementation)
            queue = asyncio.Queue()
            callback_handler = StreamingCallbackHandler(queue)
            
            # Start the agent execution in a separate task
            async def run_agent():
                try:
                    # Auto-search context before LLM processing
                    logger.info("Starting auto-context retrieval before LLM processing")
                    
                    # Pass mentioned projects to context search for filtering
                    context_summary = await self.auto_search_context(
                        user_query, 
                        queue, 
                        chat_history, 
                        mentioned_projects=mentioned_projects
                    )
                    
                    # Build enhanced input with context
                    enhanced_input = f"""INITIAL CONTEXT: The following information was found through automatic search. If this context doesn't fully address the user's question, you MUST use the available tools to explore further.

{context_summary}

IMPORTANT: 
- Use the above context as a starting point, but don't limit yourself to it
- If the context is insufficient, use tools like list_files, code_search, read_file, and run_command to explore further
- Include citations [1], [2], etc. for every piece of information you reference from files
- Be thorough in your exploration before concluding that information doesn't exist

User Query: {user_query}"""
                    
                    kwargs = {"input": enhanced_input}
                    if chat_history is not None:
                        kwargs["chat_history"] = chat_history
                    
                    result = await self.agent.ainvoke(
                        kwargs,
                        callbacks=[callback_handler]
                    )
                    # Cost estimation based on token usage
                    usage = None
                    if isinstance(result, dict):
                        usage = result.get("usage") or result.get("generation_info", {}).get("usage")

                    cost_str = ""
                    if usage:
                        model_name = self.llm.model
                        input_cost, output_cost = 0.0, 0.0
                        # Pricing in USD per 1K tokens (example values)
                        pricing = {
                            "gpt-4o": (0.005, 0.015),
                            "gpt-4o-mini": (0.01, 0.03),
                            "gpt-4-turbo-preview": (0.01, 0.03),
                            "o3": (0.005, 0.015),
                            "gpt-3.5-turbo-0125": (0.0005, 0.0015),
                        }
                        rate_in, rate_out = pricing.get(model_name, (0, 0))
                        prompt_tokens = usage.get("prompt_tokens", 0) if usage else 0
                        if usage:
                            completion_tokens = usage.get("completion_tokens", 0)
                        else:
                            completion_tokens = callback_handler.completion_tokens
                        input_cost = prompt_tokens / 1000 * rate_in
                        output_cost = completion_tokens / 1000 * rate_out
                        total_cost = input_cost + output_cost
                        cost_str = f"Cost: ${total_cost:.4f} (prompt {prompt_tokens}, completion {completion_tokens})\n\n"

                    await queue.put({
                        "type": "final_result",
                        "output": cost_str + result.get("output", "Task completed")
                    })
                except Exception as e:
                    logger.error(f"Agent error: {str(e)}")
                    await queue.put({
                        "type": "error",
                        "message": f"Agent error: {str(e)}"
                    })
                finally:
                    await queue.put(None)  # Signal completion
            
            # Start agent execution
            agent_task = asyncio.create_task(run_agent())
            
            # Yield updates from the queue
            while True:
                update = await queue.get()
                if update is None:
                    break
                yield update
            
            # Ensure the agent task is complete
            await agent_task 