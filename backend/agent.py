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
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache, SQLiteCache
from vector_store import get_vector_store
from api_providers import get_provider_manager
from hybrid_search import HybridSearchEngine
from context_delivery import ContextDeliverySystem

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
        
        # Helper function to filter results by mentioned projects
        def filter_by_projects(chunks: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
            if not mentioned_projects:
                return chunks
            
            filtered_chunks = []
            for file_path, snippet in chunks:
                # Check if the file path belongs to any of the mentioned projects
                should_include = False
                
                for project in mentioned_projects:
                    if project == "workspace":
                        # For workspace, include files that are directly in workspace root
                        # (not in any project subdirectory)
                        path_parts = file_path.split('/')
                        if len(path_parts) == 1 or not any(
                            file_path.startswith(f"{known_project}/") 
                            for known_project in ["SWE_Project", "fastapi", "sqlmodel"]
                        ):
                            should_include = True
                            break
                    else:
                        # For named projects, include files that are within the project directory
                        # This handles both direct files and nested structures
                        if file_path.startswith(f"{project}/"):
                            should_include = True
                            break
                            
                        # Special handling for known nested project structures
                        if project == "SWE_Project":
                            # Include files from the nested Java project structure
                            if (file_path.startswith("SWE_Project/obs/") or 
                                file_path.startswith("SWE_Project/.") or
                                file_path == "SWE_Project/README.md" or
                                file_path == "SWE_Project/Dockerfile" or
                                file_path == "SWE_Project/nixpacks.toml"):
                                should_include = True
                                break
                
                if should_include:
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
        """Search code and provide both raw snippets and summarized analysis with directory fallback."""
        logger.info(f"Manual code search triggered for: {query}")
        
        # Use hybrid search if available, otherwise fallback to vector search
        if self.hybrid_search:
            search_results = self.hybrid_search.search(query, k=5)
            chunks = [(result.file_path, result.snippet) for result in search_results]
        else:
            chunks = get_vector_store().query(query)
        
        # Check if we have good results or need directory fallback
        if not chunks or len(chunks) < 2:
            # Try directory-based fallback
            directory_summary = self._get_directory_fallback_summary(query)
            if directory_summary:
                return f"DIRECTORY SUMMARY (limited vector results):\n{directory_summary}"
        
        if not chunks:
            return "No relevant code found."
        
        # Raw snippets
        raw_snippets = "\n\n".join(f"{path}:\n{snippet}" for path, snippet in chunks)
        
        # Create simple summary
        try:
            context_parts = []
            for file_path, snippet in chunks:
                context_parts.append(f"File: {file_path}\n```\n{snippet}\n```")
            
            context = "\n\n".join(context_parts)
            
            prompt_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a code analysis assistant. Given code snippets and a user query, "
                        "provide a concise technical summary that directly addresses the query. "
                        "Focus on key functionality, patterns, and relevant implementation details. "
                        "Keep response under 150 tokens."
                    )
                },
                {
                    "role": "user", 
                    "content": f"User Query: {query}\n\nCode Context:\n{context}\n\nProvide a concise summary:"
                }
            ]
            
            # Use the new OpenAI client
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=prompt_messages,
                max_tokens=150,
                temperature=0.1,
            )
            summary = response.choices[0].message.content.strip()
            
            return f"SUMMARY: {summary}\n\nRAW SNIPPETS:\n{raw_snippets}"
            
        except Exception as e:
            logger.error(f"Error generating code summary: {e}")
            # Fallback to just raw snippets
            return f"SUMMARY: Found {len(chunks)} relevant snippets across files: {', '.join(set(path for path, _ in chunks))}\n\nRAW SNIPPETS:\n{raw_snippets}"
    
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
                description="Read the contents of a file. Provide the file path relative to the workspace.",
                func=lambda x: asyncio.run(self.mcp_wrapper.read_file(x)),
                coroutine=self.mcp_wrapper.read_file
            ),
            Tool(
                name="search_code",
                description="Retrieve up to 3 relevant code snippets from the project for a natural language query. Returns both raw snippets and a summarized analysis.",
                func=lambda x: self._search_code_with_summary(x),
                coroutine=lambda x: asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._search_code_with_summary(x),
                ),
            ),
            Tool(
                name="write_file",
                description="Write content to a file. Provide a JSON string with 'file_path' and 'content' keys.",
                func=lambda x: asyncio.run(self.mcp_wrapper.write_file(**json.loads(x))),
                coroutine=lambda x: self.mcp_wrapper.write_file(**json.loads(x))
            ),
            Tool(
                name="list_files",
                description="List files in a directory. Provide the directory path relative to the workspace.",
                func=lambda x: asyncio.run(self.mcp_wrapper.list_files(x)),
                coroutine=self.mcp_wrapper.list_files
            ),
            Tool(
                name="run_command",
                description="Run a shell command. Use for running tests, linters, or installing packages.",
                func=lambda x: asyncio.run(self.mcp_wrapper.run_command(x)),
                coroutine=self.mcp_wrapper.run_command
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
Available tools for direct code manipulation:
1. read_file - Analyze existing code, configurations, and documentation
2. write_file - Create new files or modify existing code
3. list_files - Explore project structure and codebase organization
4. run_command - Execute commands for testing, building, package management
</capabilities>

<response_guidelines>
- Be direct and technically precise
- Provide working code solutions, not just explanations
- Challenge inefficient or insecure approaches when spotted
- Focus on practical implementation over theory
- Don't over-explain basic programming concepts
- Give specific, actionable recommendations
- When creating code, make it production-ready with proper error handling
- Always assume project names mentioned by the user correspond to repositories or directories in the accessible workspace. If a name is unknown, search the workspace (using `search_code`, `list_files`, etc.) for closest matches (typo-tolerant) before answering generically.
- CRITICAL: When referencing code or information from files, ALWAYS include citations using [1], [2], etc. format to show exactly which files the information came from.
- In every response include the model name used in parentheses right after the section heading, e.g. "Plan (o3):" for planning steps generated by the planner model and "Answer (gpt-4o):" for the final detailed reply.
- For every new user question, extract key nouns/noun phrases (potential project or file names) and call the `search_code` tool on each before formulating the answer.
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
2. Plan the most efficient implementation approach
3. Execute with proper verification and error handling
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
            max_iterations=10
        )
    
    async def process_request(self, user_query: str, chat_history=None, mentioned_projects: List[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a user request and yield updates"""
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
                enhanced_input = f"""CRITICAL INSTRUCTION: You MUST use ONLY the information provided in the context below. Do NOT provide generic information. If the context doesn't contain relevant information, say "I don't see relevant information in the codebase for this query."

{context_summary}

IMPORTANT: Base your answer EXCLUSIVELY on the code and information shown above. Include citations [1], [2], etc. for every piece of information you reference.

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