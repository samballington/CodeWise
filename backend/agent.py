import asyncio
import os
import openai
from typing import AsyncGenerator, Dict, Any, List
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langchain.tools import Tool
from langchain.callbacks.base import AsyncCallbackHandler
import httpx
import json
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache, SQLiteCache
from vector_store import get_vector_store

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
        await self.queue.put({
            "type": "tool_start",
            "tool": serialized.get("name", "unknown"),
            "input": input_str
        })
    
    async def on_tool_end(self, output: str, **kwargs):
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

class CodeWiseAgent:
    """Main agent class for CodeWise"""
    
    def __init__(self, openai_api_key: str, mcp_server_url: str):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4-turbo-preview",
            temperature=0,
            streaming=True
        )
        self.mcp_wrapper = MCPToolWrapper(mcp_server_url)
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def _search_code_with_summary(self, query: str) -> str:
        """Search code and provide both raw snippets and summarized analysis with directory fallback."""
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
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=prompt_messages,
                max_tokens=150,
                temperature=0.1,
            )
            summary = response.choices[0].message.content.strip()
            
            return f"SUMMARY: {summary}\n\nRAW SNIPPETS:\n{raw_snippets}"
            
        except Exception as e:
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
    
    async def process_request(self, user_query: str, chat_history=None) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a user request and yield updates"""
        queue = asyncio.Queue()
        callback_handler = StreamingCallbackHandler(queue)
        
        # Start the agent execution in a separate task
        async def run_agent():
            try:
                kwargs = {"input": user_query}
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