import asyncio
from typing import AsyncGenerator, Dict, Any, List
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langchain.tools import Tool
from langchain.callbacks.base import AsyncCallbackHandler
import httpx
import json

class StreamingCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming agent updates"""
    
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
    
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
            ("system", """You are CodeWise, an AI development assistant that helps developers with their coding tasks.
            
You can:
1. Read and analyze existing code files
2. Write new code or modify existing files
3. Run commands to test code, install packages, or run linters
4. Break down complex tasks into smaller steps
5. Verify your work after completing tasks

Always:
- Plan your approach before executing
- Explain what you're doing at each step
- Verify your changes work correctly
- Handle errors gracefully and try alternative approaches"""),
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
    
    async def process_request(self, user_query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a user request and yield updates"""
        queue = asyncio.Queue()
        callback_handler = StreamingCallbackHandler(queue)
        
        # Start the agent execution in a separate task
        async def run_agent():
            try:
                result = await self.agent.ainvoke(
                    {"input": user_query},
                    callbacks=[callback_handler]
                )
                await queue.put({
                    "type": "final_result",
                    "output": result.get("output", "Task completed")
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