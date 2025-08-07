import os
import asyncio
import pytest

from agent import CodeWiseAgent

@pytest.mark.asyncio
async def test_tools_callable(tmp_path, monkeypatch):
    """Ensure code_search, file_glimpse, list_entities tools execute without error."""
    # Prepare dummy workspace file for file_glimpse using tmp_path
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    dummy_path = workspace_dir / "test_dummy.txt"
    dummy_path.write_text("hello world\n" * 5)
    
    # Set workspace environment variable for the agent
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace_dir))

    agent = CodeWiseAgent(openai_api_key="", mcp_server_url="http://mcp_server:8001")

    # Map tool names to callables - use coroutine if available, else sync func
    tool_map = {}
    for tool in agent.tools:
        if hasattr(tool, 'coroutine') and tool.coroutine:
            tool_map[tool.name] = tool.coroutine
        else:
            tool_map[tool.name] = tool.func

    # code_search should return string (even if "No results")
    if "code_search" in tool_map:
        func = tool_map["code_search"]
        if asyncio.iscoroutinefunction(func):
            code_search_result = await func("dummy search")
        else:
            code_search_result = func("dummy search")
        assert isinstance(code_search_result, str)

    # file_glimpse should return file content
    if "file_glimpse" in tool_map:
        func = tool_map["file_glimpse"]
        if asyncio.iscoroutinefunction(func):
            glimpse_result = await func("test_dummy.txt")
        else:
            glimpse_result = func("test_dummy.txt")
        assert "hello world" in glimpse_result

    # list_entities returns str (may be empty)
    if "list_entities" in tool_map:
        func = tool_map["list_entities"]
        if asyncio.iscoroutinefunction(func):
            entities_result = await func(None)
        else:
            entities_result = func(None)
        assert isinstance(entities_result, str) 